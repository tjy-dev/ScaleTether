from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scaletether.framework_counterfactual as framework_counterfactual
from scaletether.config import Architecture, NetworkTier, Parallelism, Topology
from scaletether.framework_counterfactual import (
    CONTRACT_SCHEMA,
    FrameworkCounterfactualError,
    compile_framework_counterfactual,
    search_main,
)
from scaletether.pipeline import MEGATRON_CORE_SCHEDULE_COMMIT
from scaletether.schema import TraceEvent, WorkloadTrace
from scaletether.simulator import simulate


class FrameworkCounterfactualTest(unittest.TestCase):
    @staticmethod
    def _gradient_gemm(
        event_id: str, *dependencies: str, split_k_reduction: bool = False
    ) -> TraceEvent:
        name = (
            "void cublasLt::splitKreduce_kernel<32, 16, int, float>"
            if split_k_reduction
            else event_id
        )
        return TraceEvent(
            event_id,
            name,
            "compute",
            1.0,
            dependencies=dependencies,
            metadata={
                "kernel_launch_payload": {
                    "framework_operator": {
                        "name": "aten::mm",
                        "input_dims": [[256, 224], [224, 256]],
                        "input_strides": [[1, 256], [256, 1]],
                        "output_shape": [256, 256],
                    }
                }
            },
        )

    def test_gradient_ready_selects_split_k_chain_terminals(self) -> None:
        events = [
            self._gradient_gemm("main-0"),
            self._gradient_gemm("reduce-0", "main-0", split_k_reduction=True),
            self._gradient_gemm("main-1"),
            self._gradient_gemm("reduce-1", "main-1", split_k_reduction=True),
        ]
        selected = framework_counterfactual._parameter_gradient_ready_chains(events)
        self.assertEqual(
            [(event.id, chain) for event, chain in selected],
            [
                ("reduce-0", ("main-0", "reduce-0")),
                ("reduce-1", ("main-1", "reduce-1")),
            ],
        )

    def test_gradient_ready_accepts_two_single_kernel_chains(self) -> None:
        events = [self._gradient_gemm("gemm-0"), self._gradient_gemm("gemm-1")]
        selected = framework_counterfactual._parameter_gradient_ready_chains(events)
        self.assertEqual(
            [(event.id, chain) for event, chain in selected],
            [("gemm-0", ("gemm-0",)), ("gemm-1", ("gemm-1",))],
        )

    def test_gradient_ready_excludes_square_row_major_activation_gemm(self) -> None:
        unrelated = self._gradient_gemm("activation-gemm")
        launch = dict(unrelated.metadata["kernel_launch_payload"])
        operator = dict(launch["framework_operator"])
        operator["input_strides"] = [[256, 1], [256, 1]]
        launch["framework_operator"] = operator
        unrelated = replace(
            unrelated, metadata={"kernel_launch_payload": launch}
        )
        selected = framework_counterfactual._parameter_gradient_ready_chains(
            [unrelated, self._gradient_gemm("grad-0"), self._gradient_gemm("grad-1")]
        )
        self.assertEqual([event.id for event, _ in selected], ["grad-0", "grad-1"])

    def test_gradient_ready_rejects_ambiguous_split_k_fork(self) -> None:
        events = [
            self._gradient_gemm("main-0"),
            self._gradient_gemm("reduce-0a", "main-0", split_k_reduction=True),
            self._gradient_gemm("reduce-0b", "main-0", split_k_reduction=True),
            self._gradient_gemm("main-1"),
        ]
        with self.assertRaisesRegex(
            FrameworkCounterfactualError, "not a set of linear chains"
        ):
            framework_counterfactual._parameter_gradient_ready_chains(events)

    @staticmethod
    def _trace() -> WorkloadTrace:
        events = (
            TraceEvent("fwd-gemm", "measured forward GEMM", "compute", 10.0),
            TraceEvent(
                "fwd-tp",
                "measured forward TP AllReduce",
                "collective",
                0.0,
                dependencies=("fwd-gemm",),
                collective="all_reduce",
                message_bytes=4096,
                group_role="tp",
                group_size=None,
                metadata={"observed_group_size": 2},
            ),
            TraceEvent("bwd-gemm", "measured backward GEMM", "compute", 20.0),
            TraceEvent(
                "bwd-tp",
                "measured backward TP AllReduce",
                "collective",
                0.0,
                dependencies=("bwd-gemm",),
                collective="all_reduce",
                message_bytes=4096,
                group_role="tp",
                group_size=None,
                metadata={"observed_group_size": 2},
            ),
            TraceEvent("optimizer", "measured optimizer", "compute", 3.0),
        )
        contract = {
            "schema": CONTRACT_SCHEMA,
            "framework": "megatron-core",
            "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
            "model": {
                "num_layers": 4,
                "micro_batch_size": 2,
                "global_batch_size": 8,
            },
            "schedule": {"type": "1f1b"},
            "tp_profiles": {
                "2": {
                    "target": "h100",
                    "measurement": {"status": "measured", "sample_count": 5},
                    "forward_event_ids": ["fwd-gemm", "fwd-tp"],
                    "backward_event_ids": ["bwd-gemm", "bwd-tp"],
                    "optimizer_event_ids": ["optimizer"],
                    "pipeline_activation_bytes_per_tp_rank": 8192,
                    "gradient_sync": {
                        "collective": "reduce_scatter",
                        "bytes_per_layer_per_rank": 16384,
                    },
                }
            },
        }
        return WorkloadTrace(
            events=events,
            source={"target": "h100", "kind": "measured-primitives"},
            metadata={"framework_counterfactual": contract},
        )

    def test_compiles_exact_tp_pp_dp_plan_without_candidate_execution(self) -> None:
        result = compile_framework_counterfactual(
            self._trace(), Parallelism(tp=2, pp=2, dp=2), 8, "h100"
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.summary["microbatches"], 2)
        self.assertEqual(result.summary["layers_per_stage"], 2)
        self.assertEqual({event.rank for event in result.trace.events}, set(range(8)))
        self.assertEqual(
            len({event.id for event in result.trace.events}), len(result.trace.events)
        )
        dp_events = [
            event for event in result.trace.events if event.group_role == "dp"
        ]
        self.assertEqual(len(dp_events), 8)
        self.assertTrue(
            all(event.collective == "reduce_scatter" for event in dp_events)
        )
        self.assertTrue(all(event.message_bytes == 32768 for event in dp_events))
        self.assertTrue(
            all(
                event.metadata["gradient_accumulation_boundary"]
                == "after-all-microbatches"
                for event in dp_events
            )
        )
        optimizer_events = [
            event
            for event in result.trace.events
            if event.metadata.get("pipeline_phase") == "optimizer"
        ]
        self.assertEqual(len(optimizer_events), 16)
        self.assertTrue(
            all(
                event.metadata["iteration_tail"]
                == "optimizer-after-gradient-sync-v1"
                for event in optimizer_events
            )
        )
        self.assertTrue(any(event.collective == "send" for event in result.trace.events))
        self.assertTrue(any(event.collective == "recv" for event in result.trace.events))
        rank_zero_layer_zero = [
            event
            for event in result.trace.events
            if event.rank == 0
            and event.metadata.get("logical_layer") == 0
            and event.metadata.get("pipeline_phase") == "forward"
        ]
        measured_to_compiled = {
            event.metadata["measured_primitive_event_id"]: event
            for event in rank_zero_layer_zero
        }
        self.assertEqual(
            measured_to_compiled["fwd-tp"].dependencies,
            (measured_to_compiled["fwd-gemm"].id,),
        )

        prediction = simulate(
            result.trace,
            8,
            Parallelism(tp=2, pp=2, dp=2),
            Topology(
                "two-node-test",
                4,
                2,
                NetworkTier(400.0, 1.0),
                NetworkTier(50.0, 5.0),
                {},
            ),
            Architecture("h100", "hopper", "9.0", 132),
        )
        self.assertEqual(prediction["summary"]["event_count"], len(result.trace.events))
        self.assertGreater(prediction["summary"]["step_time_us"], 0.0)
        p2p_by_instance = {}
        for event in prediction["timeline"]:
            if event["collective"] in {"send", "recv"}:
                p2p_by_instance.setdefault(event["collective_instance_id"], []).append(
                    event
                )
        self.assertEqual(
            len(p2p_by_instance),
            len(
                [
                    event
                    for event in prediction["timeline"]
                    if event["collective"] in {"send", "recv"}
                ]
            )
            // 2,
        )
        for instance, endpoints in p2p_by_instance.items():
            self.assertTrue(instance.startswith("p2p:"))
            self.assertEqual(
                {event["collective"] for event in endpoints}, {"send", "recv"}
            )
            self.assertEqual(len({event["start_us"] for event in endpoints}), 1)

    def test_absent_contract_is_an_exact_noop(self) -> None:
        trace = WorkloadTrace(events=(TraceEvent("x", "x", "compute", 1.0),))
        result = compile_framework_counterfactual(
            trace, Parallelism(tp=1, pp=1, dp=1), 1, "h100"
        )
        self.assertFalse(result.applied)
        self.assertIs(result.trace, trace)

    def test_candidate_memory_release_uses_gc_and_optional_allocator_trim(self) -> None:
        with patch.object(framework_counterfactual.gc, "collect") as collect, patch.object(
            framework_counterfactual, "_MALLOC_TRIM"
        ) as trim:
            framework_counterfactual._release_candidate_memory()

        collect.assert_called_once_with()
        trim.assert_called_once_with(0)

    def test_single_pipeline_stage_replicates_tp_dp_ranks(self) -> None:
        result = compile_framework_counterfactual(
            self._trace(), Parallelism(tp=2, pp=1, dp=2), 4, "h100"
        )
        self.assertTrue(result.applied)
        self.assertEqual(result.summary["layers_per_stage"], 4)
        self.assertFalse(
            any(
                event.collective in {"send", "recv"}
                for event in result.trace.events
            )
        )
        self.assertEqual({event.rank for event in result.trace.events}, set(range(4)))
        self.assertEqual(
            len([event for event in result.trace.events if event.group_role == "dp"]),
            4,
        )

    def test_missing_target_tp_profile_abstains(self) -> None:
        with self.assertRaisesRegex(
            FrameworkCounterfactualError, r"tp_profiles\[1\] must be a mapping"
        ):
            compile_framework_counterfactual(
                self._trace(), Parallelism(tp=1, pp=2, dp=2), 4, "h100"
            )

    def test_incompatible_batch_arithmetic_abstains(self) -> None:
        document = self._trace().to_dict()
        document["metadata"]["framework_counterfactual"]["model"][
            "global_batch_size"
        ] = 10
        trace = WorkloadTrace.from_dict(document)
        with self.assertRaisesRegex(
            FrameworkCounterfactualError,
            "global batch must be divisible",
        ):
            compile_framework_counterfactual(
                trace, Parallelism(tp=2, pp=2, dp=2), 8, "h100"
            )

    def test_unvalidated_framework_revision_abstains(self) -> None:
        document = self._trace().to_dict()
        contract = deepcopy(document["metadata"]["framework_counterfactual"])
        contract["framework_commit"] = "future-revision"
        document["metadata"]["framework_counterfactual"] = contract
        trace = WorkloadTrace.from_dict(document)
        with self.assertRaisesRegex(
            FrameworkCounterfactualError, "schedule revision"
        ):
            compile_framework_counterfactual(
                trace, Parallelism(tp=2, pp=2, dp=2), 8, "h100"
            )

    def test_profile_target_mismatch_abstains(self) -> None:
        with self.assertRaisesRegex(
            FrameworkCounterfactualError, "does not match requested target"
        ):
            compile_framework_counterfactual(
                self._trace(), Parallelism(tp=2, pp=2, dp=2), 8, "b200"
            )

    def test_unprofiled_expert_parallel_candidate_abstains(self) -> None:
        with self.assertRaisesRegex(
            FrameworkCounterfactualError,
            "no expert-parallel profile",
        ):
            compile_framework_counterfactual(
                self._trace(), Parallelism(tp=2, pp=2, dp=2, ep=2), 8, "h100"
            )

    def test_profile_collective_group_mismatch_abstains(self) -> None:
        document = self._trace().to_dict()
        for event in document["events"]:
            if event["id"] == "fwd-tp":
                event["metadata"]["observed_group_size"] = 4
        trace = WorkloadTrace.from_dict(document)
        with self.assertRaisesRegex(
            FrameworkCounterfactualError, "group size does not match target TP"
        ):
            compile_framework_counterfactual(
                trace, Parallelism(tp=2, pp=2, dp=2), 8, "h100"
            )

    def test_search_compiles_supported_candidate_without_training_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "measured.json"
            self._trace().dump(workload)
            topology = root / "cluster.json"
            topology.write_text(
                json.dumps(
                    {
                        "schema_version": "0.1",
                        "name": "test",
                        "target": "h100",
                        "gpus_per_node": 4,
                        "nodes": 2,
                        "tiers": {
                            "intra_node": {
                                "bandwidth_GBps": 400,
                                "latency_us": 1,
                            },
                            "inter_node": {
                                "bandwidth_GBps": 50,
                                "latency_us": 5,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidates = root / "candidates.json"
            candidates.write_text(
                json.dumps(
                    [
                        {"tp": 2, "pp": 2, "dp": 2},
                        {"tp": 1, "pp": 2, "dp": 2},
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "search.json"
            status = search_main(
                [
                    "--workload-trace",
                    str(workload),
                    "--topology",
                    str(topology),
                    "--target",
                    "h100",
                    "--candidate-file",
                    str(candidates),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["compiled_count"], 1)
            self.assertEqual(document["abstained_count"], 1)
            self.assertEqual(document["executed_training_candidate_count"], 0)
            self.assertEqual(
                document["results"][0]["status"], "compiled-and-simulated"
            )
            self.assertEqual(document["results"][1]["status"], "abstained")

    def test_one_measurement_bundle_compiles_and_simulates_one_hundred_candidates(
        self,
    ) -> None:
        events = []
        profiles = {}
        for tp in range(1, 26):
            forward_ids = [f"fwd-compute-tp{tp}"]
            backward_ids = [f"bwd-compute-tp{tp}"]
            events.extend(
                [
                    TraceEvent(forward_ids[0], forward_ids[0], "compute", 1.0),
                    TraceEvent(backward_ids[0], backward_ids[0], "compute", 2.0),
                ]
            )
            if tp > 1:
                forward_ids.append(f"fwd-allreduce-tp{tp}")
                backward_ids.append(f"bwd-allreduce-tp{tp}")
                for event_id in (forward_ids[-1], backward_ids[-1]):
                    events.append(
                        TraceEvent(
                            event_id,
                            event_id,
                            "collective",
                            0.0,
                            collective="all_reduce",
                            message_bytes=1024,
                            group_role="tp",
                            group_size=tp,
                        )
                    )
            profiles[str(tp)] = {
                "target": "h100",
                "measurement": {"status": "measured", "sample_count": 3},
                "forward_event_ids": forward_ids,
                "backward_event_ids": backward_ids,
                "pipeline_activation_bytes_per_tp_rank": 1024,
                "gradient_sync": {
                    "collective": "reduce_scatter",
                    "bytes_per_layer_per_rank": 2048,
                },
            }
        trace = WorkloadTrace(
            events=tuple(events),
            source={"target": "h100"},
            metadata={
                "framework_counterfactual": {
                    "schema": CONTRACT_SCHEMA,
                    "framework": "megatron-core",
                    "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
                    "model": {
                        "num_layers": 2,
                        "micro_batch_size": 1,
                        "global_batch_size": 2,
                    },
                    "schedule": {"type": "1f1b"},
                    "tp_profiles": profiles,
                }
            },
        )
        candidates = [
            Parallelism(tp=tp, pp=pp, dp=dp)
            for tp in range(1, 26)
            for pp in (1, 2)
            for dp in (1, 2)
        ]
        event_count = 0
        topology = Topology(
            "hundred-candidate-test",
            4,
            32,
            NetworkTier(400.0, 1.0),
            NetworkTier(50.0, 5.0),
            {},
        )
        accelerator = Architecture("h100", "hopper", "9.0", 132)
        simulated_count = 0
        for candidate in candidates:
            compilation = compile_framework_counterfactual(
                trace,
                candidate,
                candidate.tp * candidate.pp * candidate.dp,
                "h100",
            )
            self.assertTrue(compilation.applied)
            event_count += len(compilation.trace.events)
            prediction = simulate(
                compilation.trace,
                candidate.tp * candidate.pp * candidate.dp,
                candidate,
                topology,
                accelerator,
            )
            self.assertEqual(
                prediction["summary"]["event_count"],
                len(compilation.trace.events),
            )
            self.assertGreater(prediction["summary"]["step_time_us"], 0.0)
            simulated_count += 1
        self.assertEqual(len(candidates), 100)
        self.assertEqual(simulated_count, 100)
        self.assertGreater(event_count, 0)


if __name__ == "__main__":
    unittest.main()
