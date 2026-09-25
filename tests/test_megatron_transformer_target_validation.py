from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scaletether.megatron_transformer_target_validation import (
    COMPONENT_PATHS,
    validate_physical_pp_target,
    validate_physical_tp_dp_target,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads(
    (
        ROOT / "tests/fixtures/level2-h100-transformer-family-v1.json"
    ).read_text(encoding="utf-8")
)
V15_MATRIX = json.loads(
    (
        ROOT / "tests/fixtures/level2-h100-transformer-family-v15.json"
    ).read_text(encoding="utf-8")
)


def target_row(matrix: dict, target_id: str) -> dict:
    return next(row for row in matrix["targets"] if row["id"] == target_id)


def marker(
    rank: int,
    name: str,
    *,
    operator: str = "aten::mm",
    occurrence: int | None = None,
) -> dict:
    suffix = "" if occurrence is None else f"-occurrence-{occurrence}"
    identifier = f"r{rank}-{name}{suffix}"
    return {
        "id": identifier,
        "rank": rank,
        "kind": "compute",
        "metadata": {
            "framework_phase_marker": {
                "name": name,
                "instance_id": identifier,
            },
            "kernel_launch_payload": {"framework_operator": {"name": operator}},
        },
    }


def stage_evidence(stage: int) -> dict:
    name = f"layers.{stage}.weight"
    return {
        "pipeline_parallel_rank": stage,
        "parameter_manifest": {
            "schema": "ordered-megatron-stage-parameter-manifest-v1",
            "ordering": "module.named_parameters-before-schedule",
            "parameter_count": 1,
            "total_numel": 16,
            "parameters": [
                {
                    "order": 0,
                    "name": name,
                    "local_name": "layers.0.weight",
                    "shape": [4, 4],
                    "numel": 16,
                    "dtype": "float32",
                    "requires_grad": True,
                }
            ],
        },
        "gradient_readiness": {
            "schema": "megatron-stage-gradient-readiness-occurrences-v1",
            "observation": "unmeasured-complete-warmup-step",
            "occurrences": [name, name],
            "occurrence_count": 2,
            "unique_parameter_count": 1,
            "measured_step_repeatability_required": True,
        },
        "components": [
            f"layers.{stage}{'.' if path else ''}{path}" for path in COMPONENT_PATHS
        ]
        + (["final_layernorm"] if stage == 1 else []),
    }


def valid_workload(target_id: str = "PP-A", matrix: dict = MATRIX) -> dict:
    domain = matrix["domain"]
    target = target_row(matrix, target_id)
    sequence = target["sequence_length"]
    microbatches = target["pipeline_microbatches"]
    payload = sequence * domain.get("micro_batch_size", 1) * domain["hidden_size"] * 4
    components = [
        f"layers.{layer}{'.' if path else ''}{path}"
        for layer in range(2)
        for path in COMPONENT_PATHS
    ] + ["final_layernorm"]
    measurement = {
        "schema": "megatron-core-transformer-pipeline-measurement-v1",
        "framework": "megatron-core",
        "framework_commit": "f8e1ac64b0587ff7002a18fbaa5ecdeaeb8491be",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 1,
        "model": {
            "kind": "dense-causal-transformer-block",
            "num_layers": domain["num_layers"],
            "layers_per_stage": domain["num_layers"] // 2,
            "hidden_size": domain["hidden_size"],
            "ffn_hidden_size": domain["ffn_hidden_size"],
            "attention_heads": domain["attention_heads"],
            "sequence_length": sequence,
            "micro_batch_size": domain.get("micro_batch_size", 1),
            "global_batch_size": target["global_batch_size"],
            "parameter_dtype": domain["parameter_dtype"],
            "activation_dtype": domain["parameter_dtype"],
            "attention_mask": domain["attention_mask"],
            "dropout": domain["dropout"],
        },
        "pipeline": {
            "kind": "megatron-core-non-interleaved-1f1b",
            "microbatches": microbatches,
            "layers_per_stage": 1,
        },
        "pipeline_stage_evidence": {
            "schema": "megatron-transformer-pipeline-stage-evidence-v1",
            "ordering": "pipeline-parallel-rank",
            "stages": [stage_evidence(0), stage_evidence(1)],
        },
        "component_markers": {
            "schema": "megatron-transformer-component-markers-v1",
            "phases": ["forward", "backward"],
            "components": components,
        },
    }
    events = []
    pp_producers: dict[tuple[int, str, int], str] = {}
    for rank in range(4):
        stage = rank // 2
        events.extend(
            [
                marker(rank, "megatron_transformer_forward", operator="aten::_softmax"),
                marker(rank, "megatron_transformer_backward"),
                marker(rank, "megatron_transformer_optimizer"),
            ]
        )
        if stage == 1:
            events.append(marker(rank, "megatron_transformer_terminal_loss"))
        for phase in ("forward", "backward"):
            for path in COMPONENT_PATHS:
                component = f"layers.{stage}{'.' if path else ''}{path}"
                events.append(
                    marker(rank, f"megatron_transformer_{phase}_component:{component}")
                )
                if (
                    (phase, component)
                    == ("forward", f"layers.{stage}.mlp.linear_fc2")
                    or (phase, component)
                    == ("backward", f"layers.{stage}.self_attention.linear_qkv")
                ):
                    pp_producers[(rank, phase, 0)] = events[-1]["id"]
            if stage == 1:
                events.append(
                    marker(
                        rank,
                        f"megatron_transformer_{phase}_component:final_layernorm",
                    )
                )
        for index in range(4 * microbatches):
            events.append(
                {
                    "id": f"r{rank}-tp{index}",
                    "rank": rank,
                    "kind": "collective",
                    "collective": "all_reduce",
                    "group_role": "tp",
                    "message_bytes": payload,
                    "metadata": {},
                }
            )
        for phase, component in (
            ("forward", f"layers.{stage}.mlp.linear_fc2"),
            ("backward", f"layers.{stage}.self_attention.linear_qkv"),
        ):
            marker_name = f"megatron_transformer_{phase}_component:{component}"
            for occurrence in range(1, microbatches):
                producer = marker(rank, marker_name, occurrence=occurrence)
                pp_producers[(rank, phase, occurrence)] = producer["id"]
                events.append(producer)
    for lane in range(2):
        first, last = lane, lane + 2
        for microbatch in range(microbatches):
            for operation, source, destination, owner in (
                ("send", first, last, first),
                ("recv", first, last, last),
                ("send", last, first, last),
                ("recv", last, first, first),
            ):
                dependencies = []
                if operation == "send":
                    phase = "forward" if source < destination else "backward"
                    dependencies = [pp_producers[(owner, phase, microbatch)]]
                events.append(
                    {
                        "id": f"lane{lane}-mb{microbatch}-{operation}-{source}-{owner}",
                        "rank": owner,
                        "kind": "collective",
                        "collective": operation,
                        "group_role": "pp",
                        "group_size": 2,
                        "message_bytes": payload,
                        "dependencies": dependencies,
                        "metadata": {
                            "p2p_source_rank": source,
                            "p2p_destination_rank": destination,
                        },
                    }
                )
    rank_metadata = []
    for rank in range(4):
        observations = []
        for event in events:
            if event.get("rank") != rank or event.get("kind") != "collective":
                continue
            role = event.get("group_role")
            members = (
                ([0, 1] if rank < 2 else [2, 3])
                if role == "tp"
                else ([0, 2] if rank in {0, 2} else [1, 3])
            )
            observation = {
                "index": len(observations),
                "collective": event["collective"],
                "group_role_hint": role,
                "message_bytes": event["message_bytes"],
                "process_group_ranks": members,
                "capture_scope": "inside-declared-step",
            }
            if role == "pp":
                observation.update(
                    p2p_source_rank=event["metadata"]["p2p_source_rank"],
                    p2p_destination_rank=event["metadata"]["p2p_destination_rank"],
                )
            event["metadata"]["collective_sequence"] = observation["index"]
            observations.append(observation)
        rank_metadata.append(
            {
                "rank": rank,
                "collective_observations": observations,
                "declared_step_collective_scope": {
                    "status": "complete",
                    "inside_count": len(observations),
                    "outside_count": 0,
                    "unresolved_count": 0,
                },
                "direct_native_nccl_binding": {
                    "status": "framework-owned-fully-reconciled",
                    "operation_count": len(observations),
                    "framework_reconciliation_count": len(observations),
                },
            }
        )
    return {
        "source": {"rank_count": 4, "target": "h100", "observed_targets": ["h100"]},
        "metadata": {
            "framework_measurement": measurement,
            "rank_metadata": rank_metadata,
        },
        "events": events,
    }


def valid_tp_dp_workload(target_id: str, matrix: dict = MATRIX) -> dict:
    domain = matrix["domain"]
    target = target_row(matrix, target_id)
    sequence = target["sequence_length"]
    tp = target["parallelism"]["tp"]
    dp = target["parallelism"]["dp"]
    components = [
        f"layers.{layer}{'.' if path else ''}{path}"
        for layer in range(2)
        for path in COMPONENT_PATHS
    ] + ["final_layernorm"]
    measurement = {
        "schema": "megatron-core-transformer-block-measurement-v1",
        "framework": "megatron-core",
        "framework_commit": "f8e1ac64b0587ff7002a18fbaa5ecdeaeb8491be",
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "data_parallel_size": dp,
        "model": {
            "kind": "dense-causal-transformer-block",
            "num_layers": domain["num_layers"],
            "hidden_size": domain["hidden_size"],
            "ffn_hidden_size": domain["ffn_hidden_size"],
            "attention_heads": domain["attention_heads"],
            "sequence_length": sequence,
            "micro_batch_size": domain.get("micro_batch_size", 1),
            "global_batch_size": target["global_batch_size"],
            "parameter_dtype": domain["parameter_dtype"],
            "activation_dtype": domain["parameter_dtype"],
            "attention_mask": domain["attention_mask"],
            "dropout": domain["dropout"],
        },
        "layer": {
            "kind": "megatron-core-local-dense-transformer-layer-v1",
            "self_attention": True,
            "dense_mlp": True,
            "sequence_parallel": False,
            "transformer_engine": domain["transformer_engine"],
        },
        "parameter_manifest": {
            "schema": "ordered-megatron-parameter-manifest-v1",
            "ordering": "module.named_parameters-before-ddp-wrap",
            "parameter_count": 1,
            "total_numel": 16,
            "parameters": [
                {
                    "order": 0,
                    "name": "layers.0.weight",
                    "shape": [4, 4],
                    "numel": 16,
                    "dtype": "float32",
                    "requires_grad": True,
                }
            ],
        },
        "gradient_readiness": {
            "schema": "megatron-parameter-gradient-readiness-v1",
            "observation": "unmeasured-complete-warmup-step",
            "parameter_names": ["layers.0.weight"],
            "parameter_count": 1,
            "measured_step_repeatability_required": True,
        },
        "component_markers": {
            "schema": "megatron-transformer-component-markers-v1",
            "phases": ["forward", "backward"],
            "components": components,
        },
        "data_parallel": {
            "kind": "torch.nn.parallel.DistributedDataParallel" if dp == 2 else "none",
            "gradient_sync": dp == 2,
        },
    }
    events = []
    tp_groups = {
        rank: ([0, 1, 2, 3] if tp == 4 else ([0, 1] if rank < 2 else [2, 3]))
        for rank in range(4)
    }
    dp_groups = {0: [0, 2], 2: [0, 2], 1: [1, 3], 3: [1, 3]}
    for rank in range(4):
        events.extend(
            [
                marker(rank, "megatron_transformer_forward", operator="aten::_softmax"),
                marker(rank, "megatron_transformer_backward"),
                marker(rank, "megatron_transformer_optimizer"),
            ]
        )
        for phase in ("forward", "backward"):
            for component in components:
                events.append(
                    marker(rank, f"megatron_transformer_{phase}_component:{component}")
                )
        for index in range(8):
            events.append(
                {
                    "id": f"r{rank}-tp-{index}",
                    "rank": rank,
                    "kind": "collective",
                    "collective": "all_reduce",
                    "group_role": "tp",
                    "message_bytes": sequence * domain["hidden_size"] * 4,
                    "metadata": {"process_group_ranks": tp_groups[rank]},
                }
            )
        if dp == 2:
            for index, (operation, message_bytes) in enumerate(
                (("broadcast", 8), ("broadcast", 4), ("all_reduce", 64))
            ):
                events.append(
                    {
                        "id": f"r{rank}-dp-{index}",
                        "rank": rank,
                        "kind": "collective",
                        "collective": operation,
                        "group_role": "dp",
                        "message_bytes": message_bytes,
                        "metadata": {"process_group_ranks": dp_groups[rank]},
                    }
                )
    return {
        "source": {"rank_count": 4, "target": "h100", "observed_targets": ["h100"]},
        "metadata": {"framework_measurement": measurement},
        "events": events,
    }


def remove_pp_logical_collectives(workload: dict, predicate) -> None:
    workload["events"] = [event for event in workload["events"] if not predicate(event)]
    for record in workload["metadata"]["rank_metadata"]:
        observations = [
            observation
            for observation in record["collective_observations"]
            if not predicate({**observation, "rank": record["rank"]})
        ]
        record["collective_observations"] = observations
        record["declared_step_collective_scope"]["inside_count"] = len(observations)
        record["direct_native_nccl_binding"]["operation_count"] = len(observations)
        record["direct_native_nccl_binding"]["framework_reconciliation_count"] = len(
            observations
        )


@pytest.mark.parametrize("target_id", ["PP-A", "PP-B"])
def test_admits_both_frozen_physical_pp_targets(target_id: str) -> None:
    result = validate_physical_pp_target(valid_workload(target_id), MATRIX, target_id)
    assert result["passed"], result
    assert result["decision"] == "admit"
    assert result["unsupported_capture"] is None
    assert result["pp_send_producer_readiness"] == {
        "schema": "scaletether-physical-pp-send-producer-readiness-v1",
        "passed": True,
        "checked_send_count": 4 * target_row(MATRIX, target_id)[
            "pipeline_microbatches"
        ],
        "missing_send_count": 0,
        "missing": [],
        "evidence": "explicit-transitive-device-dependency-only",
        "inferred_edges_added": 0,
    }
    assert (
        result["claim"]
        == "physical-h100-transformer-pp-target-admitted-for-structural-scoring"
    )


def test_pp_gate_reports_typed_capture_limitation_when_send_lacks_producer() -> None:
    workload = valid_workload("PP-A")
    send = next(
        event
        for event in workload["events"]
        if event.get("group_role") == "pp"
        and event.get("collective") == "send"
        and event.get("rank") == 0
    )
    send["dependencies"] = []
    frozen = copy.deepcopy(workload)

    result = validate_physical_pp_target(workload, MATRIX, "PP-A")

    assert not result["passed"]
    assert result["decision"] == "unsupported-capture"
    assert "pp-send-producer-readiness" in result["failures"]
    assert result["unsupported_capture"] == {
        "schema": "scaletether-unsupported-capture-decision-v1",
        "reason_code": "missing-pp-send-producer-readiness",
        "action": "UPGRADE_CAPTURE",
        "inferred_edges_added": 0,
    }
    readiness = result["pp_send_producer_readiness"]
    assert readiness["passed"] is False
    assert readiness["missing_send_count"] == 1
    assert readiness["missing"][0]["physical_event_id"] == send["id"]
    assert readiness["inferred_edges_added"] == 0
    assert workload == frozen


def test_pp_gate_treats_nullable_framework_operator_as_unidentified() -> None:
    workload = valid_workload("PP-A")
    event = next(
        event
        for event in workload["events"]
        if isinstance(event.get("metadata", {}).get("kernel_launch_payload"), dict)
    )
    event["metadata"]["kernel_launch_payload"]["framework_operator"] = None
    result = validate_physical_pp_target(workload, MATRIX, "PP-A")
    assert not result["passed"]
    assert "rank-0-softmax" in result["failures"]
    assert result["claim"] == "no-physical-transformer-target-claim"


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda w: w["metadata"].pop("framework_measurement"),
            "rank-identical-framework-measurement",
        ),
        (
            lambda w: w["metadata"]["framework_measurement"]["model"].update(
                attention_mask="none"
            ),
            "model-attention_mask",
        ),
        (
            lambda w: w["metadata"]["framework_measurement"]["pipeline"].update(
                microbatches=3
            ),
            "pipeline-contract",
        ),
        (
            lambda w: w["metadata"]["framework_measurement"]["pipeline_stage_evidence"][
                "stages"
            ][0]["parameter_manifest"]["parameters"][0].update(numel=15),
            "stage-0-manifest-numel",
        ),
        (
            lambda w: w["metadata"]["framework_measurement"]["pipeline_stage_evidence"][
                "stages"
            ][1]["gradient_readiness"].update(occurrences=[]),
            "stage-1-readiness",
        ),
        (
            lambda w: remove_pp_logical_collectives(
                w, lambda event: event.get("collective") == "recv"
            ),
            "lane-0-p2p-match",
        ),
        (
            lambda w: remove_pp_logical_collectives(
                w,
                lambda event: event.get("rank") == 0
                and event.get("collective") == "all_reduce",
            ),
            "rank-0-tp-collective-count",
        ),
        (
            lambda w: w["events"].append(
                marker(0, "megatron_transformer_forward_component:layers.1")
            ),
            "rank-0-foreign-stage-component",
        ),
        (
            lambda w: w.update(
                events=[
                    event
                    for event in w["events"]
                    if not (
                        event.get("rank") == 2
                        and _event_phase(event) == "megatron_transformer_terminal_loss"
                    )
                ]
            ),
            "rank-2-terminal-loss-placement",
        ),
    ],
)
def test_malformed_physical_target_fails_closed(mutate, expected: str) -> None:
    workload = copy.deepcopy(valid_workload())
    mutate(workload)
    result = validate_physical_pp_target(workload, MATRIX, "PP-A")
    assert not result["passed"]
    assert expected in result["failures"]
    assert result["claim"] == "no-physical-transformer-target-claim"


def _event_phase(event: dict) -> str | None:
    return event.get("metadata", {}).get("framework_phase_marker", {}).get("name")


def test_tp_or_dp_target_is_rejected_by_pp_gate() -> None:
    result = validate_physical_pp_target(valid_workload(), MATRIX, "TP-A")
    assert not result["passed"]
    assert "not-frozen-pp-target" in result["failures"]


@pytest.mark.parametrize("target_id", ["TP-A", "TP-B", "DP-A", "DP-B"])
def test_admits_all_frozen_physical_tp_dp_targets(target_id: str) -> None:
    result = validate_physical_tp_dp_target(
        valid_tp_dp_workload(target_id), MATRIX, target_id
    )
    assert result["passed"], result
    assert (
        result["claim"]
        == "physical-h100-transformer-tp-dp-target-admitted-for-structural-scoring"
    )


@pytest.mark.parametrize("target_id", ["TP-A", "TP-B", "DP-A", "DP-B"])
def test_admits_versioned_v15_physical_tp_dp_targets(target_id: str) -> None:
    result = validate_physical_tp_dp_target(
        valid_tp_dp_workload(target_id, V15_MATRIX), V15_MATRIX, target_id
    )
    assert result["passed"], result


@pytest.mark.parametrize("target_id", ["PP-A", "PP-B"])
def test_admits_versioned_v15_physical_pp_targets(target_id: str) -> None:
    result = validate_physical_pp_target(
        valid_workload(target_id, V15_MATRIX), V15_MATRIX, target_id
    )
    target = target_row(V15_MATRIX, target_id)
    expected_payload = (
        target["sequence_length"]
        * V15_MATRIX["domain"]["micro_batch_size"]
        * V15_MATRIX["domain"]["hidden_size"]
        * 4
    )
    assert result["passed"], result
    assert result["pipeline_payload_bytes"] == expected_payload


def test_tp_dp_gate_accepts_exact_json_encoded_membership() -> None:
    workload = valid_tp_dp_workload("DP-A")
    for event in workload["events"]:
        metadata = event.get("metadata", {})
        if "process_group_ranks" in metadata:
            metadata["process_group_ranks"] = json.dumps(
                metadata["process_group_ranks"]
            )
    result = validate_physical_tp_dp_target(workload, MATRIX, "DP-A")
    assert result["passed"], result


@pytest.mark.parametrize("membership", ["not-json", "[0, true]", "[0, 0]"])
def test_tp_dp_gate_rejects_malformed_encoded_membership(membership: str) -> None:
    workload = valid_tp_dp_workload("TP-A")
    event = next(
        event for event in workload["events"] if event.get("group_role") == "tp"
    )
    event["metadata"]["process_group_ranks"] = membership
    result = validate_physical_tp_dp_target(workload, MATRIX, "TP-A")
    assert not result["passed"]
    assert "rank-0-collective-process-group-membership" in result["failures"]


@pytest.mark.parametrize(
    "target_id,mutate,expected",
    [
        (
            "TP-A",
            lambda w: w["metadata"]["framework_measurement"].update(
                tensor_parallel_size=2
            ),
            "measurement-tensor_parallel_size",
        ),
        (
            "TP-A",
            lambda w: w["events"][0]["metadata"]["kernel_launch_payload"][
                "framework_operator"
            ].update(name="aten::mm"),
            "rank-0-softmax",
        ),
        (
            "TP-A",
            lambda w: w["events"].append(
                {
                    "id": "dp",
                    "rank": 0,
                    "kind": "collective",
                    "collective": "all_reduce",
                    "group_role": "dp",
                    "message_bytes": 64,
                    "metadata": {"process_group_ranks": [0, 2]},
                }
            ),
            "unexpected-dp-sync",
        ),
        (
            "DP-A",
            lambda w: _mutate_first_dp(w, message_bytes=60),
            "rank-0-dp-sync-bytes",
        ),
        (
            "DP-A",
            lambda w: _mutate_first_dp(w, process_group_ranks=[0, 1]),
            "dp-communicator-membership",
        ),
        (
            "DP-A",
            lambda w: w["metadata"]["framework_measurement"]["parameter_manifest"][
                "parameters"
            ][0].update(numel=15),
            "parameter-manifest-numel",
        ),
    ],
)
def test_malformed_physical_tp_dp_target_fails_closed(
    target_id: str, mutate, expected: str
) -> None:
    workload = valid_tp_dp_workload(target_id)
    mutate(workload)
    result = validate_physical_tp_dp_target(workload, MATRIX, target_id)
    assert not result["passed"]
    assert expected in result["failures"]


def _mutate_first_dp(workload: dict, **changes) -> None:
    event = next(
        event for event in workload["events"] if event.get("group_role") == "dp"
    )
    if "process_group_ranks" in changes:
        event["metadata"]["process_group_ranks"] = changes.pop("process_group_ranks")
    event.update(changes)
