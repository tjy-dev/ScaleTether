from pathlib import Path
import json
import tempfile
import unittest

from scaletether.chakra import (
    _lower_chakra_p2p_peer_dependencies,
    _materialize_counterfactual_p2p_tags,
    _compact_local_chains,
    _expand_physical_logical_collectives,
    _validate_encoded_rank,
    export_chakra,
)
from scaletether.config import Parallelism
from scaletether.schema import TraceEvent, WorkloadTrace


try:
    from chakra.schema.protobuf import et_def_pb2
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
    from chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage

    CHAKRA_AVAILABLE = True
except Exception:
    CHAKRA_AVAILABLE = False


class ChakraLogicalProjectionTest(unittest.TestCase):
    def test_non_p2p_rank_replication_does_not_require_global_ids(self):
        replicated = TraceEvent("compute", "compute", "compute", 1.0)
        events = {0: (replicated,), 1: (replicated,)}
        lowered, report = _lower_chakra_p2p_peer_dependencies(events)
        self.assertIs(lowered, events)
        self.assertIsNone(report)

    def test_exact_cross_rank_send_recv_edge_is_lowered_to_p2p_pairing(self):
        common = {
            "p2p_source_rank": 0,
            "p2p_destination_rank": 2,
            "p2p_tag": 7,
        }
        local = TraceEvent("local", "local", "compute", 1.0, rank=2, device=2)
        send = TraceEvent(
            "send", "send", "collective", 0.0, rank=0, device=0,
            collective="send", message_bytes=16, group_role="pp", group_size=2,
            metadata=common,
        )
        recv = TraceEvent(
            "recv", "recv", "collective", 0.0, rank=2, device=2,
            dependencies=("local", "send"), collective="recv", message_bytes=16,
            group_role="pp", group_size=2, metadata=common,
        )
        lowered, report = _lower_chakra_p2p_peer_dependencies(
            {0: (send,), 1: (), 2: (local, recv), 3: ()}
        )
        assert report is not None
        self.assertEqual(report["replaced_cross_rank_edge_count"], 1)
        self.assertEqual(lowered[2][1].dependencies, ("local",))
        self.assertEqual(recv.dependencies, ("local", "send"))

    def test_unrelated_cross_rank_edge_is_not_lowered(self):
        common = {
            "p2p_source_rank": 0,
            "p2p_destination_rank": 2,
            "p2p_tag": 7,
        }
        send = TraceEvent(
            "send", "send", "collective", 0.0, rank=0, device=0,
            collective="send", message_bytes=16, group_role="pp", group_size=2,
            metadata=common,
        )
        recv = TraceEvent(
            "recv", "recv", "collective", 0.0, rank=2, device=2,
            dependencies=("other-rank-event",), collective="recv", message_bytes=16,
            group_role="pp", group_size=2, metadata=common,
        )
        other = TraceEvent(
            "other-rank-event", "other", "compute", 1.0, rank=0, device=0
        )
        with self.assertRaisesRegex(ValueError, "not its exact matched P2P send"):
            _lower_chakra_p2p_peer_dependencies(
                {0: (other, send), 1: (), 2: (recv,), 3: ()}
            )

    def test_counterfactual_pp_tags_are_lowered_from_exact_pairs(self):
        common = {
            "pipeline_phase": "forward",
            "pipeline_microbatch": 0,
            "p2p_source_rank": 0,
            "p2p_destination_rank": 2,
        }
        send = TraceEvent(
            "send", "send", "collective", 0.0, rank=0, device=0,
            collective="send", message_bytes=16, group_role="pp", group_size=2,
            metadata=common,
        )
        recv = TraceEvent(
            "recv", "recv", "collective", 0.0, rank=2, device=2,
            collective="recv", message_bytes=16, group_role="pp", group_size=2,
            metadata=common,
        )
        trace = WorkloadTrace(
            events=(send, recv),
            source={
                "kind": "framework-semantic-counterfactual",
                "candidate_training_executed": False,
            },
        )
        lowered, report = _materialize_counterfactual_p2p_tags(
            trace, {0: (send,), 1: (), 2: (recv,), 3: ()}
        )
        assert report is not None
        self.assertEqual(report["pair_count"], 1)
        self.assertEqual(lowered[0][0].metadata["p2p_tag"], 1)
        self.assertEqual(lowered[2][0].metadata["p2p_tag"], 1)
        self.assertEqual(send.metadata.get("p2p_tag"), None)

    def test_counterfactual_pp_tag_lowering_rejects_unmatched_route(self):
        send = TraceEvent(
            "send", "send", "collective", 0.0, rank=0, device=0,
            collective="send", message_bytes=16, group_role="pp", group_size=2,
            metadata={
                "pipeline_phase": "forward",
                "pipeline_microbatch": 0,
                "p2p_source_rank": 0,
                "p2p_destination_rank": 2,
            },
        )
        trace = WorkloadTrace(
            events=(send,),
            source={"kind": "framework-semantic-counterfactual"},
        )
        with self.assertRaisesRegex(ValueError, "matched send/recv"):
            _materialize_counterfactual_p2p_tags(trace, {0: (send,)})

    def test_fused_sendrecv_projects_network_ops_and_owns_duration_once(self):
        fused = TraceEvent(
            "fused",
            "ncclDevKernel_SendRecv",
            "collective",
            8.0,
            dependencies=("producer",),
            collective="send",
            message_bytes=1024,
            group_size=2,
            metadata={
                "logical_collective_operations": [
                    {
                        "schema": "logical-collective-operation-v1",
                        "collective": "send",
                        "message_bytes": 1024,
                        "group_size": 2,
                        "p2p_source_rank": 0,
                        "p2p_destination_rank": 1,
                        "p2p_tag": 0,
                    },
                    {
                        "schema": "logical-collective-operation-v1",
                        "collective": "recv",
                        "message_bytes": 1024,
                        "group_size": 2,
                        "p2p_source_rank": 1,
                        "p2p_destination_rank": 0,
                        "p2p_tag": 1,
                    },
                ]
            },
        )
        events = (
            TraceEvent("producer", "producer", "compute", 3.0),
            fused,
            TraceEvent(
                "consumer",
                "consumer",
                "compute",
                2.0,
                dependencies=("fused",),
            ),
        )
        projected = _expand_physical_logical_collectives(events)
        self.assertEqual(
            [event.id for event in projected],
            ["producer", "fused#logical-0", "fused#logical-1", "consumer"],
        )
        self.assertEqual(
            [event.duration_us for event in projected[1:3]], [8.0, 0.0]
        )
        self.assertEqual(
            [event.collective for event in projected[1:3]], ["send", "recv"]
        )
        self.assertEqual(
            projected[-1].dependencies,
            ("fused#logical-0", "fused#logical-1"),
        )

    def test_logical_projection_precedes_p2p_pair_validation(self):
        def fused(rank, event_id, source, destination, tag, collective):
            return TraceEvent(
                event_id,
                "ncclDevKernel_SendRecv",
                "collective",
                1.0,
                rank=rank,
                device=rank,
                collective=collective,
                message_bytes=16,
                group_role="pp",
                group_size=2,
                metadata={
                    # Reused top-level capture placeholder; it must not define
                    # logical identity after projection.
                    "p2p_source_rank": source,
                    "p2p_destination_rank": destination,
                    "p2p_tag": 0,
                    "logical_collective_operations": [{
                        "schema": "logical-collective-operation-v1",
                        "collective": collective,
                        "message_bytes": 16,
                        "group_size": 2,
                        "p2p_source_rank": source,
                        "p2p_destination_rank": destination,
                        "p2p_tag": tag,
                    }],
                },
            )

        raw = {
            0: (
                fused(0, "send-a", 0, 2, 11, "send"),
                fused(0, "send-b", 0, 2, 12, "send"),
            ),
            1: (),
            2: (
                fused(2, "recv-a", 0, 2, 11, "recv"),
                fused(2, "recv-b", 0, 2, 12, "recv"),
            ),
            3: (),
        }
        projected = {
            rank: _expand_physical_logical_collectives(events)
            for rank, events in raw.items()
        }
        lowered, report = _lower_chakra_p2p_peer_dependencies(projected)
        self.assertIsNone(report)
        self.assertEqual(sum(map(len, lowered.values())), 4)


@unittest.skipUnless(CHAKRA_AVAILABLE, "Chakra optional dependency is not installed")
class ChakraExportTest(unittest.TestCase):
    def test_semantic_validation_rejects_changed_encoded_node(self):
        metadata = GlobalMetadata(version="1.0.0")
        expected = Node(id=1, name="compute", type=1, duration_micros=1)
        changed = Node(id=1, name="compute", type=1, duration_micros=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.et"
            with path.open("wb") as handle:
                encodeMessage(handle, metadata)
                encodeMessage(handle, changed)
            with self.assertRaisesRegex(RuntimeError, "changed during encoding"):
                _validate_encoded_rank(
                    et_def_pb2,
                    decodeMessage,
                    path,
                    metadata,
                    [expected],
                )

    @staticmethod
    def _pipeline_trace() -> WorkloadTrace:
        return WorkloadTrace(
            events=(
                TraceEvent(
                    "s0-forward",
                    "stage 0 forward",
                    "compute",
                    10.0,
                    metadata={"pipeline_stage": 0, "pipeline_phase": "forward"},
                ),
                TraceEvent(
                    "s0-forward-send",
                    "stage 0 activation send",
                    "collective",
                    0.0,
                    dependencies=("s0-forward",),
                    collective="send",
                    message_bytes=1024,
                    group_role="pp",
                    metadata={"pipeline_stage": 0, "pipeline_phase": "forward"},
                ),
                TraceEvent(
                    "s0-backward",
                    "stage 0 backward",
                    "compute",
                    20.0,
                    metadata={"pipeline_stage": 0, "pipeline_phase": "backward"},
                ),
                TraceEvent(
                    "s1-forward",
                    "stage 1 forward",
                    "compute",
                    10.0,
                    metadata={"pipeline_stage": 1, "pipeline_phase": "forward"},
                ),
                TraceEvent(
                    "s1-backward",
                    "stage 1 backward",
                    "compute",
                    20.0,
                    metadata={"pipeline_stage": 1, "pipeline_phase": "backward"},
                ),
                TraceEvent(
                    "s1-backward-send",
                    "stage 1 gradient send",
                    "collective",
                    0.0,
                    dependencies=("s1-backward",),
                    collective="send",
                    message_bytes=1024,
                    group_role="pp",
                    metadata={"pipeline_stage": 1, "pipeline_phase": "backward"},
                ),
            ),
            metadata={"pipeline": {"schedule": "1f1b", "microbatches": 1}},
        )

    @staticmethod
    def _interleaved_pipeline_trace(microbatches: int = 2) -> WorkloadTrace:
        events = []
        for chunk in range(2):
            for stage in range(2):
                virtual_stage = chunk * 2 + stage
                common = {
                    "pipeline_stage": stage,
                    "pipeline_model_chunk": chunk,
                }
                forward = f"s{stage}c{chunk}-forward"
                events.append(
                    TraceEvent(
                        forward,
                        forward,
                        "compute",
                        1.0,
                        metadata={**common, "pipeline_phase": "forward"},
                    )
                )
                if virtual_stage < 3:
                    events.append(
                        TraceEvent(
                            f"{forward}-send",
                            f"{forward}-send",
                            "collective",
                            0.0,
                            dependencies=(forward,),
                            collective="send",
                            message_bytes=64,
                            group_role="pp",
                            metadata={**common, "pipeline_phase": "forward"},
                        )
                    )
                backward = f"s{stage}c{chunk}-backward"
                events.append(
                    TraceEvent(
                        backward,
                        backward,
                        "compute",
                        1.0,
                        metadata={**common, "pipeline_phase": "backward"},
                    )
                )
                if virtual_stage > 0:
                    events.append(
                        TraceEvent(
                            f"{backward}-send",
                            f"{backward}-send",
                            "collective",
                            0.0,
                            dependencies=(backward,),
                            collective="send",
                            message_bytes=64,
                            group_role="pp",
                            metadata={**common, "pipeline_phase": "backward"},
                        )
                    )
        return WorkloadTrace(
            events=tuple(events),
            metadata={
                "pipeline": {
                    "schedule": "interleaved_1f1b",
                    "microbatches": microbatches,
                    "virtual_stages": 2,
                    "microbatch_group_size_per_virtual_stage": 2,
                }
            },
        )

    @staticmethod
    def _decoded_nodes(path: Path):
        with path.open("rb") as handle:
            metadata = GlobalMetadata()
            if not decodeMessage(handle, metadata):
                raise AssertionError("missing Chakra metadata")
            nodes = []
            while True:
                node = Node()
                if not decodeMessage(handle, node):
                    break
                nodes.append(node)
        return nodes

    def test_exports_reduce_gather_scatter_and_barrier(self):
        collectives = ("reduce", "gather", "scatter", "barrier")
        trace = WorkloadTrace(
            events=tuple(
                TraceEvent(
                    collective,
                    collective,
                    "collective",
                    1.0,
                    collective=collective,
                    message_bytes=None if collective == "barrier" else 1024,
                    group_role="tp",
                )
                for collective in collectives
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            exported = export_chakra(
                trace,
                Path(directory) / "extended",
                ranks=2,
                parallelism=Parallelism(tp=2, pp=1, dp=1),
            )
            with Path(exported.rank_files[0]).open("rb") as handle:
                metadata = GlobalMetadata()
                self.assertTrue(decodeMessage(handle, metadata))
                nodes = []
                for _ in collectives:
                    node = Node()
                    self.assertTrue(decodeMessage(handle, node))
                    nodes.append(node)
        expected = {"reduce": 1, "gather": 3, "scatter": 4, "barrier": 9}
        for node in nodes:
            attrs = {attr.name: attr for attr in node.attr}
            self.assertEqual(attrs["comm_type"].int64_val, expected[node.name])
            self.assertEqual(
                attrs["comm_size"].int64_val,
                0 if node.name == "barrier" else 1024,
            )

    def test_exports_rank_traces_and_communicators(self):
        trace = WorkloadTrace(
            events=(
                TraceEvent("compute", "compute", "compute", 2.25),
                TraceEvent(
                    "reduce",
                    "tp allreduce",
                    "collective",
                    4.5,
                    dependencies=("compute",),
                    collective="all_reduce",
                    message_bytes=1024,
                    group_role="tp",
                ),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            exported = export_chakra(
                trace,
                Path(directory) / "workload",
                ranks=4,
                parallelism=Parallelism(tp=2, pp=1, dp=2),
                max_ctas=8,
                cta_policy={"tp": 4},
                nccl_cta_policy=1,
                nccl_nvls_ctas=6,
            )
            self.assertEqual(len(exported.rank_files), 4)
            self.assertTrue(Path(exported.communicator_file).is_file())
            with Path(exported.rank_files[0]).open("rb") as handle:
                metadata = GlobalMetadata()
                self.assertTrue(decodeMessage(handle, metadata))
                compute = Node()
                self.assertTrue(decodeMessage(handle, compute))
                collective = Node()
                self.assertTrue(decodeMessage(handle, collective))
            self.assertEqual(compute.duration_micros, 3)
            self.assertEqual(list(collective.data_deps), [compute.id])
            attrs = {attr.name: attr for attr in collective.attr}
            self.assertEqual(attrs["comm_size"].int64_val, 1024)
            self.assertEqual(attrs["scaletether.max_ctas"].int64_val, 4)
            self.assertEqual(attrs["scaletether.comm_stream_priority"].string_val, "normal")
            self.assertEqual(attrs["scaletether.nccl_cta_policy"].int64_val, 1)
            self.assertEqual(attrs["scaletether.nccl_nvls_ctas"].int64_val, 6)
            validation = exported.semantic_validation
            self.assertIsNotNone(validation)
            assert validation is not None
            self.assertEqual(validation["status"], "exact-protobuf-roundtrip")
            self.assertEqual(validation["rank_files_validated"], 4)
            self.assertEqual(validation["node_count"], 8)
            self.assertEqual(validation["data_dependency_edge_count"], 4)
            self.assertEqual(validation["collective_node_count"], 4)
            self.assertEqual(validation["p2p_node_count"], 0)
            self.assertEqual(validation["encoded_communication_bytes"], 4096)
            self.assertEqual(len(validation["rank_semantic_sha256"]), 4)
            self.assertEqual(len(validation["aggregate_semantic_sha256"]), 64)

    def test_exports_native_per_rank_capture_without_replication(self):
        trace = WorkloadTrace(
            events=(
                TraceEvent("rank-0:a", "rank zero", "compute", 2.0, rank=0),
                TraceEvent("rank-1:a", "rank one", "compute", 3.0, rank=1),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            exported = export_chakra(
                trace,
                Path(directory) / "workload",
                ranks=2,
                parallelism=Parallelism(tp=2, pp=1, dp=1),
            )
            names = []
            for rank_file in exported.rank_files:
                with Path(rank_file).open("rb") as handle:
                    metadata = GlobalMetadata()
                    self.assertTrue(decodeMessage(handle, metadata))
                    node = Node()
                    self.assertTrue(decodeMessage(handle, node))
                    names.append(node.name)
                    trailing = Node()
                    self.assertFalse(decodeMessage(handle, trailing))
            self.assertEqual(names, ["rank zero", "rank one"])

    def test_exports_pipeline_as_matched_rank_local_p2p_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            exported = export_chakra(
                self._pipeline_trace(),
                Path(directory) / "pipeline",
                ranks=8,
                parallelism=Parallelism(tp=2, pp=2, dp=2),
            )
            self.assertTrue(exported.pipeline_expanded)
            self.assertEqual(exported.p2p_pair_count, 8)
            self.assertEqual(len(exported.rank_files), 8)
            rank_zero = self._decoded_nodes(Path(exported.rank_files[0]))
            rank_two = self._decoded_nodes(Path(exported.rank_files[2]))

        def by_event_id(nodes):
            result = {}
            for node in nodes:
                attrs = {attr.name: attr for attr in node.attr}
                result[attrs["scaletether.event_id"].string_val] = (node, attrs)
            return result

        zero = by_event_id(rank_zero)
        two = by_event_id(rank_two)
        send_zero, send_zero_attrs = zero["s0-forward-send@mb0"]
        recv_two, recv_two_attrs = two["scaletether-recv::s0-forward-send@mb0"]
        self.assertEqual(send_zero.type, 5)
        self.assertEqual(recv_two.type, 6)
        for attrs in (send_zero_attrs, recv_two_attrs):
            self.assertEqual(attrs["comm_src"].int32_val, 0)
            self.assertEqual(attrs["comm_dst"].int32_val, 2)
            self.assertEqual(attrs["comm_tag"].int32_val, 1)
            self.assertEqual(attrs["comm_size"].int64_val, 1024)
        forward_two, _ = two["s1-forward@mb0"]
        self.assertIn(recv_two.id, forward_two.data_deps)

        recv_zero, recv_zero_attrs = zero["scaletether-recv::s1-backward-send@mb0"]
        send_two, send_two_attrs = two["s1-backward-send@mb0"]
        for attrs in (recv_zero_attrs, send_two_attrs):
            self.assertEqual(attrs["comm_src"].int32_val, 2)
            self.assertEqual(attrs["comm_dst"].int32_val, 0)
            self.assertEqual(attrs["comm_tag"].int32_val, 2)
        backward_zero, _ = zero["s0-backward@mb0"]
        self.assertIn(recv_zero.id, backward_zero.data_deps)

    def test_exports_explicit_pipeline_rank_layout_and_communicators(self):
        trace = self._pipeline_trace()
        stage_ranks = [[0, 2, 4, 6], [1, 3, 5, 7]]
        trace.metadata["pipeline"]["stage_ranks"] = stage_ranks
        with tempfile.TemporaryDirectory() as directory:
            exported = export_chakra(
                trace,
                Path(directory) / "pipeline-layout",
                ranks=8,
                parallelism=Parallelism(tp=2, pp=2, dp=2),
            )
            self.assertEqual(
                exported.pipeline_rank_layout_source, "explicit-stage-ranks"
            )
            self.assertEqual(
                exported.pipeline_stage_ranks, tuple(map(tuple, stage_ranks))
            )
            rank_zero = self._decoded_nodes(Path(exported.rank_files[0]))
            rank_one = self._decoded_nodes(Path(exported.rank_files[1]))
            groups = json.loads(Path(exported.communicator_file).read_text())

        memberships = {tuple(members) for members in groups.values()}
        self.assertIn((0, 2), memberships)  # TP: same stage and DP replica.
        self.assertIn((0, 1), memberships)  # PP: same lane and DP replica.
        self.assertIn((0, 4), memberships)  # DP: same lane and stage.

        def p2p_attributes(nodes, node_type):
            for node in nodes:
                if node.type == node_type:
                    attrs = {attr.name: attr for attr in node.attr}
                    if attrs["scaletether.event_id"].string_val == (
                        "s0-forward-send@mb0"
                        if node_type == 5
                        else "scaletether-recv::s0-forward-send@mb0"
                    ):
                        return attrs
            raise AssertionError("missing expected P2P node")

        for attrs in (
            p2p_attributes(rank_zero, 5),
            p2p_attributes(rank_one, 6),
        ):
            self.assertEqual(attrs["comm_src"].int32_val, 0)
            self.assertEqual(attrs["comm_dst"].int32_val, 1)
            self.assertEqual(attrs["comm_tag"].int32_val, 1)

    def test_exports_interleaved_virtual_stage_wrap_as_rank_local_recv(self):
        with tempfile.TemporaryDirectory() as directory:
            exported = export_chakra(
                self._interleaved_pipeline_trace(),
                Path(directory) / "interleaved",
                ranks=2,
                parallelism=Parallelism(tp=1, pp=2, dp=1),
            )
            self.assertTrue(exported.pipeline_expanded)
            self.assertEqual(exported.p2p_pair_count, 12)
            rank_zero = self._decoded_nodes(Path(exported.rank_files[0]))

        by_event_id = {}
        ids = {node.id for node in rank_zero}
        for node in rank_zero:
            self.assertLessEqual(set(node.data_deps), ids)
            attrs = {attr.name: attr for attr in node.attr}
            by_event_id[attrs["scaletether.event_id"].string_val] = (node, attrs)
        receive, attrs = by_event_id["scaletether-recv::s1c0-forward-send@mb0"]
        self.assertEqual(receive.type, 6)
        self.assertEqual(attrs["comm_src"].int32_val, 1)
        self.assertEqual(attrs["comm_dst"].int32_val, 0)
        consumer, _ = by_event_id["s0c1-forward@mb0"]
        self.assertIn(receive.id, consumer.data_deps)

    def test_exports_interleaved_async_source_release_with_completed_receive(self):
        trace = self._interleaved_pipeline_trace(microbatches=4)
        trace.metadata["pipeline"]["overlap_p2p_comm"] = True
        with tempfile.TemporaryDirectory() as directory:
            exported = export_chakra(
                trace,
                Path(directory) / "interleaved-overlap",
                ranks=2,
                parallelism=Parallelism(tp=1, pp=2, dp=1),
            )
            self.assertEqual(exported.p2p_pair_count, 24)
            rank_zero = self._decoded_nodes(Path(exported.rank_files[0]))

        by_event_id = {}
        ids = {node.id for node in rank_zero}
        for node in rank_zero:
            self.assertLessEqual(set(node.data_deps), ids)
            attrs = {attr.name: attr for attr in node.attr}
            by_event_id[attrs["scaletether.event_id"].string_val] = (node, attrs)

        forward, _ = by_event_id["s0c0-forward@mb2"]
        send, send_attrs = by_event_id["s0c0-forward-send@mb2"]
        backward, _ = by_event_id["s0c1-backward@mb0"]
        self.assertIn(forward.id, send.data_deps)
        self.assertIn(forward.id, backward.data_deps)
        self.assertNotIn(send.id, backward.data_deps)
        self.assertEqual(
            send_attrs["scaletether.pipeline_p2p_mode"].string_val,
            "asynchronous",
        )

        incoming, _ = by_event_id["scaletether-recv::s1c0-forward-send@mb0"]
        consumer, _ = by_event_id["s0c1-forward@mb0"]
        self.assertIn(incoming.id, consumer.data_deps)

    def test_compacts_only_linear_local_chains_with_exact_encoded_duration(self):
        events = (
            TraceEvent("a", "a", "compute", 1.1, stream="7"),
            TraceEvent(
                "b",
                "b",
                "memory",
                2.1,
                stream="7",
                dependencies=("a",),
            ),
            TraceEvent(
                "c",
                "c",
                "compute",
                3.1,
                stream="7",
                dependencies=("b",),
            ),
            TraceEvent(
                "comm",
                "comm",
                "collective",
                2.0,
                stream="25",
                collective="all_reduce",
                message_bytes=8,
                group_role="dp",
            ),
            TraceEvent(
                "copy",
                "copy",
                "memory",
                1.0,
                stream="25",
                dependencies=("comm",),
            ),
        )
        projected, report = _compact_local_chains(events)
        self.assertEqual([event.id for event in projected], ["a", "comm", "copy"])
        aggregate = projected[0]
        self.assertEqual(aggregate.duration_us, 9.0)
        self.assertEqual(aggregate.dependencies, ())
        self.assertEqual(
            aggregate.metadata["chakra_local_chain_compaction"],
            {
                "schema": "scaletether-chakra-local-chain-v1",
                "source_event_count": 3,
                "first_event_id": "a",
                "last_event_id": "c",
                "source_kind_counts": {"compute": 2, "memory": 1},
                "encoded_duration_micros": 9,
            },
        )
        self.assertEqual(
            report,
            {
                "source_nodes": 5,
                "projected_nodes": 3,
                "collapsed_nodes": 2,
                "coalesced_chains": 1,
                "source_collective_nodes": 1,
                "projected_collective_nodes": 1,
                "source_local_duration_micros": 10,
                "projected_local_duration_micros": 10,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            exported = export_chakra(
                WorkloadTrace(events=events),
                Path(directory) / "compact",
                ranks=2,
                parallelism=Parallelism(tp=1, pp=1, dp=2),
                compact_local_chains=True,
            )
            nodes = self._decoded_nodes(Path(exported.rank_files[0]))
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0].duration_micros, 9)
        self.assertEqual(list(nodes[2].data_deps), [nodes[1].id])
        self.assertEqual(
            exported.local_chain_compaction,
            {
                "schema": "scaletether-chakra-local-chain-compaction-v1",
                "mode": ("linear-compute-memory-phase-boundary-exact-encoded-duration"),
                "source_nodes": 10,
                "projected_nodes": 6,
                "collapsed_nodes": 4,
                "coalesced_chains": 2,
                "source_collective_nodes": 2,
                "projected_collective_nodes": 2,
                "source_local_duration_micros": 20,
                "projected_local_duration_micros": 20,
                "encoded_local_duration_preserved": True,
                "collective_nodes_preserved": True,
                "wall_time_equivalence": "not-claimed",
                "warning": (
                    "local-chain projection preserves encoded local-duration and "
                    "collective-node totals, but coarser backend scheduling may "
                    "change compute/communication overlap and wall cycles"
                ),
            },
        )

    def test_compaction_rejects_branches_and_cross_stream_links(self):
        events = (
            TraceEvent("root", "root", "compute", 1.0, stream="7"),
            TraceEvent(
                "left",
                "left",
                "compute",
                1.0,
                stream="7",
                dependencies=("root",),
            ),
            TraceEvent(
                "right",
                "right",
                "compute",
                1.0,
                stream="8",
                dependencies=("root",),
            ),
        )
        projected, report = _compact_local_chains(events)
        self.assertEqual(projected, events)
        self.assertEqual(report["collapsed_nodes"], 0)
        self.assertEqual(report["coalesced_chains"], 0)

        phase_events = (
            TraceEvent("before", "before", "compute", 1.0, stream="7"),
            TraceEvent(
                "phase",
                "phase",
                "collective",
                1.0,
                stream="25",
                collective="all_reduce",
                message_bytes=8,
                group_role="dp",
            ),
            TraceEvent(
                "after",
                "after",
                "compute",
                1.0,
                stream="7",
                dependencies=("before",),
            ),
        )
        projected, report = _compact_local_chains(phase_events)
        self.assertEqual(projected, phase_events)
        self.assertEqual(report["collapsed_nodes"], 0)

    def test_rejects_p2p_without_exact_rank_route(self):
        trace = WorkloadTrace(
            events=(
                TraceEvent(
                    "send",
                    "unrouted send",
                    "collective",
                    1.0,
                    collective="send",
                    message_bytes=4,
                ),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires integer p2p_source_rank"):
                export_chakra(
                    trace,
                    Path(directory) / "unrouted",
                    ranks=2,
                    parallelism=Parallelism(tp=2, pp=1, dp=1),
                )

    def test_rejects_local_chain_compaction_for_pipeline_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "does not support pipeline"):
                export_chakra(
                    self._pipeline_trace(),
                    Path(directory) / "pipeline-compact",
                    ranks=2,
                    parallelism=Parallelism(tp=1, pp=2, dp=1),
                    compact_local_chains=True,
                )


if __name__ == "__main__":
    unittest.main()
