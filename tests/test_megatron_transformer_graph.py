from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import pytest

from scaletether.megatron_transformer_graph import (
    TransformerGraphError,
    _context,
    _last_ready_component,
    _raw_context,
    _semanticize_rank,
    semanticize_pp_rank,
    compile_dp_candidate,
    compile_pp_candidate,
    compile_tp_candidate,
    main,
)
from scaletether.schema import TraceEvent, WorkloadTrace
from tests.test_megatron_transformer_family import measurement


ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads(
    (ROOT / "tests/fixtures/level2-h100-transformer-family-v1.json").read_text()
)


def source_trace(sequence: int = 256) -> WorkloadTrace:
    events = []
    payload = sequence * 256 * 4
    declared_components = measurement(sequence)["component_markers"]["components"]
    for rank in (0, 1):
        previous = None
        seen_contexts = set()
        for index in range(8):
            phase = "forward" if index < 4 else "backward"
            layer = 0 if index % 4 < 2 else 1
            component = (
                "self_attention.linear_proj" if index % 2 == 0 else "mlp.linear_fc2"
            )
            compute_id = f"r{rank}-compute-{index}"
            collective_id = f"r{rank}-collective-{index}"
            marker = {
                "process_group_ranks": "[0, 1]",
                "framework_phase_marker": {
                    "name": f"megatron_transformer_{phase}_component:layers.{layer}.{component}"
                },
                "kernel_launch_payload": {
                    "framework_operator": {"name": "aten::mm", "input_dims": [[1, 256]]}
                },
            }
            seen_contexts.add((phase, f"layers.{layer}.{component}"))
            events.append(
                TraceEvent(
                    id=compute_id,
                    name="kernel",
                    kind="compute",
                    duration_us=1.0,
                    rank=rank,
                    device=rank,
                    dependencies=() if previous is None else (previous,),
                    metadata=marker,
                )
            )
            events.append(
                TraceEvent(
                    id=collective_id,
                    name="ncclAllReduce",
                    kind="collective",
                    duration_us=2.0,
                    rank=rank,
                    device=rank,
                    dependencies=(compute_id,),
                    collective="all_reduce",
                    message_bytes=payload,
                    group_role="tp",
                    group_size=2,
                    metadata=marker,
                )
            )
            previous = collective_id
        for phase in ("forward", "backward"):
            for component in declared_components:
                if (phase, component) in seen_contexts:
                    continue
                identifier = f"r{rank}-{phase}-{component}"
                events.append(
                    TraceEvent(
                        id=identifier,
                        name="kernel",
                        kind="compute",
                        duration_us=1.0,
                        rank=rank,
                        device=rank,
                        dependencies=(previous,),
                        metadata={
                            "framework_phase_marker": {
                                "name": f"megatron_transformer_{phase}_component:{component}"
                            },
                            "kernel_launch_payload": {
                                "framework_operator": {"name": "aten::add"}
                            },
                        },
                    )
                )
                previous = identifier
        for phase in ("forward", "backward"):
            identifier = f"r{rank}-{phase}-terminal-loss"
            events.append(
                TraceEvent(
                    id=identifier,
                    name="loss-kernel",
                    kind="compute",
                    duration_us=1.0,
                    rank=rank,
                    device=rank,
                    dependencies=(previous,),
                    metadata={
                        "framework_phase_marker": {
                            "name": f"megatron_transformer_{phase}_component:terminal_loss"
                        },
                        "kernel_launch_payload": {
                            "framework_operator": {"name": "aten::mse_loss"}
                        },
                    },
                )
            )
            previous = identifier
        optimizer = f"r{rank}-optimizer"
        events.append(
            TraceEvent(
                id=optimizer,
                name="optimizer-kernel",
                kind="compute",
                duration_us=1.0,
                rank=rank,
                device=rank,
                dependencies=(previous,),
                metadata={
                    "framework_phase_marker": {
                        "name": "megatron_transformer_optimizer"
                    },
                    "kernel_launch_payload": {
                        "framework_operator": {"name": "aten::add_"}
                    },
                },
            )
        )
    return WorkloadTrace(
        events=tuple(events),
        source={"target": "h100", "rank_count": 2},
        metadata={"framework_measurement": measurement(sequence), "capture_limitations": []},
    )


def partial_physical_rank() -> list[TraceEvent]:
    events: list[TraceEvent] = []

    def add(
        operator: str,
        context: tuple[str, str] | None = None,
    ) -> None:
        metadata = {
            "kernel_launch_payload": {
                "framework_operator": {"name": operator}
            }
        }
        if context is not None:
            metadata["framework_phase_marker"] = {
                "name": f"megatron_transformer_{context[0]}_component:{context[1]}"
            }
        events.append(
            TraceEvent(
                id=f"physical-{len(events)}",
                name=operator,
                kind="compute",
                duration_us=1.0,
                rank=0,
                device=0,
                dependencies=(() if not events else (events[-1].id,)),
                metadata=metadata,
            )
        )

    add("aten::native_layer_norm")
    for layer in (0, 1):
        add("aten::mm", ("forward", f"layers.{layer}.self_attention.linear_qkv"))
        add("aten::bmm")
        add("aten::mm", ("forward", f"layers.{layer}.self_attention.linear_proj"))
        add("aten::add")
        add("aten::mm", ("forward", f"layers.{layer}.mlp.linear_fc1"))
        add("aten::gelu")
        add("aten::mm", ("forward", f"layers.{layer}.mlp.linear_fc2"))
        add("aten::add")
        if layer == 0:
            add("aten::native_layer_norm")
    add("aten::native_layer_norm", ("forward", "final_layernorm"))
    add("aten::mse_loss")
    add("aten::mse_loss_backward")
    add("aten::native_layer_norm_backward", ("backward", "final_layernorm"))
    add("aten::sum")
    for layer in (1, 0):
        add("aten::mm", ("backward", f"layers.{layer}.mlp.linear_fc2"))
        add("aten::gelu_backward")
        add("aten::mm", ("backward", f"layers.{layer}.mlp.linear_fc1"))
        add("aten::native_layer_norm_backward")
        add("aten::mm", ("backward", f"layers.{layer}.self_attention.linear_proj"))
        add("aten::bmm")
        add("aten::mm", ("backward", f"layers.{layer}.self_attention.linear_qkv"))
        add("aten::native_layer_norm_backward")
    events[-1].metadata["framework_phase_marker"] = {
        "name": "megatron_transformer_optimizer"
    }
    # Preserve the final layer tail and add a distinct terminal optimizer event.
    events[-1].metadata.pop("framework_phase_marker")
    add("aten::add_")
    events[-1].metadata["framework_phase_marker"] = {
        "name": "megatron_transformer_optimizer"
    }
    return events


def test_partial_physical_trace_is_completely_semanticized() -> None:
    semantic = _semanticize_rank(partial_physical_rank())
    assert len(semantic) == len(partial_physical_rank())
    assert all(event.metadata.get("framework_semantic_context") for event in semantic)
    contexts = [_context(event) for event in semantic]
    assert contexts[0] == ("forward", "layers.0")
    assert ("forward", "terminal_loss") in contexts
    assert ("backward", "terminal_loss") in contexts
    assert contexts[-2] == ("backward", "layers.0")
    assert contexts[-1] == ("optimizer", "step.optimizer")
    provenance = {
        event.metadata["framework_semantic_context"]["provenance"]
        for event in semantic
    }
    assert provenance == {"direct-framework-marker", "frozen-framework-rule"}


def test_same_component_marker_gap_does_not_create_a_false_reentry() -> None:
    events = partial_physical_rank()
    marker_index = next(
        index
        for index, event in enumerate(events)
        if _raw_context(event) == ("backward", "layers.0.mlp.linear_fc1")
    )
    gap = replace(
        events[marker_index],
        id="same-component-gap",
        metadata={
            "kernel_launch_payload": {
                "framework_operator": {"name": "aten::view"}
            }
        },
    )
    repeated_marker = replace(
        events[marker_index],
        id="same-component-repeated-marker",
    )
    events[marker_index + 1:marker_index + 1] = [gap, repeated_marker]

    semantic = _semanticize_rank(events)
    by_id = {event.id: event for event in semantic}
    assert _context(by_id[gap.id]) == ("backward", "layers.0.mlp.linear_fc1")
    assert (
        by_id[gap.id].metadata["framework_semantic_context"]["rule"]
        == "same-component-marker-gap"
    )


def test_partial_physical_trace_abstains_on_missing_or_reordered_landmark() -> None:
    events = partial_physical_rank()
    del events[1]
    with pytest.raises(TransformerGraphError, match="landmark order"):
        _semanticize_rank(events)


def test_partial_physical_trace_abstains_without_layer_boundary_norm() -> None:
    events = partial_physical_rank()
    boundary = next(
        event
        for event in events
        if event.id == "physical-9"
    )
    boundary.metadata["kernel_launch_payload"]["framework_operator"]["name"] = "aten::add"
    with pytest.raises(TransformerGraphError, match="layer boundary"):
        _semanticize_rank(events)


def test_last_ready_parameter_without_leaf_marker_maps_to_phase_frontier() -> None:
    source_measurement = measurement()
    source_measurement["gradient_readiness"]["parameter_names"] = [
        "layers.0.input_layernorm.bias"
    ]
    assert _last_ready_component(
        source_measurement, _semanticize_rank(partial_physical_rank())
    ) is None


def pp_stage_one_rank() -> list[TraceEvent]:
    events: list[TraceEvent] = []

    def add(phase: str, component: str | None, instance: str) -> None:
        outer = {
            "name": f"megatron_transformer_{phase}",
            "instance_id": instance,
        }
        metadata: dict = {
            "framework_phase_marker": outer,
            "framework_phase_marker_stack": [outer],
            "kernel_launch_payload": {"framework_operator": {"name": "aten::op"}},
        }
        if component is not None:
            name = (
                "megatron_transformer_terminal_loss"
                if component == "terminal_loss"
                else f"megatron_transformer_{phase}_component:{component}"
            )
            leaf = {"name": name, "instance_id": f"{instance}:{component}"}
            metadata["framework_phase_marker"] = leaf
            metadata["framework_phase_marker_stack"] = [outer, leaf]
        events.append(
            TraceEvent(
                id=f"pp-{len(events)}", name="kernel", kind="compute",
                duration_us=1.0, rank=2, device=2, metadata=metadata,
            )
        )

    leaves = [
        "layers.1.self_attention.linear_qkv",
        "layers.1.self_attention.linear_proj",
        "layers.1.mlp.linear_fc1",
        "layers.1.mlp.linear_fc2",
    ]
    add("forward", None, "f0")
    for component in leaves:
        add("forward", component, "f0")
        add("forward", None, "f0")
    add("forward", "final_layernorm", "f0")
    add("forward", "terminal_loss", "f0")
    add("backward", None, "b0")
    add("backward", "final_layernorm", "b0")
    for component in reversed(leaves):
        add("backward", None, "b0")
        add("backward", component, "b0")
    add("backward", None, "b0")
    return events


def test_pp_instance_semanticizer_covers_outer_gaps_and_terminal_loss() -> None:
    semantic = semanticize_pp_rank(pp_stage_one_rank(), 1)
    contexts = [_context(event) for event in semantic]
    assert len(contexts) == len(semantic)
    assert ("forward", "terminal_loss") in contexts
    assert ("backward", "terminal_loss") in contexts
    assert ("forward", "layers.1.self_attention") in contexts
    assert ("backward", "layers.1.mlp") in contexts


def test_pp_instance_semanticizer_rejects_leaf_reordering() -> None:
    events = pp_stage_one_rank()
    events[3] = replace(events[3], metadata=events[5].metadata)
    with pytest.raises(TransformerGraphError, match="landmark order"):
        semanticize_pp_rank(events, 1)


def test_tp_compiler_replicates_semantic_dag_without_target_timing_claim() -> None:
    source = source_trace()
    candidate, report = compile_tp_candidate(
        MATRIX, source, "TP-A", target_exists=False
    )
    source_rank_events = sum(event.rank == 0 for event in source.events)
    assert len(candidate.events) == 4 * source_rank_events
    assert report["target_event_count"] == len(candidate.events)
    assert report["target_training_executed"] is False
    assert report["timing_claim"] == "none"
    assert all(event.duration_us == 0 for event in candidate.events)
    assert all(event.observed_start_us is None for event in candidate.events)
    collectives = [event for event in candidate.events if event.kind == "collective"]
    assert len(collectives) == 32
    assert {event.group_size for event in collectives} == {4}
    assert {event.message_bytes for event in collectives} == {262144}
    assert {event.rank for event in candidate.events} == {0, 1, 2, 3}


def test_tp_compiler_derives_nondefault_tp_degree_from_matrix() -> None:
    matrix = copy.deepcopy(MATRIX)
    target = next(row for row in matrix["targets"] if row["id"] == "TP-A")
    target["parallelism"]["tp"] = 8
    candidate, report = compile_tp_candidate(
        matrix, source_trace(), "TP-A", target_exists=False
    )
    collectives = [event for event in candidate.events if event.kind == "collective"]
    assert report["source_tp"] == 2
    assert report["target_tp"] == 8
    assert {event.rank for event in candidate.events} == set(range(8))
    assert {event.group_size for event in collectives} == {8}
    assert {
        tuple(event.metadata["target_tp_group"]) for event in collectives
    } == {tuple(range(8))}


def test_tp_compiler_preserves_rank_local_dependencies() -> None:
    candidate, _ = compile_tp_candidate(MATRIX, source_trace(), "TP-A", target_exists=False)
    ids = {event.id for event in candidate.events}
    assert all(set(event.dependencies) <= ids for event in candidate.events)
    assert all(
        dependency.startswith(f"rank{event.rank}::")
        for event in candidate.events
        for dependency in event.dependencies
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda trace: trace.events[17].metadata.update(framework_phase_marker={"name": "wrong"}), "marker|symmetric"),
        (lambda trace: object.__setattr__(trace.events[1], "message_bytes", 1), "symmetric|collective"),
        (lambda trace: trace.metadata.update(capture_limitations=["missing"]), "limitations"),
    ],
)
def test_tp_compiler_abstains_on_incomplete_or_asymmetric_source(mutate, match: str) -> None:
    trace = source_trace()
    mutate(trace)
    with pytest.raises(TransformerGraphError, match=match):
        compile_tp_candidate(MATRIX, trace, "TP-A", target_exists=False)


def test_tp_compiler_rejects_malformed_or_wrong_collective_membership() -> None:
    trace = source_trace()
    collective_event = next(event for event in trace.events if event.kind == "collective")
    collective_event.metadata["process_group_ranks"] = "[0, 2]"
    with pytest.raises(TransformerGraphError, match="TP rule"):
        compile_tp_candidate(MATRIX, trace, "TP-A", target_exists=False)


def test_tp_compiler_rejects_target_leakage() -> None:
    with pytest.raises(ValueError, match="already exists"):
        compile_tp_candidate(MATRIX, source_trace(), "TP-A", target_exists=True)


def test_cli_freezes_candidate_before_absent_heldout() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        matrix = root / "matrix.json"
        source = root / "source.json"
        candidate = root / "candidate.json"
        report = root / "report.json"
        freeze = root / "freeze.json"
        heldout = root / "heldout"
        matrix.write_text(json.dumps(MATRIX), encoding="utf-8")
        source_trace().dump(source)
        assert main(
            [
                "--matrix",
                str(matrix),
                "--source",
                str(source),
                "--target-id",
                "TP-A",
                "--heldout-guard",
                str(heldout),
                "--candidate",
                str(candidate),
                "--report",
                str(report),
                "--freeze",
                str(freeze),
            ]
        ) == 0
        document = json.loads(freeze.read_text())
        assert document["status"] == "frozen-before-heldout-execution"
        assert document["candidate_training_executed"] is False
        assert document["heldout_guard_absent"] is True
        assert candidate.is_file() and report.is_file()


def test_cli_freezes_dp_candidate_before_absent_heldout(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate.json"
    report = tmp_path / "report.json"
    freeze = tmp_path / "freeze.json"
    matrix.write_text(json.dumps(MATRIX), encoding="utf-8")
    source_trace().dump(source)
    assert main(
        [
            "--matrix", str(matrix), "--source", str(source),
            "--target-id", "DP-A", "--heldout-guard", str(tmp_path / "heldout"),
            "--candidate", str(candidate), "--report", str(report),
            "--freeze", str(freeze),
        ]
    ) == 0
    generated = WorkloadTrace.load(candidate)
    assert len([event for event in generated.events if event.group_role == "dp"]) == 12


def test_cli_refuses_existing_heldout_or_prediction_output(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    source = tmp_path / "source.json"
    matrix.write_text(json.dumps(MATRIX), encoding="utf-8")
    source_trace().dump(source)
    common = [
        "--matrix", str(matrix), "--source", str(source), "--target-id", "TP-A",
        "--heldout-guard", str(tmp_path / "heldout"),
        "--candidate", str(tmp_path / "candidate.json"),
        "--report", str(tmp_path / "report.json"),
        "--freeze", str(tmp_path / "freeze.json"),
    ]
    (tmp_path / "heldout").mkdir()
    with pytest.raises(TransformerGraphError, match="held-out target exists"):
        main(common)
    (tmp_path / "heldout").rmdir()
    (tmp_path / "candidate.json").write_text("existing", encoding="utf-8")
    with pytest.raises(TransformerGraphError, match="output already exists"):
        main(common)


def test_dp_compiler_builds_orthogonal_groups_bucket_and_optimizer_gate() -> None:
    candidate, report = compile_dp_candidate(
        MATRIX, source_trace(), "DP-A", target_exists=False
    )
    assert report["dp_collective_count"] == 12
    assert report["dp_gradient_bucket_payload_bytes_per_rank"] == [32]
    assert report["dp_forward_metadata_payload_bytes_per_rank"] == [8, 4]
    dp_events = [event for event in candidate.events if event.group_role == "dp"]
    assert len(dp_events) == 12
    assert {tuple(event.metadata["target_dp_group"]) for event in dp_events} == {
        (0, 2),
        (1, 3),
    }
    assert {tuple(event.metadata["target_tp_group"]) for event in dp_events} == {
        (0, 1),
        (2, 3),
    }
    assert {event.message_bytes for event in dp_events} == {4, 8, 32}
    events = {event.id: event for event in candidate.events}

    def ancestors(identifier: str) -> set[str]:
        pending = list(events[identifier].dependencies)
        seen: set[str] = set()
        while pending:
            dependency = pending.pop()
            if dependency in seen:
                continue
            seen.add(dependency)
            if dependency in events:
                pending.extend(events[dependency].dependencies)
        return seen

    optimizer_by_rank = {
        rank: [
            event for event in candidate.events
            if event.rank == rank
            and event.metadata.get("pipeline_phase") == "optimizer"
        ]
        for rank in range(4)
    }
    assert all(optimizer_by_rank.values())
    for rank, optimizers in optimizer_by_rank.items():
        terminal_bucket = f"rank{rank}::dp-gradient-bucket-0"
        assert terminal_bucket in events
        assert all(terminal_bucket in ancestors(event.id) for event in optimizers)


def test_dp_compiler_derives_nondefault_dp_degree_from_matrix() -> None:
    matrix = copy.deepcopy(MATRIX)
    target = next(row for row in matrix["targets"] if row["id"] == "DP-A")
    target["parallelism"]["dp"] = 3
    target["global_batch_size"] = 3
    candidate, report = compile_dp_candidate(
        matrix, source_trace(), "DP-A", target_exists=False
    )
    dp_events = [event for event in candidate.events if event.group_role == "dp"]
    assert report["target_world_size"] == 6
    assert report["target_tp"] == 2
    assert report["target_dp"] == 3
    assert report["dp_collective_count"] == 18
    assert {event.group_size for event in dp_events} == {3}
    assert {tuple(event.metadata["target_dp_group"]) for event in dp_events} == {
        (0, 2, 4),
        (1, 3, 5),
    }
    assert {tuple(event.metadata["target_tp_group"]) for event in dp_events} == {
        (0, 1),
        (2, 3),
        (4, 5),
    }


def test_dp_compiler_abstains_if_last_ready_component_is_unobserved() -> None:
    trace = source_trace()
    trace.metadata["framework_measurement"]["gradient_readiness"]["parameter_names"] = [
        "not.a.component.weight"
    ]
    with pytest.raises(ValueError, match="component|manifest|readiness"):
        compile_dp_candidate(MATRIX, trace, "DP-A", target_exists=False)


@pytest.mark.parametrize(
    "target,microbatches,payload,p2p_count,compound_count",
    [("PP-A", 2, 262144, 16, 4), ("PP-B", 4, 327680, 32, 12)],
)
def test_pp_compiler_builds_schedule_routes_and_stage_optimizers(
    target: str,
    microbatches: int,
    payload: int,
    p2p_count: int,
    compound_count: int,
) -> None:
    sequence = 256 if target == "PP-A" else 320
    source = source_trace(sequence)
    source.metadata["framework_measurement"]["parameter_manifest"] = {
        "schema": "ordered-megatron-parameter-manifest-v1",
        "ordering": "module.named_parameters-before-ddp-wrap",
        "parameter_count": 3,
        "total_numel": 20,
        "parameters": [
            {"order": 0, "name": "layers.0.weight", "shape": [2, 4], "numel": 8, "dtype": "float32", "requires_grad": True},
            {"order": 1, "name": "layers.1.weight", "shape": [2, 4], "numel": 8, "dtype": "float32", "requires_grad": True},
            {"order": 2, "name": "final_layernorm.weight", "shape": [4], "numel": 4, "dtype": "float32", "requires_grad": True},
        ],
    }
    source.metadata["framework_measurement"]["gradient_readiness"] = {
        "schema": "megatron-parameter-gradient-readiness-v1",
        "observation": "unmeasured-complete-warmup-step",
        "parameter_names": [
            "final_layernorm.weight",
            "layers.1.weight",
            "layers.0.weight",
        ],
        "parameter_count": 3,
        "measured_step_repeatability_required": True,
    }
    source.metadata["framework_measurement"]["gradient_readiness"] = {
        "schema": "megatron-parameter-gradient-readiness-v1",
        "observation": "unmeasured-complete-warmup-step",
        "parameter_names": ["final_layernorm.weight", "layers.1.weight", "layers.0.weight"],
        "parameter_count": 3,
        "measured_step_repeatability_required": True,
    }
    candidate, report = compile_pp_candidate(
        MATRIX, source, target, target_exists=False
    )
    assert report["microbatches"] == microbatches
    assert report["p2p_event_count"] == p2p_count
    assert report["compound_p2p_exchange_count"] == compound_count
    p2p = [event for event in candidate.events if event.group_role == "pp"]
    assert {event.message_bytes for event in p2p} == {payload}
    sends = {
        (event.metadata["pipeline_phase"], event.metadata["pipeline_microbatch"], event.metadata["p2p_source_rank"], event.metadata["p2p_destination_rank"])
        for event in p2p if event.collective == "send"
    }
    recvs = {
        (event.metadata["pipeline_phase"], event.metadata["pipeline_microbatch"], event.metadata["p2p_source_rank"], event.metadata["p2p_destination_rank"])
        for event in p2p if event.collective == "recv"
    }
    assert sends == recvs
    tp_events = [event for event in candidate.events if event.group_role == "tp"]
    assert tp_events
    assert {
        event.metadata.get("semantic_component") for event in tp_events
    } == {"layer.self_attention.linear_proj", "layer.mlp.linear_fc2"}
    by_id = {event.id: event for event in candidate.events}
    consumers = {event.id: [] for event in candidate.events}
    for event in candidate.events:
        for dependency in event.dependencies:
            if dependency in consumers:
                consumers[dependency].append(event.id)
    for lane in (0, 1):
        for microbatch in range(1, microbatches):
            pairs = (
                (
                    f"rank{lane}::mb{microbatch}::forward-send",
                    f"rank{lane}::mb{microbatch - 1}::backward-recv",
                ),
                (
                    f"rank{lane + 2}::mb{microbatch - 1}::backward-send",
                    f"rank{lane + 2}::mb{microbatch}::forward-recv",
                ),
            )
            for send_id, recv_id in pairs:
                send = by_id[send_id]
                recv = by_id[recv_id]
                recv_microbatch = recv.metadata["pipeline_microbatch"]
                previous_recv = (
                    None
                    if recv_microbatch == 0
                    else (
                        f"rank{recv.rank}::mb{recv_microbatch - 1}::"
                        f"{recv.metadata['pipeline_phase']}-recv"
                    )
                )
                remote_send = (
                    f"rank{lane + 2}::mb{microbatch - 1}::backward-send"
                    if recv.metadata["pipeline_phase"] == "backward"
                    else f"rank{lane}::mb{microbatch}::forward-send"
                )
                assert recv.dependencies == tuple(
                    dependency
                    for dependency in (previous_recv, remote_send)
                    if dependency is not None
                )
                assert recv.metadata["pipeline_native_stream_predecessor"] == (
                    previous_recv
                )
                send_microbatch = send.metadata["pipeline_microbatch"]
                previous_send = (
                    None
                    if send_microbatch == 0
                    else (
                        f"rank{send.rank}::mb{send_microbatch - 1}::"
                        f"{send.metadata['pipeline_phase']}-send"
                    )
                )
                assert send.metadata["pipeline_native_stream_predecessor"] == (
                    previous_send
                )
                if previous_send is not None:
                    assert previous_send in send.dependencies
                joined = [
                    by_id[identifier]
                    for identifier in consumers[recv_id]
                    if by_id[identifier].rank == recv.rank
                    and send_id in by_id[identifier].dependencies
                ]
                assert len(joined) == 1
                assert (
                    send.metadata["pipeline_compound_exchange"]
                    == recv.metadata["pipeline_compound_exchange"]
                )
    # Warmup and cooldown remain one-way rather than being manufactured into
    # steady-state compound exchanges.
    for lane in (0, 1):
        terminal = microbatches - 1
        one_way = (
            f"rank{lane}::mb0::forward-send",
            f"rank{lane}::mb{terminal}::backward-recv",
            f"rank{lane + 2}::mb0::forward-recv",
            f"rank{lane + 2}::mb{terminal}::backward-send",
        )
        assert all(
            "pipeline_compound_exchange" not in by_id[identifier].metadata
            for identifier in one_way
        )
    # Every receive, including the final cooldown receive, is driven only by
    # its previous same-direction receive and exact matched remote send.  It
    # must never inherit an unrelated local compute producer.
    for recv in (event for event in p2p if event.collective == "recv"):
        microbatch = recv.metadata["pipeline_microbatch"]
        previous_recv = (
            None
            if microbatch == 0
            else (
                f"rank{recv.rank}::mb{microbatch - 1}::"
                f"{recv.metadata['pipeline_phase']}-recv"
            )
        )
        remote_send = (
            f"rank{recv.metadata['p2p_source_rank']}::mb{microbatch}::"
            f"{recv.metadata['pipeline_phase']}-send"
        )
        assert recv.dependencies == tuple(
            dependency
            for dependency in (previous_recv, remote_send)
            if dependency is not None
        )
        assert recv.metadata["pipeline_native_stream_predecessor"] == previous_recv
    # Every outgoing send retains a non-P2P local tensor producer.  Stream
    # ordering may add the previous send but never replaces that producer.
    for send in (event for event in p2p if event.collective == "send"):
        local_non_p2p = [
            by_id[dependency]
            for dependency in send.dependencies
            if by_id[dependency].rank == send.rank
            and by_id[dependency].group_role != "pp"
        ]
        assert local_non_p2p
    for event in tp_events:
        assert event.metadata.get("source_event_count") == len(
            event.metadata.get("source_event_ids", [])
        ) == 1
        assert len(event.dependencies) == 1
        predecessor = by_id[event.dependencies[0]]
        assert predecessor.metadata.get("semantic_component") == event.metadata.get(
            "semantic_component"
        )
    optimizer = [
        event for event in candidate.events
        if event.metadata.get("pipeline_phase") == "optimizer"
    ]
    assert {(event.rank, event.metadata["parameter_name"]) for event in optimizer} == {
        (0, "layers.0.weight"), (1, "layers.0.weight"),
        (2, "layers.1.weight"), (3, "layers.1.weight"),
        (2, "final_layernorm.weight"), (3, "final_layernorm.weight"),
    }
    terminal = [
        event
        for event in candidate.events
        if event.metadata.get("semantic_component") == "final_layernorm"
    ]
    assert terminal
    assert {event.rank for event in terminal} == {2, 3}


def test_pp_compiler_preserves_observed_boundary_layer_asymmetry() -> None:
    source = source_trace()
    for event in source.events:
        marker = event.metadata.get("framework_phase_marker", {}).get("name")
        if marker == (
            "megatron_transformer_backward_component:"
            "layers.0.self_attention.linear_qkv"
        ) and event.kind == "compute":
            event.metadata["kernel_launch_payload"]["framework_operator"][
                "name"
            ] = "aten::boundary_specific"
    source.metadata["framework_measurement"]["parameter_manifest"] = {
        "schema": "ordered-megatron-parameter-manifest-v1",
        "ordering": "module.named_parameters-before-ddp-wrap",
        "parameter_count": 3,
        "total_numel": 20,
        "parameters": [
            {"order": 0, "name": "layers.0.weight", "shape": [2, 4], "numel": 8, "dtype": "float32", "requires_grad": True},
            {"order": 1, "name": "layers.1.weight", "shape": [2, 4], "numel": 8, "dtype": "float32", "requires_grad": True},
            {"order": 2, "name": "final_layernorm.weight", "shape": [4], "numel": 4, "dtype": "float32", "requires_grad": True},
        ],
    }
    source.metadata["framework_measurement"]["gradient_readiness"] = {
        "schema": "megatron-parameter-gradient-readiness-v1",
        "observation": "unmeasured-complete-warmup-step",
        "parameter_names": [
            "final_layernorm.weight",
            "layers.1.weight",
            "layers.0.weight",
        ],
        "parameter_count": 3,
        "measured_step_repeatability_required": True,
    }
    candidate, report = compile_pp_candidate(
        MATRIX, source, "PP-A", target_exists=False
    )
    assert report["stage_profile_rule"].startswith("observed-global-layer-0")
    stage_zero_operators = {
        event.metadata.get("framework_operator")
        for event in candidate.events
        if event.rank == 0
    }
    stage_one_operators = {
        event.metadata.get("framework_operator")
        for event in candidate.events
        if event.rank == 2
    }
    assert "aten::boundary_specific" in stage_zero_operators
    assert "aten::boundary_specific" not in stage_one_operators
