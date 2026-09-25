from __future__ import annotations

import copy
from unittest.mock import patch

from scaletether.megatron_transformer_comparison import compare_generated_to_physical
from scaletether.megatron_transformer_graph import TransformerGraphError
from scaletether.schema import TraceEvent, WorkloadTrace


def event(
    identifier: str,
    rank: int,
    *,
    phase: str,
    component: str,
    operator: str | None,
    candidate: bool,
) -> TraceEvent:
    metadata = (
        {
            "pipeline_phase": phase,
            "semantic_component": component,
            "framework_operator": ({"name": operator} if operator else None),
            "source_kind": "compute",
            "source_event_count": 1,
        }
        if candidate
        else {
            "framework_phase_marker": {
                "name": f"megatron_transformer_{phase}_component:{component}"
            },
            "kernel_launch_payload": {
                "framework_operator": ({"name": operator} if operator else {})
            },
        }
    )
    return TraceEvent(
        id=identifier,
        name="kernel",
        kind="compute",
        duration_us=0 if candidate else 1,
        rank=rank,
        device=rank,
        metadata=metadata,
    )


def collective(identifier: str, rank: int, *, candidate: bool) -> TraceEvent:
    metadata = (
        {
            "target_tp_group": [0, 1],
            "pipeline_phase": "forward",
            "semantic_component": "layers.0.mlp",
        }
        if candidate
        else {
            "process_group_ranks": [0, 1],
            "framework_phase_marker": {
                "name": "megatron_transformer_forward_component:layers.0.mlp"
            },
        }
    )
    return TraceEvent(
        id=identifier,
        name="all_reduce",
        kind="collective",
        duration_us=0 if candidate else 1,
        rank=rank,
        device=rank,
        collective="all_reduce",
        message_bytes=1024,
        group_role="tp",
        group_size=2,
        metadata=metadata,
    )


def traces() -> tuple[WorkloadTrace, WorkloadTrace]:
    generated = []
    physical = []
    for rank in (0, 1):
        generated.append(
            event(
                f"g{rank}", rank, phase="forward", component="layers.0.mlp",
                operator=None if rank == 0 else "aten::mm", candidate=True,
            )
        )
        physical.append(
            event(
                f"p{rank}", rank, phase="forward", component="layers.0.mlp",
                operator=None if rank == 0 else "aten::mm", candidate=False,
            )
        )
        generated.append(collective(f"gc{rank}", rank, candidate=True))
        physical.append(collective(f"pc{rank}", rank, candidate=False))
    graph = {
        "target_id": "TP-A",
        "target_training_executed": False,
        "timing_claim": "none",
        "kernel_code_claim": "none",
    }
    return (
        WorkloadTrace(
            tuple(generated),
            source={"candidate_training_executed": False},
            metadata={"transformer_semantic_graph": graph},
        ),
        WorkloadTrace(tuple(physical), source={}, metadata={}),
    )


def pp_traces() -> tuple[WorkloadTrace, WorkloadTrace]:
    generated = []
    physical = []
    expected = {
        0: ["forward", "forward", "backward", "backward"],
        1: ["forward", "forward", "backward", "backward"],
        2: ["forward", "backward", "forward", "backward"],
        3: ["forward", "backward", "forward", "backward"],
    }
    for rank in range(4):
        stage = rank // 2
        generated.append(
            TraceEvent(
                id=f"g-opt-{rank}", name="optimizer", kind="compute",
                duration_us=0, rank=rank, device=rank,
                metadata={
                    "pipeline_phase": "optimizer",
                    "parameter_name": f"layers.{stage}.weight",
                },
            )
        )
        for index, phase in enumerate(expected[rank]):
            physical.append(
                TraceEvent(
                    id=f"p-phase-{rank}-{index}", name=phase, kind="compute",
                    duration_us=1, rank=rank, device=rank,
                    observed_start_us=float(index),
                    metadata={
                        "framework_phase_marker": {
                            "name": f"megatron_transformer_{phase}",
                            "instance_id": f"rank{rank}-phase{index}",
                        }
                    },
                )
            )
    graph = {
        "target_id": "PP-A",
        "target_training_executed": False,
        "timing_claim": "none",
        "kernel_code_claim": "none",
        "microbatches": 2,
    }
    stages = [
        {"parameter_manifest": {"parameters": [{"name": f"layers.{stage}.weight"}]}}
        for stage in range(2)
    ]
    return (
        WorkloadTrace(
            tuple(generated),
            source={"candidate_training_executed": False},
            metadata={"transformer_semantic_graph": graph},
        ),
        WorkloadTrace(
            tuple(physical),
            source={},
            metadata={
                "framework_measurement": {
                    "pipeline_stage_evidence": {"stages": stages}
                }
            },
        ),
    )


def dp_traces(rank_count: int = 3) -> tuple[WorkloadTrace, WorkloadTrace]:
    generated: list[TraceEvent] = []
    physical: list[TraceEvent] = []
    ranks = list(range(rank_count))
    for rank in ranks:
        generated.append(
            event(
                f"g-forward-{rank}", rank, phase="forward",
                component="layers.0.mlp", operator=None, candidate=True,
            )
        )
        physical.append(
            event(
                f"p-forward-{rank}", rank, phase="forward",
                component="layers.0.mlp", operator=None, candidate=False,
            )
        )
        generated.append(
            TraceEvent(
                id=f"g-dp-{rank}", name="all_reduce", kind="collective",
                duration_us=0, rank=rank, device=rank, collective="all_reduce",
                message_bytes=4096, group_role="dp", group_size=rank_count,
                metadata={
                    "target_dp_group": ranks,
                    "pipeline_phase": "backward",
                    "semantic_component": "layers.0.mlp",
                },
            )
        )
        physical.append(
            TraceEvent(
                id=f"p-dp-{rank}", name="all_reduce", kind="collective",
                duration_us=1, rank=rank, device=rank, collective="all_reduce",
                message_bytes=4096, group_role="dp", group_size=rank_count,
                metadata={
                    "process_group_ranks": ranks,
                    "framework_phase_marker": {
                        "name": "megatron_transformer_backward_component:layers.0.mlp"
                    },
                },
            )
        )
        generated.append(
            TraceEvent(
                id=f"g-optimizer-{rank}", name="optimizer", kind="compute",
                duration_us=0, rank=rank, device=rank,
                dependencies=(f"g-dp-{rank}",),
                metadata={
                    "pipeline_phase": "optimizer",
                    "semantic_component": "step.optimizer",
                },
            )
        )
        optimizer = event(
            f"p-optimizer-{rank}", rank, phase="optimizer",
            component="step.optimizer", operator=None, candidate=False,
        )
        optimizer.metadata["framework_semantic_context"] = {
            "phase": "optimizer", "component": "step.optimizer"
        }
        physical.append(optimizer)
    manifest = {
        "ordering": "module.named_parameters-before-ddp-wrap",
        "parameter_count": 2,
        "parameters": [
            {"name": "layers.0.weight", "numel": 16, "order": 0},
            {"name": "layers.0.bias", "numel": 4, "order": 1},
        ],
    }
    readiness = {
        "schema": "fixture-gradient-readiness-v1",
        "parameter_count": 2,
        # Backward readiness is intentionally a different order from module
        # registration; it must be an exact permutation, not the same list.
        "parameter_names": ["layers.0.bias", "layers.0.weight"],
    }
    measurement = {
        "parameter_manifest": manifest,
        "gradient_readiness": readiness,
    }
    graph = {
        "target_id": "DP-X",
        "target_training_executed": False,
        "timing_claim": "none",
        "kernel_code_claim": "none",
        "dp_collective_count": rank_count,
    }
    return (
        WorkloadTrace(
            tuple(generated),
            source={"candidate_training_executed": False},
            metadata={
                "transformer_semantic_graph": graph,
                "framework_measurement": copy.deepcopy(measurement),
            },
        ),
        WorkloadTrace(
            tuple(physical), source={},
            metadata={"framework_measurement": copy.deepcopy(measurement)},
        ),
    )


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_exact_comparison_passes_and_hashes_none_operator(_gate) -> None:
    candidate, actual = traces()
    report = compare_generated_to_physical(candidate, actual, {}, "TP-A")
    assert report["passed"] is True
    assert report["generated_compute_sha256"] == report["physical_compute_sha256"]
    assert report["generated_collective_sha256"] == report["physical_collective_sha256"]
    assert report["compute_multiset_delta"] is None
    assert report["collective_multiset_delta"] is None
    assert report["claim"] == "held-out-transformer-structural-validation-no-timing-claim"


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_operator_change_is_diagnostic_but_collective_mutation_fails(_gate) -> None:
    candidate, actual = traces()
    actual_events = list(actual.events)
    actual_events[2].metadata["kernel_launch_payload"]["framework_operator"] = {
        "name": "aten::add"
    }
    object.__setattr__(actual_events[3], "message_bytes", 2048)
    report = compare_generated_to_physical(
        candidate,
        WorkloadTrace(tuple(actual_events), actual.source, actual.metadata),
        {},
        "TP-A",
    )
    assert report["passed"] is False
    assert "framework-component-presence" not in report["failures"]
    assert "framework-component-order" not in report["failures"]
    assert "collective-role-bytes-membership-multiset" in report["failures"]
    assert report["compute_multiset_delta"] is None
    assert report["operator_diagnostic_gating"] is False
    assert report["operator_diagnostic_delta"] == {
        "generated_only_total": 1,
        "physical_only_total": 1,
        "generated_only": [
            {"key": [1, "forward", "layers.0.mlp", "compute", "aten::mm"], "count": 1}
        ],
        "physical_only": [
            {"key": [1, "forward", "layers.0.mlp", "compute", "aten::add"], "count": 1}
        ],
        "limit_per_side": 64,
        "truncated": False,
    }
    assert report["collective_multiset_delta"]["generated_only_total"] == 1
    assert report["collective_multiset_delta"]["physical_only_total"] == 1


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_component_presence_and_order_remain_fail_closed(_gate) -> None:
    candidate, actual = traces()
    physical = list(actual.events)
    physical[2].metadata["framework_phase_marker"]["name"] = (
        "megatron_transformer_forward_component:layers.0.input_layernorm"
    )
    report = compare_generated_to_physical(
        candidate,
        WorkloadTrace(tuple(physical), actual.source, actual.metadata),
        {},
        "TP-A",
    )
    assert report["passed"] is False
    assert "framework-component-presence" in report["failures"]
    assert "framework-component-order" in report["failures"]


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
@patch(
    "scaletether.megatron_transformer_comparison._semanticize_actual_tp_dp",
    side_effect=TransformerGraphError("diagnostic attribution failure"),
)
def test_semantic_attribution_failure_reports_exact_reason(_semanticize, _gate) -> None:
    candidate, actual = traces()
    report = compare_generated_to_physical(candidate, actual, {}, "TP-A")
    assert report["passed"] is False
    assert "physical-semantic-attribution" in report["failures"]
    assert report["semantic_attribution_error"] == "diagnostic attribution failure"


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_semantic_dependency_reachability_is_exact_and_fail_closed(_gate) -> None:
    candidate, actual = traces()
    generated = list(candidate.events)
    physical = list(actual.events)
    object.__setattr__(generated[1], "dependencies", (generated[0].id,))
    object.__setattr__(generated[3], "dependencies", (generated[2].id,))
    object.__setattr__(physical[1], "dependencies", (physical[0].id,))
    object.__setattr__(physical[3], "dependencies", (physical[2].id,))
    exact = compare_generated_to_physical(
        WorkloadTrace(tuple(generated), candidate.source, candidate.metadata),
        WorkloadTrace(tuple(physical), actual.source, actual.metadata),
        {},
        "TP-A",
    )
    assert exact["passed"] is True
    assert exact["dependency_matches"] is True
    assert exact["generated_dependency_sha256"] == exact["physical_dependency_sha256"]

    physical[1].metadata["framework_phase_marker"]["name"] = (
        "megatron_transformer_forward_component:layers.0.self_attention"
    )
    missing = compare_generated_to_physical(
        WorkloadTrace(tuple(generated), candidate.source, candidate.metadata),
        WorkloadTrace(tuple(physical), actual.source, actual.metadata),
        {},
        "TP-A",
    )
    assert missing["passed"] is False
    assert missing["dependency_matches"] is False
    assert "framework-semantic-dependency-reachability" in missing["failures"]


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_backend_dependency_reachability_rejects_inventory_only_match(_gate) -> None:
    candidate, actual = traces()
    generated = list(candidate.events)
    # The call inventory, enclosing component, rank membership, and occurrence
    # all still match.  Only the dependency consumed by Chakra differs.
    object.__setattr__(generated[1], "dependencies", (generated[0].id,))
    report = compare_generated_to_physical(
        WorkloadTrace(tuple(generated), candidate.source, candidate.metadata),
        actual,
        {},
        "TP-A",
    )
    assert report["dependency_matches"] is True
    assert report["backend_dependency_matches"] is False
    assert "backend-semantic-dependency-reachability" in report["failures"]
    assert report["passed"] is False


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_dependency_cycle_abstains_with_an_explicit_error(_gate) -> None:
    candidate, actual = traces()
    generated = list(candidate.events)
    object.__setattr__(generated[0], "dependencies", (generated[1].id,))
    object.__setattr__(generated[1], "dependencies", (generated[0].id,))
    report = compare_generated_to_physical(
        WorkloadTrace(tuple(generated), candidate.source, candidate.metadata),
        actual,
        {},
        "TP-A",
    )
    assert report["passed"] is False
    assert report["dependency_matches"] is False
    assert report["dependency_error"] is not None
    assert "dependency cycle" in report["dependency_error"]


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_optimizer_events_are_excluded_symmetrically(_gate) -> None:
    candidate, actual = traces()
    optimizer = event(
        "physical-optimizer",
        0,
        phase="optimizer",
        component="step.optimizer",
        operator="aten::_foreach_add_",
        candidate=False,
    )
    optimizer.metadata["framework_semantic_context"] = {
        "phase": "optimizer",
        "component": "step.optimizer",
    }
    report = compare_generated_to_physical(
        candidate,
        WorkloadTrace(actual.events + (optimizer,), actual.source, actual.metadata),
        {},
        "TP-A",
    )
    assert report["passed"] is True
    assert report["compute_multiset_delta"] is None


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_dp_optimizer_attribution_uses_generic_rank_and_manifest_contract(_gate) -> None:
    candidate, actual = dp_traces(rank_count=3)
    report = compare_generated_to_physical(candidate, actual, {}, "DP-X")
    assert report["passed"] is True
    optimizer = report["dp_optimizer_attribution"]
    assert optimizer["passed"] is True
    assert optimizer["expected_ranks"] == [0, 1, 2]
    assert optimizer["generated_optimizer_ranks"] == [0, 1, 2]
    assert optimizer["physical_optimizer_ranks"] == [0, 1, 2]
    assert optimizer["checks"]["optimizer_reaches_dp_collective"] is True
    assert optimizer["checks"]["gradient_readiness_is_manifest_permutation"] is True


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_dp_optimizer_attribution_rejects_missing_rank_and_manifest_drift(_gate) -> None:
    candidate, actual = dp_traces(rank_count=2)
    physical = tuple(
        event for event in actual.events
        if not (event.rank == 1 and event.id.startswith("p-optimizer"))
    )
    metadata = copy.deepcopy(actual.metadata)
    metadata["framework_measurement"]["parameter_manifest"]["parameters"][0][
        "numel"
    ] = 17
    report = compare_generated_to_physical(
        candidate, WorkloadTrace(physical, actual.source, metadata), {}, "DP-X"
    )
    assert report["passed"] is False
    assert "dp-optimizer-attribution" in report["failures"]
    checks = report["dp_optimizer_attribution"]["checks"]
    assert checks["parameter_manifest_exact"] is False
    assert checks["optimizer_rank_coverage"] is False
    assert checks["physical_optimizer_terminal"] is False


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_dp_optimizer_attribution_rejects_unready_or_unordered_optimizer(_gate) -> None:
    candidate, actual = dp_traces(rank_count=2)
    metadata = copy.deepcopy(actual.metadata)
    metadata["framework_measurement"]["gradient_readiness"]["parameter_names"] = []
    generated = list(candidate.events)
    optimizer = generated.pop(2)
    generated.insert(0, optimizer)
    report = compare_generated_to_physical(
        WorkloadTrace(tuple(generated), candidate.source, candidate.metadata),
        WorkloadTrace(actual.events, actual.source, metadata),
        {}, "DP-X",
    )
    assert report["passed"] is False
    checks = report["dp_optimizer_attribution"]["checks"]
    assert checks["gradient_readiness_exact"] is False
    assert checks["generated_optimizer_terminal"] is False


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_dp_optimizer_attribution_rejects_missing_bucket_dependency(_gate) -> None:
    candidate, actual = dp_traces(rank_count=5)
    generated = list(candidate.events)
    optimizer = next(event for event in generated if event.id == "g-optimizer-3")
    object.__setattr__(optimizer, "dependencies", ())
    report = compare_generated_to_physical(
        WorkloadTrace(tuple(generated), candidate.source, candidate.metadata),
        actual,
        {},
        "DP-X",
    )
    assert report["passed"] is False
    optimizer_report = report["dp_optimizer_attribution"]
    assert optimizer_report["checks"]["optimizer_reaches_dp_collective"] is False
    assert optimizer_report["logical_optimizer_ownership_count"] == 10


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_tp_dp_target",
    return_value={"passed": True, "failures": []},
)
def test_candidate_provenance_and_physical_admission_are_mandatory(gate) -> None:
    candidate, actual = traces()
    candidate.source["candidate_training_executed"] = True
    gate.return_value = {"passed": False, "failures": ["hardware"]}
    report = compare_generated_to_physical(candidate, actual, {}, "TP-A")
    assert report["passed"] is False
    assert "candidate-provenance" in report["failures"]
    assert "physical-target-admission" in report["failures"]


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_pp_target",
    return_value={"passed": True, "failures": []},
)
def test_pp_schedule_and_optimizer_placement_pass_exactly(_gate) -> None:
    candidate, actual = pp_traces()
    report = compare_generated_to_physical(candidate, actual, {}, "PP-A")
    assert report["passed"] is True
    assert report["generated_optimizer_parameter_count"] == 4
    assert report["physical_optimizer_parameter_count"] == 4
    assert report["pp_phase_orders"]["0"]["observed"] == [
        "forward", "forward", "backward", "backward"
    ]
    assert report["pp_phase_orders"]["2"]["observed"] == [
        "forward", "backward", "forward", "backward"
    ]


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_pp_target",
    return_value={"passed": True, "failures": []},
)
def test_pp_schedule_reordering_and_parameter_misplacement_fail(_gate) -> None:
    candidate, actual = pp_traces()
    candidate.events[0].metadata["parameter_name"] = "layers.1.weight"
    physical = list(actual.events)
    first = next(event for event in physical if event.rank == 0 and event.observed_start_us == 0)
    third = next(event for event in physical if event.rank == 0 and event.observed_start_us == 2)
    object.__setattr__(first, "observed_start_us", 2.0)
    object.__setattr__(third, "observed_start_us", 0.0)
    report = compare_generated_to_physical(
        candidate,
        WorkloadTrace(tuple(physical), actual.source, actual.metadata),
        {},
        "PP-A",
    )
    assert report["passed"] is False
    assert "pp-optimizer-parameter-placement" in report["failures"]
    assert "rank-0-pp-phase-order" in report["failures"]


@patch(
    "scaletether.megatron_transformer_comparison.validate_physical_pp_target",
    side_effect=ValueError("PP logical collective record is malformed"),
)
def test_malformed_pp_collective_is_a_typed_invalid_graph_rejection(_gate) -> None:
    candidate, actual = pp_traces()
    report = compare_generated_to_physical(candidate, actual, {}, "PP-A")
    assert report["passed"] is False
    assert report["failures"] == ["invalid-graph"]
    assert report["invalid_graph"]["code"] == "malformed-pp-logical-collective"
    assert report["invalid_graph"]["error_type"] == "ValueError"
