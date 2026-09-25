from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

import pytest

from scaletether.framework_counterfactual import CONTRACT_SCHEMA
from scaletether.megatron_tp_transform import (
    SCHEMA,
    MegatronTpTransformError,
    compile_megatron_tp2_to_tp4,
)
from scaletether.megatron_tp_validation import (
    MegatronTpValidationError,
    _expected_measurement,
    main as validation_main,
    validate_frozen_tp2_to_tp4,
)
from scaletether.pipeline import MEGATRON_CORE_SCHEDULE_COMMIT
from scaletether.schema import TraceEvent, WorkloadTrace


def _compute(
    identifier: str,
    phase: str,
    dimensions: list[list[int]],
    dependency: str | None = None,
    operator_name: str = "aten::mm",
) -> TraceEvent:
    def strides(shape: list[int]) -> list[int]:
        result: list[int] = []
        running = 1
        for dimension in reversed(shape):
            result.append(running)
            running *= dimension
        return list(reversed(result))

    return TraceEvent(
        identifier,
        identifier,
        "compute",
        7.0,
        dependencies=() if dependency is None else (dependency,),
        observed_start_us=10.0,
        sm_fraction=0.5,
        metadata={
            "kernel_signature": f"observed-{identifier}",
            "kernel_code_identity": {"code_object_sha256": "a" * 64},
            "framework_phase_marker": {"name": phase},
            "kernel_launch_payload": {
                "framework_operator": {
                    "name": operator_name,
                    "input_dims": dimensions,
                    "input_types": ["float", "float"],
                    "input_strides": [strides(shape) for shape in dimensions],
                }
            },
        },
    )


def _prepared() -> WorkloadTrace:
    events = (
        _compute("f-mm", "forward", [[80, 256], [256, 512]]),
        TraceEvent(
            "f-tp",
            "forward allreduce",
            "collective",
            3.0,
            dependencies=("f-mm",),
            observed_start_us=20.0,
            collective="all_reduce",
            message_bytes=80 * 256 * 4,
            group_role="tp",
            group_size=2,
            sm_fraction=0.25,
        ),
        _compute(
            "b-mm",
            "backward",
            [[80, 512], [512, 256]],
            "f-tp",
            operator_name="aten::gelu_backward",
        ),
        TraceEvent(
            "b-tp",
            "backward allreduce",
            "collective",
            3.0,
            dependencies=("f-tp",),
            observed_start_us=40.0,
            collective="all_reduce",
            message_bytes=80 * 256 * 4,
            group_role="tp",
            group_size=2,
            sm_fraction=0.25,
        ),
        _compute("opt", "optimizer", [[512, 256]], "b-tp"),
    )
    contract = {
        "schema": CONTRACT_SCHEMA,
        "framework": "megatron-core",
        "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
        "applicability": {
            "status": "certified-for-bounded-rewrite",
            "source_parallelism": {"tp": 2, "pp": 1, "dp": 1, "ep": 1},
            "fixed_dimensions": {
                "tp": 2,
                "hidden_size": 256,
                "ffn_hidden_size": 1024,
            },
        },
        "model": {
            "num_layers": 1,
            "hidden_size": 256,
            "ffn_hidden_size": 1024,
            "sequence_length": 80,
            "micro_batch_size": 1,
        },
        "tp_profiles": {
            "2": {
                "forward_event_ids": ["f-mm", "f-tp"],
                "backward_event_ids": ["b-mm", "b-tp"],
                "optimizer_event_ids": ["opt"],
            }
        },
    }
    return WorkloadTrace(
        events,
        source={"target": "h100"},
        metadata={
            "framework_counterfactual": contract,
            "framework_measurement": _expected_measurement(80),
        },
    )


def _heldout_from_generated(generated: WorkloadTrace) -> WorkloadTrace:
    events: list[TraceEvent] = []
    for index, event in enumerate(generated.events):
        metadata = dict(event.metadata)
        phase = metadata.pop("pipeline_phase")
        metadata.pop("tp_semantic_rewrite", None)
        metadata["framework_phase_marker"] = {"name": f"megatron_mlp_{phase}"}
        if event.kind == "compute":
            metadata["kernel_code_identity"] = {"code_object_sha256": "d" * 64}
        elif event.kind == "collective":
            metadata["observed_group_size"] = 4
        events.append(
            replace(
                event,
                duration_us=1.0,
                observed_start_us=float(index),
                sm_fraction=0.1,
                group_size=None if event.kind == "collective" else event.group_size,
                metadata=metadata,
            )
        )
    return WorkloadTrace(
        tuple(events),
        source={"target": "h100"},
        metadata={
            "capture_limitations": [],
            "framework_measurement": _expected_measurement(80),
        },
    )


def _as_distributed(heldout: WorkloadTrace) -> WorkloadTrace:
    return replace(
        heldout,
        source={"kind": "distributed-capture", "target": "h100"},
        metadata={
            key: value
            for key, value in heldout.metadata.items()
            if key != "capture_limitations"
        }
        | {
            "rank_metadata": [
                {"rank": rank, "capture_limitations": []} for rank in range(4)
            ]
        },
    )


def test_rewrites_tp2_operator_shapes_to_tp4_without_timing_claim() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    assert summary["schema"] == SCHEMA
    assert summary["target_parallelism"]["tp"] == 4
    assert generated.source == {
        "kind": "framework-semantic-counterfactual",
        "target": "h100",
        "framework": "megatron-core",
        "source_semantic_input_sha256": summary["source_semantic_input_sha256"],
        "candidate_training_executed": False,
    }
    assert set(generated.metadata) == {
        "megatron_tp_semantic_rewrite",
        "provenance",
    }
    assert {event.rank for event in generated.events} == {0, 1, 2, 3}
    assert len(generated.events) == 20
    rank0 = [event for event in generated.events if event.rank == 0]
    assert [event.id for event in rank0] == [
        "rank0::f-mm",
        "rank0::f-tp",
        "rank0::b-mm",
        "rank0::b-tp",
        "rank0::opt",
    ]
    compute = [event for event in rank0 if event.kind == "compute"]
    assert all(event.duration_us == 0.0 for event in compute)
    assert all(event.observed_start_us is None for event in compute)
    assert all(event.sm_fraction is None for event in compute)
    assert all(
        event.metadata["kernel_signature"].startswith("framework-operator-v1:")
        for event in compute
    )
    assert all("kernel_code_identity" not in event.metadata for event in compute)
    assert all("profiler_event_id" not in event.metadata for event in compute)
    assert compute[0].metadata["kernel_launch_payload"]["framework_operator"][
        "input_dims"
    ] == [[80, 256], [256, 256]]
    assert compute[0].metadata["kernel_launch_payload"]["framework_operator"][
        "input_strides"
    ] == [[256, 1], [256, 1]]
    collectives = [event for event in rank0 if event.kind == "collective"]
    assert all(event.group_size == 4 for event in collectives)
    assert all(event.duration_us == 0.0 for event in collectives)
    assert all("observed_group_size" not in event.metadata for event in collectives)
    assert all("communicator" not in event.metadata for event in collectives)
    assert not [event for event in rank0 if event.kind == "memory"]


def test_preserves_rank_local_dependency_graph() -> None:
    generated, _summary = compile_megatron_tp2_to_tp4(_prepared())
    rank3 = {event.id: event for event in generated.events if event.rank == 3}
    assert rank3["rank3::opt"].dependencies == ("rank3::b-tp",)
    assert rank3["rank3::b-tp"].dependencies == (
        "rank3::f-tp",
        "rank3::b-mm",
    )


def test_rejects_unmodelled_operator_dimension() -> None:
    source = _prepared()
    changed = replace(
        source,
        events=tuple(
            _compute("opt", "optimizer", [[513, 256]], "b-tp")
            if event.id == "opt"
            else event
            for event in source.events
        ),
    )
    with pytest.raises(MegatronTpTransformError, match="unmodelled dimension"):
        compile_megatron_tp2_to_tp4(changed)


def test_rejects_compute_without_framework_operator_shapes() -> None:
    source = _prepared()
    changed = replace(
        source,
        events=tuple(
            replace(event, metadata={}) if event.id == "opt" else event
            for event in source.events
        ),
    )
    with pytest.raises(MegatronTpTransformError, match="kernel launch"):
        compile_megatron_tp2_to_tp4(changed)


def test_rejects_memory_operation_after_steady_state_warmup() -> None:
    source = _prepared()
    memory = TraceEvent(
        "f-copy",
        "Memcpy HtoD (Pageable -> Device)",
        "memory",
        1.0,
        metadata={"raw_trace_category": "gpu_memcpy", "input_shapes": []},
    )
    metadata = dict(source.metadata)
    contract = dict(metadata["framework_counterfactual"])
    profiles = dict(contract["tp_profiles"])
    profile = dict(profiles["2"])
    profile["forward_event_ids"] = ["f-copy", *profile["forward_event_ids"]]
    profiles["2"] = profile
    contract["tp_profiles"] = profiles
    metadata["framework_counterfactual"] = contract
    changed = replace(
        source,
        events=(memory, *source.events),
        metadata=metadata,
    )
    with pytest.raises(MegatronTpTransformError, match="unexpectedly contains"):
        compile_megatron_tp2_to_tp4(changed)


def test_rejects_wrong_source_tp_domain() -> None:
    source = _prepared()
    metadata = dict(source.metadata)
    contract = dict(metadata["framework_counterfactual"])
    applicability = dict(contract["applicability"])
    applicability["source_parallelism"] = {"tp": 1, "pp": 1, "dp": 1, "ep": 1}
    contract["applicability"] = applicability
    metadata["framework_counterfactual"] = contract
    with pytest.raises(MegatronTpTransformError, match="outside the pinned"):
        compile_megatron_tp2_to_tp4(replace(source, metadata=metadata))


def test_rejects_dependency_outside_profile() -> None:
    source = _prepared()
    changed = replace(
        source,
        events=tuple(
            replace(event, dependencies=("setup",)) if event.id == "f-mm" else event
            for event in source.events
        ),
    )
    with pytest.raises(MegatronTpTransformError, match="depends outside"):
        compile_megatron_tp2_to_tp4(changed)


def test_heldout_tp_gate_accepts_exact_operator_collective_and_causal_structure() -> (
    None
):
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    report = validate_frozen_tp2_to_tp4(
        generated,
        _heldout_from_generated(generated),
        compilation_summary=summary,
        expected_sequence_length=80,
    )
    assert report["status"] == "passed"
    assert report["claim"].endswith("kernel-lowering-unresolved")
    assert set(report["ranks"]) == {"0", "1", "2", "3"}


def test_heldout_tp_gate_collapses_unambiguous_split_k_lowering() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    heldout = _heldout_from_generated(generated)
    events = list(heldout.events)
    root_index = next(
        index
        for index, event in enumerate(events)
        if event.rank == 0 and event.kind == "compute"
    )
    root = events[root_index]
    reduction_id = root.id + "::split-k-reduction"
    reduction = replace(
        root,
        id=reduction_id,
        name="void cublasLt::splitKreduce_kernel<32, 16, int, float>",
        dependencies=(root.id,),
        observed_start_us=(root.observed_start_us or 0.0) + 0.1,
        metadata={
            **root.metadata,
            "kernel_launch_payload": {
                **root.metadata["kernel_launch_payload"],
                "name": "void cublasLt::splitKreduce_kernel<32, 16, int, float>",
            },
        },
    )
    events = [
        replace(
            event,
            dependencies=tuple(
                reduction_id if dependency == root.id else dependency
                for dependency in event.dependencies
            ),
        )
        if event.id != reduction_id
        else event
        for event in events
    ]
    events.insert(root_index + 1, reduction)
    report = validate_frozen_tp2_to_tp4(
        generated,
        replace(heldout, events=tuple(events)),
        compilation_summary=summary,
        expected_sequence_length=80,
    )
    assert report["status"] == "passed"
    lowering = report["ranks"]["0"]["heldout"]["kernel_lowering"]
    assert lowering["status"] == "unresolved-not-compared"
    assert lowering["collapsed_kernel_count"] == 1


def test_heldout_tp_gate_collapses_cublas_execute_split_k_lowering() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    heldout = _heldout_from_generated(generated)
    events = list(heldout.events)
    root_index = next(
        index
        for index, event in enumerate(events)
        if event.rank == 0 and event.kind == "compute"
    )
    root = events[root_index]
    reduction_id = root.id + "::execute-split-k"
    split_name = "sm80_xmma_execute_split_k_kernel__5x_cublas"
    reduction = replace(
        root,
        id=reduction_id,
        name=split_name,
        dependencies=(root.id,),
        observed_start_us=(root.observed_start_us or 0.0) + 0.1,
        metadata={
            **root.metadata,
            "kernel_launch_payload": {
                **root.metadata["kernel_launch_payload"],
                "name": split_name,
            },
        },
    )
    rewritten = [
        replace(
            event,
            dependencies=tuple(
                reduction_id if dependency == root.id else dependency
                for dependency in event.dependencies
            ),
        )
        for event in events
    ]
    rewritten.insert(root_index + 1, reduction)
    report = validate_frozen_tp2_to_tp4(
        generated,
        replace(heldout, events=tuple(rewritten)),
        compilation_summary=summary,
        expected_sequence_length=80,
    )
    assert report["status"] == "passed"
    assert (
        report["ranks"]["0"]["heldout"]["kernel_lowering"]["collapsed_kernel_count"]
        == 1
    )


def test_heldout_tp_gate_rejects_ambiguous_split_k_lowering() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    heldout = _heldout_from_generated(generated)
    root = next(
        event for event in heldout.events if event.rank == 0 and event.kind == "compute"
    )
    reduction = replace(
        root,
        id=root.id + "::ambiguous-split-k",
        name="void cublasLt::splitKreduce_kernel<32, 16, int, float>",
        dependencies=(
            root.id,
            next(
                event.id
                for event in heldout.events
                if event.rank == 0 and event.id != root.id
            ),
        ),
    )
    with pytest.raises(
        MegatronTpValidationError, match="one exact semantic predecessor"
    ):
        validate_frozen_tp2_to_tp4(
            generated,
            replace(heldout, events=(*heldout.events, reduction)),
            compilation_summary=summary,
            expected_sequence_length=80,
        )


def test_heldout_tp_gate_accepts_distributed_rank_limitation_inventories() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    report = validate_frozen_tp2_to_tp4(
        generated,
        _as_distributed(_heldout_from_generated(generated)),
        compilation_summary=summary,
        expected_sequence_length=80,
    )
    assert report["status"] == "passed"


def test_heldout_tp_gate_rejects_incomplete_distributed_rank() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    heldout = _as_distributed(_heldout_from_generated(generated))
    rank_metadata = [dict(item) for item in heldout.metadata["rank_metadata"]]
    rank_metadata[2]["capture_limitations"] = ["missing edge"]
    heldout = replace(
        heldout,
        metadata={**heldout.metadata, "rank_metadata": rank_metadata},
    )
    with pytest.raises(MegatronTpValidationError, match="rank 2 capture"):
        validate_frozen_tp2_to_tp4(
            generated,
            heldout,
            compilation_summary=summary,
            expected_sequence_length=80,
        )


def test_heldout_tp_gate_rejects_operator_shape_mismatch() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    heldout = _heldout_from_generated(generated)
    changed_events = list(heldout.events)
    event_index = next(
        index
        for index, candidate in enumerate(changed_events)
        if candidate.kind == "compute"
    )
    event = changed_events[event_index]
    metadata = dict(event.metadata)
    launch = dict(metadata["kernel_launch_payload"])
    operator = dict(launch["framework_operator"])
    operator["input_dims"] = [[80, 256], [256, 257]]
    launch["framework_operator"] = operator
    metadata["kernel_launch_payload"] = launch
    changed_events[event_index] = replace(event, metadata=metadata)
    with pytest.raises(MegatronTpValidationError, match="operator-shape"):
        validate_frozen_tp2_to_tp4(
            generated,
            replace(heldout, events=tuple(changed_events)),
            compilation_summary=summary,
            expected_sequence_length=80,
        )


def test_heldout_tp_gate_rejects_causal_mismatch() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    heldout = _heldout_from_generated(generated)
    changed = replace(
        heldout,
        events=tuple(
            replace(event, dependencies=()) if event.id == "rank2::opt" else event
            for event in heldout.events
        ),
    )
    with pytest.raises(MegatronTpValidationError, match="causal reachability"):
        validate_frozen_tp2_to_tp4(
            generated,
            changed,
            compilation_summary=summary,
            expected_sequence_length=80,
        )


def test_heldout_tp_gate_rejects_post_warmup_memory_operation() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    heldout = _heldout_from_generated(generated)
    memory = TraceEvent(
        "rank1::unexpected-copy",
        "Memcpy HtoD (Pageable -> Device)",
        "memory",
        1.0,
        rank=1,
        device=1,
        metadata={
            "raw_trace_category": "gpu_memcpy",
            "framework_phase_marker": {"name": "megatron_mlp_forward"},
        },
    )
    changed = replace(heldout, events=(*heldout.events, memory))
    with pytest.raises(MegatronTpValidationError, match="steady-state.*memory"):
        validate_frozen_tp2_to_tp4(
            generated,
            changed,
            compilation_summary=summary,
            expected_sequence_length=80,
        )


def test_heldout_tp_gate_rejects_generated_timing_claim() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    changed = replace(
        generated,
        events=(replace(generated.events[0], duration_us=1.0), *generated.events[1:]),
    )
    with pytest.raises(MegatronTpValidationError, match="unauthorized timing"):
        validate_frozen_tp2_to_tp4(
            changed,
            _heldout_from_generated(generated),
            compilation_summary=summary,
            expected_sequence_length=80,
        )


def test_heldout_tp_gate_rejects_inherited_source_runtime_metadata() -> None:
    generated, summary = compile_megatron_tp2_to_tp4(_prepared())
    changed = replace(
        generated,
        metadata={**generated.metadata, "rank_sources": [{"device_name": "H100"}]},
    )
    with pytest.raises(MegatronTpValidationError, match="provenance is not bound"):
        validate_frozen_tp2_to_tp4(
            changed,
            _heldout_from_generated(generated),
            compilation_summary=summary,
            expected_sequence_length=80,
        )


def test_tp_cli_freezes_before_scoring() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepared = root / "prepared.json"
        generated = root / "generated.json"
        freeze = root / "freeze.json"
        heldout = root / "heldout.json"
        report = root / "report.json"
        _prepared().dump(prepared)
        assert (
            validation_main(
                [
                    "freeze",
                    "--prepared-source",
                    str(prepared),
                    "--generated-output",
                    str(generated),
                    "--output",
                    str(freeze),
                    "--expected-sequence-length",
                    "80",
                ]
            )
            == 0
        )
        _heldout_from_generated(WorkloadTrace.load(generated)).dump(heldout)
        assert (
            validation_main(
                [
                    "score",
                    "--generated",
                    str(generated),
                    "--freeze",
                    str(freeze),
                    "--actual-target",
                    str(heldout),
                    "--output",
                    str(report),
                    "--expected-sequence-length",
                    "80",
                ]
            )
            == 0
        )
        frozen = freeze.read_text(encoding="utf-8")
        freeze.write_text(frozen.replace("labeled-causal", "weakened-causal"))
        with pytest.raises(MegatronTpValidationError, match="freeze is invalid"):
            validation_main(
                [
                    "score",
                    "--generated",
                    str(generated),
                    "--freeze",
                    str(freeze),
                    "--actual-target",
                    str(heldout),
                    "--output",
                    str(report),
                    "--expected-sequence-length",
                    "80",
                ]
            )


def test_opened_target_audit_is_explicitly_nonprospective() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepared = root / "prepared.json"
        generated = root / "generated.json"
        freeze = root / "freeze.json"
        heldout = root / "heldout.json"
        report = root / "audit.json"
        _prepared().dump(prepared)
        assert (
            validation_main(
                [
                    "freeze",
                    "--prepared-source",
                    str(prepared),
                    "--generated-output",
                    str(generated),
                    "--output",
                    str(freeze),
                    "--expected-sequence-length",
                    "80",
                ]
            )
            == 0
        )
        _as_distributed(_heldout_from_generated(WorkloadTrace.load(generated))).dump(
            heldout
        )
        assert (
            validation_main(
                [
                    "audit-opened-target",
                    "--generated",
                    str(generated),
                    "--freeze",
                    str(freeze),
                    "--actual-target",
                    str(heldout),
                    "--output",
                    str(report),
                    "--expected-sequence-length",
                    "80",
                ]
            )
            == 0
        )
        document = json.loads(report.read_text(encoding="utf-8"))
        assert document["status"] == "descriptive-structural-pass"
        assert document["prospective_claim"] is False


def test_opened_target_audit_accepts_but_does_not_promote_legacy_freeze() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepared = root / "prepared.json"
        generated = root / "generated.json"
        freeze = root / "freeze.json"
        heldout = root / "heldout.json"
        report = root / "audit.json"
        _prepared().dump(prepared)
        assert (
            validation_main(
                [
                    "freeze",
                    "--prepared-source",
                    str(prepared),
                    "--generated-output",
                    str(generated),
                    "--output",
                    str(freeze),
                    "--expected-sequence-length",
                    "80",
                ]
            )
            == 0
        )
        frozen = json.loads(freeze.read_text(encoding="utf-8"))
        frozen["schema"] = "scaletether-megatron-tp-prediction-freeze-v4"
        frozen["required_validation_gates"] = [
            "framework-operator-shape-multiset-v1",
            "steady-state-bounded-ddp-memory-structure-v2",
            "tp-collective-order-bytes-v1",
            "labeled-causal-reachability-v1",
        ]
        freeze.write_text(json.dumps(frozen) + "\n", encoding="utf-8")
        _as_distributed(_heldout_from_generated(WorkloadTrace.load(generated))).dump(
            heldout
        )
        assert (
            validation_main(
                [
                    "audit-opened-target",
                    "--generated",
                    str(generated),
                    "--freeze",
                    str(freeze),
                    "--actual-target",
                    str(heldout),
                    "--output",
                    str(report),
                    "--expected-sequence-length",
                    "80",
                ]
            )
            == 0
        )
        document = json.loads(report.read_text(encoding="utf-8"))
        assert document["status"] == "descriptive-structural-pass"
        assert document["prospective_claim"] is False
        assert document["audited_original_required_validation_gates"][0].endswith(
            "multiset-v1"
        )


def test_opened_target_audit_reports_structural_difference() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepared = root / "prepared.json"
        generated = root / "generated.json"
        freeze = root / "freeze.json"
        heldout = root / "heldout.json"
        report = root / "audit.json"
        _prepared().dump(prepared)
        assert (
            validation_main(
                [
                    "freeze",
                    "--prepared-source",
                    str(prepared),
                    "--generated-output",
                    str(generated),
                    "--output",
                    str(freeze),
                    "--expected-sequence-length",
                    "80",
                ]
            )
            == 0
        )
        actual = _as_distributed(_heldout_from_generated(WorkloadTrace.load(generated)))
        events = list(actual.events)
        compute_index = next(
            index for index, event in enumerate(events) if event.kind == "compute"
        )
        event = events[compute_index]
        metadata = dict(event.metadata)
        launch = dict(metadata["kernel_launch_payload"])
        operator = dict(launch["framework_operator"])
        operator["input_dims"] = [[80, 256], [256, 257]]
        launch["framework_operator"] = operator
        metadata["kernel_launch_payload"] = launch
        events[compute_index] = replace(event, metadata=metadata)
        replace(actual, events=tuple(events)).dump(heldout)
        assert (
            validation_main(
                [
                    "audit-opened-target",
                    "--generated",
                    str(generated),
                    "--freeze",
                    str(freeze),
                    "--actual-target",
                    str(heldout),
                    "--output",
                    str(report),
                    "--expected-sequence-length",
                    "80",
                ]
            )
            == 0
        )
        document = json.loads(report.read_text(encoding="utf-8"))
        assert document["status"] == "descriptive-structural-rejection"
        assert document["prospective_claim"] is False
        assert document["rank_differences"]["0"]["missing_generated_event_keys"]
        assert document["rank_differences"]["0"]["unexpected_heldout_event_keys"]
