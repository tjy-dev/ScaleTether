from __future__ import annotations

import json

import pytest

from scaletether.megatron_layout_transform import (
    MegatronLayoutTransformError,
    compile_megatron_mlp_layout_candidate,
    main,
    validate_megatron_mlp_layout_candidate,
)
from scaletether.schema import TraceEvent, WorkloadTrace


def _source(*, tp: int = 4, hidden: int = 2048, ffn: int = 8192) -> WorkloadTrace:
    tokens = 2048
    payload = tokens * hidden * 4
    events: list[TraceEvent] = []
    for rank in range(tp):
        prefix = f"rank{rank}"
        events.extend(
            [
                TraceEvent(
                    id=f"{prefix}::forward-compute",
                    name="measured column/GELU/row forward",
                    kind="compute",
                    duration_us=10.0,
                    rank=rank,
                    device=rank,
                ),
                TraceEvent(
                    id=f"{prefix}::forward-all-reduce",
                    name="measured row-parallel output reduction",
                    kind="collective",
                    duration_us=2.0,
                    rank=rank,
                    device=rank,
                    dependencies=(f"{prefix}::forward-compute",),
                    collective="all_reduce",
                    message_bytes=payload,
                    group_role="tp",
                    group_size=tp,
                    metadata={"process_group_ranks": list(range(tp))},
                ),
                TraceEvent(
                    id=f"{prefix}::backward-compute",
                    name="measured column/GELU/row backward",
                    kind="compute",
                    duration_us=20.0,
                    rank=rank,
                    device=rank,
                    dependencies=(f"{prefix}::forward-all-reduce",),
                ),
                TraceEvent(
                    id=f"{prefix}::backward-all-reduce",
                    name="measured column-parallel input-gradient reduction",
                    kind="collective",
                    duration_us=2.0,
                    rank=rank,
                    device=rank,
                    dependencies=(f"{prefix}::backward-compute",),
                    collective="all_reduce",
                    message_bytes=payload,
                    group_role="tp",
                    group_size=tp,
                    metadata={"process_group_ranks": list(range(tp))},
                ),
            ]
        )
    return WorkloadTrace(
        events=tuple(events),
        source={"kind": "physical", "target": "h100", "rank_count": tp},
        metadata={
            "framework_measurement": {
                "schema": "megatron-core-mlp-measurement-v1",
                "framework": "megatron-core",
                "framework_commit": "f8e1ac64b0587ff7002a18fbaa5ecdeaeb8491be",
                "tensor_parallel_size": tp,
                "data_parallel_size": 1,
                "model": {
                    "num_layers": 1,
                    "hidden_size": hidden,
                    "ffn_hidden_size": ffn,
                    "sequence_length": 2048,
                    "micro_batch_size": 1,
                    "global_batch_size": 1,
                    "parameter_dtype": "float32",
                    "parameter_element_bytes": 4,
                    "activation_dtype": "float32",
                    "activation_element_bytes": 4,
                    "bias": False,
                },
                "layer": {
                    "kind": "megatron-core-column-gelu-row-mlp-v1",
                    "column_gather_output": False,
                    "row_input_is_parallel": True,
                    "sequence_parallel": False,
                },
            }
        },
    )


def test_generates_row_gelu_column_layout_without_target_timing() -> None:
    candidate, report = compile_megatron_mlp_layout_candidate(_source())

    assert report["source_layout"] == "column-gelu-row"
    assert report["target_layout"] == "row-gelu-column"
    assert report["target_training_executed"] is False
    assert report["timing_status"] == "unresolved-requires-target-calibration"
    assert report["validation"]["status"] == "passed-structural-validation"
    assert candidate.source["candidate_training_executed"] is False

    rank_zero = [event for event in candidate.events if event.rank == 0]
    assert len(rank_zero) == 13
    collectives = [event for event in rank_zero if event.kind == "collective"]
    assert [event.collective for event in collectives] == [
        "all_reduce",
        "all_gather",
        "all_reduce",
        "all_gather",
    ]
    assert [event.message_bytes for event in collectives] == [
        2048 * 8192 * 4,
        2048 * (2048 // 4) * 4,
        2048 * 8192 * 4,
        2048 * (2048 // 4) * 4,
    ]
    assert all(event.duration_us == 0.0 for event in candidate.events)
    assert all(event.observed_start_us is None for event in candidate.events)

    fc1 = rank_zero[0].metadata["layout"]
    fc2 = rank_zero[3].metadata["layout"]
    assert fc1["weight_shape"] == [8192, 512]
    assert fc2["weight_shape"] == [512, 8192]


def test_validator_rejects_a_missing_layout_collective() -> None:
    candidate, _report = compile_megatron_mlp_layout_candidate(_source())
    damaged = WorkloadTrace(
        events=tuple(
            event
            for event in candidate.events
            if event.id != "rank0::layout::layer0-forward-output-all-gather"
        ),
        source=candidate.source,
        metadata=candidate.metadata,
    )
    with pytest.raises((MegatronLayoutTransformError, ValueError)):
        validate_megatron_mlp_layout_candidate(damaged)


def test_rejects_noncanonical_source_layout() -> None:
    source = _source()
    measurement = dict(source.metadata["framework_measurement"])
    measurement["layer"] = {
        **measurement["layer"],
        "column_gather_output": True,
    }
    changed = WorkloadTrace(
        events=source.events,
        source=source.source,
        metadata={"framework_measurement": measurement},
    )
    with pytest.raises(MegatronLayoutTransformError, match="supported column"):
        compile_megatron_mlp_layout_candidate(changed)


def test_rejects_dimension_that_does_not_divide_tp() -> None:
    with pytest.raises(MegatronLayoutTransformError, match="divide the TP width"):
        compile_megatron_mlp_layout_candidate(_source(ffn=8194))


def test_rejects_source_with_incomplete_collective_evidence() -> None:
    source = _source()
    changed = WorkloadTrace(
        events=tuple(
            event
            for event in source.events
            if event.id != "rank2::backward-all-reduce"
        ),
        source=source.source,
        metadata=source.metadata,
    )
    with pytest.raises(MegatronLayoutTransformError, match="two canonical"):
        compile_megatron_mlp_layout_candidate(changed)


def test_cli_writes_candidate_and_report(tmp_path) -> None:
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "candidate.json"
    report_path = tmp_path / "report.json"
    _source().dump(source_path)

    assert (
        main(
            [
                "--source-workload",
                str(source_path),
                "--target-layout",
                "row-gelu-column",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    candidate = WorkloadTrace.load(output_path)
    report = json.loads(report_path.read_text())
    assert len(candidate.events) == 52
    assert report["validation"]["status"] == "passed-structural-validation"
