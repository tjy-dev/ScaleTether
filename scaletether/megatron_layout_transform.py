"""Layout-aware rewrite for a pinned Megatron-Core tensor-parallel MLP.

The initial rule changes the ordinary column-parallel/GELU/row-parallel MLP
into a row-parallel/GELU/column-parallel MLP at the same TP width.  It emits a
semantic graph only: target kernel identities and durations remain unresolved
until a physical implementation is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import TraceEvent, WorkloadTrace


SCHEMA = "scaletether-megatron-mlp-layout-rewrite-v1"
SOURCE_MEASUREMENT_SCHEMA = "megatron-core-mlp-measurement-v1"
SOURCE_LAYOUT = "column-gelu-row"
TARGET_LAYOUT = "row-gelu-column"
SOURCE_LAYER_KIND = "megatron-core-column-gelu-row-mlp-v1"


class MegatronLayoutTransformError(ValueError):
    """The measured source or requested layout is outside the bounded rule."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MegatronLayoutTransformError(f"{label} must be a positive integer")
    return value


def _measurement(source: WorkloadTrace) -> dict[str, Any]:
    measurement = source.metadata.get("framework_measurement")
    if not isinstance(measurement, dict):
        raise MegatronLayoutTransformError("source lacks a framework measurement")
    if measurement.get("schema") != SOURCE_MEASUREMENT_SCHEMA:
        raise MegatronLayoutTransformError(
            "layout rewriting requires a Megatron-Core MLP measurement"
        )
    if measurement.get("framework") != "megatron-core":
        raise MegatronLayoutTransformError("source framework is not Megatron-Core")
    layer = measurement.get("layer")
    if not isinstance(layer, dict) or layer != {
        "kind": SOURCE_LAYER_KIND,
        "column_gather_output": False,
        "row_input_is_parallel": True,
        "sequence_parallel": False,
    }:
        raise MegatronLayoutTransformError(
            "source is not the supported column/GELU/row MLP layout"
        )
    if measurement.get("data_parallel_size") != 1:
        raise MegatronLayoutTransformError("layout rewriting currently requires DP=1")
    return measurement


def _model_spec(measurement: dict[str, Any]) -> dict[str, int]:
    model = measurement.get("model")
    if not isinstance(model, dict):
        raise MegatronLayoutTransformError("framework measurement lacks model dimensions")
    if model.get("parameter_dtype") != "float32" or model.get(
        "activation_dtype"
    ) != "float32":
        raise MegatronLayoutTransformError("layout rewriting currently requires FP32")
    if model.get("bias") is not False:
        raise MegatronLayoutTransformError("layout rewriting currently requires bias=False")
    spec = {
        "tp": _positive_integer(
            measurement.get("tensor_parallel_size"), "tensor parallel size"
        ),
        "layers": _positive_integer(model.get("num_layers"), "number of layers"),
        "hidden": _positive_integer(model.get("hidden_size"), "hidden size"),
        "ffn": _positive_integer(model.get("ffn_hidden_size"), "FFN hidden size"),
        "sequence": _positive_integer(model.get("sequence_length"), "sequence length"),
        "micro_batch": _positive_integer(
            model.get("micro_batch_size"), "micro batch size"
        ),
        "element_bytes": _positive_integer(
            model.get("activation_element_bytes"), "activation element bytes"
        ),
    }
    if spec["tp"] < 2:
        raise MegatronLayoutTransformError(
            "row/column reorganization requires tensor parallel size at least two"
        )
    if spec["hidden"] % spec["tp"] or spec["ffn"] % spec["tp"]:
        raise MegatronLayoutTransformError(
            "hidden and FFN dimensions must both divide the TP width"
        )
    return spec


def _validate_source_collectives(source: WorkloadTrace, spec: dict[str, int]) -> None:
    """Require the two canonical TP reductions on every measured source rank."""

    tp = spec["tp"]
    activation_bytes = (
        spec["sequence"]
        * spec["micro_batch"]
        * spec["hidden"]
        * spec["element_bytes"]
    )
    ranks = {event.rank for event in source.events}
    if ranks != set(range(tp)):
        raise MegatronLayoutTransformError(
            "source does not contain exactly the measured TP ranks"
        )
    for rank in range(tp):
        collectives = [
            event
            for event in source.events
            if event.rank == rank and event.group_role == "tp"
        ]
        if len(collectives) != 2:
            raise MegatronLayoutTransformError(
                f"source rank {rank} does not contain the two canonical TP reductions"
            )
        for event in collectives:
            if (
                event.collective != "all_reduce"
                or event.message_bytes != activation_bytes
                or event.group_size not in (None, tp)
            ):
                raise MegatronLayoutTransformError(
                    f"source rank {rank} TP collective differs from the canonical layout"
                )


def _candidate_rank_events(rank: int, spec: dict[str, int]) -> list[TraceEvent]:
    tp = spec["tp"]
    tokens = spec["sequence"] * spec["micro_batch"]
    hidden = spec["hidden"]
    ffn = spec["ffn"]
    element_bytes = spec["element_bytes"]
    group = list(range(tp))
    events: list[TraceEvent] = []
    previous: str | None = None

    def add_compute(
        key: str,
        name: str,
        phase: str,
        component: str,
        **layout: object,
    ) -> None:
        nonlocal previous
        event_id = f"rank{rank}::layout::{key}"
        events.append(
            TraceEvent(
                id=event_id,
                name=name,
                kind="compute",
                duration_us=0.0,
                rank=rank,
                device=rank,
                dependencies=() if previous is None else (previous,),
                metadata={
                    "schema": SCHEMA,
                    "pipeline_phase": phase,
                    "semantic_component": component,
                    "source_layout": SOURCE_LAYOUT,
                    "target_layout": TARGET_LAYOUT,
                    "layout": layout,
                    "provenance": "pinned-megatron-layout-rule",
                    "timing_status": "unresolved-requires-target-calibration",
                    "kernel_code_status": "unresolved-requires-target-observation",
                },
            )
        )
        previous = event_id

    def add_collective(
        key: str,
        name: str,
        phase: str,
        component: str,
        collective: str,
        message_bytes: int,
        tensor_layout: str,
    ) -> None:
        nonlocal previous
        event_id = f"rank{rank}::layout::{key}"
        events.append(
            TraceEvent(
                id=event_id,
                name=name,
                kind="collective",
                duration_us=0.0,
                rank=rank,
                device=rank,
                dependencies=() if previous is None else (previous,),
                collective=collective,
                message_bytes=message_bytes,
                group_role="tp",
                group_size=tp,
                metadata={
                    "schema": SCHEMA,
                    "pipeline_phase": phase,
                    "semantic_component": component,
                    "source_layout": SOURCE_LAYOUT,
                    "target_layout": TARGET_LAYOUT,
                    "process_group_ranks": group,
                    "tensor_layout": tensor_layout,
                    "message_bytes_semantics": "per-rank-input-payload",
                    "provenance": "pinned-megatron-layout-rule",
                    "timing_status": "unresolved-requires-target-calibration",
                },
            )
        )
        previous = event_id

    for layer in range(spec["layers"]):
        prefix = f"layers.{layer}.mlp"
        add_compute(
            f"layer{layer}-forward-fc1-row",
            f"Megatron layer {layer} forward row-parallel FC1",
            "forward",
            f"{prefix}.linear_fc1",
            weight_shape=[ffn, hidden // tp],
            input_shape=[tokens, hidden // tp],
            output_shape=[tokens, ffn],
            output_state="partial",
        )
        add_collective(
            f"layer{layer}-forward-fc1-all-reduce",
            f"Megatron layer {layer} row-parallel FC1 output reduction",
            "forward",
            f"{prefix}.linear_fc1",
            "all_reduce",
            tokens * ffn * element_bytes,
            "partial-to-replicated-ffn",
        )
        add_compute(
            f"layer{layer}-forward-gelu",
            f"Megatron layer {layer} replicated GELU",
            "forward",
            f"{prefix}.gelu",
            input_shape=[tokens, ffn],
            output_shape=[tokens, ffn],
            placement="replicated",
        )
        add_compute(
            f"layer{layer}-forward-fc2-column",
            f"Megatron layer {layer} forward column-parallel FC2",
            "forward",
            f"{prefix}.linear_fc2",
            weight_shape=[hidden // tp, ffn],
            input_shape=[tokens, ffn],
            output_shape=[tokens, hidden // tp],
            output_state="sharded-hidden",
        )
        add_collective(
            f"layer{layer}-forward-output-all-gather",
            f"Megatron layer {layer} column-parallel FC2 output gather",
            "forward",
            f"{prefix}.linear_fc2",
            "all_gather",
            tokens * (hidden // tp) * element_bytes,
            "sharded-to-replicated-hidden",
        )
    add_compute(
        "forward-loss",
        "Layout-validation loss",
        "forward",
        "terminal_loss",
        input_shape=[tokens, hidden],
        placement="replicated",
    )
    add_compute(
        "backward-loss",
        "Layout-validation loss backward",
        "backward",
        "terminal_loss",
        output_shape=[tokens, hidden],
        placement="replicated",
    )
    for layer in reversed(range(spec["layers"])):
        prefix = f"layers.{layer}.mlp"
        add_compute(
            f"layer{layer}-backward-fc2-column",
            f"Megatron layer {layer} backward column-parallel FC2",
            "backward",
            f"{prefix}.linear_fc2",
            weight_shape=[hidden // tp, ffn],
            input_gradient_shape=[tokens, ffn],
            input_gradient_state="partial",
        )
        add_collective(
            f"layer{layer}-backward-fc2-all-reduce",
            f"Megatron layer {layer} column-parallel FC2 input-gradient reduction",
            "backward",
            f"{prefix}.linear_fc2",
            "all_reduce",
            tokens * ffn * element_bytes,
            "partial-to-replicated-ffn-gradient",
        )
        add_compute(
            f"layer{layer}-backward-gelu",
            f"Megatron layer {layer} replicated GELU backward",
            "backward",
            f"{prefix}.gelu",
            input_shape=[tokens, ffn],
            placement="replicated",
        )
        add_compute(
            f"layer{layer}-backward-fc1-row",
            f"Megatron layer {layer} backward row-parallel FC1",
            "backward",
            f"{prefix}.linear_fc1",
            weight_shape=[ffn, hidden // tp],
            input_gradient_shape=[tokens, hidden // tp],
            input_gradient_state="sharded-hidden",
        )
        add_collective(
            f"layer{layer}-backward-input-all-gather",
            f"Megatron layer {layer} row-parallel FC1 input-gradient gather",
            "backward",
            f"{prefix}.linear_fc1",
            "all_gather",
            tokens * (hidden // tp) * element_bytes,
            "sharded-to-replicated-hidden-gradient",
        )
    add_compute(
        "optimizer",
        "Megatron optimizer step",
        "optimizer",
        "step.optimizer",
        parameter_layout="row-fc1-column-fc2",
    )
    return events


def validate_megatron_mlp_layout_candidate(candidate: WorkloadTrace) -> dict[str, Any]:
    report = candidate.metadata.get("megatron_mlp_layout_rewrite")
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise MegatronLayoutTransformError("candidate lacks a layout rewrite report")
    tp = _positive_integer(report.get("tensor_parallel_size"), "candidate TP")
    expected_group = list(range(tp))
    if {event.rank for event in candidate.events} != set(expected_group):
        raise MegatronLayoutTransformError("candidate rank set differs from target TP")
    model = report.get("model")
    if not isinstance(model, dict):
        raise MegatronLayoutTransformError("candidate layout report lacks model dimensions")
    layers = _positive_integer(model.get("num_layers"), "candidate layer count")
    expected_collectives = (
        ["all_reduce", "all_gather"] * layers
        + ["all_reduce", "all_gather"] * layers
    )
    for rank in range(tp):
        rank_events = [event for event in candidate.events if event.rank == rank]
        if len(rank_events) != 10 * layers + 3:
            raise MegatronLayoutTransformError(
                f"candidate rank {rank} does not contain the complete layout plan"
            )
        for index, event in enumerate(rank_events):
            expected_dependencies = () if index == 0 else (rank_events[index - 1].id,)
            if event.dependencies != expected_dependencies:
                raise MegatronLayoutTransformError(
                    f"candidate rank {rank} layout chain is incomplete"
                )
            if (
                event.duration_us != 0.0
                or event.observed_start_us is not None
                or event.sm_fraction is not None
                or event.metadata.get("timing_status")
                != "unresolved-requires-target-calibration"
            ):
                raise MegatronLayoutTransformError(
                    "candidate improperly inherits target timing information"
                )
        collectives = [event for event in rank_events if event.kind == "collective"]
        if [event.collective for event in collectives] != expected_collectives:
            raise MegatronLayoutTransformError(
                f"candidate rank {rank} collective order is invalid"
            )
        if any(
            event.group_role != "tp"
            or event.group_size != tp
            or event.metadata.get("process_group_ranks") != expected_group
            for event in collectives
        ):
            raise MegatronLayoutTransformError(
                f"candidate rank {rank} communicator is invalid"
            )
    candidate.validate()
    return {
        "schema": "scaletether-megatron-mlp-layout-validation-v1",
        "status": "passed-structural-validation",
        "rank_count": tp,
        "event_count": len(candidate.events),
        "collective_count": sum(
            event.kind == "collective" for event in candidate.events
        ),
        "timing_status": "unresolved-requires-target-calibration",
    }


def compile_megatron_mlp_layout_candidate(
    source: WorkloadTrace,
    *,
    target_layout: str = TARGET_LAYOUT,
) -> tuple[WorkloadTrace, dict[str, Any]]:
    """Generate a row/GELU/column MLP graph from a measured column/GELU/row MLP."""

    if target_layout != TARGET_LAYOUT:
        raise MegatronLayoutTransformError(
            f"unsupported target layout {target_layout!r}; expected {TARGET_LAYOUT!r}"
        )
    measurement = _measurement(source)
    spec = _model_spec(measurement)
    _validate_source_collectives(source, spec)
    source_sha256 = _canonical_sha256(source.to_dict())
    events = tuple(
        event
        for rank in range(spec["tp"])
        for event in _candidate_rank_events(rank, spec)
    )
    report = {
        "schema": SCHEMA,
        "status": "generated-structural-candidate",
        "source_layout": SOURCE_LAYOUT,
        "target_layout": TARGET_LAYOUT,
        "tensor_parallel_size": spec["tp"],
        "source_parallelism": {"tp": spec["tp"], "pp": 1, "dp": 1, "ep": 1},
        "target_parallelism": {"tp": spec["tp"], "pp": 1, "dp": 1, "ep": 1},
        "model": {
            "num_layers": spec["layers"],
            "hidden_size": spec["hidden"],
            "ffn_hidden_size": spec["ffn"],
            "sequence_length": spec["sequence"],
            "micro_batch_size": spec["micro_batch"],
            "activation_element_bytes": spec["element_bytes"],
        },
        "source_workload_sha256": source_sha256,
        "target_training_executed": False,
        "kernel_code_status": "unresolved-requires-target-observation",
        "timing_status": "unresolved-requires-target-calibration",
        "supported_claim": (
            "framework-derived MLP layout structure only; no target timing or "
            "physical-equivalence claim"
        ),
    }
    candidate = WorkloadTrace(
        events=events,
        source={
            "kind": "framework-semantic-counterfactual",
            "target": str(source.source.get("target", "")).lower(),
            "rank_count": spec["tp"],
            "framework": "megatron-core",
            "candidate_training_executed": False,
            "source_sha256": source_sha256,
        },
        metadata={
            "framework_measurement": measurement,
            "megatron_mlp_layout_rewrite": report,
            "provenance": {
                "observed": "source-column-gelu-row-layout",
                "transformed": "row-gelu-column-layout-shapes-collectives-dependencies",
                "estimated": [],
                "unresolved": ["target-kernel-code", "target-timing"],
            },
        },
    )
    validation = validate_megatron_mlp_layout_candidate(candidate)
    report["validation"] = validation
    candidate = WorkloadTrace(
        events=candidate.events,
        source=candidate.source,
        metadata={**candidate.metadata, "megatron_mlp_layout_rewrite": report},
    )
    return candidate, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compile-megatron-layout",
        description=(
            "Generate a structural row/GELU/column Megatron MLP workload from "
            "a measured column/GELU/row workload."
        ),
    )
    parser.add_argument("--source-workload", type=Path, required=True)
    parser.add_argument(
        "--target-layout", choices=(TARGET_LAYOUT,), default=TARGET_LAYOUT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.report.exists():
            raise MegatronLayoutTransformError("output or report already exists")
        source = WorkloadTrace.load(args.source_workload)
        candidate, report = compile_megatron_mlp_layout_candidate(
            source, target_layout=args.target_layout
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        candidate.dump(args.output)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
