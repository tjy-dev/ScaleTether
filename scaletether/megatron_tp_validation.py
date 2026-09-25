"""Prospective structural validation for the bounded Megatron TP rewrite."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from .megatron_tp_transform import (
    FFN_HIDDEN,
    HIDDEN,
    SCHEMA as TRANSFORM_SCHEMA,
    SOURCE_TP,
    TARGET_TP,
    compile_megatron_tp2_to_tp4,
)
from .pipeline import MEGATRON_CORE_SCHEDULE_COMMIT
from .schema import TraceEvent, WorkloadTrace


SCHEMA = "scaletether-megatron-heldout-tp-validation-v4"
FREEZE_SCHEMA = "scaletether-megatron-tp-prediction-freeze-v5"
OPENED_TARGET_AUDIT_SCHEMA = "scaletether-megatron-tp-opened-target-audit-v1"
AUDITABLE_FREEZE_SCHEMAS = {
    "scaletether-megatron-tp-prediction-freeze-v2",
    "scaletether-megatron-tp-prediction-freeze-v3",
    "scaletether-megatron-tp-prediction-freeze-v4",
    FREEZE_SCHEMA,
}
LEGACY_KERNEL_EVENT_GATE_SETS = (
    (
        "framework-operator-shape-multiset-v1",
        "steady-state-no-device-memory-events-v1",
        "tp-collective-order-bytes-v1",
        "labeled-causal-reachability-v1",
    ),
    (
        "framework-operator-shape-multiset-v1",
        "steady-state-bounded-ddp-memory-structure-v2",
        "tp-collective-order-bytes-v1",
        "labeled-causal-reachability-v1",
    ),
)
REQUIRED_GATES = (
    "framework-semantic-operator-occurrence-v2",
    "kernel-lowering-explicitly-unresolved-v1",
    "steady-state-bounded-ddp-memory-structure-v2",
    "tp-collective-order-bytes-v1",
    "labeled-causal-reachability-v1",
)


class MegatronTpValidationError(ValueError):
    """The generated and held-out TP workloads fail the structural gate."""


def _require_complete_capture(trace: WorkloadTrace, expected_ranks: set[int]) -> None:
    if trace.source.get("kind") == "distributed-capture":
        raw = trace.metadata.get("rank_metadata")
        if not isinstance(raw, list) or len(raw) != len(expected_ranks):
            raise MegatronTpValidationError(
                "held-out distributed capture lacks exact rank metadata"
            )
        ranks: set[int] = set()
        for item in raw:
            if not isinstance(item, dict):
                raise MegatronTpValidationError(
                    "held-out distributed rank metadata is malformed"
                )
            rank = item.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise MegatronTpValidationError(
                    "held-out distributed rank metadata has an invalid rank"
                )
            ranks.add(rank)
            if item.get("capture_limitations") != []:
                raise MegatronTpValidationError(
                    f"held-out target rank {rank} capture is incomplete"
                )
        if ranks != expected_ranks:
            raise MegatronTpValidationError(
                "held-out distributed rank metadata does not cover four ranks"
            )
        return
    if trace.metadata.get("capture_limitations") != []:
        raise MegatronTpValidationError("held-out target capture is incomplete")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase(event: TraceEvent, *, generated: bool) -> str:
    if generated:
        value = event.metadata.get("pipeline_phase")
        if value in {"forward", "backward", "optimizer"}:
            return str(value)
    else:
        marker = event.metadata.get("framework_phase_marker")
        name = marker.get("name") if isinstance(marker, dict) else None
        if isinstance(name, str):
            for phase in ("forward", "backward", "optimizer"):
                if name.endswith(phase):
                    return phase
    raise MegatronTpValidationError(f"event {event.id!r} lacks an exact phase")


def _operator(event: TraceEvent) -> dict[str, Any]:
    launch = event.metadata.get("kernel_launch_payload")
    operator = launch.get("framework_operator") if isinstance(launch, dict) else None
    if not isinstance(operator, dict):
        raise MegatronTpValidationError(
            f"compute event {event.id!r} lacks framework operator evidence"
        )
    projected = {
        key: operator.get(key)
        for key in ("name", "input_dims", "input_types", "input_strides")
    }
    if (
        not isinstance(projected["name"], str)
        or not isinstance(projected["input_dims"], list)
        or not isinstance(projected["input_types"], list)
        or not isinstance(projected["input_strides"], list)
    ):
        raise MegatronTpValidationError(
            f"compute event {event.id!r} has an incomplete operator signature"
        )
    return projected


def _group_size(event: TraceEvent) -> int | None:
    value = event.group_size
    if value is None:
        value = event.metadata.get("observed_group_size")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _base_label(event: TraceEvent, *, generated: bool) -> str:
    phase = _phase(event, generated=generated)
    if event.kind == "compute":
        payload = json.dumps(_operator(event), sort_keys=True, separators=(",", ":"))
        return f"{phase}:compute:{payload}"
    if event.kind == "collective":
        if (
            event.group_role != "tp"
            or not isinstance(event.collective, str)
            or not isinstance(event.message_bytes, int)
            or _group_size(event) != TARGET_TP
        ):
            raise MegatronTpValidationError(
                f"collective {event.id!r} lacks an exact TP=4 signature"
            )
        return (
            f"{phase}:collective:{event.collective}:{event.message_bytes}:{TARGET_TP}"
        )
    if event.kind == "memory":
        if generated:
            operation = event.metadata.get("memory_operation")
            name = operation.get("name") if isinstance(operation, dict) else None
            category = (
                operation.get("raw_trace_category")
                if isinstance(operation, dict)
                else None
            )
            bytes_status = (
                operation.get("bytes_status") if isinstance(operation, dict) else None
            )
            if bytes_status != "unobserved":
                raise MegatronTpValidationError(
                    f"generated memory event {event.id!r} claims transfer bytes"
                )
        else:
            name = event.name
            category = event.metadata.get("raw_trace_category")
        if (
            name != "Memcpy HtoD (Pageable -> Device)"
            or category != "gpu_memcpy"
            or event.message_bytes is not None
        ):
            raise MegatronTpValidationError(
                f"memory event {event.id!r} is outside the opaque transfer gate"
            )
        return f"{phase}:memory:{category}:{name}:bytes-unobserved"
    raise MegatronTpValidationError(
        f"event {event.id!r} has unsupported kind {event.kind!r}"
    )


def _topological_order(events: list[TraceEvent]) -> list[TraceEvent]:
    by_id = {event.id: event for event in events}
    if len(by_id) != len(events):
        raise MegatronTpValidationError("rank contains duplicate event ids")
    missing = sorted(
        {
            dependency
            for event in events
            for dependency in event.dependencies
            if dependency not in by_id
        }
    )
    if missing:
        raise MegatronTpValidationError(
            f"rank dependencies leave the local graph: {missing[:8]}"
        )
    index = {event.id: position for position, event in enumerate(events)}
    remaining = set(by_id)
    emitted: set[str] = set()
    ordered: list[TraceEvent] = []
    while remaining:
        ready = [
            by_id[identifier]
            for identifier in remaining
            if set(by_id[identifier].dependencies).issubset(emitted)
        ]
        if not ready:
            raise MegatronTpValidationError("rank dependency graph contains a cycle")
        ready.sort(
            key=lambda event: (
                event.observed_start_us is None,
                event.observed_start_us or 0.0,
                index[event.id],
                event.id,
            )
        )
        for event in ready:
            ordered.append(event)
            emitted.add(event.id)
            remaining.remove(event.id)
    return ordered


def _is_split_k_reduction(event: TraceEvent) -> bool:
    launch = event.metadata.get("kernel_launch_payload")
    launch_name = launch.get("name") if isinstance(launch, dict) else None
    patterns = ("cublasLt::splitKreduce_kernel<", "_execute_split_k_kernel")
    return any(pattern in event.name for pattern in patterns) or (
        isinstance(launch_name, str)
        and any(pattern in launch_name for pattern in patterns)
    )


def _collapse_split_k_lowering(
    events: list[TraceEvent], *, generated: bool
) -> tuple[list[TraceEvent], dict[str, Any]]:
    """Project an unambiguous cuBLAS split-K chain to one operator occurrence.

    Kineto associates both the main GEMM kernel and its dependent split-K
    reduction with one framework operator.  We collapse only a direct
    one-predecessor chain with identical operator and phase evidence.  Other
    decompositions remain outside the bounded rule and fail closed.
    """

    by_id = {event.id: event for event in events}
    aliases: dict[str, str] = {}
    collapsed: list[dict[str, Any]] = []
    for event in _topological_order(events):
        if not _is_split_k_reduction(event):
            continue
        if event.kind != "compute":
            raise MegatronTpValidationError(
                f"split-K event {event.id!r} is not a compute event"
            )
        matching = [
            dependency
            for dependency in event.dependencies
            if dependency in by_id
            and by_id[dependency].kind == "compute"
            and _operator(by_id[dependency]) == _operator(event)
            and _phase(by_id[dependency], generated=generated)
            == _phase(event, generated=generated)
        ]
        if len(event.dependencies) != 1 or len(matching) != 1:
            raise MegatronTpValidationError(
                f"split-K event {event.id!r} lacks one exact semantic predecessor"
            )
        predecessor = aliases.get(matching[0], matching[0])
        if predecessor not in by_id or _is_split_k_reduction(by_id[predecessor]):
            raise MegatronTpValidationError(
                f"split-K event {event.id!r} has an unsupported decomposition chain"
            )
        aliases[event.id] = predecessor
        collapsed.append(
            {
                "framework_event_id": predecessor,
                "lowering_event_id": event.id,
                "lowering_kind": "cublasLt-split-k-reduction",
            }
        )

    projected: list[TraceEvent] = []
    for event in events:
        if event.id in aliases:
            continue
        dependencies = tuple(
            dict.fromkeys(
                aliases.get(dependency, dependency) for dependency in event.dependencies
            )
        )
        if event.id in dependencies:
            raise MegatronTpValidationError(
                f"semantic projection creates a self-edge at {event.id!r}"
            )
        projected.append(replace(event, dependencies=dependencies))
    return projected, {
        "status": "unresolved-not-compared",
        "collapsed_kernel_count": len(collapsed),
        "collapsed_decompositions": collapsed,
    }


def _rank_projection(
    trace: WorkloadTrace, rank: int, *, generated: bool
) -> dict[str, Any]:
    events = [event for event in trace.events if event.rank == rank]
    events, lowering = _collapse_split_k_lowering(events, generated=generated)
    ordered = _topological_order(events)
    occurrences: Counter[str] = Counter()
    key_by_id: dict[str, str] = {}
    labels: list[str] = []
    for event in ordered:
        label = _base_label(event, generated=generated)
        occurrence = occurrences[label]
        occurrences[label] += 1
        key = f"{label}#{occurrence}"
        key_by_id[event.id] = key
        labels.append(key)
    by_id = {event.id: event for event in events}
    reachability: set[tuple[str, str]] = set()
    for destination in ordered:
        pending = list(destination.dependencies)
        visited: set[str] = set()
        while pending:
            source = pending.pop()
            if source in visited:
                continue
            visited.add(source)
            reachability.add((key_by_id[source], key_by_id[destination.id]))
            pending.extend(by_id[source].dependencies)
    return {
        "ordered_event_keys": labels,
        "event_key_multiset": dict(sorted(Counter(labels).items())),
        "collective_sequence": [
            key_by_id[event.id] for event in ordered if event.kind == "collective"
        ],
        "causal_reachability": [list(edge) for edge in sorted(reachability)],
        "kernel_lowering": lowering,
    }


def _projection_difference(
    predicted: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    predicted_events = Counter(predicted["ordered_event_keys"])
    observed_events = Counter(observed["ordered_event_keys"])
    predicted_reachability = {tuple(edge) for edge in predicted["causal_reachability"]}
    observed_reachability = {tuple(edge) for edge in observed["causal_reachability"]}
    missing_reachability = sorted(predicted_reachability - observed_reachability)
    unexpected_reachability = sorted(observed_reachability - predicted_reachability)
    return {
        "missing_generated_event_keys": list(
            (predicted_events - observed_events).elements()
        ),
        "unexpected_heldout_event_keys": list(
            (observed_events - predicted_events).elements()
        ),
        "generated_collective_sequence": predicted["collective_sequence"],
        "heldout_collective_sequence": observed["collective_sequence"],
        "missing_generated_reachability_count": len(missing_reachability),
        "unexpected_heldout_reachability_count": len(unexpected_reachability),
        "missing_generated_reachability_preview": [
            list(edge) for edge in missing_reachability[:64]
        ],
        "unexpected_heldout_reachability_preview": [
            list(edge) for edge in unexpected_reachability[:64]
        ],
    }


def _expected_measurement(sequence_length: int) -> dict[str, Any]:
    return {
        "schema": "megatron-core-mlp-measurement-v1",
        "framework": "megatron-core",
        "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
        "environment_constraints": {"NO_VCS_VERSION": "1"},
        "measurement_protocol": {
            "schema": "steady-state-optimizer-step-v1",
            "warmup_steps": 1,
            "warmup_scope": "forward-backward-optimizer",
            "quiescence_before_measured_step": True,
        },
        "tensor_parallel_size": TARGET_TP,
        "data_parallel_size": 1,
        "model": {
            "num_layers": 1,
            "hidden_size": HIDDEN,
            "ffn_hidden_size": FFN_HIDDEN,
            "sequence_length": sequence_length,
            "micro_batch_size": 1,
            "global_batch_size": 2,
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
        "optimizer": {"kind": "torch.optim.AdamW"},
        "data_parallel": {"kind": "none", "gradient_sync": False},
        "phase_markers": {
            "forward": "megatron_mlp_forward",
            "backward": "megatron_mlp_backward",
            "optimizer": "megatron_mlp_optimizer",
        },
    }


def validate_frozen_tp2_to_tp4(
    generated: WorkloadTrace,
    actual_target: WorkloadTrace,
    *,
    compilation_summary: dict[str, Any],
    expected_sequence_length: int,
) -> dict[str, Any]:
    if expected_sequence_length <= 0:
        raise MegatronTpValidationError("expected sequence length must be positive")
    expected_ranks = set(range(TARGET_TP))
    _require_complete_capture(actual_target, expected_ranks)
    if actual_target.metadata.get("framework_measurement") != _expected_measurement(
        expected_sequence_length
    ):
        raise MegatronTpValidationError(
            "held-out capture is not the exact pinned TP=4 Megatron MLP"
        )
    embedded = generated.metadata.get("megatron_tp_semantic_rewrite")
    expected_source = {
        "kind": "framework-semantic-counterfactual",
        "target": "h100",
        "framework": "megatron-core",
        "source_semantic_input_sha256": compilation_summary.get(
            "source_semantic_input_sha256"
        ),
        "candidate_training_executed": False,
    }
    expected_metadata = {
        "megatron_tp_semantic_rewrite": compilation_summary,
        "provenance": {
            "observed": "source-tp2-only",
            "transformed": "operator-shapes-collectives-dependencies",
            "estimated": [],
            "unresolved": ["target-kernel-code", "target-timing"],
        },
    }
    if (
        embedded != compilation_summary
        or compilation_summary.get("schema") != TRANSFORM_SCHEMA
        or generated.source != expected_source
        or generated.metadata != expected_metadata
    ):
        raise MegatronTpValidationError(
            "compilation summary or provenance is not bound to the generated workload"
        )
    expected_summary = {
        "source_parallelism": {"tp": SOURCE_TP, "pp": 1, "dp": 1, "ep": 1},
        "target_parallelism": {"tp": TARGET_TP, "pp": 1, "dp": 1, "ep": 1},
        "gpus": TARGET_TP,
    }
    if any(
        compilation_summary.get(key) != value for key, value in expected_summary.items()
    ):
        raise MegatronTpValidationError("compilation summary is outside the TP gate")
    if compilation_summary.get("model") != {
        "num_layers": 1,
        "hidden_size": HIDDEN,
        "ffn_hidden_size": FFN_HIDDEN,
        "sequence_length": expected_sequence_length,
        "micro_batch_size": 1,
    }:
        raise MegatronTpValidationError(
            "compilation summary has wrong model dimensions"
        )
    if {event.rank for event in generated.events} != expected_ranks or {
        event.rank for event in actual_target.events
    } != expected_ranks:
        raise MegatronTpValidationError("both workloads must cover exactly four ranks")
    for event in generated.events:
        if (
            event.duration_us != 0.0
            or event.observed_start_us is not None
            or event.sm_fraction is not None
            or "kernel_code_identity" in event.metadata
        ):
            raise MegatronTpValidationError(
                "generated TP candidate contains an unauthorized timing or code claim"
            )
    if any(event.kind == "memory" for event in generated.events):
        raise MegatronTpValidationError(
            "generated steady-state TP candidate contains a memory event"
        )
    if any(event.kind == "memory" for event in actual_target.events):
        raise MegatronTpValidationError(
            "held-out steady-state TP capture contains a memory event"
        )
    ranks: dict[str, Any] = {}
    for rank in sorted(expected_ranks):
        predicted = _rank_projection(generated, rank, generated=True)
        observed = _rank_projection(actual_target, rank, generated=False)
        if predicted["event_key_multiset"] != observed["event_key_multiset"]:
            raise MegatronTpValidationError(
                f"rank {rank} framework operator-shape multiset mismatch"
            )
        if predicted["collective_sequence"] != observed["collective_sequence"]:
            raise MegatronTpValidationError(
                f"rank {rank} TP collective order or bytes mismatch"
            )
        if predicted["causal_reachability"] != observed["causal_reachability"]:
            raise MegatronTpValidationError(
                f"rank {rank} labeled causal reachability mismatch"
            )
        ranks[str(rank)] = {"generated": predicted, "heldout": observed}
    return {
        "schema": SCHEMA,
        "status": "passed",
        "claim": "bounded-tp-semantic-structural-validation-kernel-lowering-unresolved",
        "candidate_training_executed_before_prediction_freeze": False,
        "source_parallelism": {"tp": SOURCE_TP, "pp": 1, "dp": 1, "ep": 1},
        "heldout_parallelism": {"tp": TARGET_TP, "pp": 1, "dp": 1, "ep": 1},
        "required_validation_gates": list(REQUIRED_GATES),
        "compilation_summary": compilation_summary,
        "ranks": ranks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--prepared-source", type=Path, required=True)
    freeze.add_argument("--generated-output", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--expected-sequence-length", type=int, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--generated", type=Path, required=True)
    score.add_argument("--freeze", type=Path, required=True)
    score.add_argument("--actual-target", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--expected-sequence-length", type=int, required=True)
    audit = subparsers.add_parser("audit-opened-target")
    audit.add_argument("--generated", type=Path, required=True)
    audit.add_argument("--freeze", type=Path, required=True)
    audit.add_argument("--actual-target", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--expected-sequence-length", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        generated, summary = compile_megatron_tp2_to_tp4(
            WorkloadTrace.load(args.prepared_source)
        )
        if (
            summary.get("model", {}).get("sequence_length")
            != args.expected_sequence_length
        ):
            raise MegatronTpValidationError("prepared source sequence length is wrong")
        generated.dump(args.generated_output)
        document = {
            "schema": FREEZE_SCHEMA,
            "status": "frozen-before-heldout-execution",
            "candidate_training_executed": False,
            "generated_workload_sha256": _sha256(args.generated_output),
            "compilation_summary": summary,
            "required_validation_gates": list(REQUIRED_GATES),
        }
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(document, sort_keys=True))
        return 0
    document = json.loads(args.freeze.read_text(encoding="utf-8"))
    if args.command == "audit-opened-target":
        frozen_gates = document.get("required_validation_gates")
        if (
            not isinstance(document, dict)
            or document.get("schema") not in AUDITABLE_FREEZE_SCHEMAS
            or document.get("status") != "frozen-before-heldout-execution"
            or document.get("candidate_training_executed") is not False
            or document.get("generated_workload_sha256") != _sha256(args.generated)
            or frozen_gates
            not in (
                list(REQUIRED_GATES),
                *(list(gates) for gates in LEGACY_KERNEL_EVENT_GATE_SETS),
            )
            or not isinstance(document.get("compilation_summary"), dict)
        ):
            raise MegatronTpValidationError(
                "opened-target audit freeze is invalid or was modified"
            )
        audit_document: dict[str, Any] = {
            "schema": OPENED_TARGET_AUDIT_SCHEMA,
            "prospective_claim": False,
            "reason": "target-was-opened-before-this-corrected-audit",
            "prediction_freeze_sha256": _sha256(args.freeze),
            "generated_workload_sha256": _sha256(args.generated),
            "actual_target_sha256": _sha256(args.actual_target),
            "audited_original_required_validation_gates": frozen_gates,
            "audit_required_validation_gates": list(REQUIRED_GATES),
        }
        generated_trace = WorkloadTrace.load(args.generated)
        actual_trace = WorkloadTrace.load(args.actual_target)
        try:
            report = validate_frozen_tp2_to_tp4(
                generated_trace,
                actual_trace,
                compilation_summary=document["compilation_summary"],
                expected_sequence_length=args.expected_sequence_length,
            )
        except MegatronTpValidationError as error:
            audit_document["status"] = "descriptive-structural-rejection"
            audit_document["rejection"] = str(error)
            differences: dict[str, Any] = {}
            for rank in range(TARGET_TP):
                try:
                    predicted = _rank_projection(generated_trace, rank, generated=True)
                    observed = _rank_projection(actual_trace, rank, generated=False)
                except MegatronTpValidationError as projection_error:
                    differences[str(rank)] = {"projection_error": str(projection_error)}
                else:
                    differences[str(rank)] = _projection_difference(predicted, observed)
            audit_document["rank_differences"] = differences
        else:
            audit_document["status"] = "descriptive-structural-pass"
            audit_document["validation"] = report
        args.output.write_text(
            json.dumps(audit_document, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(audit_document, sort_keys=True))
        return 0
    if (
        not isinstance(document, dict)
        or document.get("schema") != FREEZE_SCHEMA
        or document.get("status") != "frozen-before-heldout-execution"
        or document.get("candidate_training_executed") is not False
        or document.get("generated_workload_sha256") != _sha256(args.generated)
        or document.get("required_validation_gates") != list(REQUIRED_GATES)
        or not isinstance(document.get("compilation_summary"), dict)
    ):
        raise MegatronTpValidationError("prediction freeze is invalid or was modified")
    report = validate_frozen_tp2_to_tp4(
        WorkloadTrace.load(args.generated),
        WorkloadTrace.load(args.actual_target),
        compilation_summary=document["compilation_summary"],
        expected_sequence_length=args.expected_sequence_length,
    )
    report["prediction_freeze_sha256"] = _sha256(args.freeze)
    report["generated_workload_sha256"] = _sha256(args.generated)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
