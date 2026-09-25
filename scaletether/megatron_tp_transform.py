"""Bounded semantic TP-width rewrite for a pinned Megatron-Core MLP."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any

from .framework_counterfactual import CONTRACT_SCHEMA
from .pipeline import MEGATRON_CORE_SCHEDULE_COMMIT
from .schema import TraceEvent, WorkloadTrace


SCHEMA = "scaletether-megatron-tp-semantic-rewrite-v3"
SOURCE_TP = 2
TARGET_TP = 4
HIDDEN = 256
FFN_HIDDEN = 1024
SOURCE_LOCAL_FFN = FFN_HIDDEN // SOURCE_TP
TARGET_LOCAL_FFN = FFN_HIDDEN // TARGET_TP


class MegatronTpTransformError(ValueError):
    """The source is outside the exact TP-width rewrite contract."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MegatronTpTransformError(f"{name} must be a mapping")
    return value


def _semantic_signature(operator: dict[str, Any]) -> str:
    payload = json.dumps(operator, sort_keys=True, separators=(",", ":")).encode()
    return "framework-operator-v1:" + hashlib.sha256(payload).hexdigest()


def _trace_sha256(trace: WorkloadTrace) -> str:
    payload = json.dumps(
        trace.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _rewrite_dimensions(value: Any) -> tuple[Any, int, set[int]]:
    if isinstance(value, bool):
        return value, 0, set()
    if isinstance(value, int):
        if value == SOURCE_LOCAL_FFN:
            return TARGET_LOCAL_FFN, 1, {value}
        return value, 0, {value}
    if isinstance(value, list):
        rewritten: list[Any] = []
        replacements = 0
        observed: set[int] = set()
        for item in value:
            changed, count, dimensions = _rewrite_dimensions(item)
            rewritten.append(changed)
            replacements += count
            observed.update(dimensions)
        return rewritten, replacements, observed
    raise MegatronTpTransformError("framework operator dimensions are not integral")


def _rewrite_compute(
    event: TraceEvent,
    *,
    tokens: int,
    phase: str,
) -> TraceEvent:
    launch = _mapping(event.metadata.get("kernel_launch_payload"), "kernel launch")
    operator = _mapping(launch.get("framework_operator"), "framework operator")
    dimensions = operator.get("input_dims")
    if not isinstance(dimensions, list):
        raise MegatronTpTransformError(
            f"compute event {event.id!r} lacks framework input dimensions"
        )
    rewritten_dims, replacements, observed = _rewrite_dimensions(dimensions)
    allowed = {0, 1, HIDDEN, SOURCE_LOCAL_FFN, tokens}
    if not observed.issubset(allowed):
        raise MegatronTpTransformError(
            f"compute event {event.id!r} contains an unmodelled dimension: "
            f"{sorted(observed - allowed)}"
        )
    rewritten_operator = dict(operator)
    rewritten_operator["input_dims"] = rewritten_dims
    stride_replacements = 0
    if "input_strides" in operator:
        rewritten_strides, stride_replacements, _observed_strides = _rewrite_dimensions(
            operator["input_strides"]
        )
        rewritten_operator["input_strides"] = rewritten_strides
    metadata = {
        "kernel_launch_payload": {"framework_operator": rewritten_operator},
        "kernel_signature": _semantic_signature(rewritten_operator),
        "pipeline_phase": phase,
        "source_observation_provenance": {
            "source_event_id": event.id,
            "source_tp": SOURCE_TP,
            "status": "shape-rule-input-only-not-target-runtime-evidence",
        },
        "tp_semantic_rewrite": {
            "schema": SCHEMA,
            "source_tp": SOURCE_TP,
            "target_tp": TARGET_TP,
            "source_local_ffn": SOURCE_LOCAL_FFN,
            "target_local_ffn": TARGET_LOCAL_FFN,
            "dimension_replacements": replacements,
            "stride_replacements": stride_replacements,
            "timing_status": "unresolved-requires-target-calibration",
            "kernel_code_status": "unresolved-requires-target-observation",
        },
    }
    return replace(
        event,
        duration_us=0.0,
        observed_start_us=None,
        sm_fraction=None,
        metadata=metadata,
    )


def _rewrite_memory(event: TraceEvent, *, phase: str) -> TraceEvent:
    if (
        event.name != "Memcpy HtoD (Pageable -> Device)"
        or event.metadata.get("raw_trace_category") != "gpu_memcpy"
        or event.metadata.get("input_shapes") != []
        or event.message_bytes is not None
    ):
        raise MegatronTpTransformError(
            f"memory event {event.id!r} is outside the opaque transfer contract"
        )
    return replace(
        event,
        duration_us=0.0,
        observed_start_us=None,
        stream="opaque-memory-copy",
        sm_fraction=None,
        metadata={
            "pipeline_phase": phase,
            "memory_operation": {
                "name": event.name,
                "raw_trace_category": "gpu_memcpy",
                "bytes_status": "unobserved",
            },
            "source_observation_provenance": {
                "source_event_id": event.id,
                "source_tp": SOURCE_TP,
                "status": "opaque-operation-input-only-not-target-runtime-evidence",
            },
            "tp_semantic_rewrite": {
                "schema": SCHEMA,
                "source_tp": SOURCE_TP,
                "target_tp": TARGET_TP,
                "operation_status": "replicated-from-certified-rank-symmetry",
                "timing_status": "unresolved-requires-target-calibration",
                "bytes_status": "unresolved-not-captured",
            },
        },
    )


def compile_megatron_tp2_to_tp4(
    prepared_source: WorkloadTrace,
) -> tuple[WorkloadTrace, dict[str, Any]]:
    contract = _mapping(
        prepared_source.metadata.get("framework_counterfactual"),
        "framework counterfactual contract",
    )
    model = _mapping(contract.get("model"), "model")
    applicability = _mapping(contract.get("applicability"), "applicability")
    fixed = _mapping(applicability.get("fixed_dimensions"), "fixed dimensions")
    measurement = _mapping(
        prepared_source.metadata.get("framework_measurement"),
        "framework measurement",
    )
    steady_state_protocol = {
        "schema": "steady-state-optimizer-step-v1",
        "warmup_steps": 1,
        "warmup_scope": "forward-backward-optimizer",
        "quiescence_before_measured_step": True,
    }
    source_parallelism = applicability.get("source_parallelism")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("framework") != "megatron-core"
        or contract.get("framework_commit") != MEGATRON_CORE_SCHEDULE_COMMIT
        or applicability.get("status") != "certified-for-bounded-rewrite"
        or source_parallelism != {"tp": SOURCE_TP, "pp": 1, "dp": 1, "ep": 1}
        or fixed.get("tp") != SOURCE_TP
        or fixed.get("hidden_size") != HIDDEN
        or fixed.get("ffn_hidden_size") != FFN_HIDDEN
        or model.get("num_layers") != 1
        or model.get("hidden_size") != HIDDEN
        or model.get("ffn_hidden_size") != FFN_HIDDEN
        or measurement.get("measurement_protocol") != steady_state_protocol
    ):
        raise MegatronTpTransformError("source is outside the pinned TP2-to-TP4 domain")
    sequence = model.get("sequence_length")
    micro_batch = model.get("micro_batch_size")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
        or isinstance(micro_batch, bool)
        or not isinstance(micro_batch, int)
        or micro_batch <= 0
    ):
        raise MegatronTpTransformError("source has invalid sequence or microbatch size")
    tokens = sequence * micro_batch
    profiles = _mapping(contract.get("tp_profiles"), "TP profiles")
    profile = _mapping(profiles.get(str(SOURCE_TP)), "source TP profile")
    event_by_id = {event.id: event for event in prepared_source.events}
    if len(event_by_id) != len(prepared_source.events):
        raise MegatronTpTransformError("source contains duplicate event ids")
    phase_ids: list[tuple[str, str]] = []
    for phase, key in (
        ("forward", "forward_event_ids"),
        ("backward", "backward_event_ids"),
        ("optimizer", "optimizer_event_ids"),
    ):
        identifiers = profile.get(key)
        if not isinstance(identifiers, list) or not identifiers:
            raise MegatronTpTransformError(f"source profile lacks {key}")
        phase_ids.extend((phase, str(identifier)) for identifier in identifiers)
    identifiers = [identifier for _phase, identifier in phase_ids]
    if len(set(identifiers)) != len(identifiers) or any(
        identifier not in event_by_id for identifier in identifiers
    ):
        raise MegatronTpTransformError("source TP profile is incomplete or overlapping")
    identifier_set = set(identifiers)
    rewritten_template: list[TraceEvent] = []
    for phase, identifier in phase_ids:
        event = event_by_id[identifier]
        if any(dependency not in identifier_set for dependency in event.dependencies):
            raise MegatronTpTransformError(
                f"source event {identifier!r} depends outside the phase profile"
            )
        if event.kind == "compute":
            rewritten = _rewrite_compute(event, tokens=tokens, phase=phase)
        elif event.kind == "memory":
            raise MegatronTpTransformError(
                f"steady-state source event {identifier!r} unexpectedly contains "
                "a memory transfer"
            )
        elif event.kind == "collective":
            if event.group_role != "tp" or event.group_size not in {None, SOURCE_TP}:
                raise MegatronTpTransformError(
                    f"collective {identifier!r} is not an exact source TP operation"
                )
            metadata = {
                "pipeline_phase": phase,
                "message_bytes_semantics": "payload",
                "source_observation_provenance": {
                    "source_event_id": event.id,
                    "source_tp": SOURCE_TP,
                    "status": "framework-rule-input-only-not-target-runtime-evidence",
                },
                "tp_semantic_rewrite": {
                    "schema": SCHEMA,
                    "source_tp": SOURCE_TP,
                    "target_tp": TARGET_TP,
                    "message_bytes_status": "framework-rule-derived",
                    "timing_status": "unresolved-requires-target-calibration",
                },
            }
            rewritten = replace(
                event,
                duration_us=0.0,
                observed_start_us=None,
                group_size=TARGET_TP,
                sm_fraction=None,
                metadata=metadata,
            )
        else:
            raise MegatronTpTransformError(
                f"source event {identifier!r} has unsupported kind {event.kind!r}"
            )
        rewritten_template.append(rewritten)

    # Megatron's column-parallel backward AllReduce consumes the gradient that
    # exits GELU backward.  The TP=2 anchor retained only NCCL-stream ordering
    # from the forward collective, while the opened TP=4 development trace in
    # job 8238004 exposed the missing framework event edge.  Treat this as an
    # explicit pinned-framework rule, not as evidence inferred from the anchor.
    backward_collectives = [
        event
        for event in rewritten_template
        if event.kind == "collective"
        and event.metadata.get("pipeline_phase") == "backward"
    ]
    gelu_backward = [
        event
        for event in rewritten_template
        if event.kind == "compute"
        and event.metadata.get("pipeline_phase") == "backward"
        and _mapping(
            _mapping(
                event.metadata.get("kernel_launch_payload"), "kernel launch"
            ).get("framework_operator"),
            "framework operator",
        ).get("name")
        == "aten::gelu_backward"
    ]
    if len(backward_collectives) != 1 or len(gelu_backward) != 1:
        raise MegatronTpTransformError(
            "pinned TP rule requires exactly one backward collective and one "
            "aten::gelu_backward operation"
        )
    collective = backward_collectives[0]
    dependency = gelu_backward[0].id
    if dependency not in collective.dependencies:
        rewritten_template[rewritten_template.index(collective)] = replace(
            collective,
            dependencies=(*collective.dependencies, dependency),
            metadata={
                **collective.metadata,
                "framework_causal_rule": {
                    "schema": "megatron-column-backward-tp-causal-rule-v1",
                    "dependency": "aten::gelu_backward",
                    "discovery_job": 8238004,
                    "evidence_status": (
                        "opened-target-development-evidence-not-anchor-observation"
                    ),
                },
            },
        )
    generated: list[TraceEvent] = []
    for rank in range(TARGET_TP):
        for event in rewritten_template:
            generated.append(
                replace(
                    event,
                    id=f"rank{rank}::{event.id}",
                    rank=rank,
                    device=rank,
                    dependencies=tuple(
                        f"rank{rank}::{dependency}" for dependency in event.dependencies
                    ),
                    metadata={
                        **event.metadata,
                        "pipeline_tp_lane": rank,
                        "pipeline_physical_device": rank,
                    },
                )
            )
    summary = {
        "schema": SCHEMA,
        "status": "generated-structural-candidate",
        "source_parallelism": {"tp": SOURCE_TP, "pp": 1, "dp": 1, "ep": 1},
        "target_parallelism": {"tp": TARGET_TP, "pp": 1, "dp": 1, "ep": 1},
        "gpus": TARGET_TP,
        "model": {
            "num_layers": 1,
            "hidden_size": HIDDEN,
            "ffn_hidden_size": FFN_HIDDEN,
            "sequence_length": sequence,
            "micro_batch_size": micro_batch,
        },
        "operator_shape_rule": f"replace-local-ffn-{SOURCE_LOCAL_FFN}-with-{TARGET_LOCAL_FFN}",
        "measurement_protocol": steady_state_protocol,
        "memory_rule": "reject-device-memory-events-after-one-complete-warmup-step",
        "backward_collective_causal_rule": {
            "schema": "megatron-column-backward-tp-causal-rule-v1",
            "dependency": "aten::gelu_backward",
            "discovery_job": 8238004,
            "evidence_status": "opened-target-development-evidence-not-anchor-observation",
        },
        "source_semantic_input_sha256": _trace_sha256(prepared_source),
        "timing_claim": "none-until-target-tp-calibration",
        "kernel_code_claim": "none-until-target-tp-observation",
    }
    result = WorkloadTrace(
        events=tuple(generated),
        source={
            "kind": "framework-semantic-counterfactual",
            "target": "h100",
            "framework": "megatron-core",
            "source_semantic_input_sha256": summary["source_semantic_input_sha256"],
            "candidate_training_executed": False,
        },
        metadata={
            "megatron_tp_semantic_rewrite": summary,
            "provenance": {
                "observed": "source-tp2-only",
                "transformed": "operator-shapes-collectives-dependencies",
                "estimated": [],
                "unresolved": ["target-kernel-code", "target-timing"],
            },
        },
    )
    result.validate()
    return result, summary
