"""Semantic event-graph compilation for the frozen Megatron Transformer family."""

from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .megatron_transformer_rules import (
    TransformerRuleError,
    build_rule_plan,
    compare_megatron_tp_parameter_manifest,
    derive_megatron_tp_parameter_manifest,
    derive_megatron_tp_rule,
)
from .schema import TraceEvent, WorkloadTrace


SCHEMA = "scaletether-megatron-transformer-semantic-graph-v1"
_COMPONENT = re.compile(
    r"^megatron_transformer_(forward|backward)_component:"
    r"(layers\.(?:0|[1-9]\d*)(?:\..+)?|final_layernorm|terminal_loss)$"
)
_OUTER = {
    "megatron_transformer_forward": "forward",
    "megatron_transformer_backward": "backward",
    "megatron_transformer_optimizer": "optimizer",
}

_LEAF_COMPONENTS = tuple(
    [
        f"layers.{layer}.{leaf}"
        for layer in range(2)
        for leaf in (
            "self_attention.linear_qkv",
            "self_attention.linear_proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        )
    ]
    + ["final_layernorm"]
)

_LANDMARKS = (
    *[("forward", component) for component in _LEAF_COMPONENTS],
    ("backward", "final_layernorm"),
    *[
        ("backward", f"layers.{layer}.{leaf}")
        for layer in (1, 0)
        for leaf in (
            "mlp.linear_fc2",
            "mlp.linear_fc1",
            "self_attention.linear_proj",
            "self_attention.linear_qkv",
        )
    ],
    ("optimizer", "step.optimizer"),
)


def _leaf_components(num_layers: int) -> tuple[str, ...]:
    return tuple(
        [
            f"layers.{layer}.{leaf}"
            for layer in range(num_layers)
            for leaf in (
                "self_attention.linear_qkv",
                "self_attention.linear_proj",
                "mlp.linear_fc1",
                "mlp.linear_fc2",
            )
        ]
        + ["final_layernorm"]
    )


def _landmarks(num_layers: int) -> tuple[tuple[str, str], ...]:
    leaves = _leaf_components(num_layers)
    return (
        *[("forward", component) for component in leaves],
        ("backward", "final_layernorm"),
        *[
            ("backward", f"layers.{layer}.{leaf}")
            for layer in reversed(range(num_layers))
            for leaf in (
                "mlp.linear_fc2",
                "mlp.linear_fc1",
                "self_attention.linear_proj",
                "self_attention.linear_qkv",
            )
        ],
        ("optimizer", "step.optimizer"),
    )


class TransformerGraphError(ValueError):
    """The source trace cannot support the requested semantic graph rewrite."""

    def __init__(self, message: str, *, code: str = "transformer-graph-rejected") -> None:
        super().__init__(message)
        self.code = code


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_context(event: TraceEvent) -> tuple[str, str] | None:
    marker = event.metadata.get("framework_phase_marker")
    name = marker.get("name") if isinstance(marker, dict) else None
    if not isinstance(name, str):
        return None
    if name == "megatron_transformer_terminal_loss":
        return "forward", "terminal_loss"
    match = _COMPONENT.fullmatch(name)
    if match:
        return match.group(1), match.group(2)
    phase = _OUTER.get(name)
    if phase is None:
        raise TransformerGraphError(
            f"event {event.id!r} has unsupported framework marker {name!r}"
        )
    return phase, f"step.{phase}"


def _outer_phase_instance(event: TraceEvent) -> tuple[str, str] | None:
    markers = event.metadata.get("framework_phase_marker_stack")
    if not isinstance(markers, list):
        marker = event.metadata.get("framework_phase_marker")
        markers = [marker] if isinstance(marker, dict) else []
    matches: list[tuple[str, str]] = []
    for marker in markers:
        if not isinstance(marker, dict):
            continue
        phase = _OUTER.get(marker.get("name"))
        instance = marker.get("instance_id")
        if phase in {"forward", "backward"} and isinstance(instance, str):
            matches.append((phase, instance))
    if len(set(matches)) > 1:
        raise TransformerGraphError(
            f"event {event.id!r} belongs to multiple outer phase instances"
        )
    return matches[0] if matches else None


def semanticize_pp_rank(events: list[TraceEvent], stage: int) -> list[TraceEvent]:
    """Attribute repeated one-layer PP phase instances from outer+leaf markers."""

    if stage not in {0, 1}:
        raise TransformerGraphError("frozen PP semanticizer requires stage 0 or 1")
    grouped: dict[tuple[str, str], list[TraceEvent]] = {}
    for event in events:
        outer = _outer_phase_instance(event)
        if outer is not None:
            grouped.setdefault(outer, []).append(event)
    if not grouped:
        raise TransformerGraphError("PP rank lacks outer forward/backward instances")
    replacements: dict[str, TraceEvent] = {}
    layer = f"layers.{stage}"
    forward_leaves = [
        f"{layer}.self_attention.linear_qkv",
        f"{layer}.self_attention.linear_proj",
        f"{layer}.mlp.linear_fc1",
        f"{layer}.mlp.linear_fc2",
    ]
    backward_leaves = list(reversed(forward_leaves))
    expected_counts: dict[str, int] = {"forward": 0, "backward": 0}
    for (phase, _instance), instance_events in grouped.items():
        expected_counts[phase] += 1
        raw = [_raw_context(event) for event in instance_events]
        blocks: list[tuple[tuple[str, str], int, int]] = []
        for index, context in enumerate(raw):
            if context is None or context[1].startswith("step."):
                continue
            if blocks and blocks[-1][0] == context and blocks[-1][2] == index - 1:
                old = blocks[-1]
                blocks[-1] = (old[0], old[1], index)
            else:
                blocks.append((context, index, index))
        expected_components = (
            [
                *forward_leaves,
                *(["final_layernorm", "terminal_loss"] if stage == 1 else []),
            ]
            if phase == "forward"
            else [*(["final_layernorm"] if stage == 1 else []), *backward_leaves]
        )
        if [context[1] for context, _, _ in blocks] != expected_components or any(
            context[0] != phase for context, _, _ in blocks
        ):
            raise TransformerGraphError(
                f"PP stage {stage} {phase} landmark order differs from frozen rule"
            )
        assigned: list[tuple[str, str, str, str] | None] = [None] * len(instance_events)
        for context, start, end in blocks:
            for index in range(start, end + 1):
                assigned[index] = (
                    *context,
                    "direct-framework-marker",
                    "direct-pp-marker",
                )

        def fill(start: int, end: int, component: str, rule: str) -> None:
            for index in range(start, end):
                if assigned[index] is not None:
                    raise TransformerGraphError("PP semantic attribution overlaps")
                assigned[index] = (phase, component, "frozen-framework-rule", rule)

        first_start = blocks[0][1]
        prefix_component = (
            "terminal_loss"
            if phase == "backward" and stage == 1
            else (
                "final_layernorm"
                if phase == "backward" and blocks[0][0][1] == "final_layernorm"
                else layer
            )
        )
        fill(0, first_start, prefix_component, f"pp-{phase}-prefix")
        for block_index in range(len(blocks) - 1):
            previous, _, previous_end = blocks[block_index]
            following, following_start, _ = blocks[block_index + 1]
            start, end = previous_end + 1, following_start
            if start >= end:
                continue
            component, next_component = previous[1], following[1]
            if phase == "forward":
                if component.endswith("linear_qkv") and next_component.endswith(
                    "linear_proj"
                ):
                    owner = f"{layer}.self_attention"
                elif component.endswith("linear_proj") and next_component.endswith(
                    "linear_fc1"
                ):
                    owner = layer
                elif component.endswith("linear_fc1") and next_component.endswith(
                    "linear_fc2"
                ):
                    owner = f"{layer}.mlp"
                elif (
                    component.endswith("linear_fc2")
                    and next_component == "final_layernorm"
                ):
                    owner = layer
                elif (
                    component == "final_layernorm" and next_component == "terminal_loss"
                ):
                    owner = "terminal_loss"
                else:
                    raise TransformerGraphError(
                        "unsupported PP forward landmark transition"
                    )
            else:
                if component == "final_layernorm" and next_component.endswith(
                    "linear_fc2"
                ):
                    owner = "final_layernorm"
                elif component.endswith("linear_fc2") and next_component.endswith(
                    "linear_fc1"
                ):
                    owner = f"{layer}.mlp"
                elif component.endswith("linear_fc1") and next_component.endswith(
                    "linear_proj"
                ):
                    owner = layer
                elif component.endswith("linear_proj") and next_component.endswith(
                    "linear_qkv"
                ):
                    owner = f"{layer}.self_attention"
                else:
                    raise TransformerGraphError(
                        "unsupported PP backward landmark transition"
                    )
            fill(start, end, owner, f"pp-{phase}-gap")
        tail_component = "terminal_loss" if phase == "forward" and stage == 1 else layer
        fill(
            blocks[-1][2] + 1, len(instance_events), tail_component, f"pp-{phase}-tail"
        )
        if any(value is None for value in assigned):
            raise TransformerGraphError("PP semantic attribution is incomplete")
        for event, value in zip(instance_events, assigned, strict=True):
            if value is not None:
                replacements[event.id] = _semantic_event(
                    event, value[0], value[1], provenance=value[2], rule=value[3]
                )
    if not all(expected_counts.values()):
        raise TransformerGraphError("PP rank lacks a forward or backward phase")
    return [replacements.get(event.id, event) for event in events]


def _context(event: TraceEvent) -> tuple[str, str]:
    semantic = event.metadata.get("framework_semantic_context")
    if isinstance(semantic, dict):
        phase = semantic.get("phase")
        component = semantic.get("component")
        if isinstance(phase, str) and isinstance(component, str):
            return phase, component
    context = _raw_context(event)
    if context is None:
        raise TransformerGraphError(f"event {event.id!r} lacks semantic attribution")
    return context


def _operator(event: TraceEvent) -> dict[str, Any] | None:
    launch = event.metadata.get("kernel_launch_payload")
    if not isinstance(launch, dict):
        return None
    operator = launch.get("framework_operator")
    return dict(operator) if isinstance(operator, dict) else None


def _operator_name(event: TraceEvent) -> str | None:
    operator = _operator(event)
    name = operator.get("name") if isinstance(operator, dict) else None
    return name if isinstance(name, str) else None


def _process_group_ranks(event: TraceEvent) -> tuple[int, ...]:
    raw = event.metadata.get("process_group_ranks")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise TransformerGraphError(
                f"collective {event.id!r} has malformed process-group evidence"
            ) from error
    if not isinstance(raw, list) or not all(isinstance(rank, int) for rank in raw):
        raise TransformerGraphError(
            f"collective {event.id!r} lacks exact process-group membership"
        )
    ranks = tuple(raw)
    if len(set(ranks)) != len(ranks):
        raise TransformerGraphError(
            f"collective {event.id!r} has duplicate process-group ranks"
        )
    return ranks


def _semantic_event(
    event: TraceEvent,
    phase: str,
    component: str,
    *,
    provenance: str,
    rule: str,
) -> TraceEvent:
    return replace(
        event,
        metadata={
            **event.metadata,
            "framework_semantic_context": {
                "schema": "megatron-transformer-semantic-attribution-v1",
                "phase": phase,
                "component": component,
                "provenance": provenance,
                "rule": rule,
            },
        },
    )


def _semanticize_rank(
    events: list[TraceEvent], num_layers: int = 2
) -> list[TraceEvent]:
    """Attribute every event using the measured dense-Megatron landmark rule.

    Leaf markers are observations. Attribution between leaves is an explicit
    framework rule, not an inferred module-hook hierarchy. Any change to the
    landmark order, contiguity, or layer-boundary LayerNorm causes abstention.
    """

    if not events:
        raise TransformerGraphError("source rank is empty")
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers <= 0
    ):
        raise TransformerGraphError("number of Transformer layers must be positive")
    raw = [_raw_context(event) for event in events]
    # Fully marked synthetic/replay inputs remain useful, but physical partial
    # traces must satisfy the stricter landmark protocol below.
    if all(
        context is not None
        and (
            not context[1].startswith("step.")
            or context == ("optimizer", "step.optimizer")
        )
        for context in raw
    ):
        return [
            _semantic_event(
                event,
                context[0],
                context[1],
                provenance="direct-framework-marker",
                rule="direct-marker",
            )
            for event, context in zip(events, raw, strict=True)
        ]

    blocks: list[tuple[tuple[str, str], int, int]] = []
    for index, context in enumerate(raw):
        if context is None or context[1].startswith("step."):
            continue
        if blocks and blocks[-1][0] == context and blocks[-1][2] == index - 1:
            previous = blocks[-1]
            blocks[-1] = (previous[0], previous[1], index)
        else:
            blocks.append((context, index, index))
    optimizer_indices = [
        index
        for index, context in enumerate(raw)
        if context == ("optimizer", "step.optimizer")
    ]
    if not optimizer_indices:
        raise TransformerGraphError("source lacks optimizer landmark")
    first_optimizer = min(optimizer_indices)
    if optimizer_indices != list(range(first_optimizer, len(events))):
        raise TransformerGraphError(
            "optimizer landmark is not one terminal contiguous run"
        )
    coalesced_landmarks: list[tuple[str, str]] = []
    for context, _, _ in blocks:
        if not coalesced_landmarks or coalesced_landmarks[-1] != context:
            coalesced_landmarks.append(context)
    observed = tuple(coalesced_landmarks) + (("optimizer", "step.optimizer"),)
    expected_landmarks = _landmarks(num_layers)
    if observed != expected_landmarks:
        raise TransformerGraphError(
            "source leaf landmark order differs from the measured dense-Megatron rule: "
            f"observed={observed!r} expected={expected_landmarks!r}"
        )

    assigned: list[tuple[str, str, str, str] | None] = [None] * len(events)
    for context, start, end in blocks:
        for index in range(start, end + 1):
            assigned[index] = (
                *context,
                "direct-framework-marker",
                "direct-leaf-marker",
            )
    for index in optimizer_indices:
        assigned[index] = (
            "optimizer",
            "step.optimizer",
            "direct-framework-marker",
            "direct-optimizer-marker",
        )

    def fill(start: int, end: int, phase: str, component: str, rule: str) -> None:
        for index in range(start, end):
            if assigned[index] is not None:
                raise TransformerGraphError("semantic attribution rules overlap")
            assigned[index] = (phase, component, "frozen-framework-rule", rule)

    # Prefix is the input LayerNorm of layer 0.
    fill(0, blocks[0][1], "forward", "layers.0", "forward-layer-prefix")
    for block_index in range(len(blocks) - 1):
        previous, _, previous_end = blocks[block_index]
        following, following_start, _ = blocks[block_index + 1]
        start, end = previous_end + 1, following_start
        if start >= end:
            continue
        phase, component = previous
        next_phase, next_component = following
        if previous == following:
            fill(start, end, phase, component, "same-component-marker-gap")
        elif phase == next_phase == "forward":
            if component.endswith("linear_qkv") and next_component.endswith(
                "linear_proj"
            ):
                fill(
                    start,
                    end,
                    phase,
                    component.rsplit(".", 1)[0],
                    "forward-attention-core-gap",
                )
            elif component.endswith("linear_proj") and next_component.endswith(
                "linear_fc1"
            ):
                fill(
                    start,
                    end,
                    phase,
                    component.split(".self_attention", 1)[0],
                    "forward-post-attention-gap",
                )
            elif component.endswith("linear_fc1") and next_component.endswith(
                "linear_fc2"
            ):
                fill(
                    start,
                    end,
                    phase,
                    component.rsplit(".", 1)[0],
                    "forward-activation-gap",
                )
            elif component.endswith("linear_fc2") and next_component.endswith(
                "linear_qkv"
            ):
                boundary = end - 1
                if _operator_name(events[boundary]) != "aten::native_layer_norm":
                    raise TransformerGraphError(
                        "forward layer boundary lacks the frozen native_layer_norm landmark"
                    )
                fill(
                    start,
                    boundary,
                    phase,
                    component.split(".mlp", 1)[0],
                    "forward-layer-tail-gap",
                )
                fill(
                    boundary,
                    end,
                    phase,
                    next_component.split(".self_attention", 1)[0],
                    "forward-next-layernorm",
                )
            elif (
                component.endswith("linear_fc2") and next_component == "final_layernorm"
            ):
                fill(
                    start,
                    end,
                    phase,
                    component.split(".mlp", 1)[0],
                    "forward-final-layer-tail-gap",
                )
            else:
                raise TransformerGraphError("unsupported forward landmark transition")
        elif (
            phase == "forward"
            and component == "final_layernorm"
            and next_phase == "backward"
        ):
            backward_loss = [
                index
                for index in range(start, end)
                if _operator_name(events[index]) == "aten::mse_loss_backward"
            ]
            if len(backward_loss) != 1:
                raise TransformerGraphError(
                    "terminal loss boundary is not uniquely identified"
                )
            boundary = backward_loss[0]
            fill(start, boundary, "forward", "terminal_loss", "terminal-loss-forward")
            fill(boundary, end, "backward", "terminal_loss", "terminal-loss-backward")
        elif phase == next_phase == "backward":
            if component == "final_layernorm" and next_component.endswith("linear_fc2"):
                fill(start, end, phase, component, "backward-final-layernorm-tail")
            elif component.endswith("linear_fc2") and next_component.endswith(
                "linear_fc1"
            ):
                fill(
                    start,
                    end,
                    phase,
                    component.rsplit(".", 1)[0],
                    "backward-activation-gap",
                )
            elif component.endswith("linear_fc1") and next_component.endswith(
                "linear_proj"
            ):
                fill(
                    start,
                    end,
                    phase,
                    component.split(".mlp", 1)[0],
                    "backward-post-mlp-gap",
                )
            elif component.endswith("linear_proj") and next_component.endswith(
                "linear_qkv"
            ):
                fill(
                    start,
                    end,
                    phase,
                    component.rsplit(".", 1)[0],
                    "backward-attention-core-gap",
                )
            elif component.endswith("linear_qkv") and next_component.endswith(
                "linear_fc2"
            ):
                fill(
                    start,
                    end,
                    phase,
                    component.split(".self_attention", 1)[0],
                    "backward-layer-tail-gap",
                )
            else:
                raise TransformerGraphError("unsupported backward landmark transition")
        else:
            raise TransformerGraphError(
                "unsupported phase transition in source landmarks"
            )

    last_end = blocks[-1][2] + 1
    fill(
        last_end,
        first_optimizer,
        "backward",
        "layers.0",
        "backward-input-layernorm-tail",
    )
    if any(value is None for value in assigned):
        raise TransformerGraphError("semantic attribution did not cover every event")
    return [
        _semantic_event(
            event,
            value[0],
            value[1],
            provenance=value[2],
            rule=value[3],
        )
        for event, value in zip(events, assigned, strict=True)
        if value is not None
    ]


def _rank_signature(events: list[TraceEvent]) -> list[dict[str, Any]]:
    indices = {event.id: index for index, event in enumerate(events)}
    if len(indices) != len(events):
        raise TransformerGraphError("rank contains duplicate event ids")
    signature = []
    for event in events:
        if any(dependency not in indices for dependency in event.dependencies):
            raise TransformerGraphError("source has a cross-rank or unknown dependency")
        phase, component = _context(event)
        signature.append(
            {
                "kind": event.kind,
                "name": event.name,
                "phase": phase,
                "component": component,
                "dependencies": [indices[value] for value in event.dependencies],
                "collective": event.collective,
                "message_bytes": event.message_bytes,
                "group_role": event.group_role,
                "group_size": event.group_size,
                "operator": _operator(event),
            }
        )
    return signature


def _symmetric_source(
    trace: WorkloadTrace,
    measurement: dict[str, Any],
    *,
    source_tp: int = 2,
) -> list[TraceEvent]:
    physical_target = trace.source.get("target")
    if not isinstance(physical_target, str) or not physical_target.strip():
        raise TransformerGraphError("source lacks a physical accelerator target")
    limitations = trace.metadata.get("capture_limitations", [])
    if limitations not in ([], None):
        raise TransformerGraphError("source capture has unresolved limitations")
    by_rank = {
        rank: [event for event in trace.events if event.rank == rank]
        for rank in sorted({event.rank for event in trace.events})
    }
    expected_ranks = set(range(source_tp))
    if set(by_rank) != expected_ranks or any(not events for events in by_rank.values()):
        raise TransformerGraphError(
            f"source must cover exactly TP ranks {sorted(expected_ranks)}"
        )
    component_contract = measurement.get("component_markers")
    if not isinstance(component_contract, dict):
        raise TransformerGraphError("source lacks component marker contract")
    components = component_contract.get("components")
    phases = component_contract.get("phases")
    model = measurement.get("model")
    num_layers = model.get("num_layers") if isinstance(model, dict) else None
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers <= 0
    ):
        raise TransformerGraphError("source measurement has invalid layer count")
    expected_components = list(_leaf_components(num_layers))
    if components != expected_components or phases != ["forward", "backward"]:
        raise TransformerGraphError("source component marker contract is invalid")
    expected_contexts = {
        (phase, component) for phase in phases for component in components
    }
    semantic_by_rank: dict[int, list[TraceEvent]] = {}
    for rank, events in by_rank.items():
        semantic_events = _semanticize_rank(events, num_layers=num_layers)
        semantic_by_rank[rank] = semantic_events
        contexts = {_context(event) for event in semantic_events}
        missing = expected_contexts - contexts
        if missing or ("optimizer", "step.optimizer") not in contexts:
            raise TransformerGraphError(
                f"source rank {rank} lacks complete component or phase coverage"
            )
    reference = _rank_signature(semantic_by_rank[0])
    for rank in range(1, source_tp):
        if _rank_signature(semantic_by_rank[rank]) != reference:
            raise TransformerGraphError(
                "source TP ranks are not semantically symmetric"
            )
    return semantic_by_rank[0]


def _abstract_event(event: TraceEvent, *, group_size: int) -> TraceEvent:
    phase, component = _context(event)
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "pipeline_phase": phase,
        "semantic_component": component,
        "source_event": {"id": event.id, "rank": event.rank},
        "framework_operator": _operator(event),
        "provenance": "source-semantic-structure-not-target-runtime-observation",
        "timing_status": "unresolved-requires-post-admission-calibration",
        "kernel_code_status": "unresolved-requires-target-observation",
    }
    if event.kind == "collective":
        metadata["message_bytes_semantics"] = "payload"
    return replace(
        event,
        duration_us=0.0,
        observed_start_us=None,
        group_size=group_size if event.kind == "collective" else event.group_size,
        sm_fraction=None,
        metadata=metadata,
    )


def _parameter_rows_for_component(
    rows: list[dict[str, Any]], component: str
) -> list[dict[str, Any]]:
    """Bind semantic components to parameters without assuming a root prefix."""

    marker = f".{component}."
    return [
        row
        for row in rows
        if isinstance(row.get("name"), str)
        and (
            row["name"] == component
            or row["name"].startswith(f"{component}.")
            or marker in row["name"]
        )
    ]


def _semantic_tp_template(
    rule: dict[str, Any],
    target_manifest: dict[str, Any],
    *,
    include_terminal_loss: bool,
) -> list[TraceEvent]:
    """Build one framework-level target lane, including derived TP sites."""

    layers = rule["model"]["num_layers"]
    target_tp = rule["target_tp"]
    payload = rule["collective_rule"]["payload_bytes"]
    manifest_rows = target_manifest["parameters"]
    events: list[TraceEvent] = []
    previous: str | None = None

    def add_compute(phase: str, component: str) -> None:
        nonlocal previous
        event_id = f"semantic::{phase}::{component}"
        relevant = _parameter_rows_for_component(manifest_rows, component)
        events.append(
            TraceEvent(
                id=event_id,
                name=f"Megatron {phase} {component}",
                kind="compute",
                duration_us=0.0,
                dependencies=() if previous is None else (previous,),
                metadata={
                    "schema": SCHEMA,
                    "pipeline_phase": phase,
                    "semantic_component": component,
                    "provenance": "pinned-megatron-semantic-component-rule",
                    "target_tensor_contract": relevant,
                    "target_parameter_manifest_sha256": target_manifest[
                        "canonical_sha256"
                    ],
                    "timing_status": "unresolved-requires-target-calibration",
                    "kernel_code_status": "unresolved-requires-target-observation",
                },
            )
        )
        previous = event_id

    def add_collective(phase: str, component: str, site: str) -> None:
        nonlocal previous
        if target_tp == 1:
            return
        event_id = f"semantic::{phase}::{component}::tp-all-reduce"
        events.append(
            TraceEvent(
                id=event_id,
                name=f"Megatron TP all-reduce: {site}",
                kind="collective",
                duration_us=0.0,
                dependencies=() if previous is None else (previous,),
                collective="all_reduce",
                message_bytes=payload,
                group_role="tp",
                group_size=target_tp,
                metadata={
                    "schema": SCHEMA,
                    "pipeline_phase": phase,
                    "semantic_component": component,
                    "framework_collective_site": site,
                    "message_bytes_semantics": "payload",
                    "provenance": "pinned-megatron-tp-collective-rule",
                    "timing_status": "unresolved-requires-target-calibration",
                },
            )
        )
        previous = event_id

    for layer in range(layers):
        prefix = f"layers.{layer}"
        add_compute("forward", f"{prefix}.self_attention.linear_qkv")
        add_compute("forward", f"{prefix}.self_attention.linear_proj")
        add_collective(
            "forward",
            f"{prefix}.self_attention.linear_proj",
            "attention-projection-forward",
        )
        add_compute("forward", f"{prefix}.mlp.linear_fc1")
        add_compute("forward", f"{prefix}.mlp.linear_fc2")
        add_collective("forward", f"{prefix}.mlp.linear_fc2", "mlp-fc2-forward")
    add_compute("forward", "final_layernorm")
    if include_terminal_loss:
        add_compute("forward", "terminal_loss")
        add_compute("backward", "terminal_loss")
    add_compute("backward", "final_layernorm")
    for layer in reversed(range(layers)):
        prefix = f"layers.{layer}"
        add_compute("backward", f"{prefix}.mlp.linear_fc2")
        add_compute("backward", f"{prefix}.mlp.linear_fc1")
        add_collective(
            "backward",
            f"{prefix}.mlp.linear_fc1",
            "mlp-fc1-input-gradient",
        )
        add_compute("backward", f"{prefix}.self_attention.linear_proj")
        add_compute("backward", f"{prefix}.self_attention.linear_qkv")
        add_collective(
            "backward",
            f"{prefix}.self_attention.linear_qkv",
            "attention-qkv-input-gradient",
        )
    add_compute("optimizer", "step.optimizer")
    return events


def compile_megatron_tp_candidate(
    source: WorkloadTrace,
    *,
    source_tp: int,
    target_tp: int,
    target: str,
) -> tuple[WorkloadTrace, dict[str, Any]]:
    """Compile a dense Megatron semantic TP candidate for any legal width.

    The compiler is deliberately structural: it consumes only the physical
    source capture and its bound framework measurement.  Target durations,
    target code objects, and target traces are neither accepted nor emitted.
    """

    if isinstance(source_tp, bool) or not isinstance(source_tp, int) or source_tp <= 0:
        raise TransformerGraphError("source TP must be a positive integer")
    if isinstance(target_tp, bool) or not isinstance(target_tp, int) or target_tp <= 0:
        raise TransformerGraphError("target TP must be a positive integer")
    if not isinstance(target, str) or not target.strip():
        raise TransformerGraphError("target accelerator must be a non-empty string")
    target = target.strip().lower()
    measurement = source.metadata.get("framework_measurement")
    if not isinstance(measurement, dict):
        raise TransformerGraphError("source lacks framework measurement")
    if measurement.get("tensor_parallel_size") != source_tp:
        raise TransformerGraphError(
            "requested source TP differs from the signed framework measurement"
        )
    try:
        rule = derive_megatron_tp_rule(measurement, target_tp=target_tp)
        target_manifest = derive_megatron_tp_parameter_manifest(
            measurement, target_tp=target_tp
        )
    except TransformerRuleError as error:
        raise TransformerGraphError(
            f"unsupported TP rewrite: {error}", code=error.code
        ) from error
    template = _symmetric_source(source, measurement, source_tp=source_tp)
    collective_rule = rule["collective_rule"]
    collectives = [event for event in template if event.kind == "collective"]
    expected_source_collectives = (
        0 if source_tp == 1 else collective_rule["per_rank_operation_count"]
    )
    if len(collectives) != expected_source_collectives:
        raise TransformerGraphError(
            "source TP collective count differs from the derived framework rule"
        )
    source_group = tuple(rule["source_tp_group"])
    for event in collectives:
        if (
            event.collective != collective_rule["operation"]
            or event.group_role != "tp"
            or event.group_size not in (None, source_tp)
            or _process_group_ranks(event) != source_group
            or event.message_bytes != collective_rule["payload_bytes"]
        ):
            raise TransformerGraphError(
                f"source collective {event.id!r} differs from the derived TP rule"
            )
    include_terminal_loss = any(
        _context(event)[1] == "terminal_loss" for event in template
    )
    abstract = _semantic_tp_template(
        rule,
        target_manifest,
        include_terminal_loss=include_terminal_loss,
    )
    target_group = list(range(target_tp))
    generated: list[TraceEvent] = []
    for rank in range(target_tp):
        for event in abstract:
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
                        "target_rank": rank,
                        "target_tp_group": target_group,
                        **(
                            {"process_group_ranks": target_group}
                            if event.kind == "collective"
                            else {}
                        ),
                    },
                )
            )
    source_sha256 = _canonical_sha256(source.to_dict())
    report = {
        "schema": "scaletether-generic-megatron-tp-rewrite-v1",
        "status": "generated-semantic-candidate-awaiting-heldout-validation",
        "source_tp": source_tp,
        "target_tp": target_tp,
        "source_parallelism": {"tp": source_tp, "pp": 1, "dp": 1},
        "target_parallelism": {"tp": target_tp, "pp": 1, "dp": 1},
        "target": target,
        "source_sha256": source_sha256,
        "source_rank_event_count": len(template),
        "target_rank_event_count": len(abstract),
        "target_event_count": len(generated),
        "framework_rule_sha256": _canonical_sha256(rule),
        "framework_rule": rule,
        "target_parameter_manifest": target_manifest,
        "target_parameter_manifest_sha256": target_manifest["canonical_sha256"],
        "target_training_executed": False,
        "timing_claim": "none",
        "kernel_code_claim": "none",
        "claim": "generated-framework-semantic-graph-no-physical-validity-claim",
    }
    candidate = WorkloadTrace(
        events=tuple(generated),
        source={
            "kind": "framework-semantic-counterfactual",
            "target": target,
            "rank_count": target_tp,
            "framework": "megatron-core",
            "candidate_training_executed": False,
            "source_sha256": source_sha256,
        },
        metadata={
            "framework_measurement": measurement,
            "generic_megatron_tp_rewrite": report,
            "provenance": {
                "observed": (
                    f"source-tp{source_tp}-component-and-dependency-structure"
                ),
                "transformed": (
                    "tp-rank-replication-communicator-width-and-framework-shapes"
                ),
                "estimated": [],
                "unresolved": ["target-kernel-code", "target-timing"],
            },
        },
    )
    candidate.validate()
    validate_megatron_tp_candidate(candidate)
    return candidate, report


def validate_megatron_tp_candidate(candidate: WorkloadTrace) -> dict[str, Any]:
    """Validate the generic candidate's derivation and fail-closed invariants.

    This is a pre-physical structural gate.  Passing it never asserts timing
    accuracy or equality with a held-out target execution.
    """

    report = candidate.metadata.get("generic_megatron_tp_rewrite")
    if not isinstance(report, dict):
        raise TransformerGraphError("candidate lacks generic TP rewrite report")
    source_tp = report.get("source_tp")
    target_tp = report.get("target_tp")
    if (
        isinstance(source_tp, bool)
        or not isinstance(source_tp, int)
        or source_tp <= 0
        or isinstance(target_tp, bool)
        or not isinstance(target_tp, int)
        or target_tp <= 0
    ):
        raise TransformerGraphError("candidate TP widths are invalid")
    measurement = candidate.metadata.get("framework_measurement")
    if not isinstance(measurement, dict):
        raise TransformerGraphError("candidate lacks its source framework measurement")
    try:
        expected_rule = derive_megatron_tp_rule(measurement, target_tp=target_tp)
        expected_manifest = derive_megatron_tp_parameter_manifest(
            measurement, target_tp=target_tp
        )
    except TransformerRuleError as error:
        raise TransformerGraphError(
            f"candidate is outside the pinned TP applicability domain: {error}"
        ) from error
    rule = report.get("framework_rule")
    if not isinstance(rule, dict):
        raise TransformerGraphError("candidate lacks derived framework rule")
    if (
        rule.get("source_tp") != source_tp
        or rule.get("target_tp") != target_tp
        or rule != expected_rule
        or report.get("framework_rule_sha256") != _canonical_sha256(rule)
    ):
        raise TransformerGraphError("candidate framework rule binding is invalid")
    target_manifest = report.get("target_parameter_manifest")
    if (
        not isinstance(target_manifest, dict)
        or report.get("target_parameter_manifest_sha256")
        != target_manifest.get("canonical_sha256")
        or target_manifest != expected_manifest
    ):
        raise TransformerGraphError("candidate target parameter manifest is unbound")
    unsigned_manifest = {
        key: value
        for key, value in target_manifest.items()
        if key != "canonical_sha256"
    }
    if target_manifest.get("canonical_sha256") != _canonical_sha256(unsigned_manifest):
        raise TransformerGraphError(
            "candidate target parameter manifest hash is invalid"
        )
    if (
        candidate.source.get("kind") != "framework-semantic-counterfactual"
        or candidate.source.get("candidate_training_executed") is not False
        or candidate.source.get("rank_count") != target_tp
        or report.get("target_training_executed") is not False
        or report.get("timing_claim") != "none"
        or report.get("kernel_code_claim") != "none"
    ):
        raise TransformerGraphError("candidate provenance or claim boundary is invalid")
    expected_ranks = set(range(target_tp))
    by_rank = {
        rank: [event for event in candidate.events if event.rank == rank]
        for rank in sorted({event.rank for event in candidate.events})
    }
    if set(by_rank) != expected_ranks or any(not events for events in by_rank.values()):
        raise TransformerGraphError("candidate does not cover every target TP rank")
    expected_count = report.get("target_rank_event_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or report.get("target_event_count") != expected_count * target_tp
        or len(candidate.events) != expected_count * target_tp
    ):
        raise TransformerGraphError("candidate event counts violate the TP rule")
    target_group = list(range(target_tp))
    manifest_rows = target_manifest.get("parameters")
    if not isinstance(manifest_rows, list):
        raise TransformerGraphError("candidate target parameter rows are invalid")
    collective_rule = rule.get("collective_rule")
    if not isinstance(collective_rule, dict):
        raise TransformerGraphError("candidate lacks TP collective rule")
    for rank, events in by_rank.items():
        if len(events) != expected_count:
            raise TransformerGraphError("candidate TP ranks have unequal event counts")
        identifiers = {event.id for event in events}
        if len(identifiers) != len(events):
            raise TransformerGraphError("candidate rank contains duplicate event ids")
        for event in events:
            if (
                event.duration_us != 0.0
                or event.observed_start_us is not None
                or event.sm_fraction is not None
                or any(
                    dependency not in identifiers for dependency in event.dependencies
                )
            ):
                raise TransformerGraphError(
                    "candidate inherited target-unauthorized runtime or dependency data"
                )
            if (
                event.metadata.get("target_rank") != rank
                or event.metadata.get("target_tp_group") != target_group
            ):
                raise TransformerGraphError("candidate target-rank metadata is invalid")
            if event.kind == "compute":
                component = event.metadata.get("semantic_component")
                expected_tensor_contract = (
                    _parameter_rows_for_component(manifest_rows, component)
                    if isinstance(component, str)
                    else []
                )
                if (
                    event.metadata.get("target_parameter_manifest_sha256")
                    != target_manifest.get("canonical_sha256")
                    or event.metadata.get("target_tensor_contract")
                    != expected_tensor_contract
                ):
                    raise TransformerGraphError(
                        "candidate event tensor contract is invalid"
                    )
        collectives = [event for event in events if event.kind == "collective"]
        expected_collectives = (
            0 if target_tp == 1 else collective_rule.get("per_rank_operation_count")
        )
        if len(collectives) != expected_collectives:
            raise TransformerGraphError("candidate collective count is invalid")
        for event in collectives:
            if (
                event.collective != collective_rule.get("operation")
                or event.group_role != "tp"
                or event.group_size != target_tp
                or event.message_bytes != collective_rule.get("payload_bytes")
                or event.metadata.get("process_group_ranks") != target_group
            ):
                raise TransformerGraphError("candidate collective contract is invalid")
    return {
        "schema": "scaletether-generic-megatron-tp-candidate-validation-v1",
        "passed": True,
        "source_tp": source_tp,
        "target_tp": target_tp,
        "rank_count": target_tp,
        "event_count": len(candidate.events),
        "claim": "generated-structure-only-no-target-timing-claim",
    }


def compare_megatron_tp_candidate_to_physical(
    candidate: WorkloadTrace, physical: WorkloadTrace
) -> dict[str, Any]:
    """Score a frozen generic candidate against a subsequently opened target."""

    candidate_validation = validate_megatron_tp_candidate(candidate)
    rewrite = candidate.metadata["generic_megatron_tp_rewrite"]
    target_tp = rewrite["target_tp"]
    physical_measurement = physical.metadata.get("framework_measurement")
    failures: list[str] = []
    if not isinstance(physical_measurement, dict):
        failures.append("physical-framework-measurement")
        physical_measurement = {}
    try:
        manifest_comparison = compare_megatron_tp_parameter_manifest(
            rewrite["target_parameter_manifest"], physical_measurement
        )
    except TransformerRuleError as error:
        manifest_comparison = {
            "schema": "scaletether-megatron-tp-parameter-manifest-comparison-v1",
            "passed": False,
            "failures": [f"malformed-physical-manifest:{error}"],
        }
    manifest_match = manifest_comparison["passed"] is True
    if not manifest_match:
        failures.append("parameter-manifest")

    try:
        physical_template = _symmetric_source(
            physical, physical_measurement, source_tp=target_tp
        )
    except TransformerGraphError as error:
        physical_template = []
        failures.append(f"physical-semantic-source:{error}")
    candidate_lane = [event for event in candidate.events if event.rank == 0]

    def component_order(events: list[TraceEvent], *, generated: bool) -> list[str]:
        result: list[str] = []
        for event in events:
            if event.kind != "compute":
                continue
            if generated:
                phase = event.metadata.get("pipeline_phase")
                component = event.metadata.get("semantic_component")
            else:
                phase, component = _context(event)
            if not isinstance(phase, str) or not isinstance(component, str):
                continue
            admitted = component in {
                "final_layernorm",
                "terminal_loss",
                "step.optimizer",
            } or component.endswith(
                (
                    "self_attention.linear_qkv",
                    "self_attention.linear_proj",
                    "mlp.linear_fc1",
                    "mlp.linear_fc2",
                )
            )
            label = f"{phase}::{component}"
            if admitted and (not result or result[-1] != label):
                result.append(label)
        return result

    candidate_order = component_order(candidate_lane, generated=True)
    physical_order = component_order(physical_template, generated=False)
    semantic_nodes_match = bool(physical_template) and sorted(
        candidate_order
    ) == sorted(physical_order)
    if not semantic_nodes_match:
        failures.append("semantic-component-nodes")
    semantic_order_match = bool(physical_template) and physical_order == candidate_order
    if not semantic_order_match:
        failures.append("semantic-component-order")

    # Verify that each consecutive normalized physical component is causally
    # reachable through the full captured DAG, including ignored lowering nodes.
    physical_by_id = {event.id: event for event in physical_template}
    ancestor_memo: dict[str, set[str]] = {}

    def physical_ancestors(identifier: str) -> set[str]:
        if identifier in ancestor_memo:
            return ancestor_memo[identifier]
        event = physical_by_id[identifier]
        value: set[str] = set(event.dependencies)
        for dependency in event.dependencies:
            if dependency in physical_by_id:
                value.update(physical_ancestors(dependency))
        ancestor_memo[identifier] = value
        return value

    dependency_match = bool(physical_template) and semantic_order_match
    if dependency_match:
        blocks: list[tuple[str, set[str]]] = []
        for event in physical_template:
            if event.kind != "compute":
                continue
            phase, component = _context(event)
            label = f"{phase}::{component}"
            if label not in candidate_order:
                continue
            if blocks and blocks[-1][0] == label:
                blocks[-1][1].add(event.id)
            else:
                blocks.append((label, {event.id}))
        for previous, following in zip(blocks, blocks[1:]):
            if not any(
                physical_ancestors(identifier) & previous[1]
                for identifier in following[1]
            ):
                dependency_match = False
                break
    if not dependency_match:
        failures.append("semantic-dependencies")

    target_group = tuple(range(target_tp))
    physical_collectives = [
        event
        for event in physical_template
        if event.kind == "collective" and event.group_role == "tp"
    ]
    candidate_collectives = [
        event
        for event in candidate_lane
        if event.kind == "collective" and event.group_role == "tp"
    ]
    expected_collective_count = (
        0
        if target_tp == 1
        else rewrite["framework_rule"]["collective_rule"]["per_rank_operation_count"]
    )
    collective_contract_match = len(physical_collectives) == expected_collective_count
    for event in physical_collectives:
        try:
            members = _process_group_ranks(event)
        except TransformerGraphError:
            members = ()
        if (
            event.collective != "all_reduce"
            or event.message_bytes
            != rewrite["framework_rule"]["collective_rule"]["payload_bytes"]
            or event.group_size not in (None, target_tp)
            or members != target_group
        ):
            collective_contract_match = False
    if not collective_contract_match:
        failures.append("collective-contract")

    def normalized_collective(event: TraceEvent, *, generated: bool) -> dict[str, Any]:
        if generated:
            phase = event.metadata.get("pipeline_phase")
            component = event.metadata.get("semantic_component")
        else:
            phase, component = _context(event)
        try:
            members = list(_process_group_ranks(event))
        except TransformerGraphError:
            members = []
        return {
            "phase": phase,
            "component": component,
            "operation": event.collective,
            "payload_bytes": event.message_bytes,
            "group_role": event.group_role,
            "group_size": (
                event.group_size if event.group_size is not None else len(members)
            ),
            "group_members": members,
        }

    expected_collective_sequence = [
        normalized_collective(event, generated=True) for event in candidate_collectives
    ]
    observed_collective_sequence = [
        normalized_collective(event, generated=False) for event in physical_collectives
    ]
    collective_sites_order_match = (
        observed_collective_sequence == expected_collective_sequence
    )
    if not collective_sites_order_match:
        failures.append("collective-sites-or-order")

    # A matching list of collectives is not enough: each physical collective
    # must be causally downstream of its producing semantic component and
    # upstream of the following semantic component.  This rejects traces that
    # preserve counts and bytes while moving communication to another point in
    # the training step.
    collective_dependencies_match = collective_sites_order_match
    if collective_dependencies_match:
        physical_compute_ids: dict[str, set[str]] = {}
        for event in physical_template:
            if event.kind != "compute":
                continue
            phase, component = _context(event)
            physical_compute_ids.setdefault(f"{phase}::{component}", set()).add(
                event.id
            )
        candidate_indices = {
            event.id: index for index, event in enumerate(candidate_lane)
        }
        for candidate_event, physical_event in zip(
            candidate_collectives, physical_collectives, strict=True
        ):
            phase = candidate_event.metadata.get("pipeline_phase")
            component = candidate_event.metadata.get("semantic_component")
            producer_ids = physical_compute_ids.get(f"{phase}::{component}", set())
            if not producer_ids or not (
                physical_ancestors(physical_event.id) & producer_ids
            ):
                collective_dependencies_match = False
                break
            following_label = None
            for event in candidate_lane[candidate_indices[candidate_event.id] + 1 :]:
                if event.kind == "compute":
                    following_label = (
                        f"{event.metadata.get('pipeline_phase')}::"
                        f"{event.metadata.get('semantic_component')}"
                    )
                    break
            following_ids = physical_compute_ids.get(following_label or "", set())
            if following_label is None or not any(
                physical_event.id in physical_ancestors(identifier)
                for identifier in following_ids
            ):
                collective_dependencies_match = False
                break
    if not collective_dependencies_match:
        failures.append("collective-dependencies")

    gates = {
        "candidate_internal": {
            "passed": candidate_validation["passed"] is True,
        },
        "target_parameter_manifest": {
            "passed": manifest_match,
            "failures": manifest_comparison.get("failures", []),
        },
        "semantic_nodes": {
            "passed": semantic_nodes_match,
            "expected": sorted(candidate_order),
            "observed": sorted(physical_order),
        },
        "semantic_order": {
            "passed": semantic_order_match,
            "expected": candidate_order,
            "observed": physical_order,
        },
        "semantic_dependencies": {
            "passed": dependency_match,
            "contract": "each-consecutive-semantic-block-is-causally-reachable",
        },
        "collective_contract": {
            "passed": collective_contract_match,
            "expected_per_rank": expected_collective_count,
            "observed_per_rank": len(physical_collectives),
            "payload_bytes": rewrite["framework_rule"]["collective_rule"][
                "payload_bytes"
            ],
            "group": list(target_group),
        },
        "collective_sites_order": {
            "passed": collective_sites_order_match,
            "expected": expected_collective_sequence,
            "observed": observed_collective_sequence,
        },
        "collective_dependencies": {
            "passed": collective_dependencies_match,
            "contract": (
                "producer-component-precedes-collective-and-collective-precedes-"
                "following-component"
            ),
        },
    }
    passed = all(gate["passed"] is True for gate in gates.values()) and not failures
    return {
        "schema": "scaletether-generic-megatron-tp-physical-comparison-v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "source_tp": rewrite["source_tp"],
        "target_tp": target_tp,
        "candidate_sha256": _canonical_sha256(candidate.to_dict()),
        "physical_sha256": _canonical_sha256(physical.to_dict()),
        "gates": gates,
        "failures": failures,
        "parameter_manifest_comparison": manifest_comparison,
        "semantic_component_order": gates["semantic_order"],
        "collective_contract": gates["collective_contract"],
        "claim": (
            "heldout-physical-structure-matches-derived-tp-candidate"
            if passed
            else "no-heldout-physical-structure-validity-claim"
        ),
    }


def compile_tp_candidate(
    matrix: dict[str, Any],
    source: WorkloadTrace,
    target_id: str,
    *,
    target_exists: bool,
) -> tuple[WorkloadTrace, dict[str, Any]]:
    if not target_id.startswith("TP-"):
        raise TransformerGraphError("TP compiler requires a frozen TP target")
    measurement = source.metadata.get("framework_measurement")
    if not isinstance(measurement, dict):
        raise TransformerGraphError("source lacks framework measurement")
    plan = build_rule_plan(matrix, measurement, target_id, target_exists=target_exists)
    admission = plan.get("admission")
    if not isinstance(admission, dict):
        raise TransformerGraphError("TP plan lacks generation admission")
    source_parallelism = admission.get("source_parallelism")
    target_parallelism = admission.get("target_parallelism")
    if not isinstance(source_parallelism, dict) or not isinstance(
        target_parallelism, dict
    ):
        raise TransformerGraphError("TP plan lacks parallelism contracts")
    source_tp = source_parallelism.get("tp")
    target_tp = target_parallelism.get("tp")
    if (
        isinstance(source_tp, bool)
        or not isinstance(source_tp, int)
        or source_tp <= 0
        or isinstance(target_tp, bool)
        or not isinstance(target_tp, int)
        or target_tp <= 0
        or target_parallelism.get("pp") != 1
        or target_parallelism.get("dp") != 1
    ):
        raise TransformerGraphError("TP plan has invalid target parallelism")
    template = _symmetric_source(source, measurement, source_tp=source_tp)
    collective_rule = plan["framework_rule"]["collective_rule"]
    collectives = [event for event in template if event.kind == "collective"]
    if len(collectives) != collective_rule["per_rank_operation_count"]:
        raise TransformerGraphError(
            "source TP collective count differs from framework rule"
        )
    for event in collectives:
        if (
            event.collective != collective_rule["operation"]
            or event.group_role != "tp"
            or event.group_size not in (None, source_tp)
            or _process_group_ranks(event) != tuple(range(source_tp))
            or event.message_bytes != collective_rule["payload_bytes"]
        ):
            raise TransformerGraphError(
                f"source collective {event.id!r} differs from the frozen TP rule"
            )
    abstract = [_abstract_event(event, group_size=target_tp) for event in template]
    generated: list[TraceEvent] = []
    target_group = list(range(target_tp))
    for rank in target_group:
        for event in abstract:
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
                        "target_rank": rank,
                        "target_tp_group": target_group,
                    },
                )
            )
    source_sha256 = _canonical_sha256(source.to_dict())
    report = {
        "schema": SCHEMA,
        "status": "generated-semantic-candidate-awaiting-heldout-validation",
        "target_id": target_id,
        "source_sha256": source_sha256,
        "source_rank_event_count": len(template),
        "target_rank_event_count": len(template),
        "target_event_count": len(generated),
        "source_tp": source_tp,
        "target_tp": target_tp,
        "framework_rule_sha256": plan["framework_rule_sha256"],
        "target_training_executed": False,
        "timing_claim": "none",
        "kernel_code_claim": "none",
        "claim": "generated-framework-semantic-graph-no-physical-validity-claim",
    }
    candidate = WorkloadTrace(
        events=tuple(generated),
        source={
            "kind": "framework-semantic-counterfactual",
            "target": "h100",
            "framework": "megatron-core",
            "candidate_training_executed": False,
            "source_sha256": source_sha256,
        },
        metadata={
            "framework_measurement": measurement,
            "transformer_rule_plan": plan,
            "transformer_semantic_graph": report,
            "provenance": {
                "observed": "source-tp2-component-and-dependency-structure",
                "transformed": "tp-rank-replication-and-communicator-width",
                "estimated": [],
                "unresolved": ["target-kernel-code", "target-timing"],
            },
        },
    )
    candidate.validate()
    return candidate, report


def _last_ready_component(
    measurement: dict[str, Any],
    template: list[TraceEvent],
    parameter: str | None = None,
) -> str | None:
    readiness = measurement.get("gradient_readiness")
    component_contract = measurement.get("component_markers")
    if not isinstance(readiness, dict) or not isinstance(component_contract, dict):
        raise TransformerGraphError("DP source lacks readiness or component contract")
    names = readiness.get("parameter_names")
    declared_components = component_contract.get("components")
    if (
        not isinstance(names, list)
        or not names
        or not isinstance(declared_components, list)
    ):
        raise TransformerGraphError("DP source readiness or components are incomplete")
    parameter = names[-1] if parameter is None else parameter
    if parameter not in names:
        raise TransformerGraphError(
            "gradient-ready parameter is absent from the measured readiness order"
        )
    # Only explicitly declared leaf markers can own a framework call.  Broad
    # semanticized parents such as ``layers.0`` are attribution helpers, not
    # active module scopes.  PyTorch may invoke a reducer hook after the final
    # leaf scope closes; that is represented by ``None`` (the backward-phase
    # frontier), not a fabricated parent component.
    components = {
        str(component)
        for component in declared_components
        if isinstance(component, str)
    }
    matches = [
        component
        for component in components
        if parameter == component or parameter.startswith(f"{component}.")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _compile_dp_candidate_from_plan(
    source: WorkloadTrace,
    target_id: str,
    plan: dict[str, Any],
) -> tuple[WorkloadTrace, dict[str, Any]]:
    if not target_id.startswith("DP-"):
        raise TransformerGraphError("DP compiler requires a frozen DP target")
    measurement = source.metadata.get("framework_measurement")
    if not isinstance(measurement, dict):
        raise TransformerGraphError("source lacks framework measurement")
    symmetric_template = _symmetric_source(source, measurement)
    admission = plan.get("admission")
    if not isinstance(admission, dict):
        raise TransformerGraphError("DP plan lacks generation admission")
    source_parallelism = admission.get("source_parallelism")
    target_parallelism = admission.get("target_parallelism")
    if not isinstance(source_parallelism, dict) or not isinstance(
        target_parallelism, dict
    ):
        raise TransformerGraphError("DP plan lacks parallelism contracts")
    source_tp = source_parallelism.get("tp")
    target_tp = target_parallelism.get("tp")
    target_pp = target_parallelism.get("pp")
    target_dp = target_parallelism.get("dp")
    if (
        isinstance(source_tp, bool)
        or not isinstance(source_tp, int)
        or source_tp <= 0
        or target_tp != source_tp
        or target_pp != 1
        or isinstance(target_dp, bool)
        or not isinstance(target_dp, int)
        or target_dp <= 1
    ):
        raise TransformerGraphError(
            "DP plan must preserve positive TP, preserve PP=1, and increase DP"
        )
    target_world_size = target_tp * target_dp
    by_rank = {
        rank: _semanticize_rank(
            [event for event in source.events if event.rank == rank]
        )
        for rank in range(source_tp)
    }
    if any(not events for events in by_rank.values()):
        raise TransformerGraphError("DP source is missing a declared TP rank")
    ddp_contract = plan["framework_rule"]["ddp_contract"]
    gradient_buckets = ddp_contract["gradient_buckets"]
    forward_broadcasts = ddp_contract["forward_metadata_broadcasts"]
    bucket_components = [
        _last_ready_component(
            measurement, symmetric_template, bucket["last_ready_parameter"]
        )
        for bucket in gradient_buckets
    ]
    generated: list[TraceEvent] = []
    for target_rank in range(target_world_size):
        source_rank = target_rank % target_tp
        replica = target_rank // target_tp
        tp_group = list(
            range(replica * target_tp, (replica + 1) * target_tp)
        )
        dp_group = [
            source_rank + dp_replica * target_tp
            for dp_replica in range(target_dp)
        ]
        template = by_rank[source_rank]
        local_ids = {event.id for event in template}
        abstract = [_abstract_event(event, group_size=target_tp) for event in template]
        prefixed: list[TraceEvent] = []
        for event in abstract:
            metadata = {
                **event.metadata,
                "target_rank": target_rank,
                "target_tp_group": tp_group,
                "target_dp_group": dp_group,
                "target_dp_replica": replica,
            }
            prefixed.append(
                replace(
                    event,
                    id=f"rank{target_rank}::{event.id}",
                    rank=target_rank,
                    device=target_rank,
                    dependencies=tuple(
                        f"rank{target_rank}::{dependency}"
                        if dependency in local_ids
                        else dependency
                        for dependency in event.dependencies
                    ),
                    metadata=metadata,
                )
            )
        original_ids = {event.id for event in prefixed}
        original_roots = {
            event.id
            for event in prefixed
            if not any(dependency in original_ids for dependency in event.dependencies)
        }
        if not original_roots:
            raise TransformerGraphError("source semantic graph has no DAG root")

        dp_events: list[TraceEvent] = []
        previous_broadcast: str | None = None
        for broadcast in forward_broadcasts:
            broadcast_id = (
                f"rank{target_rank}::dp-forward-metadata-{broadcast['index']}"
            )
            dp_events.append(
                TraceEvent(
                    id=broadcast_id,
                    name="framework-derived-ddp-forward-metadata-broadcast",
                    kind="collective",
                    duration_us=0.0,
                    rank=target_rank,
                    device=target_rank,
                    dependencies=(previous_broadcast,) if previous_broadcast else (),
                    collective="broadcast",
                    message_bytes=broadcast["payload_bytes"],
                    group_role="dp",
                    group_size=target_dp,
                    metadata={
                        "schema": SCHEMA,
                        "pipeline_phase": "forward",
                        "semantic_component": "step.ddp_bucket_rebuild",
                        "target_rank": target_rank,
                        "target_tp_group": tp_group,
                        "target_dp_group": dp_group,
                        "ddp_metadata_purpose": broadcast["purpose"],
                        "torch_version": ddp_contract["torch_version"],
                        "provenance": "explicit-pinned-ddp-warm-reducer-framework-rule",
                        "timing_status": "unresolved-requires-post-admission-calibration",
                    },
                )
            )
            previous_broadcast = broadcast_id
        if previous_broadcast is None:
            raise TransformerGraphError(
                "DDP contract has no forward metadata broadcasts"
            )
        prefixed = [
            replace(event, dependencies=(*event.dependencies, previous_broadcast))
            if event.id in original_roots
            else event
            for event in prefixed
        ]

        previous_bucket: str | None = None
        for bucket, component in zip(gradient_buckets, bucket_components, strict=True):
            backward_component = [
                event
                for event in prefixed
                if event.metadata.get("pipeline_phase") == "backward"
                and (
                    component is None
                    or event.metadata.get("semantic_component") == component
                )
            ]
            if not backward_component:
                raise TransformerGraphError(
                    "gradient-bucket last-ready component has no backward source events"
                )
            component_ids = {event.id for event in backward_component}
            depended = {
                dependency
                for event in backward_component
                for dependency in event.dependencies
                if dependency in component_ids
            }
            component_exits = sorted(component_ids - depended)
            if not component_exits:
                raise TransformerGraphError(
                    "gradient-bucket last-ready component has no DAG exit"
                )
            bucket_id = f"rank{target_rank}::dp-gradient-bucket-{bucket['index']}"
            dependencies = tuple(
                dict.fromkeys(
                    (*component_exits, *((previous_bucket,) if previous_bucket else ()))
                )
            )
            dp_events.append(
                TraceEvent(
                    id=bucket_id,
                    name="framework-derived-ddp-gradient-all-reduce",
                    kind="collective",
                    duration_us=0.0,
                    rank=target_rank,
                    device=target_rank,
                    dependencies=dependencies,
                    collective="all_reduce",
                    message_bytes=bucket["payload_bytes"],
                    group_role="dp",
                    group_size=target_dp,
                    metadata={
                        "schema": SCHEMA,
                        "pipeline_phase": "backward",
                        "semantic_component": component,
                        "target_rank": target_rank,
                        "target_tp_group": tp_group,
                        "target_dp_group": dp_group,
                        "parameter_gradient_ready": bucket["last_ready_parameter"],
                        "bucket_parameter_names_sha256": bucket[
                            "parameter_names_sha256"
                        ],
                        "parameter_readiness_order_sha256": ddp_contract[
                            "gradient_readiness_order_sha256"
                        ],
                        "torch_version": ddp_contract["torch_version"],
                        "provenance": "explicit-pinned-ddp-warm-reducer-framework-rule",
                        "timing_status": "unresolved-requires-post-admission-calibration",
                    },
                )
            )
            previous_bucket = bucket_id
        optimizer_ids = {
            event.id
            for event in prefixed
            if event.metadata.get("pipeline_phase") == "optimizer"
        }
        optimizer_roots = {
            event.id
            for event in prefixed
            if event.id in optimizer_ids
            and not any(
                dependency in optimizer_ids for dependency in event.dependencies
            )
        }
        if not optimizer_roots:
            raise TransformerGraphError("source has no optimizer DAG root")
        if previous_bucket is None:
            raise TransformerGraphError("DDP contract has no gradient bucket")
        prefixed = [
            replace(event, dependencies=(*event.dependencies, previous_bucket))
            if event.id in optimizer_roots
            else event
            for event in prefixed
        ]
        generated.extend([*prefixed, *dp_events])
    source_sha256 = _canonical_sha256(source.to_dict())
    report = {
        "schema": SCHEMA,
        "status": "generated-semantic-candidate-awaiting-heldout-validation",
        "target_id": target_id,
        "source_sha256": source_sha256,
        "target_event_count": len(generated),
        "dp_collective_count": (
            target_world_size * ddp_contract["per_rank_operation_count"]
        ),
        "target_world_size": target_world_size,
        "target_tp": target_tp,
        "target_dp": target_dp,
        "dp_gradient_bucket_payload_bytes_per_rank": [
            bucket["payload_bytes"] for bucket in gradient_buckets
        ],
        "dp_forward_metadata_payload_bytes_per_rank": [
            broadcast["payload_bytes"] for broadcast in forward_broadcasts
        ],
        "last_gradient_ready_parameters": [
            bucket["last_ready_parameter"] for bucket in gradient_buckets
        ],
        "last_gradient_ready_components": bucket_components,
        "framework_rule_sha256": plan["framework_rule_sha256"],
        "target_training_executed": False,
        "timing_claim": "none",
        "kernel_code_claim": "none",
        "claim": "generated-framework-semantic-graph-no-physical-validity-claim",
    }
    candidate = WorkloadTrace(
        events=tuple(generated),
        source={
            "kind": "framework-semantic-counterfactual",
            "target": "h100",
            "framework": "megatron-core",
            "candidate_training_executed": False,
            "source_sha256": source_sha256,
        },
        metadata={
            "framework_measurement": measurement,
            "transformer_rule_plan": plan,
            "transformer_semantic_graph": report,
            "provenance": {
                "observed": "source-tp2-component-dependency-and-readiness-structure",
                "transformed": "dp-replication-groups-warm-reducer-buckets-broadcasts-and-optimizer-dependency",
                "estimated": [],
                "unresolved": ["target-kernel-code", "target-timing"],
            },
        },
    )
    candidate.validate()
    return candidate, report


def compile_dp_candidate(
    matrix: dict[str, Any],
    source: WorkloadTrace,
    target_id: str,
    *,
    target_exists: bool,
) -> tuple[WorkloadTrace, dict[str, Any]]:
    """Compile a matrix-admitted fixed-TP DP semantic candidate."""

    if not target_id.startswith("DP-"):
        raise TransformerGraphError("DP compiler requires a frozen DP target")
    measurement = source.metadata.get("framework_measurement")
    if not isinstance(measurement, dict):
        raise TransformerGraphError("source lacks framework measurement")
    plan = build_rule_plan(matrix, measurement, target_id, target_exists=target_exists)
    return _compile_dp_candidate_from_plan(source, target_id, plan)


def _named_component_segments(
    template: list[TraceEvent],
    phase: str,
    component_root: str,
    relative_root: str,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    collective_index = 0
    for event in template:
        event_phase, component = _context(event)
        if event_phase != phase or not (
            component == component_root or component.startswith(f"{component_root}.")
        ):
            continue
        relative = relative_root + component[len(component_root) :]
        if event.kind == "collective":
            if event.group_role != "tp" or event.collective != "all_reduce":
                raise TransformerGraphError(
                    f"PP source component {component_root} contains unsupported "
                    f"collective {event.id!r}"
                )
            segments.append(
                {
                    "operation": "tp_collective",
                    "semantic_component": relative,
                    "collective_role_index": collective_index,
                    "source_event_ids": [event.id],
                }
            )
            collective_index += 1
            continue
        operator = _operator(event)
        signature = {
            "component": relative,
            "kind": event.kind,
            "collective": event.collective,
            "operator": operator.get("name") if isinstance(operator, dict) else None,
        }
        if (
            segments
            and segments[-1].get("operation") == "compute_segment"
            and segments[-1]["signature"] == signature
        ):
            segments[-1]["source_event_count"] += 1
            segments[-1]["source_event_ids"].append(event.id)
        else:
            segments.append(
                {
                    "operation": "compute_segment",
                    "signature": signature,
                    "source_event_count": 1,
                    "source_event_ids": [event.id],
                }
            )
    if not segments:
        raise TransformerGraphError(
            f"source lacks {phase} segments for component {component_root}"
        )
    return segments


def _component_segments(
    template: list[TraceEvent], phase: str, layer: int
) -> list[dict[str, Any]]:
    return _named_component_segments(template, phase, f"layers.{layer}", "layer")


def _pp_schedule(stage: int, microbatches: int) -> list[tuple[str, int]]:
    if stage == 0:
        operations: list[tuple[str, int]] = [("forward", 0)]
        for microbatch in range(1, microbatches):
            operations.extend([("forward", microbatch), ("backward", microbatch - 1)])
        operations.append(("backward", microbatches - 1))
        return operations
    return [
        operation
        for microbatch in range(microbatches)
        for operation in (("forward", microbatch), ("backward", microbatch))
    ]


def compile_pp_candidate(
    matrix: dict[str, Any],
    source: WorkloadTrace,
    target_id: str,
    *,
    target_exists: bool,
) -> tuple[WorkloadTrace, dict[str, Any]]:
    if not target_id.startswith("PP-"):
        raise TransformerGraphError("PP compiler requires a frozen PP target")
    measurement = source.metadata.get("framework_measurement")
    if not isinstance(measurement, dict):
        raise TransformerGraphError("source lacks framework measurement")
    plan = build_rule_plan(matrix, measurement, target_id, target_exists=target_exists)
    template = _symmetric_source(source, measurement)
    rule = plan["framework_rule"]
    microbatches = rule["microbatches"]
    segment_profiles: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for phase in ("forward", "backward"):
        left = _component_segments(template, phase, 0)
        right = _component_segments(template, phase, 1)
        terminal = _named_component_segments(
            template, phase, "final_layernorm", "final_layernorm"
        )
        loss = _named_component_segments(
            template, phase, "terminal_loss", "terminal_loss"
        )
        segment_profiles[(phase, 0)] = left
        segment_profiles[(phase, 1)] = (
            [*right, *terminal, *loss]
            if phase == "forward"
            else [*loss, *terminal, *right]
        )
    for (phase, stage), operations in sorted(segment_profiles.items()):
        observed_tp = sum(
            operation.get("operation") == "tp_collective" for operation in operations
        )
        if observed_tp != 2:
            raise TransformerGraphError(
                f"PP source {phase} stage {stage} requires exactly two "
                f"component-bound TP collectives, observed {observed_tp}"
            )
    manifest = measurement["parameter_manifest"]
    parameters = manifest["parameters"]
    stage_parameters: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for parameter in parameters:
        name = parameter["name"]
        if name.startswith("layers.0."):
            stage_parameters[0].append(parameter)
        elif name.startswith("layers.1.") or name.startswith("final_layernorm."):
            stage_parameters[1].append(parameter)
        else:
            raise TransformerGraphError(
                f"parameter {name!r} has no frozen PP stage placement"
            )
    if any(not values for values in stage_parameters.values()):
        raise TransformerGraphError("each PP stage requires at least one parameter")

    generated: list[TraceEvent] = []
    phase_entry: dict[tuple[int, int, str, int], str] = {}
    phase_exit: dict[tuple[int, int, str, int], str] = {}
    send_forward: dict[tuple[int, int], str] = {}
    recv_forward: dict[tuple[int, int], str] = {}
    send_backward: dict[tuple[int, int], str] = {}
    recv_backward: dict[tuple[int, int], str] = {}
    for rank in range(4):
        stage = rank // 2
        lane = rank % 2
        previous: str | None = None
        for phase, microbatch in _pp_schedule(stage, microbatches):
            if phase == "forward" and stage == 1:
                recv_id = f"rank{rank}::mb{microbatch}::forward-recv"
                recv_forward[(lane, microbatch)] = recv_id
                generated.append(
                    TraceEvent(
                        id=recv_id,
                        name="framework-derived-pipeline-recv-forward",
                        kind="collective",
                        duration_us=0.0,
                        rank=rank,
                        device=rank,
                        dependencies=() if previous is None else (previous,),
                        collective="recv",
                        message_bytes=rule["p2p"]["payload_bytes"],
                        group_role="pp",
                        group_size=2,
                        metadata={
                            "pipeline_phase": phase,
                            "pipeline_stage": stage,
                            "pipeline_microbatch": microbatch,
                            "p2p_source_rank": lane,
                            "p2p_destination_rank": rank,
                            "provenance": "explicit-megatron-1f1b-rule",
                        },
                    )
                )
                previous = recv_id
            if phase == "backward" and stage == 0:
                recv_id = f"rank{rank}::mb{microbatch}::backward-recv"
                recv_backward[(lane, microbatch)] = recv_id
                generated.append(
                    TraceEvent(
                        id=recv_id,
                        name="framework-derived-pipeline-recv-backward",
                        kind="collective",
                        duration_us=0.0,
                        rank=rank,
                        device=rank,
                        dependencies=() if previous is None else (previous,),
                        collective="recv",
                        message_bytes=rule["p2p"]["payload_bytes"],
                        group_role="pp",
                        group_size=2,
                        metadata={
                            "pipeline_phase": phase,
                            "pipeline_stage": stage,
                            "pipeline_microbatch": microbatch,
                            "p2p_source_rank": lane + 2,
                            "p2p_destination_rank": rank,
                            "provenance": "explicit-megatron-1f1b-rule",
                        },
                    )
                )
                previous = recv_id
            segments = segment_profiles[(phase, stage)]
            phase_key = (rank, stage, phase, microbatch)
            for index, segment in enumerate(segments):
                if segment.get("operation") == "tp_collective":
                    collective_index = segment["collective_role_index"]
                    collective_id = (
                        f"rank{rank}::mb{microbatch}::{phase}::tp{collective_index}"
                    )
                    generated.append(
                        TraceEvent(
                            id=collective_id,
                            name="framework-derived-tp-all-reduce",
                            kind="collective",
                            duration_us=0.0,
                            rank=rank,
                            device=rank,
                            dependencies=() if previous is None else (previous,),
                            collective="all_reduce",
                            message_bytes=rule["tp_collectives"]["payload_bytes"],
                            group_role="tp",
                            group_size=2,
                            metadata={
                                "pipeline_phase": phase,
                                "pipeline_stage": stage,
                                "pipeline_microbatch": microbatch,
                                "semantic_component": segment["semantic_component"],
                                "target_tp_group": [
                                    stage * 2,
                                    stage * 2 + 1,
                                ],
                                "collective_role_index": collective_index,
                                "source_event_ids": segment["source_event_ids"],
                                "source_event_count": len(segment["source_event_ids"]),
                                "provenance": (
                                    "explicit-megatron-tp-layer-rule-at-"
                                    "observed-component-boundary"
                                ),
                            },
                        )
                    )
                    previous = collective_id
                    continue
                event_id = f"rank{rank}::mb{microbatch}::{phase}::segment{index}"
                generated.append(
                    TraceEvent(
                        id=event_id,
                        name=f"semantic-{segment['signature']['component']}",
                        kind="compute",
                        duration_us=0.0,
                        rank=rank,
                        device=rank,
                        dependencies=() if previous is None else (previous,),
                        metadata={
                            "schema": SCHEMA,
                            "pipeline_phase": phase,
                            "pipeline_stage": stage,
                            "pipeline_microbatch": microbatch,
                            "semantic_component": segment["signature"]["component"],
                            "framework_operator": segment["signature"]["operator"],
                            "source_kind": segment["signature"]["kind"],
                            "source_event_count": segment["source_event_count"],
                            "source_event_ids": segment["source_event_ids"],
                            "provenance": "source-component-segment-not-target-runtime-observation",
                            "timing_status": "unresolved-requires-post-admission-calibration",
                        },
                    )
                )
                if index == 0:
                    phase_entry[phase_key] = event_id
                previous = event_id
            if phase == "forward" and stage == 0:
                send_id = f"rank{rank}::mb{microbatch}::forward-send"
                send_forward[(lane, microbatch)] = send_id
                generated.append(
                    TraceEvent(
                        id=send_id,
                        name="framework-derived-pipeline-send-forward",
                        kind="collective",
                        duration_us=0.0,
                        rank=rank,
                        device=rank,
                        dependencies=(previous,),
                        collective="send",
                        message_bytes=rule["p2p"]["payload_bytes"],
                        group_role="pp",
                        group_size=2,
                        metadata={
                            "pipeline_phase": phase,
                            "pipeline_stage": stage,
                            "pipeline_microbatch": microbatch,
                            "p2p_source_rank": rank,
                            "p2p_destination_rank": rank + 2,
                            "provenance": "explicit-megatron-1f1b-rule",
                        },
                    )
                )
                previous = send_id
            if phase == "backward" and stage == 1:
                send_id = f"rank{rank}::mb{microbatch}::backward-send"
                send_backward[(lane, microbatch)] = send_id
                generated.append(
                    TraceEvent(
                        id=send_id,
                        name="framework-derived-pipeline-send-backward",
                        kind="collective",
                        duration_us=0.0,
                        rank=rank,
                        device=rank,
                        dependencies=(previous,),
                        collective="send",
                        message_bytes=rule["p2p"]["payload_bytes"],
                        group_role="pp",
                        group_size=2,
                        metadata={
                            "pipeline_phase": phase,
                            "pipeline_stage": stage,
                            "pipeline_microbatch": microbatch,
                            "p2p_source_rank": rank,
                            "p2p_destination_rank": rank - 2,
                            "provenance": "explicit-megatron-1f1b-rule",
                        },
                    )
                )
                previous = send_id
            phase_exit[phase_key] = previous
        for parameter in stage_parameters[stage]:
            optimizer_id = f"rank{rank}::optimizer::{parameter['order']}"
            generated.append(
                TraceEvent(
                    id=optimizer_id,
                    name="semantic-adamw-parameter-update",
                    kind="compute",
                    duration_us=0.0,
                    rank=rank,
                    device=rank,
                    dependencies=(previous,),
                    metadata={
                        "pipeline_phase": "optimizer",
                        "pipeline_stage": stage,
                        "parameter_name": parameter["name"],
                        "parameter_numel": parameter["numel"],
                        "provenance": "parameter-manifest-and-optimizer-rule",
                        "timing_status": "unresolved-requires-post-admission-calibration",
                    },
                )
            )
            previous = optimizer_id

    replacements: dict[str, tuple[str, ...]] = {}
    # Megatron's non-interleaved 1F1B schedule issues the steady-state
    # send/receive transition as one grouped bidirectional exchange.  Model
    # the outgoing send as a child of its tensor producer.  The incoming recv
    # has no local tensor producer: its only producer is the matched remote
    # send added by the exact route-replacement pass below.  The following
    # compute region joins both logical operations.  The warmup and cooldown
    # boundaries remain one-way operations.
    #
    # This is a framework schedule rule.  It depends only on the stage, lane,
    # and microbatch order above; no target identifier or measured timing is
    # consulted.
    by_id = {event.id: event for event in generated}
    consumers: dict[str, list[str]] = {event.id: [] for event in generated}
    for event in generated:
        for dependency in event.dependencies:
            consumers[dependency].append(event.id)
    compound_exchanges: list[tuple[str, str]] = []
    for lane in (0, 1):
        for microbatch in range(1, microbatches):
            compound_exchanges.extend(
                [
                    (
                        send_forward[(lane, microbatch)],
                        recv_backward[(lane, microbatch - 1)],
                    ),
                    (
                        send_backward[(lane, microbatch - 1)],
                        recv_forward[(lane, microbatch)],
                    ),
                ]
            )
    exchange_by_event: dict[str, tuple[str, str]] = {}
    joins: dict[str, str] = {}
    for send_id, recv_id in compound_exchanges:
        send = by_id[send_id]
        recv = by_id[recv_id]
        if recv.dependencies != (send_id,):
            raise TransformerGraphError(
                "steady-state PP receive is not immediately after its paired send"
            )
        recv_consumers = consumers[recv_id]
        if len(recv_consumers) != 1:
            raise TransformerGraphError(
                "steady-state PP receive must have exactly one local consumer"
            )
        exchange_by_event[send_id] = (send_id, recv_id)
        exchange_by_event[recv_id] = (send_id, recv_id)
        joins[recv_consumers[0]] = send_id

    # Grouped SendRecv calls are separate NCCL launches on one native stream.
    # Send and recv within one group are logically parallel, but the same
    # direction/route in the next group follows its prior launch.  Build that
    # ordering from rank, direction, phase, route, and microbatch only.
    p2p_by_sequence: dict[tuple[int, str, str, int, int, int], str] = {}
    for event in generated:
        if event.group_role != "pp" or event.collective not in {"send", "recv"}:
            continue
        phase = event.metadata.get("pipeline_phase")
        microbatch = event.metadata.get("pipeline_microbatch")
        p2p_source = event.metadata.get("p2p_source_rank")
        destination = event.metadata.get("p2p_destination_rank")
        if (
            not isinstance(phase, str)
            or isinstance(microbatch, bool)
            or not isinstance(microbatch, int)
            or isinstance(p2p_source, bool)
            or not isinstance(p2p_source, int)
            or isinstance(destination, bool)
            or not isinstance(destination, int)
        ):
            raise TransformerGraphError(
                "PP native-stream ordering requires exact phase, microbatch, and route"
            )
        key = (
            event.rank,
            event.collective,
            phase,
            p2p_source,
            destination,
            microbatch,
        )
        if key in p2p_by_sequence:
            raise TransformerGraphError("PP native-stream ordering is not one-to-one")
        p2p_by_sequence[key] = event.id

    def previous_same_direction(event: TraceEvent) -> str | None:
        phase = event.metadata["pipeline_phase"]
        microbatch = event.metadata["pipeline_microbatch"]
        source = event.metadata["p2p_source_rank"]
        destination = event.metadata["p2p_destination_rank"]
        if microbatch == 0:
            return None
        return p2p_by_sequence.get(
            (
                event.rank,
                event.collective,
                phase,
                source,
                destination,
                microbatch - 1,
            )
        )

    rewritten: list[TraceEvent] = []
    for event in generated:
        pair = exchange_by_event.get(event.id)
        metadata = event.metadata
        dependencies = event.dependencies
        if event.group_role == "pp" and event.collective in {"send", "recv"}:
            stream_predecessor = previous_same_direction(event)
            if event.collective == "recv":
                # No incoming tensor is produced by preceding local compute,
                # including at warmup/cooldown boundaries.  Preserve only the
                # prior receive on this native stream; the exact matched remote
                # send is added below and then represented by Chakra P2P pairing.
                dependencies = (
                    () if stream_predecessor is None else (stream_predecessor,)
                )
            elif stream_predecessor is not None:
                # Outgoing sends retain their tensor producer and the native
                # stream order between successive sends on the same route.
                dependencies = tuple(
                    dict.fromkeys((*dependencies, stream_predecessor))
                )
            metadata = {
                **metadata,
                "pipeline_native_stream_predecessor": stream_predecessor,
            }
            if pair is not None:
                send_id, recv_id = pair
                metadata = {
                    **metadata,
                    "pipeline_compound_exchange": {
                        "schema": "megatron-1f1b-compound-p2p-exchange-v1",
                        "send_event_id": send_id,
                        "recv_event_id": recv_id,
                        "dependency_semantics": (
                            "send-local-producer+recv-remote-producer+local-join"
                        ),
                    },
                }
        join_send = joins.get(event.id)
        if join_send is not None:
            dependencies = tuple(dict.fromkeys((*dependencies, join_send)))
        rewritten.append(replace(event, dependencies=dependencies, metadata=metadata))
    generated = rewritten

    for lane in (0, 1):
        for microbatch in range(microbatches):
            replacements[recv_forward[(lane, microbatch)]] = (
                send_forward[(lane, microbatch)],
            )
            replacements[recv_backward[(lane, microbatch)]] = (
                send_backward[(lane, microbatch)],
            )
    generated = [
        replace(
            event,
            dependencies=tuple(
                dict.fromkeys((*event.dependencies, *replacements.get(event.id, ())))
            ),
        )
        for event in generated
    ]
    source_sha256 = _canonical_sha256(source.to_dict())
    p2p_events = [event for event in generated if event.group_role == "pp"]
    report = {
        "schema": SCHEMA,
        "status": "generated-semantic-candidate-awaiting-heldout-validation",
        "target_id": target_id,
        "source_sha256": source_sha256,
        "target_event_count": len(generated),
        "microbatches": microbatches,
        "p2p_event_count": len(p2p_events),
        "compound_p2p_exchange_count": len(compound_exchanges),
        "p2p_payload_bytes": rule["p2p"]["payload_bytes"],
        "framework_rule_sha256": plan["framework_rule_sha256"],
        "stage_profile_rule": (
            "observed-global-layer-0-to-stage-0-and-global-layer-1-to-stage-1"
        ),
        "target_training_executed": False,
        "timing_claim": "none",
        "kernel_code_claim": "none",
        "claim": "generated-framework-semantic-graph-no-physical-validity-claim",
    }
    candidate = WorkloadTrace(
        events=tuple(generated),
        source={
            "kind": "framework-semantic-counterfactual",
            "target": "h100",
            "framework": "megatron-core",
            "candidate_training_executed": False,
            "source_sha256": source_sha256,
        },
        metadata={
            "framework_measurement": measurement,
            "transformer_rule_plan": plan,
            "transformer_semantic_graph": report,
            "provenance": {
                "observed": "source-tp2-layer-component-segments",
                "transformed": "pp-placement-1f1b-p2p-tp-and-stage-optimizer",
                "estimated": [],
                "unresolved": ["target-kernel-code", "target-timing"],
            },
        },
    )
    candidate.validate()
    return candidate, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--heldout-guard", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = (args.candidate, args.report, args.freeze)
    if args.heldout_guard.exists():
        raise TransformerGraphError("held-out target exists before prediction freeze")
    if any(path.exists() for path in outputs):
        raise TransformerGraphError("prediction output already exists")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    source = WorkloadTrace.load(args.source)
    if args.target_id.startswith("TP-"):
        compiler = compile_tp_candidate
    elif args.target_id.startswith("DP-"):
        compiler = compile_dp_candidate
    elif args.target_id.startswith("PP-"):
        compiler = compile_pp_candidate
    else:
        raise TransformerGraphError(
            "no event-graph compiler exists for the requested frozen target"
        )
    candidate, report = compiler(matrix, source, args.target_id, target_exists=False)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    candidate.dump(args.candidate)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze = {
        "schema": "scaletether-transformer-pre-heldout-freeze-v1",
        "status": "frozen-before-heldout-execution",
        "target_id": args.target_id,
        "candidate_training_executed": False,
        "heldout_guard_absent": not args.heldout_guard.exists(),
        "matrix_sha256": _file_sha256(args.matrix),
        "source_sha256": _file_sha256(args.source),
        "candidate_sha256": _file_sha256(args.candidate),
        "report_sha256": _file_sha256(args.report),
    }
    args.freeze.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
