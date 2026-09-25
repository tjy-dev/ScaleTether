"""Exact post-admission structural comparison for frozen Transformer targets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Hashable

from .schema import TraceEvent, WorkloadTrace
from .megatron_transformer_graph import (
    TransformerGraphError,
    _semanticize_rank,
    semanticize_pp_rank,
)
from .megatron_transformer_target_validation import (
    physical_pp_logical_collectives,
    validate_physical_pp_target,
    validate_physical_tp_dp_target,
)


SCHEMA = "scaletether-transformer-generated-physical-comparison-v4"
_COMPONENT = re.compile(
    r"^megatron_transformer_(forward|backward)_component:"
    r"(layers\.[01](?:\..+)?|final_layernorm|terminal_loss)$"
)


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter_sha256(counter: Counter[Any]) -> str:
    """Hash a multiset without relying on heterogeneous tuple ordering."""
    rows = [
        {"key": list(key) if isinstance(key, tuple) else key, "count": count}
        for key, count in counter.items()
    ]
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    return _sha256(rows)


def _counter_delta(
    generated: Counter[Any], physical: Counter[Any], *, limit: int = 64
) -> dict[str, Any]:
    """Return a deterministic, bounded explanation of an exact multiset mismatch."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("counter delta limit must be a positive integer")

    def rows(counter: Counter[Any]) -> list[dict[str, Any]]:
        result = [
            {
                "key": list(key) if isinstance(key, tuple) else key,
                "count": count,
            }
            for key, count in counter.items()
            if count > 0
        ]
        result.sort(
            key=lambda row: json.dumps(
                row, sort_keys=True, separators=(",", ":"), default=str
            )
        )
        return result

    generated_only = rows(generated - physical)
    physical_only = rows(physical - generated)
    return {
        "generated_only_total": sum(row["count"] for row in generated_only),
        "physical_only_total": sum(row["count"] for row in physical_only),
        "generated_only": generated_only[:limit],
        "physical_only": physical_only[:limit],
        "limit_per_side": limit,
        "truncated": len(generated_only) > limit or len(physical_only) > limit,
    }


def _operator_from_actual(event: TraceEvent) -> str | None:
    launch = event.metadata.get("kernel_launch_payload")
    operator = launch.get("framework_operator") if isinstance(launch, dict) else None
    name = operator.get("name") if isinstance(operator, dict) else None
    return name if isinstance(name, str) else None


def _actual_component(event: TraceEvent, *, pp: bool) -> tuple[str, str] | None:
    semantic = event.metadata.get("framework_semantic_context")
    if isinstance(semantic, dict):
        phase = semantic.get("phase")
        component = semantic.get("component")
        if isinstance(phase, str) and isinstance(component, str):
            if pp and component.startswith("layers."):
                stage = event.rank // 2
                prefix = f"layers.{stage}"
                if not (component == prefix or component.startswith(prefix + ".")):
                    return phase, "foreign-stage"
                component = "layer" + component[len(prefix) :]
            return phase, component
    marker = event.metadata.get("framework_phase_marker")
    name = marker.get("name") if isinstance(marker, dict) else None
    if not isinstance(name, str):
        return None
    match = _COMPONENT.fullmatch(name)
    if not match:
        return None
    phase, component = match.groups()
    if pp and component.startswith("layers."):
        stage = event.rank // 2
        prefix = f"layers.{stage}"
        if not (component == prefix or component.startswith(prefix + ".")):
            return phase, "foreign-stage"
        component = "layer" + component[len(prefix) :]
    return phase, component


def _semanticize_actual_tp_dp(trace: WorkloadTrace) -> WorkloadTrace:
    # A completely direct-marked fixture needs no gap attribution. Real captures
    # enter the frozen two-layer rule whenever at least one compute event is
    # unmarked.
    unmarked_compute = any(
        event.kind != "collective"
        and _actual_component(event, pp=False) is None
        for event in trace.events
    )
    if not unmarked_compute:
        return trace
    by_rank = {
        rank: [event for event in trace.events if event.rank == rank]
        for rank in sorted({event.rank for event in trace.events})
    }
    replacements = {
        event.id: event
        for events in by_rank.values()
        for event in _semanticize_rank(events)
    }
    return WorkloadTrace(
        tuple(replacements[event.id] for event in trace.events),
        trace.source,
        trace.metadata,
    )


def _semanticize_actual_pp(trace: WorkloadTrace) -> WorkloadTrace:
    # The pipeline model declares and emits exact leaf-component markers for
    # every microbatch.  Parent ``layer``/``attention``/``mlp`` segments in the
    # generated graph are synthetic hierarchy, not additional runtime facts.
    # Compare the directly observed leaves and keep PP schedule/collective
    # checks separate instead of attempting gap attribution across fused P2P
    # kernels.
    return trace


def _is_pp_leaf_component(component: str) -> bool:
    return component in {
        "layer.self_attention.linear_qkv",
        "layer.self_attention.linear_proj",
        "layer.mlp.linear_fc1",
        "layer.mlp.linear_fc2",
        "final_layernorm",
    }


def _semantic_compute_counter(
    trace: WorkloadTrace, *, candidate: bool, pp: bool
) -> Counter[tuple[int, str, str, str, str | None]]:
    counter: Counter[tuple[int, str, str, str, str | None]] = Counter()
    for event in trace.events:
        if event.kind == "collective":
            continue
        if candidate:
            phase = event.metadata.get("pipeline_phase")
            component = event.metadata.get("semantic_component")
            if not isinstance(phase, str) or not isinstance(component, str):
                continue
            if component.startswith("step.") or phase == "optimizer":
                continue
            operator = event.metadata.get("framework_operator")
            if isinstance(operator, dict):
                operator = operator.get("name")
            if operator is not None and not isinstance(operator, str):
                operator = None
            kind = event.metadata.get("source_kind", event.kind)
            count = event.metadata.get("source_event_count", 1)
            if not isinstance(kind, str) or not isinstance(count, int) or count <= 0:
                raise ValueError("candidate semantic segment metadata is malformed")
        else:
            context = _actual_component(event, pp=pp)
            if context is None:
                continue
            phase, component = context
            if component.startswith("step.") or phase == "optimizer":
                continue
            operator = _operator_from_actual(event)
            kind = event.kind
            count = 1
        counter[(event.rank, phase, component, kind, operator)] += count
    return counter


def _semantic_component_counter(
    trace: WorkloadTrace, *, candidate: bool, pp: bool
) -> Counter[tuple[int, str, str]]:
    """Return framework-semantic component presence, not target kernel identity.

    A TP change may legally alter fusion and the number or direction labels of
    ATen/CUDA kernels inside one Megatron component.  The counterfactual graph
    explicitly makes no target-kernel-code claim, so admission must not require
    exact operator multiplicity or source-kind multiplicity.  PyTorch DDP can
    insert memory/runtime events inside an otherwise identical Megatron
    component.  It does require every declared component to be present on
    every expected rank; component order and all collectives remain separate
    gates.
    """

    present: set[tuple[int, str, str]] = set()
    for event in trace.events:
        if event.kind == "collective":
            continue
        if candidate:
            phase = event.metadata.get("pipeline_phase")
            component = event.metadata.get("semantic_component")
        else:
            context = _actual_component(event, pp=pp)
            if context is None:
                continue
            phase, component = context
        if not all(isinstance(item, str) for item in (phase, component)):
            continue
        if component.startswith("step.") or phase == "optimizer":
            continue
        if pp and not _is_pp_leaf_component(component):
            continue
        present.add((event.rank, phase, component))
    return Counter({item: 1 for item in present})


def _semantic_component_order(
    trace: WorkloadTrace, *, candidate: bool, pp: bool
) -> dict[int, list[tuple[str, str]]]:
    """Return each rank's consecutive-deduplicated component sequence."""

    by_rank: dict[int, list[TraceEvent]] = {}
    for event in trace.events:
        if event.kind != "collective":
            by_rank.setdefault(event.rank, []).append(event)
    result: dict[int, list[tuple[str, str]]] = {}
    for rank, events in sorted(by_rank.items()):
        if not candidate and all(event.observed_start_us is not None for event in events):
            events = sorted(
                events,
                key=lambda event: (float(event.observed_start_us), event.id),
            )
        sequence: list[tuple[str, str]] = []
        for event in events:
            if candidate:
                phase = event.metadata.get("pipeline_phase")
                component = event.metadata.get("semantic_component")
            else:
                context = _actual_component(event, pp=pp)
                if context is None:
                    continue
                phase, component = context
            if not isinstance(phase, str) or not isinstance(component, str):
                continue
            if component.startswith("step.") or phase == "optimizer":
                continue
            if pp and not _is_pp_leaf_component(component):
                continue
            current = (phase, component)
            if not sequence or sequence[-1] != current:
                sequence.append(current)
        result[rank] = sequence
    return result


def _members(event: TraceEvent, *, candidate: bool) -> tuple[int, ...]:
    if event.group_role == "pp":
        source = event.metadata.get("p2p_source_rank")
        destination = event.metadata.get("p2p_destination_rank")
        return (source, destination) if isinstance(source, int) and isinstance(destination, int) else ()
    if candidate:
        key = "target_dp_group" if event.group_role == "dp" else "target_tp_group"
        raw = event.metadata.get(key)
    else:
        raw = event.metadata.get("process_group_ranks")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return ()
    return tuple(sorted(raw)) if isinstance(raw, list) and all(isinstance(item, int) for item in raw) else ()


def _collective_counter(
    trace: WorkloadTrace, *, candidate: bool, pp: bool = False
) -> Counter[tuple[int, str | None, str | None, int | None, tuple[int, ...]]]:
    counter: Counter[tuple[int, str | None, str | None, int | None, tuple[int, ...]]] = Counter()
    rank_metadata = trace.metadata.get("rank_metadata")
    if pp and not candidate and isinstance(rank_metadata, list):
        for operation in physical_pp_logical_collectives(trace.to_dict()):
            members = (
                (
                    operation["p2p_source_rank"],
                    operation["p2p_destination_rank"],
                )
                if operation["group_role"] == "pp"
                else tuple(sorted(operation["process_group_ranks"]))
            )
            counter[(
                operation["rank"],
                operation["group_role"],
                operation["collective"],
                operation["message_bytes"],
                members,
            )] += 1
        return counter
    for event in trace.events:
        if event.kind != "collective":
            continue
        counter[(
            event.rank,
            event.group_role,
            event.collective,
            event.message_bytes,
            _members(event, candidate=candidate),
        )] += 1
    return counter


def _pp_phase_order(trace: WorkloadTrace, rank: int) -> list[str]:
    instances: dict[str, tuple[float, str]] = {}
    for event in trace.events:
        if event.rank != rank or event.observed_start_us is None:
            continue
        marker = event.metadata.get("framework_phase_marker")
        if not isinstance(marker, dict):
            continue
        name = marker.get("name")
        instance = marker.get("instance_id")
        phase = {
            "megatron_transformer_forward": "forward",
            "megatron_transformer_backward": "backward",
        }.get(name)
        if phase is None or not isinstance(instance, str) or not instance:
            continue
        current = instances.get(instance)
        if current is not None and current[1] != phase:
            raise ValueError("one framework marker instance has conflicting PP phases")
        if current is None or event.observed_start_us < current[0]:
            instances[instance] = (event.observed_start_us, phase)
    return [phase for _, phase in sorted(instances.values())]


def _expected_pp_phase_order(stage: int, microbatches: int) -> list[str]:
    if stage == 0:
        result = ["forward"]
        for microbatch in range(1, microbatches):
            result.extend(["forward", "backward"])
        result.append("backward")
        return result
    return [phase for _ in range(microbatches) for phase in ("forward", "backward")]


def _all_causal_ancestors(
    by_id: dict[str, TraceEvent],
) -> dict[str, frozenset[str]]:
    """Compute transitive ancestors with deterministic Kahn topological order."""

    children: dict[str, list[str]] = {identifier: [] for identifier in by_id}
    indegree: dict[str, int] = {}
    for identifier, event in by_id.items():
        unknown = set(event.dependencies) - set(by_id)
        if unknown:
            raise ValueError(
                f"event {identifier!r} has unresolved dependencies "
                f"{sorted(unknown)[:8]}"
            )
        indegree[identifier] = len(event.dependencies)
        for dependency in event.dependencies:
            children[dependency].append(identifier)
    ready = sorted(identifier for identifier, degree in indegree.items() if degree == 0)
    ancestors: dict[str, frozenset[str]] = {}
    while ready:
        identifier = ready.pop(0)
        values: set[str] = set()
        for dependency in by_id[identifier].dependencies:
            values.add(dependency)
            values.update(ancestors[dependency])
        ancestors[identifier] = frozenset(values)
        for child in sorted(children[identifier]):
            indegree[child] -= 1
            if indegree[child] == 0:
                # Workloads are small enough that maintaining a sorted list is
                # clearer and fully deterministic.
                ready.append(child)
                ready.sort()
    if len(ancestors) != len(by_id):
        cyclic = sorted(identifier for identifier in by_id if identifier not in ancestors)
        raise ValueError(f"dependency cycle involving {cyclic[:8]}")
    return ancestors


def _pp_logical_operations_by_event(
    trace: WorkloadTrace,
) -> dict[str, list[dict[str, Any]]]:
    """Bind every reconciled PP framework call to its physical trace event."""

    operations = physical_pp_logical_collectives(trace.to_dict())
    by_rank_index = {
        (operation["rank"], operation["index"]): operation
        for operation in operations
    }
    result: dict[str, list[dict[str, Any]]] = {}
    metadata = trace.metadata.get("rank_metadata")
    if not isinstance(metadata, list):
        raise ValueError("PP dependency projection lacks rank metadata")
    for record in metadata:
        if not isinstance(record, dict) or not isinstance(record.get("rank"), int):
            raise ValueError("PP dependency projection has malformed rank metadata")
        rank = record["rank"]
        binding = record.get("direct_native_nccl_binding")
        if not isinstance(binding, dict):
            raise ValueError("PP dependency projection lacks native binding")
        mappings: dict[int, str] = {}
        for key in ("bindings", "general_fallback_bindings"):
            rows = binding.get(key, [])
            if not isinstance(rows, list):
                raise ValueError("PP dependency binding rows are malformed")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("PP dependency binding row is malformed")
                event_id = row.get("event_id")
                sequences = row.get("framework_collective_sequences")
                if not isinstance(event_id, str) or not isinstance(sequences, list):
                    continue
                merged_id = event_id if event_id.startswith("rank-") else f"rank-{rank}:{event_id}"
                for sequence in sequences:
                    if isinstance(sequence, bool) or not isinstance(sequence, int):
                        raise ValueError("PP dependency binding sequence is malformed")
                    if sequence in mappings and mappings[sequence] != merged_id:
                        raise ValueError("PP logical call maps to multiple physical events")
                    mappings[sequence] = merged_id
        # Ordinary one-kernel collectives carry their framework sequence on the
        # event itself.  This also covers captures whose exact binding did not
        # need a fallback row.
        for event in trace.events:
            if event.rank != rank or event.kind != "collective":
                continue
            sequence = event.metadata.get("collective_sequence")
            if isinstance(sequence, int) and not isinstance(sequence, bool):
                mappings.setdefault(sequence, event.id)
        expected = {index for observed_rank, index in by_rank_index if observed_rank == rank}
        if set(mappings) != expected:
            raise ValueError(
                f"rank {rank} PP logical/event binding is incomplete: "
                f"missing={sorted(expected - set(mappings))[:8]} "
                f"extra={sorted(set(mappings) - expected)[:8]}"
            )
        for sequence in sorted(expected):
            result.setdefault(mappings[sequence], []).append(by_rank_index[(rank, sequence)])
    return result


def _collective_dependency_label(
    event: TraceEvent | None,
    operation: dict[str, Any] | None,
    *,
    candidate: bool,
) -> tuple[Hashable, ...]:
    if operation is not None:
        rank = operation["rank"]
        role = operation["group_role"]
        collective_name = operation["collective"]
        message_bytes = operation["message_bytes"]
        members = (
            (operation["p2p_source_rank"], operation["p2p_destination_rank"])
            if role == "pp"
            else tuple(sorted(operation["process_group_ranks"]))
        )
    elif event is not None:
        rank = event.rank
        role = event.group_role
        collective_name = event.collective
        message_bytes = event.message_bytes
        members = _members(event, candidate=candidate)
    else:  # pragma: no cover - internal misuse guard
        raise AssertionError("collective dependency label lacks an operation")
    return ("collective", rank, role, collective_name, message_bytes, members)


def _dependency_projection(
    trace: WorkloadTrace, *, candidate: bool, pp: bool
) -> dict[str, Any]:
    """Project a device DAG onto framework components and logical calls.

    Repeated equal labels are occurrence-indexed.  Consecutive device events
    inside one component marker form one semantic node.  For PP, independently
    reconciled framework calls split a fused physical SendRecv event into the
    parallel logical operations used by Chakra export.  Reachability otherwise
    comes only from explicit trace dependencies.  This projection therefore
    checks the schedule that the backend consumes, not merely call inventory.
    """

    by_id = {event.id: event for event in trace.events}
    if len(by_id) != len(trace.events):
        raise ValueError("dependency projection requires unique event ids")
    cache = _all_causal_ancestors(by_id)

    pp_operations = (
        _pp_logical_operations_by_event(trace)
        if pp
        and not candidate
        and any(event.kind == "collective" for event in trace.events)
        else {}
    )
    ranks = sorted({event.rank for event in trace.events})
    reports: dict[str, Any] = {}
    for rank in ranks:
        events = [event for event in trace.events if event.rank == rank]
        if not candidate and all(event.observed_start_us is not None for event in events):
            events.sort(key=lambda event: (float(event.observed_start_us), event.id))
        raw_nodes: list[dict[str, Any]] = []
        for event in events:
            logical = pp_operations.get(event.id)
            if logical is not None:
                for operation in logical:
                    raw_nodes.append(
                        {
                            "base": _collective_dependency_label(
                                None, operation, candidate=False
                            ),
                            "event_ids": [event.id],
                            "ledger_order": operation["index"],
                        }
                    )
                continue
            if event.kind == "collective":
                # In a fully reconciled PP capture every physical collective is
                # represented above.  Seeing another one is an unresolved
                # physical/logical boundary and must abstain.
                if pp and not candidate:
                    raise ValueError(
                        f"PP collective {event.id!r} lacks a logical binding"
                    )
                raw_nodes.append(
                    {
                        "base": _collective_dependency_label(
                            event, None, candidate=candidate
                        ),
                        "event_ids": [event.id],
                        "ledger_order": None,
                    }
                )
                continue
            if candidate:
                phase = event.metadata.get("pipeline_phase")
                component = event.metadata.get("semantic_component")
            else:
                context = _actual_component(event, pp=pp)
                if context is None:
                    continue
                phase, component = context
            if not isinstance(phase, str) or not isinstance(component, str):
                continue
            if component.startswith("step.") or phase == "optimizer":
                continue
            if pp and not _is_pp_leaf_component(component):
                continue
            base = ("component", rank, phase, component)
            if raw_nodes and raw_nodes[-1]["base"] == base:
                raw_nodes[-1]["event_ids"].append(event.id)
            else:
                raw_nodes.append(
                    {"base": base, "event_ids": [event.id], "ledger_order": None}
                )

        occurrences: Counter[tuple[Hashable, ...]] = Counter()
        nodes: list[dict[str, Any]] = []
        for raw in raw_nodes:
            base = raw["base"]
            occurrence = occurrences[base]
            occurrences[base] += 1
            nodes.append({**raw, "key": (*base, occurrence)})

        # Distinct operations in one compound exchange are parallel, so their
        # source-list order is not a backend semantic.  Canonicalize after
        # assigning repeated-label occurrences to compare the graph relation
        # rather than an incidental send/recv list order.
        nodes.sort(
            key=lambda node: json.dumps(
                node["key"], sort_keys=True, separators=(",", ":")
            )
        )

        reachability: list[list[int]] = []
        for destination_index, destination in enumerate(nodes):
            destination_ancestors = set().union(
                *(cache[identifier] for identifier in destination["event_ids"])
            )
            ancestors: list[int] = []
            for source_index, source in enumerate(nodes):
                if source_index == destination_index:
                    continue
                device_path = any(
                    identifier in destination_ancestors
                    for identifier in source["event_ids"]
                )
                if device_path:
                    ancestors.append(source_index)
            reachability.append(ancestors)
        reports[str(rank)] = {
            "nodes": [list(node["key"]) for node in nodes],
            "reachability": reachability,
        }
    return {
        "schema": "megatron-framework-semantic-dependency-projection-v1",
        "boundary": "component-occurrence-and-logical-collective-call",
        "relation": "existential-device-causal-path-with-parallel-fused-p2p",
        "ranks": reports,
    }


def _marker_stack_context(
    stack: Any, *, rank: int, pp: bool
) -> tuple[str, str | None]:
    if not isinstance(stack, list) or not stack or any(
        not isinstance(item, str) for item in stack
    ):
        raise ValueError("logical collective lacks an active framework marker stack")
    phase: str | None = None
    component: str | None = None
    for name in stack:
        match = _COMPONENT.fullmatch(name)
        if match:
            phase, component = match.groups()
        elif name == "megatron_transformer_forward":
            phase = "forward"
        elif name == "megatron_transformer_backward":
            phase = "backward"
    if phase is None:
        raise ValueError("logical collective marker stack has no forward/backward phase")
    if pp and isinstance(component, str) and component.startswith("layers."):
        stage = rank // 2
        prefix = f"layers.{stage}"
        if not (component == prefix or component.startswith(prefix + ".")):
            raise ValueError("logical collective marker stack names a foreign PP stage")
        component = "layer" + component[len(prefix) :]
    return phase, component


def _event_marker_stack(event: TraceEvent) -> list[str]:
    stack = event.metadata.get("framework_phase_marker_stack")
    names: list[str] = []
    if isinstance(stack, list):
        for row in stack:
            name = row.get("name") if isinstance(row, dict) else None
            if isinstance(name, str):
                names.append(name)
    marker = event.metadata.get("framework_phase_marker")
    name = marker.get("name") if isinstance(marker, dict) else None
    if isinstance(name, str) and name not in names:
        names.append(name)
    return names


def _framework_dependency_projection(
    trace: WorkloadTrace, *, candidate: bool, pp: bool
) -> dict[str, Any]:
    """Project exact collective dependencies at the framework-call boundary.

    The relation is the enclosing Transformer phase/component occurrence for
    every logical collective call, plus its exact role-local ordinal and P2P
    send-to-recv pairing.  This deliberately does not equate a fused or
    delayed Kineto NCCL activity with the Python framework call that owns it.
    """

    if candidate:
        # Generated graphs themselves must remain closed and acyclic even
        # though physical device-activity binding is outside this projection.
        _all_causal_ancestors({event.id: event for event in trace.events})
    rows: list[dict[str, Any]] = []
    if pp and not candidate and any(
        event.kind == "collective" for event in trace.events
    ):
        operations = physical_pp_logical_collectives(trace.to_dict())
        operations.sort(key=lambda row: (row["rank"], row["index"]))
        for operation in operations:
            if operation["group_role"] == "pp":
                source_stage = operation["p2p_source_rank"] // 2
                destination_stage = operation["p2p_destination_rank"] // 2
                if (source_stage, destination_stage) == (0, 1):
                    phase, component = "forward", None
                elif (source_stage, destination_stage) == (1, 0):
                    phase, component = "backward", None
                else:
                    raise ValueError("logical P2P route does not cross PP stages")
                stack = operation.get("framework_marker_stack")
                if stack not in (None, []):
                    observed_phase, observed_component = _marker_stack_context(
                        stack, rank=operation["rank"], pp=True
                    )
                    if observed_phase != phase or observed_component is not None:
                        raise ValueError(
                            "logical P2P marker context conflicts with its route"
                        )
            else:
                phase, component = _marker_stack_context(
                    operation.get("framework_marker_stack"),
                    rank=operation["rank"],
                    pp=True,
                )
            members = (
                [operation["p2p_source_rank"], operation["p2p_destination_rank"]]
                if operation["group_role"] == "pp"
                else sorted(operation["process_group_ranks"])
            )
            rows.append(
                {
                    "rank": operation["rank"],
                    "role": operation["group_role"],
                    "collective": operation["collective"],
                    "message_bytes": operation["message_bytes"],
                    "members": members,
                    "phase": phase,
                    "component": component,
                    "source_order": operation["index"],
                }
            )
    else:
        events = [event for event in trace.events if event.kind == "collective"]
        if not candidate and all(event.observed_start_us is not None for event in events):
            events.sort(
                key=lambda event: (event.rank, float(event.observed_start_us), event.id)
            )
        for source_order, event in enumerate(events):
            if candidate:
                phase = event.metadata.get("pipeline_phase")
                component = event.metadata.get("semantic_component")
                if not isinstance(phase, str):
                    raise ValueError(
                        f"candidate collective {event.id!r} lacks a pipeline phase"
                    )
                if not isinstance(component, str) or component.startswith("step."):
                    component = None
            else:
                phase, component = _marker_stack_context(
                    _event_marker_stack(event), rank=event.rank, pp=False
                )
            rows.append(
                {
                    "rank": event.rank,
                    "role": event.group_role,
                    "collective": event.collective,
                    "message_bytes": event.message_bytes,
                    "members": list(_members(event, candidate=candidate)),
                    "phase": phase,
                    "component": component,
                    "source_order": source_order,
                }
            )

    # Occurrence indices turn repeated calls into a bijective set without
    # relying on target kernel identity or wall-clock timing.
    occurrences: Counter[tuple[Any, ...]] = Counter()
    canonical: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["rank"], item["source_order"])):
        base = (
            row["rank"],
            row["role"],
            row["collective"],
            row["message_bytes"],
            tuple(row["members"]),
            row["phase"],
            row["component"],
        )
        occurrence = occurrences[base]
        occurrences[base] += 1
        canonical.append(
            {
                **{key: value for key, value in row.items() if key != "source_order"},
                "occurrence": occurrence,
            }
        )
    canonical.sort(
        key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))
    )
    p2p: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    for row in canonical:
        if row["role"] != "pp":
            continue
        key = (
            tuple(row["members"]),
            row["message_bytes"],
            row["phase"],
            row["occurrence"],
        )
        p2p.setdefault(key, {"send": [], "recv": []})[row["collective"]].append(
            row
        )
    malformed = [
        key
        for key, pair in p2p.items()
        if len(pair["send"]) != 1 or len(pair["recv"]) != 1
    ]
    if malformed:
        raise ValueError(f"framework dependency projection has unmatched P2P {malformed[:4]}")
    p2p_edges = [
        {
            "route": list(key[0]),
            "message_bytes": key[1],
            "phase": key[2],
            "occurrence": key[3],
        }
        for key in sorted(p2p)
    ]
    return {
        "schema": "megatron-framework-call-dependency-projection-v1",
        "boundary": "logical-framework-collective-call",
        "relations": [
            "enclosing-transformer-phase-component",
            "role-local-occurrence",
            "matched-send-to-recv",
        ],
        "calls": canonical,
        "p2p_edges": p2p_edges,
    }


def _pp_optimizer_parameters(
    candidate: WorkloadTrace, actual: WorkloadTrace
) -> tuple[set[tuple[int, str]], set[tuple[int, str]]]:
    generated = {
        (event.rank, event.metadata["parameter_name"])
        for event in candidate.events
        if event.metadata.get("pipeline_phase") == "optimizer"
        and isinstance(event.metadata.get("parameter_name"), str)
    }
    measurement = actual.metadata.get("framework_measurement", {})
    contract = measurement.get("pipeline_stage_evidence", {}) if isinstance(measurement, dict) else {}
    stages = contract.get("stages", []) if isinstance(contract, dict) else []
    physical: set[tuple[int, str]] = set()
    if isinstance(stages, list):
        for stage, row in enumerate(stages):
            manifest = row.get("parameter_manifest", {}) if isinstance(row, dict) else {}
            parameters = manifest.get("parameters", []) if isinstance(manifest, dict) else []
            for parameter in parameters if isinstance(parameters, list) else []:
                name = parameter.get("name") if isinstance(parameter, dict) else None
                if isinstance(name, str):
                    physical.add((stage * 2, name))
                    physical.add((stage * 2 + 1, name))
    return generated, physical


def _optimizer_ranks(trace: WorkloadTrace, *, candidate: bool) -> set[int]:
    ranks: set[int] = set()
    for event in trace.events:
        if event.kind == "collective":
            continue
        if candidate:
            phase = event.metadata.get("pipeline_phase")
            component = event.metadata.get("semantic_component")
        else:
            context = _actual_component(event, pp=False)
            if context is None:
                continue
            phase, component = context
        if phase == "optimizer" or component == "step.optimizer":
            ranks.add(event.rank)
    return ranks


def _optimizer_is_terminal(trace: WorkloadTrace, *, candidate: bool) -> bool:
    """Require every rank's optimizer region to follow its training regions."""

    by_rank: dict[int, list[tuple[str, str]]] = {}
    for event in trace.events:
        if event.kind == "collective":
            continue
        if candidate:
            phase = event.metadata.get("pipeline_phase")
            component = event.metadata.get("semantic_component")
            context = (phase, component)
        else:
            context = _actual_component(event, pp=False)
        if (
            context is not None
            and isinstance(context[0], str)
            and isinstance(context[1], str)
        ):
            by_rank.setdefault(event.rank, []).append(context)
    for contexts in by_rank.values():
        optimizer = [
            index
            for index, (phase, component) in enumerate(contexts)
            if phase == "optimizer" or component == "step.optimizer"
        ]
        training = [
            index
            for index, (phase, component) in enumerate(contexts)
            if phase in {"forward", "backward"}
            and not component.startswith("step.")
        ]
        if not optimizer or not training or min(optimizer) <= max(training):
            return False
        if any(
            phase != "optimizer" and component != "step.optimizer"
            for phase, component in contexts[min(optimizer) :]
        ):
            return False
    return bool(by_rank)


def _dp_optimizer_reaches_gradient_collective(candidate: WorkloadTrace) -> bool:
    """Check that each rank's optimizer is downstream of a DP collective.

    This follows serialized dependency identifiers and does not assume a DP
    degree, rank layout, collective count, or optimizer-kernel multiplicity.
    """

    events = {event.id: event for event in candidate.events}
    ranks = {event.rank for event in candidate.events}
    reached_ranks: set[int] = set()
    terminal_dp_by_rank: dict[int, str] = {}
    for event in candidate.events:
        if event.group_role == "dp":
            terminal_dp_by_rank[event.rank] = event.id
    optimizer_events = [
        event
        for event in candidate.events
        if event.metadata.get("pipeline_phase") == "optimizer"
        or event.metadata.get("semantic_component") == "step.optimizer"
    ]
    for optimizer in optimizer_events:
        pending = list(optimizer.dependencies)
        seen: set[str] = set()
        reached_terminal = False
        while pending:
            identifier = pending.pop()
            if identifier in seen:
                continue
            seen.add(identifier)
            dependency = events.get(identifier)
            if dependency is None:
                continue
            if identifier == terminal_dp_by_rank.get(optimizer.rank):
                reached_terminal = True
                break
            pending.extend(dependency.dependencies)
        if not reached_terminal:
            return False
        reached_ranks.add(optimizer.rank)
    return reached_ranks == ranks and bool(ranks)


def _dp_optimizer_contract(
    candidate: WorkloadTrace, semantic_actual: WorkloadTrace, actual: WorkloadTrace
) -> dict[str, Any]:
    """Compare recorded DP optimizer ownership without kernel identity claims."""

    generated_measurement = candidate.metadata.get("framework_measurement")
    physical_measurement = actual.metadata.get("framework_measurement")
    generated_measurement = (
        generated_measurement if isinstance(generated_measurement, dict) else {}
    )
    physical_measurement = (
        physical_measurement if isinstance(physical_measurement, dict) else {}
    )
    generated_manifest = generated_measurement.get("parameter_manifest")
    physical_manifest = physical_measurement.get("parameter_manifest")
    generated_readiness = generated_measurement.get("gradient_readiness")
    physical_readiness = physical_measurement.get("gradient_readiness")
    expected_ranks = {event.rank for event in candidate.events}
    physical_ranks = {event.rank for event in actual.events}
    generated_optimizer_ranks = _optimizer_ranks(candidate, candidate=True)
    physical_optimizer_ranks = _optimizer_ranks(
        semantic_actual, candidate=False
    )
    generated_parameters = (
        generated_manifest.get("parameters", [])
        if isinstance(generated_manifest, dict)
        else []
    )
    physical_parameters = (
        physical_manifest.get("parameters", [])
        if isinstance(physical_manifest, dict)
        else []
    )
    generated_parameter_names = [
        row.get("name") for row in generated_parameters if isinstance(row, dict)
    ]
    physical_parameter_names = [
        row.get("name") for row in physical_parameters if isinstance(row, dict)
    ]
    generated_readiness_names = (
        generated_readiness.get("parameter_names", [])
        if isinstance(generated_readiness, dict)
        else []
    )
    physical_readiness_names = (
        physical_readiness.get("parameter_names", [])
        if isinstance(physical_readiness, dict)
        else []
    )
    manifest_well_formed = bool(generated_parameter_names) and all(
        isinstance(name, str) and name for name in generated_parameter_names
    )
    checks = {
        "parameter_manifest_well_formed": (
            manifest_well_formed
            and len(set(generated_parameter_names)) == len(generated_parameter_names)
            and generated_manifest.get("parameter_count")
            == len(generated_parameter_names)
            and physical_manifest.get("parameter_count")
            == len(physical_parameter_names)
        ),
        "parameter_manifest_exact": (
            isinstance(generated_manifest, dict)
            and generated_manifest == physical_manifest
        ),
        "gradient_readiness_exact": (
            isinstance(generated_readiness, dict)
            and generated_readiness == physical_readiness
        ),
        "gradient_readiness_is_manifest_permutation": (
            len(generated_readiness_names) == len(generated_parameter_names)
            and len(set(generated_readiness_names))
            == len(generated_readiness_names)
            and set(generated_readiness_names) == set(generated_parameter_names)
            and len(physical_readiness_names) == len(physical_parameter_names)
            and len(set(physical_readiness_names))
            == len(physical_readiness_names)
            and set(physical_readiness_names) == set(physical_parameter_names)
        ),
        "rank_set_exact": expected_ranks == physical_ranks and bool(expected_ranks),
        "optimizer_rank_coverage": (
            generated_optimizer_ranks
            == physical_optimizer_ranks
            == expected_ranks
        ),
        "generated_optimizer_terminal": _optimizer_is_terminal(
            candidate, candidate=True
        ),
        "physical_optimizer_terminal": _optimizer_is_terminal(
            semantic_actual, candidate=False
        ),
        "optimizer_reaches_dp_collective": (
            _dp_optimizer_reaches_gradient_collective(candidate)
        ),
    }
    return {
        "schema": "scaletether-dp-optimizer-attribution-contract-v2",
        "passed": all(checks.values()),
        "checks": checks,
        "expected_ranks": sorted(expected_ranks),
        "generated_optimizer_ranks": sorted(generated_optimizer_ranks),
        "physical_optimizer_ranks": sorted(physical_optimizer_ranks),
        "logical_optimizer_ownership_count": (
            len(expected_ranks) * len(generated_parameter_names)
            if manifest_well_formed
            else 0
        ),
        "generated_parameter_manifest_sha256": (
            _sha256(generated_manifest) if isinstance(generated_manifest, dict) else None
        ),
        "physical_parameter_manifest_sha256": (
            _sha256(physical_manifest) if isinstance(physical_manifest, dict) else None
        ),
        "generated_gradient_readiness_sha256": (
            _sha256(generated_readiness) if isinstance(generated_readiness, dict) else None
        ),
        "physical_gradient_readiness_sha256": (
            _sha256(physical_readiness) if isinstance(physical_readiness, dict) else None
        ),
        "kernel_identity_gating": False,
    }


def _compare_generated_to_physical(
    candidate: WorkloadTrace,
    actual: WorkloadTrace,
    matrix: dict[str, Any],
    target_id: str,
) -> dict[str, Any]:
    pp = target_id.startswith("PP-")
    dp = not pp and any(event.group_role == "dp" for event in candidate.events)
    physical_gate = (
        validate_physical_pp_target(actual.to_dict(), matrix, target_id)
        if pp
        else validate_physical_tp_dp_target(actual.to_dict(), matrix, target_id)
    )
    failures: list[str] = []
    if not physical_gate["passed"]:
        failures.append("physical-target-admission")
    semantic_actual = actual
    semantic_attribution_error: str | None = None
    try:
        if pp:
            semantic_actual = _semanticize_actual_pp(actual)
        else:
            semantic_actual = _semanticize_actual_tp_dp(actual)
    except TransformerGraphError as error:
        semantic_attribution_error = str(error)
        failures.append("physical-semantic-attribution")
    graph = candidate.metadata.get("transformer_semantic_graph")
    if (
        candidate.source.get("candidate_training_executed") is not False
        or not isinstance(graph, dict)
        or graph.get("target_id") != target_id
        or graph.get("target_training_executed") is not False
        or graph.get("timing_claim") != "none"
        or graph.get("kernel_code_claim") != "none"
    ):
        failures.append("candidate-provenance")

    generated_operator_diagnostic = _semantic_compute_counter(
        candidate, candidate=True, pp=pp
    )
    physical_operator_diagnostic = _semantic_compute_counter(
        semantic_actual, candidate=False, pp=pp
    )
    operator_delta = (
        None
        if generated_operator_diagnostic == physical_operator_diagnostic
        else _counter_delta(
            generated_operator_diagnostic, physical_operator_diagnostic
        )
    )
    generated_compute = _semantic_component_counter(
        candidate, candidate=True, pp=pp
    )
    physical_compute = _semantic_component_counter(
        semantic_actual, candidate=False, pp=pp
    )
    if generated_compute != physical_compute:
        failures.append("framework-component-presence")
        compute_delta = _counter_delta(generated_compute, physical_compute)
    else:
        compute_delta = None
    generated_component_order = _semantic_component_order(
        candidate, candidate=True, pp=pp
    )
    physical_component_order = _semantic_component_order(
        semantic_actual, candidate=False, pp=pp
    )
    component_order_matches = generated_component_order == physical_component_order
    if not component_order_matches:
        failures.append("framework-component-order")
    generated_collectives = _collective_counter(candidate, candidate=True, pp=pp)
    physical_collectives = _collective_counter(actual, candidate=False, pp=pp)
    if generated_collectives != physical_collectives:
        failures.append("collective-role-bytes-membership-multiset")
        collective_delta = _counter_delta(generated_collectives, physical_collectives)
    else:
        collective_delta = None

    generated_dependencies: dict[str, Any] | None = None
    physical_dependencies: dict[str, Any] | None = None
    dependency_error: str | None = None
    dependency_matches = False
    try:
        generated_dependencies = _framework_dependency_projection(
            candidate, candidate=True, pp=pp
        )
        physical_dependencies = _framework_dependency_projection(
            semantic_actual, candidate=False, pp=pp
        )
        dependency_matches = generated_dependencies == physical_dependencies
        if not dependency_matches:
            failures.append("framework-semantic-dependency-reachability")
    except ValueError as error:
        dependency_error = str(error)
        failures.append("framework-semantic-dependency-reachability")

    backend_generated_dependencies: dict[str, Any] | None = None
    backend_physical_dependencies: dict[str, Any] | None = None
    backend_dependency_error: str | None = None
    backend_dependency_matches = False
    if pp and physical_gate.get("decision") == "unsupported-capture":
        backend_dependency_error = "missing-pp-send-producer-readiness"
        failures.append("backend-semantic-dependency-reachability")
    else:
        try:
            backend_generated_dependencies = _dependency_projection(
                candidate, candidate=True, pp=pp
            )
            backend_physical_dependencies = _dependency_projection(
                semantic_actual, candidate=False, pp=pp
            )
            backend_dependency_matches = (
                backend_generated_dependencies == backend_physical_dependencies
            )
            if not backend_dependency_matches:
                failures.append("backend-semantic-dependency-reachability")
        except ValueError as error:
            backend_dependency_error = str(error)
            failures.append("backend-semantic-dependency-reachability")

    generated_optimizer: set[tuple[int, str]] = set()
    physical_optimizer: set[tuple[int, str]] = set()
    phase_orders: dict[str, Any] = {}
    dp_optimizer: dict[str, Any] | None = None
    if pp:
        generated_optimizer, physical_optimizer = _pp_optimizer_parameters(candidate, actual)
        if generated_optimizer != physical_optimizer:
            failures.append("pp-optimizer-parameter-placement")
        microbatches = graph.get("microbatches") if isinstance(graph, dict) else None
        if not isinstance(microbatches, int) or microbatches <= 0:
            failures.append("pp-microbatch-contract")
        else:
            for rank in range(4):
                observed = _pp_phase_order(actual, rank)
                expected = _expected_pp_phase_order(rank // 2, microbatches)
                phase_orders[str(rank)] = {"expected": expected, "observed": observed}
                if observed != expected:
                    failures.append(f"rank-{rank}-pp-phase-order")
    elif dp and semantic_attribution_error is None:
        dp_optimizer = _dp_optimizer_contract(candidate, semantic_actual, actual)
        if not dp_optimizer["passed"]:
            failures.append("dp-optimizer-attribution")

    unique_failures = sorted(set(failures))
    return {
        "schema": SCHEMA,
        "passed": not unique_failures,
        "failures": unique_failures,
        "target_id": target_id,
        "physical_gate": physical_gate,
        "generated_compute_sha256": _counter_sha256(generated_compute),
        "physical_compute_sha256": _counter_sha256(physical_compute),
        "compute_multiset_delta": compute_delta,
        "generated_component_order": {
            str(rank): [list(item) for item in sequence]
            for rank, sequence in generated_component_order.items()
        },
        "physical_component_order": {
            str(rank): [list(item) for item in sequence]
            for rank, sequence in physical_component_order.items()
        },
        "component_order_matches": component_order_matches,
        "semantic_attribution_error": semantic_attribution_error,
        "generated_operator_diagnostic_sha256": _counter_sha256(
            generated_operator_diagnostic
        ),
        "physical_operator_diagnostic_sha256": _counter_sha256(
            physical_operator_diagnostic
        ),
        "operator_diagnostic_delta": operator_delta,
        "operator_diagnostic_gating": False,
        "comparison_contract": {
            "schema": "megatron-framework-semantic-comparison-contract-v1",
            "gating": [
                "framework-component-presence",
                "framework-component-order",
                "collective-role-bytes-membership-multiset",
                "framework-semantic-dependency-reachability",
                "backend-semantic-dependency-reachability",
                *( ["dp-optimizer-attribution"] if dp else [] ),
            ],
            "non_gating": [
                "target-aten-operator-identity",
                "target-kernel-multiplicity",
                "target-kernel-code",
            ],
            "reason": "candidate kernel_code_claim is none",
        },
        "generated_collective_sha256": _counter_sha256(generated_collectives),
        "physical_collective_sha256": _counter_sha256(physical_collectives),
        "collective_multiset_delta": collective_delta,
        "generated_dependency_sha256": (
            _sha256(generated_dependencies)
            if generated_dependencies is not None
            else None
        ),
        "physical_dependency_sha256": (
            _sha256(physical_dependencies)
            if physical_dependencies is not None
            else None
        ),
        "dependency_matches": dependency_matches,
        "dependency_error": dependency_error,
        "generated_dependency_projection": generated_dependencies,
        "physical_dependency_projection": physical_dependencies,
        "backend_generated_dependency_sha256": (
            _sha256(backend_generated_dependencies)
            if backend_generated_dependencies is not None
            else None
        ),
        "backend_physical_dependency_sha256": (
            _sha256(backend_physical_dependencies)
            if backend_physical_dependencies is not None
            else None
        ),
        "backend_dependency_matches": backend_dependency_matches,
        "backend_dependency_error": backend_dependency_error,
        "backend_generated_dependency_projection": backend_generated_dependencies,
        "backend_physical_dependency_projection": backend_physical_dependencies,
        "generated_optimizer_parameter_count": len(generated_optimizer),
        "physical_optimizer_parameter_count": len(physical_optimizer),
        "dp_optimizer_attribution": dp_optimizer,
        "pp_phase_orders": phase_orders,
        "claim": (
            "held-out-transformer-structural-validation-no-timing-claim"
            if not unique_failures
            else "no-held-out-transformer-structural-claim"
        ),
    }


def _invalid_graph_code(error: Exception) -> str:
    """Map expected document/contract failures to stable public reason codes."""

    message = str(error)
    if "PP logical collective record is malformed" in message:
        return "malformed-pp-logical-collective"
    if "collective" in message.lower():
        return "malformed-collective"
    if "dependenc" in message.lower():
        return "malformed-dependency-graph"
    return "malformed-workload"


def compare_generated_to_physical(
    candidate: WorkloadTrace,
    actual: WorkloadTrace,
    matrix: dict[str, Any],
    target_id: str,
) -> dict[str, Any]:
    """Compare traces, converting malformed inputs into a typed rejection."""

    try:
        return _compare_generated_to_physical(candidate, actual, matrix, target_id)
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        return {
            "schema": SCHEMA,
            "passed": False,
            "failures": ["invalid-graph"],
            "target_id": target_id,
            "invalid_graph": {
                "code": _invalid_graph_code(error),
                "error_type": type(error).__name__,
                "message": str(error),
            },
            "claim": "no-held-out-transformer-structural-claim",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("target_id")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    report = compare_generated_to_physical(
        WorkloadTrace.load(args.candidate),
        WorkloadTrace.load(args.actual),
        json.loads(args.matrix.read_text(encoding="utf-8")),
        args.target_id,
    )
    report["matrix_sha256"] = _file_sha256(args.matrix)
    report["generated_candidate_sha256"] = _file_sha256(args.candidate)
    report["physical_target_sha256"] = _file_sha256(args.actual)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
