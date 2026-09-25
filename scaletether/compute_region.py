from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any, Iterable

from .schema import TraceEvent, WorkloadTrace


REGION_SCHEMA = "compute-region-v1"
CANDIDATE_SCHEMA = "compute-region-candidates-v1"
CANDIDATE_POLICY = "qualified-contiguous-mixed-kernel-windows-v1"
DECLARATION_SOURCES = {
    "captured-profiler-marker",
    "cli-user-declaration",
}


def compute_region_signature(kernel_signatures: Iterable[str]) -> str:
    signatures = tuple(str(item).strip() for item in kernel_signatures)
    if not signatures or any(not item for item in signatures):
        raise ValueError("compute region requires qualified kernel signatures")
    payload = json.dumps(
        {"schema": REGION_SCHEMA, "kernel_signatures": signatures},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{REGION_SCHEMA}:{hashlib.sha256(payload).hexdigest()[:24]}"


def compute_region_metadata(event: TraceEvent) -> dict[str, Any] | None:
    raw = event.metadata.get("compute_region")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{event.id}: compute_region metadata must be an object")
    required = {
        "schema",
        "signature",
        "instance_id",
        "member_index",
        "member_count",
        "kernel_signatures",
    }
    if required - set(raw):
        raise ValueError(f"{event.id}: compute_region metadata is incomplete")
    has_name = "name" in raw
    has_source = "declaration_source" in raw
    if has_name != has_source:
        raise ValueError(f"{event.id}: compute_region display metadata is incomplete")
    signatures = raw["kernel_signatures"]
    if (
        raw["schema"] != REGION_SCHEMA
        or not isinstance(raw["signature"], str)
        or not raw["signature"]
        or not isinstance(raw["instance_id"], str)
        or not raw["instance_id"]
        or not isinstance(raw["member_index"], int)
        or not isinstance(raw["member_count"], int)
        or not isinstance(signatures, list)
        or not all(isinstance(item, str) and item for item in signatures)
    ):
        raise ValueError(f"{event.id}: compute_region metadata is invalid")
    if has_name and (
        not isinstance(raw["name"], str)
        or not raw["name"].strip()
        or any(character in raw["name"] for character in "=,")
        or raw["name"] != raw["name"].strip()
        or raw["declaration_source"] not in DECLARATION_SOURCES
    ):
        raise ValueError(f"{event.id}: compute_region display metadata is invalid")
    if (
        raw["member_count"] <= 0
        or not 0 <= raw["member_index"] < raw["member_count"]
        or len(signatures) != raw["member_count"]
        or raw["signature"] != compute_region_signature(signatures)
    ):
        raise ValueError(f"{event.id}: compute_region identity is inconsistent")
    kernel_signature = event.metadata.get("kernel_signature")
    if kernel_signature != signatures[raw["member_index"]]:
        raise ValueError(
            f"{event.id}: compute_region member does not match its kernel signature"
        )
    return raw


def event_compute_identity(event: TraceEvent) -> str | None:
    region = compute_region_metadata(event)
    if region is not None:
        return str(region["signature"])
    signature = event.metadata.get("kernel_signature")
    return None if signature in (None, "") else str(signature)


def compute_region_members(
    events: Iterable[TraceEvent], member: TraceEvent
) -> tuple[TraceEvent, ...]:
    materialized = tuple(events)
    metadata = compute_region_metadata(member)
    if metadata is None:
        return ()
    candidates: list[tuple[int, TraceEvent]] = []
    for event in materialized:
        if event.kind != "compute":
            continue
        other = compute_region_metadata(event)
        if other is None:
            continue
        if (
            event.rank == member.rank
            and event.device == member.device
            and other["instance_id"] == metadata["instance_id"]
        ):
            if (
                event.stream != member.stream
                or other["signature"] != metadata["signature"]
                or other["member_count"] != metadata["member_count"]
                or other["kernel_signatures"] != metadata["kernel_signatures"]
                or other.get("name") != metadata.get("name")
                or other.get("declaration_source") != metadata.get("declaration_source")
            ):
                raise ValueError(
                    f"{member.id}: compute region mixes stream or identity domains"
                )
            candidates.append((int(other["member_index"]), event))
    if len(candidates) != metadata["member_count"] or [
        index for index, _ in candidates
    ] != list(range(metadata["member_count"])):
        raise ValueError(
            f"{member.id}: compute region membership is incomplete or reordered"
        )
    member_ids = {event.id for _, event in candidates}
    domain_compute = [
        event
        for event in materialized
        if event.kind == "compute"
        and event.rank == member.rank
        and event.device == member.device
        and event.stream == member.stream
    ]
    positions = [
        index for index, event in enumerate(domain_compute) if event.id in member_ids
    ]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError(
            f"{member.id}: compute region omits an intervening same-stream kernel"
        )
    return tuple(event for _, event in candidates)


def validate_compute_regions(events: Iterable[TraceEvent]) -> None:
    materialized = tuple(events)
    checked: set[tuple[int, int, str]] = set()
    for event in materialized:
        metadata = compute_region_metadata(event)
        if metadata is None:
            continue
        key = (event.rank, event.device, str(metadata["instance_id"]))
        if key not in checked:
            compute_region_members(materialized, event)
            checked.add(key)


def suggest_compute_regions(
    events: Iterable[TraceEvent],
    *,
    max_members: int = 4,
    max_candidates: int = 128,
) -> dict[str, Any]:
    """Return bounded explicit-region suggestions without annotating the trace."""

    if max_members < 2:
        raise ValueError(
            "compute-region candidate windows require at least two members"
        )
    if max_candidates <= 0:
        raise ValueError("compute-region candidate limit must be positive")
    materialized = tuple(events)
    validate_compute_regions(materialized)
    policy = {
        "schema": CANDIDATE_POLICY,
        "minimum_members": 2,
        "maximum_members": max_members,
        "maximum_retained_candidates": max_candidates,
        "ranking": "descending-observed-duration-then-captured-order",
        "selection_semantics": (
            "suggestions only; users must choose semantically meaningful, "
            "mutually disjoint declarations"
        ),
    }
    if any(compute_region_metadata(event) is not None for event in materialized):
        return {
            "schema": CANDIDATE_SCHEMA,
            "status": "locked-workload",
            "reason": (
                "workload already contains compute-region metadata; replay it "
                "without adding declarations"
            ),
            "policy": policy,
            "total_candidate_count": 0,
            "truncated": False,
            "candidates": [],
        }

    domains: dict[tuple[int, int, str], list[tuple[int, TraceEvent]]] = {}
    for captured_index, event in enumerate(materialized):
        signature = event.metadata.get("kernel_signature")
        if event.kind != "compute" or not isinstance(signature, str) or not signature:
            continue
        domains.setdefault((event.rank, event.device, event.stream), []).append(
            (captured_index, event)
        )

    windows: list[dict[str, Any]] = []
    for (rank, device, stream), domain_events in domains.items():
        for start in range(len(domain_events)):
            maximum = min(max_members, len(domain_events) - start)
            for member_count in range(2, maximum + 1):
                window = domain_events[start : start + member_count]
                members = [event for _, event in window]
                signatures = [
                    str(event.metadata["kernel_signature"]) for event in members
                ]
                if len(set(signatures)) < 2:
                    continue
                windows.append(
                    {
                        "_captured_index": window[0][0],
                        "signature": compute_region_signature(signatures),
                        "event_ids": [event.id for event in members],
                        "kernel_signatures": signatures,
                        "kernel_names": [event.name for event in members],
                        "member_count": member_count,
                        "rank": rank,
                        "device": device,
                        "stream": stream,
                        "observed_duration_us": sum(
                            float(event.duration_us) for event in members
                        ),
                    }
                )
    windows.sort(
        key=lambda item: (
            -float(item["observed_duration_us"]),
            int(item["_captured_index"]),
            int(item["member_count"]),
            tuple(item["event_ids"]),
        )
    )
    total = len(windows)
    candidates = []
    for index, window in enumerate(windows[:max_candidates]):
        candidate = dict(window)
        candidate.pop("_captured_index")
        name = f"candidate-{index:04d}"
        candidate["name"] = name
        candidate["declaration"] = f"{name}=" + ",".join(candidate["event_ids"])
        candidates.append(candidate)
    return {
        "schema": CANDIDATE_SCHEMA,
        "status": "available",
        "reason": None,
        "policy": policy,
        "total_candidate_count": total,
        "truncated": total > len(candidates),
        "candidates": candidates,
    }


def compute_region_marker_declarations(
    events: Iterable[TraceEvent],
) -> tuple[dict[str, Any], ...]:
    """Validate captured marker evidence and return ordered declarations."""

    materialized = tuple(events)
    groups: dict[str, list[TraceEvent]] = {}
    evidence_by_instance: dict[str, dict[str, Any]] = {}
    first_position: dict[str, int] = {}
    for position, event in enumerate(materialized):
        raw = event.metadata.get("compute_region_marker")
        if raw is None:
            continue
        if event.kind != "compute" or not isinstance(raw, dict):
            raise ValueError(f"{event.id}: compute-region marker evidence is invalid")
        required = {
            "schema",
            "name",
            "instance_id",
            "occurrence",
            "host_duration_us",
            "gpu_annotation_count",
            "gpu_streams",
            "membership_evidence",
        }
        if required - set(raw) or (
            raw["schema"] != "compute-region-marker-v1"
            or not isinstance(raw["name"], str)
            or not raw["name"]
            or any(character in raw["name"] for character in "=,")
            or not isinstance(raw["instance_id"], str)
            or not raw["instance_id"]
            or not isinstance(raw["occurrence"], int)
            or raw["occurrence"] < 0
            or not isinstance(raw["host_duration_us"], (int, float))
            or float(raw["host_duration_us"]) <= 0.0
            or not isinstance(raw["gpu_annotation_count"], int)
            or raw["gpu_annotation_count"] < 0
            or not isinstance(raw["gpu_streams"], list)
            or not all(isinstance(item, str) for item in raw["gpu_streams"])
            or raw["membership_evidence"]
            != "correlated-cuda-launch-inside-host-marker-v1"
        ):
            raise ValueError(f"{event.id}: compute-region marker evidence is invalid")
        instance_id = str(raw["instance_id"])
        canonical = dict(raw)
        if (
            instance_id in evidence_by_instance
            and canonical != evidence_by_instance[instance_id]
        ):
            raise ValueError(
                f"{event.id}: compute-region marker instance evidence is inconsistent"
            )
        evidence_by_instance[instance_id] = canonical
        groups.setdefault(instance_id, []).append(event)
        first_position.setdefault(instance_id, position)

    ordered_instances = sorted(groups, key=lambda item: first_position[item])
    name_counts: dict[str, int] = {}
    name_ranks: dict[str, set[int]] = {}
    for instance_id in ordered_instances:
        name = str(evidence_by_instance[instance_id]["name"])
        name_counts[name] = name_counts.get(name, 0) + 1
        name_ranks.setdefault(name, set()).update(
            event.rank for event in groups[instance_id]
        )
    declarations = []
    for instance_id in ordered_instances:
        members = groups[instance_id]
        evidence = evidence_by_instance[instance_id]
        if not members:
            raise ValueError(
                f"compute-region marker {evidence['name']!r} has no members"
            )
        domains = {(event.rank, event.device, event.stream) for event in members}
        if len(domains) != 1:
            raise ValueError(
                f"compute-region marker {evidence['name']!r} mixes domains"
            )
        rank, device, stream = next(iter(domains))
        member_ids = {event.id for event in members}
        domain_compute = [
            event
            for event in materialized
            if event.kind == "compute"
            and event.rank == rank
            and event.device == device
            and event.stream == stream
        ]
        positions = [
            index
            for index, event in enumerate(domain_compute)
            if event.id in member_ids
        ]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError(
                f"compute-region marker {evidence['name']!r} omits an "
                "intervening same-stream kernel"
            )
        marker_name = str(evidence["name"])
        if name_counts[marker_name] == 1:
            declaration_name = marker_name
        elif len(name_ranks[marker_name]) > 1:
            declaration_name = f"{marker_name}-rank{rank}-{int(evidence['occurrence'])}"
        else:
            declaration_name = f"{marker_name}-{int(evidence['occurrence'])}"
        declarations.append(
            {
                "name": declaration_name,
                "marker_name": marker_name,
                "instance_id": instance_id,
                "event_ids": [event.id for event in members],
                "evidence": evidence,
            }
        )
    return tuple(declarations)


def annotate_compute_region(
    trace: WorkloadTrace,
    event_ids: Iterable[str],
    *,
    instance_id: str | None = None,
    name: str | None = None,
    declaration_source: str | None = None,
) -> tuple[WorkloadTrace, str]:
    requested = tuple(event_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("compute region event IDs must be non-empty and unique")
    requested_set = set(requested)
    selected = [event for event in trace.events if event.id in requested_set]
    missing = requested_set - {event.id for event in selected}
    if missing:
        raise ValueError(
            "compute region references unknown events: " + ", ".join(sorted(missing))
        )
    if any(event.kind != "compute" for event in selected):
        raise ValueError("compute region may contain only compute events")
    domains = {(event.rank, event.device, event.stream) for event in selected}
    if len(domains) != 1:
        raise ValueError("compute region must use one rank, device, and stream")
    if any(event.metadata.get("compute_region") is not None for event in selected):
        raise ValueError("compute event already belongs to a region")
    kernel_signatures = tuple(
        str(event.metadata.get("kernel_signature", "")).strip() for event in selected
    )
    signature = compute_region_signature(kernel_signatures)
    resolved_instance = (
        f"{signature}:{selected[0].id}" if instance_id is None else instance_id.strip()
    )
    if not resolved_instance:
        raise ValueError("compute region instance ID must be non-empty")
    if (name is None) != (declaration_source is None):
        raise ValueError(
            "compute region name and declaration source must be supplied together"
        )
    display_name = None if name is None else name.strip()
    if display_name is not None and (
        not display_name
        or display_name != name
        or any(character in display_name for character in "=,")
    ):
        raise ValueError("compute region name is invalid")
    if declaration_source is not None and declaration_source not in DECLARATION_SOURCES:
        raise ValueError("compute region declaration source is invalid")
    replacements: dict[str, TraceEvent] = {}
    for index, event in enumerate(selected):
        metadata = dict(event.metadata)
        metadata["compute_region"] = {
            "schema": REGION_SCHEMA,
            "signature": signature,
            "instance_id": resolved_instance,
            "member_index": index,
            "member_count": len(selected),
            "kernel_signatures": list(kernel_signatures),
        }
        if display_name is not None:
            metadata["compute_region"].update(
                {
                    "name": display_name,
                    "declaration_source": declaration_source,
                }
            )
        replacements[event.id] = replace(event, metadata=metadata)
    annotated = replace(
        trace,
        events=tuple(replacements.get(event.id, event) for event in trace.events),
    )
    validate_compute_regions(annotated.events)
    return annotated, signature
