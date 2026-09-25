from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, log2
from typing import Any

from .calibration import (
    OverlapCalibrationModel,
    nvls_runtime_evidence,
    select_calibration_environment,
)
from .config import (
    Architecture,
    NetworkTier,
    Parallelism,
    Topology,
    assess_hardware_identity,
)
from .compute_transfer import ComputeTransferCalibration
from .compute_region import (
    compute_region_members,
    compute_region_metadata,
    event_compute_identity,
    validate_compute_regions,
)
from .network_calibration import NetworkContentionCalibration
from .nccl_policy import (
    cta_policy_name,
    parse_cga_cluster_size,
    parse_cta_policy_mode,
)
from .pipeline import expand_pipeline
from .schema import TraceEvent, WorkloadTrace
from .uncertainty import attach_uncertainty


@dataclass(frozen=True)
class PreparedEvent:
    event: TraceEvent
    isolated_duration_us: float
    duration_source: str
    group_role: str | None
    group_size: int | None
    max_ctas: int | None
    comm_stream_priority: str | None
    network_resources: tuple[str, ...]
    collective_instance_id: str | None
    nccl_cta_policy: int | None
    nccl_nvls_ctas: int | None
    nccl_cga_cluster_size: int | None
    network_tier: str | None
    p2p_source_device: int | None
    p2p_destination_device: int | None

    @property
    def stream_key(self) -> tuple[int, int, str]:
        return (self.event.rank, self.event.device, self.event.stream)


def _network_contention_operation(item: PreparedEvent) -> dict[str, Any]:
    return {
        "collective": item.event.collective or "unknown",
        "message_bytes": (
            0 if item.event.collective == "barrier" else item.event.message_bytes
        ),
        "group_size": item.group_size,
        "group_role": item.group_role,
        "network_resources": list(item.network_resources),
        "network_tier": item.network_tier,
        "max_ctas": item.max_ctas,
        "comm_stream_priority": item.comm_stream_priority or "normal",
        "nccl_cta_policy": item.nccl_cta_policy or 0,
        "nccl_nvls_ctas": item.nccl_nvls_ctas,
        "nccl_cga_cluster_size": item.nccl_cga_cluster_size or 0,
    }


@dataclass(frozen=True)
class ScheduledEvent:
    id: str
    name: str
    kind: str
    stream: str
    start_us: float
    end_us: float
    duration_us: float
    isolated_duration_us: float
    effective_slowdown: float
    duration_source: str
    launch_release_us: float
    launch_release_source: str
    dependencies: tuple[str, ...]
    dependency_evidence: tuple[dict[str, Any], ...]
    interference_sources: tuple[str, ...]
    collective: str | None
    group_role: str | None
    group_size: int | None
    message_bytes: int | None
    max_ctas: int | None
    comm_stream_priority: str | None
    rank: int
    device: int
    sm_fraction: float | None
    kernel_signature: str | None
    cuda_stream_priority_binding: dict[str, Any] | None
    compute_region: dict[str, Any] | None
    overlap_region: dict[str, Any] | None
    collective_region: dict[str, Any] | None
    kernel_resource: dict[str, Any]
    network_resources: tuple[str, ...]
    collective_instance_id: str | None
    nccl_cta_policy: int | None
    nccl_cta_policy_name: str | None
    nccl_nvls_ctas: int | None
    nccl_cga_cluster_size: int | None
    kernel_sm_footprint: dict[str, Any] | None
    nccl_sm_footprint: dict[str, Any] | None
    zero_cta_eligible: bool | None
    network_tier: str | None
    p2p_source_device: int | None
    p2p_destination_device: int | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dependencies"] = list(self.dependencies)
        result["dependency_evidence"] = list(self.dependency_evidence)
        result["interference_sources"] = list(self.interference_sources)
        result["network_resources"] = list(self.network_resources)
        return result


def _realized_critical_path(
    timeline: list[ScheduledEvent], step_time_us: float
) -> dict[str, Any]:
    """Extract one deterministic causal chain from the realized schedule.

    Explicit DAG dependencies and same-rank/device/stream serialization are
    the only start constraints attributed here. Interference and routed
    contention are already reflected in each scheduled event's wall duration;
    this function does not pretend to assign those slowdowns to a predecessor.
    """

    schema = "realized-schedule-critical-path-v1"
    interpretation = (
        "One representative zero-slack chain through the realized schedule. "
        "Edges are explicit dependencies or same-stream serialization; "
        "a controlling captured launch release is represented as a wait segment; "
        "interference and network contention are folded into event wall times, "
        "not attributed as causal edges. Tied critical terminals are retained."
    )
    if not timeline:
        return {
            "schema": schema,
            "status": "empty-workload",
            "step_time_us": step_time_us,
            "accounted_duration_us": 0.0,
            "gpu_event_duration_us": 0.0,
            "launch_release_wait_us": 0.0,
            "unexplained_idle_us": step_time_us,
            "accounted_step_fraction": 1.0 if step_time_us == 0 else 0.0,
            "event_ids": [],
            "event_count": 0,
            "edges": [],
            "segments": [],
            "terminal_event_id": None,
            "tied_terminal_event_ids": [],
            "composition_us": {},
            "bottleneck_kind": None,
            "bottleneck_segment": None,
            "bottleneck_event": None,
            "interpretation": interpretation,
        }

    by_id = {event.id: event for event in timeline}
    tolerance = max(1e-9, abs(step_time_us) * 1e-9)
    constraints: dict[str, dict[str, set[str]]] = {event.id: {} for event in timeline}

    for event in timeline:
        for dependency in event.dependencies:
            predecessor = by_id.get(dependency)
            if predecessor is None:
                continue
            if abs(predecessor.end_us - event.start_us) <= tolerance:
                constraints[event.id].setdefault(dependency, set()).add(
                    "explicit-dependency"
                )

    streams: dict[tuple[int, int, str], list[ScheduledEvent]] = {}
    for event in timeline:
        streams.setdefault((event.rank, event.device, event.stream), []).append(event)
    for events in streams.values():
        ordered = sorted(events, key=lambda item: (item.start_us, item.end_us, item.id))
        for index, event in enumerate(ordered):
            if index == 0:
                continue
            predecessor = ordered[index - 1]
            if abs(predecessor.end_us - event.start_us) > tolerance:
                continue
            constraints[event.id].setdefault(predecessor.id, set()).add(
                "same-stream-serialization"
            )

    ordered_timeline = sorted(
        timeline, key=lambda item: (item.end_us, item.start_us, item.id)
    )
    accounted: dict[str, float] = {}
    selected_predecessor: dict[str, str | None] = {}
    selected_release: dict[str, bool] = {}
    for event in ordered_timeline:
        candidates = [
            predecessor
            for predecessor in constraints[event.id]
            if predecessor in accounted
        ]
        predecessor = (
            None
            if not candidates
            else max(candidates, key=lambda item: (accounted[item], item))
        )
        predecessor_accounted = 0.0 if predecessor is None else accounted[predecessor]
        release_controls = (
            abs(event.launch_release_us - event.start_us) <= tolerance
            and event.launch_release_us > predecessor_accounted + tolerance
        )
        selected_release[event.id] = release_controls
        selected_predecessor[event.id] = None if release_controls else predecessor
        accounted[event.id] = event.duration_us + (
            event.launch_release_us if release_controls else predecessor_accounted
        )

    terminals = sorted(
        event.id for event in timeline if abs(event.end_us - step_time_us) <= tolerance
    )
    terminal = max(terminals, key=lambda item: (accounted[item], item))
    reverse_chain = []
    current: str | None = terminal
    while current is not None:
        reverse_chain.append(current)
        current = selected_predecessor[current]
    event_ids = list(reversed(reverse_chain))

    edges = []
    for source, target in zip(event_ids, event_ids[1:]):
        edges.append(
            {
                "source": source,
                "target": target,
                "constraint_types": sorted(constraints[target][source]),
            }
        )
    composition: dict[str, float] = {}
    segments = []
    first = by_id[event_ids[0]]
    release_wait = first.launch_release_us if selected_release[first.id] else 0.0
    if release_wait > tolerance:
        composition["launch-release-wait"] = release_wait
        segments.append(
            {
                "kind": "launch-release-wait",
                "duration_us": release_wait,
                "target_event": first.id,
                "source": first.launch_release_source,
            }
        )
    for event_id in event_ids:
        event = by_id[event_id]
        composition[event.kind] = composition.get(event.kind, 0.0) + event.duration_us
        segments.append(
            {
                "kind": event.kind,
                "duration_us": event.duration_us,
                "event_id": event.id,
                "name": event.name,
                "duration_source": event.duration_source,
            }
        )
    gpu_event_duration = sum(by_id[event_id].duration_us for event_id in event_ids)
    accounted_duration = accounted[terminal]
    bottleneck_kind = max(
        sorted(composition), key=lambda kind: (composition[kind], kind)
    )
    bottleneck = max(
        (by_id[event_id] for event_id in event_ids),
        key=lambda event: (event.duration_us, event.id),
    )
    bottleneck_segment = max(
        segments,
        key=lambda segment: (
            float(segment["duration_us"]),
            str(segment.get("event_id", "")),
        ),
    )
    unexplained_idle = max(0.0, step_time_us - accounted_duration)
    return {
        "schema": schema,
        "status": (
            "complete" if unexplained_idle <= tolerance else "partial-unexplained-idle"
        ),
        "step_time_us": step_time_us,
        "accounted_duration_us": accounted_duration,
        "gpu_event_duration_us": gpu_event_duration,
        "launch_release_wait_us": release_wait,
        "unexplained_idle_us": unexplained_idle,
        "accounted_step_fraction": (
            1.0 if step_time_us == 0 else accounted_duration / step_time_us
        ),
        "event_ids": event_ids,
        "event_count": len(event_ids),
        "edges": edges,
        "segments": segments,
        "terminal_event_id": terminal,
        "tied_terminal_event_ids": terminals,
        "composition_us": dict(sorted(composition.items())),
        "bottleneck_kind": bottleneck_kind,
        "bottleneck_segment": bottleneck_segment,
        "bottleneck_event": {
            "id": bottleneck.id,
            "name": bottleneck.name,
            "kind": bottleneck.kind,
            "duration_us": bottleneck.duration_us,
            "duration_source": bottleneck.duration_source,
        },
        "interpretation": interpretation,
    }


def _transfer_us(bytes_: float, tier: NetworkTier) -> float:
    return bytes_ / (tier.bandwidth_gbps * 1_000.0)


def _collective_tier_duration(
    collective: str,
    bytes_: int,
    ranks: int,
    tier: NetworkTier,
    byte_semantics: str | None,
) -> tuple[float, str]:
    if ranks <= 1:
        return 0.0, "single-rank"
    if collective == "all_reduce":
        transfer_bytes = 2.0 * (ranks - 1) / ranks * bytes_
        steps = 2 * (ranks - 1)
        algorithm = "ring-all-reduce"
    elif collective == "all_gather":
        if byte_semantics in (None, "local_input_contribution"):
            transfer_bytes = (ranks - 1) * bytes_
            algorithm = "ring-all-gather-local-input"
        else:
            transfer_bytes = (ranks - 1) / ranks * bytes_
            algorithm = "ring-all-gather-global-output"
        steps = ranks - 1
    elif collective in {"reduce_scatter", "all_to_all"}:
        transfer_bytes = (ranks - 1) / ranks * bytes_
        steps = ranks - 1
        algorithm = f"ring-{collective.replace('_', '-')}"
    elif collective == "broadcast":
        steps = ceil(log2(ranks))
        transfer_bytes = steps * bytes_
        algorithm = "binomial-tree-broadcast"
    elif collective == "reduce":
        steps = ceil(log2(ranks))
        transfer_bytes = steps * bytes_
        algorithm = "binomial-tree-reduce"
    elif collective == "gather":
        steps = ranks - 1
        transfer_bytes = (ranks - 1) * bytes_
        algorithm = "ring-gather-local-input"
    elif collective == "scatter":
        steps = ceil(log2(ranks))
        transfer_bytes = steps * bytes_
        algorithm = "binomial-tree-scatter-local-output"
    elif collective == "barrier":
        steps = ceil(log2(ranks))
        transfer_bytes = 0.0
        algorithm = "dissemination-barrier"
    elif collective in {"send", "recv"}:
        transfer_bytes = bytes_
        steps = 1
        algorithm = f"point-to-point-{collective}"
    else:
        raise ValueError(f"unsupported analytical collective {collective!r}")
    return _transfer_us(transfer_bytes, tier) + steps * tier.latency_us, algorithm


def _ring_duration(bytes_: int, ranks: int, tier: NetworkTier) -> float:
    # Retain the original helper for callers/tests of the AllReduce fallback.
    return _collective_tier_duration(
        "all_reduce", bytes_, ranks, tier, "in_place_tensor"
    )[0]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _kernel_resource_hardware_domain(
    event: TraceEvent,
    trace_source: dict[str, Any],
    target: Architecture,
) -> dict[str, Any]:
    trace_assessment = assess_hardware_identity(trace_source, target)
    if trace_assessment["status"] == "exact-product-match":
        return {**trace_assessment, "evidence_source": "trace-source"}
    resource = event.metadata.get("kernel_resource", {})
    evidence = (
        resource.get("max_active_blocks_per_sm_evidence")
        if isinstance(resource, dict)
        else None
    )
    if isinstance(evidence, dict):
        profile_assessment = assess_hardware_identity(
            {
                "target": evidence.get("target"),
                "device_name": evidence.get("device_name"),
                "compute_capability": evidence.get("compute_capability"),
                "sm_count": evidence.get("sm_count"),
            },
            target,
        )
        if profile_assessment["status"] == "exact-product-match":
            return {
                **profile_assessment,
                "evidence_source": "exact-occupancy-profile",
                "profile_sha256": evidence.get("profile_sha256"),
            }
    return {**trace_assessment, "evidence_source": "trace-source"}


def _nccl_sm_footprint(
    event: TraceEvent,
    target: Architecture,
    max_ctas: int | None,
    cga_cluster_size: int | None,
    trace_source: dict[str, Any],
) -> dict[str, Any] | None:
    if event.kind != "collective":
        return None
    resource = event.metadata.get("kernel_resource", {})
    if not isinstance(resource, dict):
        resource = {}
    hardware_domain = _kernel_resource_hardware_domain(event, trace_source, target)
    exact_hardware = hardware_domain["status"] == "exact-product-match"
    source_observed_grid_blocks = _positive_int(resource.get("grid_blocks"))
    observed_grid_blocks = source_observed_grid_blocks if exact_hardware else None
    configured_cta_budget = _positive_int(max_ctas)
    if configured_cta_budget is not None:
        full_grid_ctas = configured_cta_budget
        count_source = "configured-max-ctas-upper-bound"
    elif observed_grid_blocks is not None:
        full_grid_ctas = observed_grid_blocks
        count_source = "observed-kernel-grid"
    else:
        full_grid_ctas = None
        count_source = "unknown"
    raw_grid_blocks_per_sm = resource.get("blocks_per_sm")
    try:
        grid_blocks_per_sm = (
            None if raw_grid_blocks_per_sm is None else float(raw_grid_blocks_per_sm)
        )
    except (TypeError, ValueError):
        grid_blocks_per_sm = None
    source_blocks_per_sm_capacity = _positive_int(
        resource.get("max_active_blocks_per_sm")
    )
    blocks_per_sm_capacity = source_blocks_per_sm_capacity if exact_hardware else None
    full_grid_min_sms = (
        None
        if full_grid_ctas is None or blocks_per_sm_capacity is None
        else ceil(full_grid_ctas / blocks_per_sm_capacity)
    )
    full_grid_max_sms = (
        None if full_grid_ctas is None else min(full_grid_ctas, target.sm_count)
    )
    full_grid_residency_feasible = (
        None
        if full_grid_ctas is None or blocks_per_sm_capacity is None
        else full_grid_ctas <= blocks_per_sm_capacity * target.sm_count
    )
    if full_grid_residency_feasible is False:
        full_grid_max_sms = None
    minimum_residency_waves = (
        None
        if full_grid_ctas is None or blocks_per_sm_capacity is None
        else ceil(full_grid_ctas / (blocks_per_sm_capacity * target.sm_count))
    )
    cluster_size = _positive_int(cga_cluster_size)
    cluster_divisible = (
        None
        if cluster_size is None or full_grid_ctas is None
        else full_grid_ctas % cluster_size == 0
    )
    if (
        configured_cta_budget is None
        and source_observed_grid_blocks is not None
        and not exact_hardware
    ):
        status = "source-hardware-outside-target-domain"
    elif full_grid_ctas is None:
        status = "unknown-cta-count"
    elif blocks_per_sm_capacity is None:
        status = "cta-count-known-occupancy-unknown"
    elif full_grid_residency_feasible:
        status = "full-grid-residency-bounded-placement-unknown"
    else:
        status = "full-grid-requires-multiple-waves"
    return {
        "schema": "nccl-sm-footprint-v1",
        "status": status,
        "configured_max_ctas": configured_cta_budget,
        "observed_grid_blocks": observed_grid_blocks,
        "source_observed_grid_blocks": source_observed_grid_blocks,
        "full_grid_ctas": full_grid_ctas,
        "count_source": count_source,
        "observed_grid_blocks_per_sm": (grid_blocks_per_sm if exact_hardware else None),
        "source_observed_grid_blocks_per_sm": grid_blocks_per_sm,
        "measured_max_active_blocks_per_sm": blocks_per_sm_capacity,
        "source_observed_max_active_blocks_per_sm": source_blocks_per_sm_capacity,
        "capacity_evidence": (
            resource.get("max_active_blocks_per_sm_evidence")
            if exact_hardware
            else None
        ),
        "hardware_evidence_domain": hardware_domain,
        "full_grid_min_sms_if_concurrently_resident": full_grid_min_sms,
        "full_grid_max_sms_if_concurrently_resident": full_grid_max_sms,
        "full_grid_residency_feasible_from_observed_capacity": (
            full_grid_residency_feasible
        ),
        "target_sm_count": target.sm_count,
        "minimum_residency_waves": minimum_residency_waves,
        "cga_cluster_size": cluster_size,
        "cta_count_divisible_by_cluster_size": cluster_divisible,
        "exclusive_sm_assignment": None,
        "exclusive_sm_reservation": False,
        "sm_identity_set": None,
        "placement": "unknown",
        "residency_model": "static-capacity-bounds-only",
        "scheduler_assignment": "dynamic-unknown",
        "timing_effect_source": "matched-calibration-only",
        "interpretation": (
            "SM bounds describe a fully resident CTA grid, not exclusive SM "
            "ownership, a fixed SM identity set, or achieved placement; calibrated "
            "overlap remains authoritative"
        ),
    }


def _kernel_sm_footprint(
    event: TraceEvent,
    target: Architecture,
    trace_source: dict[str, Any],
) -> dict[str, Any] | None:
    resource = event.metadata.get("kernel_resource")
    if not isinstance(resource, dict):
        return None
    source_grid_blocks = _positive_int(resource.get("grid_blocks"))
    if source_grid_blocks is None:
        return None
    hardware_domain = _kernel_resource_hardware_domain(event, trace_source, target)
    exact_hardware = hardware_domain["status"] == "exact-product-match"
    source_capacity = _positive_int(resource.get("max_active_blocks_per_sm"))
    grid_blocks = source_grid_blocks if exact_hardware else None
    capacity = source_capacity if exact_hardware else None
    minimum_sms = None if capacity is None else ceil(grid_blocks / capacity)
    maximum_sms = None if grid_blocks is None else min(grid_blocks, target.sm_count)
    feasible = (
        None
        if capacity is None or grid_blocks is None
        else grid_blocks <= capacity * target.sm_count
    )
    if feasible is False:
        maximum_sms = None
    minimum_residency_waves = (
        None if capacity is None else ceil(grid_blocks / (capacity * target.sm_count))
    )
    return {
        "schema": "kernel-sm-footprint-v1",
        "status": (
            "source-hardware-outside-target-domain"
            if not exact_hardware
            else (
                "grid-known-occupancy-unknown"
                if capacity is None
                else (
                    "full-grid-residency-bounded-placement-unknown"
                    if feasible
                    else "full-grid-requires-multiple-waves"
                )
            )
        ),
        "observed_grid_blocks": grid_blocks,
        "source_observed_grid_blocks": source_grid_blocks,
        "observed_grid_blocks_per_sm": (
            resource.get("blocks_per_sm") if exact_hardware else None
        ),
        "source_observed_grid_blocks_per_sm": resource.get("blocks_per_sm"),
        "measured_max_active_blocks_per_sm": capacity,
        "source_observed_max_active_blocks_per_sm": source_capacity,
        "capacity_evidence": (
            resource.get("max_active_blocks_per_sm_evidence")
            if exact_hardware
            else None
        ),
        "hardware_evidence_domain": hardware_domain,
        "full_grid_min_sms_if_concurrently_resident": minimum_sms,
        "full_grid_max_sms_if_concurrently_resident": maximum_sms,
        "full_grid_residency_feasible_from_observed_capacity": feasible,
        "target_sm_count": target.sm_count,
        "minimum_residency_waves": minimum_residency_waves,
        "exclusive_sm_assignment": None,
        "exclusive_sm_reservation": False,
        "sm_identity_set": None,
        "placement": "unknown",
        "residency_model": "static-capacity-bounds-only",
        "scheduler_assignment": "dynamic-unknown",
        "timing_effect_source": "matched-calibration-only",
        "interpretation": (
            "bounds describe the observed kernel's fully resident block grid, "
            "not exclusive SM ownership, a fixed SM identity set, achieved placement, "
            "or slowdown"
        ),
    }


def _point_to_point_route(
    event: TraceEvent, topology: Topology
) -> tuple[str, int, int] | None:
    if event.collective not in {"send", "recv"}:
        return None
    raw_source = event.metadata.get("p2p_source_device")
    raw_destination = event.metadata.get("p2p_destination_device")
    if raw_source is None and raw_destination is None:
        return None
    if raw_source is None or raw_destination is None:
        raise ValueError(f"event {event.id!r} must provide both P2P endpoint devices")
    if (
        isinstance(raw_source, bool)
        or not isinstance(raw_source, int)
        or isinstance(raw_destination, bool)
        or not isinstance(raw_destination, int)
    ):
        raise ValueError(f"event {event.id!r} P2P endpoint devices must be integers")
    if raw_source < 0 or raw_destination < 0 or raw_source == raw_destination:
        raise ValueError(
            f"event {event.id!r} P2P endpoint devices must be distinct and non-negative"
        )
    if topology.nodes is not None:
        capacity = topology.nodes * topology.gpus_per_node
        if raw_source >= capacity or raw_destination >= capacity:
            raise ValueError(
                f"event {event.id!r} P2P endpoint exceeds topology capacity {capacity}"
            )
    tier = (
        "intra_node"
        if raw_source // topology.gpus_per_node
        == raw_destination // topology.gpus_per_node
        else "inter_node"
    )
    return tier, raw_source, raw_destination


def _collective_network_resources(
    event: TraceEvent, group_size: int | None, topology: Topology
) -> tuple[str, ...]:
    explicit = event.metadata.get("network_resources")
    if explicit is not None:
        if not isinstance(explicit, (list, tuple)) or not explicit:
            raise ValueError(
                f"event {event.id!r} network_resources must be a non-empty list"
            )
        resources = tuple(str(item).strip() for item in explicit)
        if any(not item for item in resources) or len(set(resources)) != len(resources):
            raise ValueError(
                f"event {event.id!r} network_resources has empty or duplicate entries"
            )
        return resources
    point_to_point = _point_to_point_route(event, topology)
    if point_to_point is not None:
        tier, _, _ = point_to_point
        return topology.route_resources(tier)
    if group_size is None or group_size <= 1:
        return ()
    tiers = ["intra_node"]
    if group_size > topology.gpus_per_node:
        tiers.append("inter_node")
    resources: list[str] = []
    for tier in tiers:
        for resource in topology.route_resources(tier):
            if resource not in resources:
                resources.append(resource)
    return tuple(resources)


def _collective_instance_id(event: TraceEvent) -> str | None:
    if event.kind != "collective":
        return None
    explicit = event.metadata.get("collective_instance_id")
    if explicit is not None:
        value = str(explicit).strip()
        if not value:
            raise ValueError(
                f"event {event.id!r} collective_instance_id cannot be empty"
            )
        return value
    p2p_pair = event.metadata.get("p2p_pair_id")
    if event.collective in {"send", "recv"} and p2p_pair is not None:
        value = str(p2p_pair).strip()
        if not value:
            raise ValueError(f"event {event.id!r} p2p_pair_id cannot be empty")
        # Send and Recv are the two endpoints of one transfer, not two
        # independently contending network operations.
        return f"p2p:{value}"
    sequence = event.metadata.get("collective_sequence")
    communicator = event.metadata.get("communicator")
    if sequence is not None and communicator is not None:
        return f"{communicator}:{sequence}:{event.collective}"
    # A semantic trace may contain only one representative rank. Treat each
    # event as a distinct operation unless it explicitly supplies an instance.
    return event.id


def _collective_duration(
    event: TraceEvent,
    parallelism: Parallelism,
    topology: Topology,
    target: Architecture,
    max_ctas: int | None,
    calibration: OverlapCalibrationModel | None,
    comm_stream_priority: str,
    nccl_cta_policy: int,
    nccl_nvls_ctas: int | None,
    nccl_cga_cluster_size: int,
) -> tuple[float, str, int | None, str | None]:
    role = event.group_role
    group_size = event.group_size
    if group_size is None and role is not None:
        group_size = parallelism.group_size(role)
    point_to_point = _point_to_point_route(event, topology)
    if point_to_point is not None and (group_size is None or group_size <= 1):
        group_size = 2
    if group_size is None or (
        event.message_bytes is None and event.collective != "barrier"
    ):
        return (
            event.duration_us,
            "observed-missing-collective-metadata",
            group_size,
            role,
        )
    supported = {
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "broadcast",
        "all_to_all",
        "reduce",
        "gather",
        "scatter",
        "barrier",
        "send",
        "recv",
    }
    if event.collective not in supported:
        return (
            event.duration_us,
            "observed-unsupported-collective-model",
            group_size,
            role,
        )
    message_bytes = 0 if event.collective == "barrier" else event.message_bytes
    assert message_bytes is not None
    calibration_group_size = 2 if event.collective in {"send", "recv"} else group_size
    calibration_network_tier = None if point_to_point is None else point_to_point[0]
    if calibration is not None:
        lookup = calibration.collective_duration(
            message_bytes,
            calibration_group_size,
            target.name,
            max_ctas,
            comm_stream_priority,
            nccl_cta_policy,
            nccl_nvls_ctas,
            nccl_cga_cluster_size,
            event.collective,
            calibration_network_tier,
        )
        if lookup is not None:
            return lookup.duration_us, lookup.source, group_size, role
    if point_to_point is not None:
        tier_name, _, _ = point_to_point
        tier = topology.intra_node if tier_name == "intra_node" else topology.inter_node
        duration, algorithm = _collective_tier_duration(
            event.collective,
            message_bytes,
            2,
            tier,
            event.metadata.get("message_bytes_semantics"),
        )
        return (
            duration,
            f"analytical-{algorithm}-{tier_name.replace('_', '-')}",
            group_size,
            role,
        )
    if group_size <= topology.gpus_per_node:
        duration, algorithm = _collective_tier_duration(
            event.collective,
            message_bytes,
            group_size,
            topology.intra_node,
            event.metadata.get("message_bytes_semantics"),
        )
        return (
            duration,
            f"analytical-{algorithm}-intra-node",
            group_size,
            role,
        )
    node_count = ceil(group_size / topology.gpus_per_node)
    local_ranks = min(group_size, topology.gpus_per_node)
    # Transparent fallback only. It does not claim to reproduce NCCL's
    # topology-dependent algorithm and protocol selection.
    local, local_algorithm = _collective_tier_duration(
        event.collective,
        message_bytes,
        local_ranks,
        topology.intra_node,
        event.metadata.get("message_bytes_semantics"),
    )
    inter, inter_algorithm = _collective_tier_duration(
        event.collective,
        message_bytes,
        node_count,
        topology.inter_node,
        event.metadata.get("message_bytes_semantics"),
    )
    return (
        local + inter,
        f"analytical-hierarchical-{local_algorithm}+{inter_algorithm}",
        group_size,
        role,
    )


def _prepare_events(
    trace: WorkloadTrace,
    parallelism: Parallelism,
    topology: Topology,
    target: Architecture,
    max_ctas: int | None,
    cta_policy: dict[str, int],
    calibration: OverlapCalibrationModel | None,
    comm_stream_priority: str,
    warnings: set[str],
    compute_transfer: ComputeTransferCalibration | None,
    nccl_cta_policy: int,
    nccl_nvls_ctas: int | None,
    nccl_cga_cluster_size: int,
) -> dict[str, PreparedEvent]:
    prepared: dict[str, PreparedEvent] = {}
    for event in trace.events:
        duration = event.duration_us
        source = "observed"
        variability = event.metadata.get("capture_duration_variability", {})
        unstable_capture = (
            isinstance(variability, dict) and variability.get("status") == "unstable"
        )
        if unstable_capture:
            warnings.add(
                f"{event.id}: identical qualified kernel signature had unstable repeated timing"
            )
        group_size = event.group_size
        group_role = event.group_role
        event_max_ctas = (
            cta_policy.get(group_role, max_ctas) if group_role is not None else max_ctas
        )
        source_target = trace.source.get("target")
        if (
            event.kind == "compute"
            and source_target not in (None, target.name)
            and compute_transfer is not None
            and not unstable_capture
        ):
            raw_signature = event.metadata.get("kernel_signature")
            lookup = compute_transfer.lookup(
                None if raw_signature is None else str(raw_signature),
                str(source_target),
                target.name,
                event.duration_us,
                trace.source,
            )
            if lookup is None:
                warnings.add(
                    f"{event.id}: compute event has no exact signature transfer from "
                    f"{source_target} to {target.name}"
                )
            else:
                duration = lookup.duration_us
                source = lookup.source
        elif (
            event.kind == "compute"
            and source_target not in (None, target.name)
            and compute_transfer is not None
            and unstable_capture
        ):
            warnings.add(
                f"{event.id}: cross-target compute transfer was skipped because capture timing is unstable"
            )
        if event.kind == "collective":
            duration, source, group_size, group_role = _collective_duration(
                event,
                parallelism,
                topology,
                target,
                event_max_ctas,
                calibration,
                comm_stream_priority,
                nccl_cta_policy,
                nccl_nvls_ctas,
                nccl_cga_cluster_size,
            )
            if source.startswith("observed-"):
                warnings.add(
                    f"{event.id}: collective used observed timing because its model metadata was incomplete"
                )
            elif source.startswith("analytical-"):
                warnings.add(
                    f"{event.id}: collective used an uncalibrated analytical fallback"
                )
        point_to_point = (
            _point_to_point_route(event, topology)
            if event.kind == "collective"
            else None
        )
        prepared[event.id] = PreparedEvent(
            event=event,
            isolated_duration_us=duration,
            duration_source=source,
            group_role=group_role,
            group_size=group_size,
            max_ctas=event_max_ctas if event.kind == "collective" else None,
            comm_stream_priority=(
                comm_stream_priority if event.kind == "collective" else None
            ),
            network_resources=(
                _collective_network_resources(event, group_size, topology)
                if event.kind == "collective"
                else ()
            ),
            collective_instance_id=_collective_instance_id(event),
            nccl_cta_policy=(nccl_cta_policy if event.kind == "collective" else None),
            nccl_nvls_ctas=(nccl_nvls_ctas if event.kind == "collective" else None),
            nccl_cga_cluster_size=(
                nccl_cga_cluster_size if event.kind == "collective" else None
            ),
            network_tier=(None if point_to_point is None else point_to_point[0]),
            p2p_source_device=(None if point_to_point is None else point_to_point[1]),
            p2p_destination_device=(
                None if point_to_point is None else point_to_point[2]
            ),
        )
    return prepared


def _duration_evidence(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    """Explain the authority and counterfactual scope of every duration."""

    entries = []
    counts: dict[str, int] = {}
    for event in sorted(timeline, key=lambda item: (item.start_us, item.id)):
        source = str(event.duration_source)
        if source.startswith("observed-missing-"):
            status = "abstained"
            scope = "observed-run-only"
            explanation = (
                "Required collective type, message, or group metadata is missing; "
                "the captured duration is retained but not scaled counterfactually."
            )
        elif source.startswith("observed-unsupported-"):
            status = "abstained"
            scope = "observed-run-only"
            explanation = (
                "No counterfactual timing model supports this collective; the "
                "captured duration is retained unchanged."
            )
        elif source.startswith("analytical-"):
            status = "uncalibrated-analytical"
            scope = "exploratory-counterfactual"
            explanation = (
                "A topology-aware analytical fallback supplies this duration, "
                "but no matching measured calibration validates its error."
            )
        elif source.startswith("interpolated-"):
            status = "interpolated-calibration"
            scope = "bounded-calibration-domain"
            explanation = (
                "This duration is interpolated between measured calibration points."
            )
        elif source.startswith("measured-"):
            status = "measured-calibration"
            scope = "matched-calibration-domain"
            explanation = (
                "A matching measured calibration record supplies this duration."
            )
        elif source == "observed":
            status = "observed-capture"
            scope = "captured-kernel-replay"
            explanation = (
                "The device duration is taken from the captured executable workload."
            )
        else:
            status = "modeled"
            scope = "source-specific"
            explanation = (
                "The named duration source supplies this value; consult its lock and "
                "event metadata for the exact domain."
            )
        counts[status] = counts.get(status, 0) + 1
        entries.append(
            {
                "event_id": event.id,
                "kind": event.kind,
                "duration_source": source,
                "status": status,
                "counterfactual_scope": scope,
                "explanation": explanation,
            }
        )
    limited = {
        "abstained",
        "uncalibrated-analytical",
        "interpolated-calibration",
    }
    return {
        "schema": "duration-evidence-v1",
        "status_counts": dict(sorted(counts.items())),
        "fully_measured_or_observed": not any(
            entry["status"] in limited for entry in entries
        ),
        "limited_event_count": sum(entry["status"] in limited for entry in entries),
        "events": entries,
    }


def _schedule(
    prepared: dict[str, PreparedEvent],
    target: Architecture,
    topology: Topology,
    trace_source: dict[str, Any],
    calibration: OverlapCalibrationModel | None,
    network_calibration: NetworkContentionCalibration | None,
    warnings: set[str],
) -> tuple[
    list[ScheduledEvent],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    remaining = set(prepared)
    completed: set[str] = set()
    dependents: dict[str, list[str]] = {event_id: [] for event_id in prepared}
    pending_dependency_counts: dict[str, int] = {}
    for event_id, item in prepared.items():
        missing = [
            dependency
            for dependency in item.event.dependencies
            if dependency not in prepared
        ]
        if missing:
            raise ValueError(
                f"event {event_id!r} references missing dependencies {missing}"
            )
        pending_dependency_counts[event_id] = len(item.event.dependencies)
        for dependency in item.event.dependencies:
            dependents[dependency].append(event_id)

    p2p_pair_members: dict[str, list[str]] = {}
    for event_id, item in prepared.items():
        pair_id = item.event.metadata.get("p2p_pair_id")
        if pair_id is None:
            continue
        if item.event.collective not in {"send", "recv"}:
            raise ValueError(
                f"event {event_id!r} supplies p2p_pair_id but is not Send or Recv"
            )
        value = str(pair_id).strip()
        if not value:
            raise ValueError(f"event {event_id!r} p2p_pair_id cannot be empty")
        p2p_pair_members.setdefault(value, []).append(event_id)

    p2p_partner: dict[str, str] = {}
    for pair_id, members in sorted(p2p_pair_members.items()):
        if len(members) != 2:
            raise ValueError(
                f"P2P pair {pair_id!r} requires exactly one Send and one Recv"
            )
        first, second = (prepared[event_id] for event_id in members)
        if {first.event.collective, second.event.collective} != {"send", "recv"}:
            raise ValueError(
                f"P2P pair {pair_id!r} requires exactly one Send and one Recv"
            )
        route_fields = (
            "p2p_source_device",
            "p2p_destination_device",
            "message_bytes",
            "network_resources",
        )
        first_route = (
            first.p2p_source_device,
            first.p2p_destination_device,
            first.event.message_bytes,
            first.network_resources,
        )
        second_route = (
            second.p2p_source_device,
            second.p2p_destination_device,
            second.event.message_bytes,
            second.network_resources,
        )
        if first_route != second_route:
            mismatch = ", ".join(
                name
                for name, left, right in zip(
                    route_fields, first_route, second_route, strict=True
                )
                if left != right
            )
            raise ValueError(
                f"P2P pair {pair_id!r} has inconsistent {mismatch}"
            )
        if first.event.rank == second.event.rank:
            raise ValueError(f"P2P pair {pair_id!r} endpoints must have distinct ranks")
        p2p_partner[members[0]] = members[1]
        p2p_partner[members[1]] = members[0]
    ready = {
        event_id
        for event_id, count in pending_dependency_counts.items()
        if count == 0
    }
    active: dict[str, float] = {}
    calibrated_rate_caps: dict[str, float] = {}
    active_streams: set[tuple[int, int, str]] = set()
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    interference_sources: dict[str, set[str]] = {
        event_id: set() for event_id in prepared
    }
    matched_pairs: dict[tuple[str, str, str], dict[str, Any]] = {}
    unresolved_admission_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    network_contention_regions: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    current_us = 0.0
    epsilon = 1.0e-9

    def resource_summary(event: TraceEvent) -> dict[str, Any]:
        resource = event.metadata.get("kernel_resource", {})
        if not isinstance(resource, dict):
            return {}
        block = resource.get("block")
        block_threads = None
        if isinstance(block, (list, tuple)) and len(block) == 3:
            try:
                block_threads = int(block[0]) * int(block[1]) * int(block[2])
            except (TypeError, ValueError):
                block_threads = None
        keys = (
            "grid_blocks",
            "registers_per_thread",
            "shared_memory_bytes",
            "blocks_per_sm",
            "max_active_blocks_per_sm",
            "warps_per_sm",
            "estimated_achieved_occupancy_percent",
            "grid_waves",
            "grid_sm_coverage_upper_bound",
        )
        summary = {key: resource[key] for key in keys if resource.get(key) is not None}
        if block_threads is not None:
            summary["block_threads"] = block_threads
        return summary

    def release_constraint(item: PreparedEvent) -> tuple[float, str]:
        host_release = item.event.metadata.get("host_release_us")
        if isinstance(host_release, (int, float)):
            return max(0.0, float(host_release)), "captured-host-release"
        if item.event.dependencies:
            # A device start timestamp includes queueing behind the dependency
            # and must not become a fixed counterfactual release constraint.
            return 0.0, "dependency-ready"
        if item.event.observed_start_us is None:
            return 0.0, "step-origin"
        return max(0.0, item.event.observed_start_us), "observed-root-start"

    def release_time(item: PreparedEvent) -> float:
        return release_constraint(item)[0]

    def ready_admission_groups() -> list[tuple[str, ...]]:
        groups: list[tuple[str, ...]] = []
        visited: set[str] = set()
        for event_id in sorted(ready):
            if event_id in visited:
                continue
            partner = p2p_partner.get(event_id)
            group = (
                (event_id,)
                if partner is None
                else tuple(sorted((event_id, partner)))
            )
            visited.update(group)
            if any(member not in ready for member in group):
                continue
            stream_keys = [prepared[member].stream_key for member in group]
            if len(set(stream_keys)) != len(stream_keys):
                pair_id = prepared[group[0]].event.metadata.get("p2p_pair_id")
                raise ValueError(
                    f"P2P pair {pair_id!r} endpoints cannot share a stream"
                )
            if any(stream_key in active_streams for stream_key in stream_keys):
                continue
            groups.append(group)
        return groups

    def group_release_time(group: tuple[str, ...]) -> float:
        return max(release_time(prepared[event_id]) for event_id in group)

    while len(completed) < len(prepared):
        # Launch every ready event whose stream is free. The trace carries
        # stream-order dependencies, but this guard also makes malformed traces
        # deterministic rather than allowing two events onto one stream.
        while True:
            candidates = [
                group
                for group in ready_admission_groups()
                if group_release_time(group) <= current_us + epsilon
            ]
            if not candidates:
                break
            group = min(
                candidates,
                key=lambda candidate: (group_release_time(candidate), candidate),
            )
            for event_id in group:
                item = prepared[event_id]
                remaining.remove(event_id)
                ready.remove(event_id)
                active[event_id] = item.isolated_duration_us
                calibrated_rate_caps.setdefault(event_id, 1.0)
                active_streams.add(item.stream_key)
                starts[event_id] = current_us

        if not active:
            future = [
                group_release_time(group)
                for group in ready_admission_groups()
            ]
            if not future:
                cycle = ", ".join(sorted(remaining))
                raise ValueError(
                    "trace dependency/P2P rendezvous graph contains a cycle "
                    f"involving {cycle}"
                )
            current_us = max(current_us, min(future))
            continue

        rates = {
            event_id: calibrated_rate_caps.get(event_id, 1.0) for event_id in active
        }
        compute_ids = [
            event_id
            for event_id in active
            if prepared[event_id].event.kind == "compute"
        ]
        collective_ids = [
            event_id
            for event_id in active
            if prepared[event_id].event.kind == "collective"
        ]
        resource_instances: dict[str, set[str]] = {}
        instance_items: dict[str, PreparedEvent] = {}
        for collective_id in collective_ids:
            item = prepared[collective_id]
            instance = item.collective_instance_id
            if instance is None:
                continue
            instance_items.setdefault(instance, item)
            for resource in item.network_resources:
                resource_instances.setdefault(resource, set()).add(instance)
        pending_contention: list[
            tuple[str, tuple[str, ...], tuple[str, ...], str, str, int | None]
        ] = []
        for resource, instances in resource_instances.items():
            if len(instances) <= 1:
                continue
            sorted_instances = tuple(sorted(instances))
            affected = tuple(
                sorted(
                    event_id
                    for event_id in collective_ids
                    if resource in prepared[event_id].network_resources
                )
            )
            operation_domains = [
                _network_contention_operation(instance_items[instance])
                for instance in sorted_instances
            ]
            contention_lookup = (
                None
                if network_calibration is None
                else network_calibration.lookup(
                    resource,
                    operation_domains,
                    target.name,
                    topology.name,
                    trace_source,
                )
            )
            if contention_lookup is None:
                pending_contention.append(
                    (
                        resource,
                        sorted_instances,
                        affected,
                        "unmatched",
                        "equal fluid fair share over the whole isolated event",
                        None,
                    )
                )
                for event_id in affected:
                    rates[event_id] = min(rates[event_id], 1.0 / len(instances))
                    interference_sources[event_id].add(
                        f"fluid-network-contention:{resource}:{len(instances)}-operations"
                    )
                warnings.add(
                    "overlapping collectives shared a routed network resource using the fluid fair-share approximation"
                )
            else:
                rates_by_instance = dict(
                    zip(sorted_instances, contention_lookup.progress_rates)
                )
                pending_contention.append(
                    (
                        resource,
                        sorted_instances,
                        affected,
                        "matched",
                        "measured exact-operation-set progress rates",
                        contention_lookup.samples,
                    )
                )
                for event_id in affected:
                    instance = prepared[event_id].collective_instance_id
                    assert instance is not None
                    rates[event_id] = min(rates[event_id], rates_by_instance[instance])
                    interference_sources[event_id].add(
                        f"{contention_lookup.source}:{resource}:{len(instances)}-operations"
                    )
        for compute_id in compute_ids:
            compute = prepared[compute_id]
            signature = event_compute_identity(compute.event)
            for collective_id in collective_ids:
                collective = prepared[collective_id]
                if (
                    compute.event.rank,
                    compute.event.device,
                ) != (
                    collective.event.rank,
                    collective.event.device,
                ):
                    # Device-resource interference is local. Concurrent work
                    # observed on another rank/GPU must not consume this GPU's
                    # progress budget.
                    continue
                message_bytes = collective.event.message_bytes
                group_size = collective.group_size
                lookup = None
                if (
                    calibration is not None
                    and message_bytes is not None
                    and group_size is not None
                ):
                    calibration_group_size = (
                        2
                        if collective.event.collective in {"send", "recv"}
                        else group_size
                    )
                    calibration_network_tier = collective.network_tier
                    lookup = calibration.interference(
                        message_bytes,
                        calibration_group_size,
                        target.name,
                        collective.max_ctas,
                        signature,
                        collective.comm_stream_priority or "normal",
                        collective.nccl_cta_policy or 0,
                        collective.nccl_nvls_ctas,
                        collective.nccl_cga_cluster_size or 0,
                        collective.event.collective or "unknown",
                        calibration_network_tier,
                    )
                if lookup is None:
                    warnings.add(
                        "an overlapping compute/collective pair was outside the interference calibration domain"
                    )
                    resource = compute.event.metadata.get("kernel_resource", {})
                    raw_grid_blocks = (
                        resource.get("grid_blocks")
                        if isinstance(resource, dict)
                        else None
                    )
                    try:
                        source_grid_blocks = (
                            None if raw_grid_blocks is None else int(raw_grid_blocks)
                        )
                    except (TypeError, ValueError):
                        source_grid_blocks = None
                    compute_hardware_domain = _kernel_resource_hardware_domain(
                        compute.event, trace_source, target
                    )
                    grid_blocks = (
                        source_grid_blocks
                        if compute_hardware_domain["status"] == "exact-product-match"
                        else None
                    )
                    sm_count = target.sm_count
                    minimum_grid_free_sms = (
                        None if grid_blocks is None else max(0, sm_count - grid_blocks)
                    )
                    requested_ctas = collective.max_ctas
                    compute_resources = resource_summary(compute.event)
                    collective_resources = resource_summary(collective.event)
                    sm_footprint = _nccl_sm_footprint(
                        collective.event,
                        target,
                        requested_ctas,
                        collective.nccl_cga_cluster_size,
                        trace_source,
                    )
                    footprint_min_sms = (
                        None
                        if sm_footprint is None
                        else sm_footprint.get(
                            "full_grid_min_sms_if_concurrently_resident"
                        )
                    )
                    footprint_max_sms = (
                        None
                        if sm_footprint is None
                        else sm_footprint.get(
                            "full_grid_max_sms_if_concurrently_resident"
                        )
                    )
                    if minimum_grid_free_sms is None or footprint_max_sms is None:
                        aggregate_test = "unknown"
                    elif minimum_grid_free_sms >= footprint_max_sms:
                        aggregate_test = (
                            "full-grid-upper-bound-count-sufficient-but-not-predictive"
                        )
                    elif footprint_min_sms is None:
                        aggregate_test = "indeterminate-missing-nccl-occupancy"
                    elif minimum_grid_free_sms >= footprint_min_sms:
                        aggregate_test = (
                            "full-grid-count-possible-only-with-cta-packing"
                        )
                    else:
                        aggregate_test = (
                            "full-grid-count-insufficient-in-guaranteed-free-subset"
                        )
                    unresolved_admission_pairs[(compute_id, collective_id)] = {
                        "compute_event": compute_id,
                        "collective_event": collective_id,
                        "status": "unresolved-no-exact-interference-calibration",
                        "target": target.name,
                        "compute_grid_blocks": grid_blocks,
                        "source_observed_compute_grid_blocks": source_grid_blocks,
                        "hardware_evidence_domain": compute_hardware_domain,
                        "compute_kernel_signature": signature,
                        "compute_resources": compute_resources,
                        "device_sm_count": sm_count,
                        "minimum_grid_free_sms": minimum_grid_free_sms,
                        "minimum_grid_free_sms_semantics": (
                            "instantaneous-count-lower-bound-not-a-fixed-sm-subset"
                            if minimum_grid_free_sms is not None
                            else "unknown"
                        ),
                        "requested_nccl_ctas": requested_ctas,
                        "nccl_sm_footprint": sm_footprint,
                        "observed_nccl_grid_blocks": collective_resources.get(
                            "grid_blocks"
                        ),
                        "collective_kernel_signature": collective.event.metadata.get(
                            "kernel_signature"
                        ),
                        "collective_resources": collective_resources,
                        "aggregate_capacity_test": aggregate_test,
                        "interpretation": (
                            "full-grid resident-SM bounds and the instantaneous count of "
                            "SMs without compute blocks do not identify a fixed free-SM "
                            "subset or establish achieved NCCL placement, partial "
                            "progress, or slowdown; exact overlap evidence remains required"
                        ),
                    }
                    continue
                region_event_ids: list[str] = []
                rebase = compute.event.metadata.get("captured_overlap_rebase", {})
                region_id = (
                    rebase.get("region_id") if isinstance(rebase, dict) else None
                )
                if region_id is not None and lookup.compute_event_count is not None:
                    # The measured ratio belongs to the declared compute
                    # region, including when the measured execution was
                    # serialized. Preserve the complete region identity in
                    # the boundary-compatible prediction.
                    region_event_ids = [
                        event_id
                        for event_id, member in prepared.items()
                        if isinstance(
                            member.event.metadata.get("captured_overlap_rebase", {}),
                            dict,
                        )
                        and member.event.metadata.get(
                            "captured_overlap_rebase", {}
                        ).get("region_id")
                        == region_id
                    ]
                    if len(region_event_ids) != lookup.compute_event_count:
                        raise ValueError(
                            f"{compute_id}: calibrated compute region count "
                            "does not match the rebased captured region"
                        )
                elif lookup.compute_event_count is not None:
                    region = compute_region_metadata(compute.event)
                    if region is not None:
                        members = compute_region_members(
                            (item.event for item in prepared.values()),
                            compute.event,
                        )
                        if len(members) != lookup.compute_event_count:
                            raise ValueError(
                                f"{compute_id}: calibrated compute region count "
                                "does not match the captured region"
                            )
                        region_event_ids = [event.id for event in members]
                if lookup.regime == "serialized_compute_first":
                    rates[compute_id] = min(rates[compute_id], 1.0)
                    rates[collective_id] = 0.0
                else:
                    calibrated_rate_caps[compute_id] = min(
                        calibrated_rate_caps.get(compute_id, 1.0),
                        1.0 / lookup.compute_slowdown,
                    )
                    if region_event_ids:
                        # Keep the measured region slowdown on members that
                        # launch after the collective has completed.
                        for event_id in region_event_ids:
                            calibrated_rate_caps[event_id] = min(
                                calibrated_rate_caps.get(event_id, 1.0),
                                1.0 / lookup.compute_slowdown,
                            )
                    calibrated_rate_caps[collective_id] = min(
                        calibrated_rate_caps.get(collective_id, 1.0),
                        1.0 / lookup.collective_slowdown,
                    )
                    rates[compute_id] = min(
                        rates[compute_id], calibrated_rate_caps[compute_id]
                    )
                    rates[collective_id] = min(
                        rates[collective_id], calibrated_rate_caps[collective_id]
                    )
                source = (
                    f"{lookup.source}:{compute_id}+{collective_id}:"
                    f"{lookup.lower_bytes}-{lookup.upper_bytes}B"
                )
                interference_sources[compute_id].add(source)
                interference_sources[collective_id].add(source)
                matched_pairs[(compute_id, collective_id, source)] = {
                    "compute_event": compute_id,
                    "collective_event": collective_id,
                    "compute_slowdown": lookup.compute_slowdown,
                    "collective_slowdown": lookup.collective_slowdown,
                    "regime": lookup.regime,
                    "source": lookup.source,
                    "lower_bytes": lookup.lower_bytes,
                    "upper_bytes": lookup.upper_bytes,
                    "application": (
                        "sticky-whole-compute-region-effective-slowdown-v0"
                        if region_event_ids
                        else "sticky-whole-event-effective-slowdown-v0"
                    ),
                    "compute_region_event_ids": region_event_ids,
                    "compute_alone_us": lookup.compute_alone_us,
                    "collective_alone_us": lookup.collective_alone_us,
                    "overlap_device_us": lookup.overlap_device_us,
                    "validation_evidence_status": (lookup.validation_evidence_status),
                    "validation_evidence_sha256": (lookup.validation_evidence_sha256),
                    "validation_timeline_outcome": (lookup.validation_timeline_outcome),
                }

        completion_delta = min(
            (
                active[event_id] / rates[event_id]
                if rates[event_id] > 0
                else float("inf")
            )
            for event_id in active
        )
        future_releases = [
            group_release_time(group) - current_us
            for group in ready_admission_groups()
            if group_release_time(group) > current_us + epsilon
        ]
        delta_us = min(completion_delta, min(future_releases, default=float("inf")))
        if delta_us < 0 or delta_us == float("inf"):
            raise RuntimeError("event scheduler failed to make forward progress")
        for event_id in active:
            active[event_id] = max(0.0, active[event_id] - delta_us * rates[event_id])
        for resource, instances, affected, status, rule, samples in pending_contention:
            key = (resource, instances)
            region = network_contention_regions.setdefault(
                key,
                {
                    "resource": resource,
                    "collective_instances": list(instances),
                    "operation_count": len(instances),
                    "event_ids": list(affected),
                    "contended_wall_time_us": 0.0,
                    "sharing_rule": rule,
                    "calibration_status": status,
                    "calibration_samples": samples,
                },
            )
            region["contended_wall_time_us"] += delta_us
        current_us += delta_us

        finished = [
            event_id
            for event_id, remaining_work in active.items()
            if remaining_work <= epsilon
        ]
        for event_id in finished:
            item = prepared[event_id]
            del active[event_id]
            calibrated_rate_caps.pop(event_id, None)
            active_streams.remove(item.stream_key)
            completed.add(event_id)
            ends[event_id] = current_us
            for dependent in dependents[event_id]:
                if dependent not in remaining:
                    continue
                pending_dependency_counts[dependent] -= 1
                if pending_dependency_counts[dependent] == 0:
                    ready.add(dependent)

    timeline = []
    for event_id, item in prepared.items():
        wall_duration = ends[event_id] - starts[event_id]
        isolated = item.isolated_duration_us
        region_metadata = compute_region_metadata(item.event)
        timeline.append(
            ScheduledEvent(
                id=event_id,
                name=item.event.name,
                kind=item.event.kind,
                stream=item.event.stream,
                start_us=starts[event_id],
                end_us=ends[event_id],
                duration_us=wall_duration,
                isolated_duration_us=isolated,
                effective_slowdown=(1.0 if isolated == 0 else wall_duration / isolated),
                duration_source=item.duration_source,
                launch_release_us=release_constraint(item)[0],
                launch_release_source=release_constraint(item)[1],
                dependencies=item.event.dependencies,
                dependency_evidence=tuple(
                    dict(value)
                    for value in (
                        *item.event.metadata.get("cuda_event_dependencies", ()),
                        *item.event.metadata.get(
                            "cuda_host_synchronization_dependencies", ()
                        ),
                        *item.event.metadata.get("collective_work_dependencies", ()),
                    )
                    if isinstance(value, dict)
                ),
                interference_sources=tuple(sorted(interference_sources[event_id])),
                collective=item.event.collective,
                group_role=item.group_role,
                group_size=item.group_size,
                message_bytes=item.event.message_bytes,
                max_ctas=item.max_ctas,
                comm_stream_priority=item.comm_stream_priority,
                rank=item.event.rank,
                device=item.event.device,
                sm_fraction=item.event.sm_fraction,
                kernel_signature=(
                    None
                    if item.event.metadata.get("kernel_signature") is None
                    else str(item.event.metadata["kernel_signature"])
                ),
                cuda_stream_priority_binding=(
                    dict(item.event.metadata["cuda_stream_priority_binding"])
                    if isinstance(
                        item.event.metadata.get("cuda_stream_priority_binding"), dict
                    )
                    else None
                ),
                compute_region=(
                    None if region_metadata is None else dict(region_metadata)
                ),
                overlap_region=(
                    dict(item.event.metadata["overlap_region_marker"])
                    if isinstance(
                        item.event.metadata.get("overlap_region_marker"), dict
                    )
                    else None
                ),
                collective_region=(
                    dict(item.event.metadata["collective_region_marker"])
                    if isinstance(
                        item.event.metadata.get("collective_region_marker"), dict
                    )
                    else None
                ),
                kernel_resource=dict(item.event.metadata.get("kernel_resource", {})),
                network_resources=item.network_resources,
                collective_instance_id=item.collective_instance_id,
                nccl_cta_policy=item.nccl_cta_policy,
                nccl_cta_policy_name=(
                    None
                    if item.nccl_cta_policy is None
                    else cta_policy_name(item.nccl_cta_policy)
                ),
                nccl_nvls_ctas=item.nccl_nvls_ctas,
                nccl_cga_cluster_size=item.nccl_cga_cluster_size,
                kernel_sm_footprint=_kernel_sm_footprint(
                    item.event, target, trace_source
                ),
                nccl_sm_footprint=_nccl_sm_footprint(
                    item.event,
                    target,
                    item.max_ctas,
                    item.nccl_cga_cluster_size,
                    trace_source,
                ),
                zero_cta_eligible=(
                    None
                    if item.event.kind != "collective"
                    else bool(
                        item.event.metadata.get("symmetric_registered_buffers", False)
                        and item.event.collective in {"all_gather", "all_to_all"}
                    )
                ),
                network_tier=item.network_tier,
                p2p_source_device=item.p2p_source_device,
                p2p_destination_device=item.p2p_destination_device,
            )
        )
    return (
        timeline,
        list(matched_pairs.values()),
        list(unresolved_admission_pairs.values()),
        list(network_contention_regions.values()),
    )


def _calibrated_overlap_region_predictions(
    timeline: list[ScheduledEvent],
    matched_pairs: list[dict[str, Any]],
    calibration: OverlapCalibrationModel | None,
    target: Architecture,
) -> list[dict[str, Any]]:
    """Expose measurement-boundary-compatible overlap predictions.

    The full trace scheduler preserves captured host release constraints and is
    an iteration/step prediction.  An overlap microbenchmark instead measures
    from a common CUDA event gate until one declared compute region and one
    collective have completed.  Exact overlap calibration already records that
    makespan; keeping it as a separate, provenance-rich result prevents callers
    from comparing unlike timing boundaries.
    """

    by_id = {event.id: event for event in timeline}
    grouped: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = {}
    abstentions: list[dict[str, Any]] = []
    explicit_groups: dict[str, list[ScheduledEvent]] = {}
    for event in timeline:
        marker = event.overlap_region
        if marker is None:
            continue
        required_marker_fields = {
            "schema",
            "name",
            "instance_id",
            "occurrence",
            "host_duration_us",
            "membership_evidence",
        }
        if (
            required_marker_fields - set(marker)
            or marker.get("schema") != "overlap-region-marker-v1"
            or not isinstance(marker.get("name"), str)
            or not str(marker["name"]).strip()
            or any(character in str(marker["name"]) for character in "=,")
            or not isinstance(marker.get("occurrence"), int)
            or int(marker["occurrence"]) < 0
            or not isinstance(marker.get("host_duration_us"), (int, float))
            or float(marker["host_duration_us"]) <= 0.0
            or marker.get("membership_evidence")
            != "correlated-cuda-launch-inside-overlap-marker-v1"
        ):
            raise ValueError(f"{event.id}: overlap-region marker evidence is invalid")
        instance_id = marker.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"{event.id}: overlap-region marker lacks an instance ID")
        explicit_groups.setdefault(instance_id, []).append(event)

    explicit_keys: set[tuple[tuple[str, ...], str]] = set()
    for instance_id, events in sorted(explicit_groups.items()):
        markers = [event.overlap_region for event in events]
        if any(marker != markers[0] for marker in markers[1:]):
            raise ValueError(
                f"overlap-region marker {instance_id!r} has inconsistent evidence"
            )
        computes = sorted(
            (event for event in events if event.kind == "compute"),
            key=lambda event: (event.start_us, event.id),
        )
        all_collectives = [event for event in events if event.kind == "collective"]
        explicitly_selected = [
            event for event in all_collectives if event.collective_region is not None
        ]
        if explicitly_selected:
            if len(explicitly_selected) != 1:
                raise ValueError(
                    f"overlap-region marker {instance_id!r} has multiple "
                    "explicitly selected collectives"
                )
            selection = explicitly_selected[0].collective_region or {}
            if (
                selection.get("schema") != "collective-region-marker-v1"
                or not isinstance(selection.get("name"), str)
                or not str(selection["name"]).strip()
                or selection.get("membership_evidence")
                != "correlated-cuda-launch-inside-collective-marker-v1"
            ):
                raise ValueError(
                    f"overlap-region marker {instance_id!r} has invalid "
                    "explicit collective evidence"
                )
            collectives = explicitly_selected
        else:
            collectives = all_collectives
        if not computes or len(collectives) != 1:
            raise ValueError(
                f"overlap-region marker {instance_id!r} requires compute work "
                "and exactly one collective"
            )
        collective = collectives[0]
        if len({(event.rank, event.device) for event in events}) != 1:
            raise ValueError(
                f"overlap-region marker {instance_id!r} mixes rank/device domains"
            )
        identities = {
            str(event.compute_region["signature"])
            if event.compute_region is not None
            else event.kernel_signature
            for event in computes
        }
        identities.discard(None)
        compute_ids = tuple(event.id for event in computes)
        key = (compute_ids, collective.id)
        explicit_keys.add(key)
        marker = markers[0] or {}
        if (
            calibration is None
            or len(identities) != 1
            or collective.message_bytes is None
            or collective.group_size is None
            or collective.collective is None
        ):
            abstentions.append(
                {
                    "status": "abstained-no-exact-calibration-domain",
                    "scope": "one-declared-compute-region-plus-one-collective",
                    "compute_event_ids": list(compute_ids),
                    "collective_event": collective.id,
                    "overlap_region_marker": dict(marker),
                }
            )
            continue
        lookup = calibration.interference(
            collective.message_bytes,
            collective.group_size,
            target.name,
            collective.max_ctas,
            next(iter(identities)),
            collective.comm_stream_priority or "normal",
            collective.nccl_cta_policy or 0,
            collective.nccl_nvls_ctas,
            collective.nccl_cga_cluster_size or 0,
            collective.collective,
            collective.network_tier,
        )
        if lookup is None or (
            lookup.compute_event_count is not None
            and lookup.compute_event_count != len(computes)
        ):
            abstentions.append(
                {
                    "status": "abstained-no-exact-calibration-domain",
                    "scope": "one-declared-compute-region-plus-one-collective",
                    "compute_event_ids": list(compute_ids),
                    "collective_event": collective.id,
                    "overlap_region_marker": dict(marker),
                }
            )
            continue
        grouped[key] = [
            {
                "compute_event": computes[0].id,
                "collective_event": collective.id,
                "compute_region_event_ids": list(compute_ids),
                "compute_slowdown": lookup.compute_slowdown,
                "collective_slowdown": lookup.collective_slowdown,
                "regime": lookup.regime,
                "source": lookup.source,
                "lower_bytes": lookup.lower_bytes,
                "upper_bytes": lookup.upper_bytes,
                "compute_alone_us": lookup.compute_alone_us,
                "collective_alone_us": lookup.collective_alone_us,
                "overlap_device_us": lookup.overlap_device_us,
                "overlap_region_marker": dict(marker),
                "boundary_identity_source": "explicit-profiler-marker-v1",
            }
        ]
    for match in matched_pairs:
        predicted = match.get("overlap_device_us")
        if not isinstance(predicted, (int, float)) or float(predicted) <= 0.0:
            continue
        compute_ids = tuple(match.get("compute_region_event_ids") or ())
        if not compute_ids:
            compute_event = match.get("compute_event")
            if not isinstance(compute_event, str) or not compute_event:
                continue
            compute_ids = (compute_event,)
        collective_id = match.get("collective_event")
        if not isinstance(collective_id, str) or not collective_id:
            continue
        key = (compute_ids, collective_id)
        if key not in explicit_keys:
            grouped.setdefault(key, []).append(match)

    predictions = []
    for index, ((compute_ids, collective_id), matches) in enumerate(
        sorted(grouped.items(), key=lambda item: item[0])
    ):
        fields = (
            "overlap_device_us",
            "compute_alone_us",
            "collective_alone_us",
            "source",
            "lower_bytes",
            "upper_bytes",
            "regime",
        )
        reference = matches[0]
        if any(
            any(match.get(field) != reference.get(field) for field in fields)
            for match in matches[1:]
        ):
            # Multiple incompatible calibrations for the same declared region
            # are not combined into a point prediction.
            continue
        event_ids = (*compute_ids, collective_id)
        if any(event_id not in by_id for event_id in event_ids):
            continue
        events = [by_id[event_id] for event_id in event_ids]
        compute_events = [by_id[event_id] for event_id in compute_ids]
        collective = by_id[collective_id]
        start_us = min(event.start_us for event in events)
        end_us = max(event.end_us for event in events)
        lower_bytes = int(reference["lower_bytes"])
        upper_bytes = int(reference["upper_bytes"])
        predictions.append(
            {
                "id": f"overlap-region-{index:04d}",
                "status": "calibrated",
                "scope": "one-declared-compute-region-plus-one-collective",
                "measurement_boundary": "common-cuda-event-gate-to-both-operations-complete",
                "excluded_from_boundary": [
                    "captured host launch release skew",
                    "setup kernels outside the declared compute region",
                    "other step events",
                ],
                "model": "calibrated-overlap-device-makespan-v1",
                "predicted_device_us": float(reference["overlap_device_us"]),
                "full_trace_scheduler_window_us": end_us - start_us,
                "compute_event_ids": list(compute_ids),
                "collective_event": collective_id,
                "rank": compute_events[0].rank,
                "overlap_region_marker": reference.get("overlap_region_marker"),
                "boundary_identity_source": reference.get(
                    "boundary_identity_source", "scheduled-interference-match-v1"
                ),
                "device": compute_events[0].device,
                "collective": collective.collective,
                "message_bytes": collective.message_bytes,
                "compute_alone_us": float(reference["compute_alone_us"]),
                "collective_alone_us": float(reference["collective_alone_us"]),
                "overlap_regime": reference["regime"],
                "calibration_source": reference["source"],
                "lower_bytes": lower_bytes,
                "upper_bytes": upper_bytes,
                "lookup": "exact" if lower_bytes == upper_bytes else "interpolated",
                "matched_pair_count": len(matches),
            }
        )
    for abstention in abstentions:
        abstention["id"] = f"overlap-region-{len(predictions):04d}"
        abstention["measurement_boundary"] = (
            "common-cuda-event-gate-to-both-operations-complete"
        )
        predictions.append(abstention)
    return predictions


def _observed_overlap(left: TraceEvent, right: TraceEvent) -> bool:
    if left.observed_start_us is None or right.observed_start_us is None:
        return False
    return (
        left.observed_start_us < right.observed_start_us + right.duration_us
        and right.observed_start_us < left.observed_start_us + left.duration_us
    )


def _rebase_captured_overlap(
    trace: WorkloadTrace,
    parallelism: Parallelism,
    calibration: OverlapCalibrationModel | None,
    warnings: set[str],
) -> tuple[WorkloadTrace, dict[str, Any]]:
    """Recover isolated compute time from a verified captured overlap policy."""

    validate_compute_regions(trace.events)
    computes = [event for event in trace.events if event.kind == "compute"]
    collectives = [event for event in trace.events if event.kind == "collective"]
    overlap_pairs = [
        (compute, collective)
        for compute in computes
        for collective in collectives
        if compute.rank == collective.rank
        and compute.device == collective.device
        and _observed_overlap(compute, collective)
    ]
    policy = trace.source.get("capture_policy")
    report: dict[str, Any] = {
        "method": "verified-capture-policy-deslow-v0",
        "observed_overlap_pair_count": len(overlap_pairs),
        "capture_policy": policy,
        "rebased_compute_event_count": 0,
        "rebased_compute_region_count": 0,
        "unmatched_compute_event_count": 0,
        "ambiguous_compute_event_count": 0,
    }
    if not overlap_pairs:
        report["status"] = "no-observed-overlap"
        return trace, report
    if calibration is None:
        report["status"] = "no-calibration"
        return trace, report
    if not isinstance(policy, dict) or policy.get("verified") is not True:
        warnings.add(
            "captured compute overlaps communication but its source NCCL policy is not verified; durations were not de-slowed"
        )
        report["status"] = "unverified-capture-policy"
        return trace, report

    capture_max_ctas = policy.get("max_ctas")
    capture_priority = str(policy.get("stream_priority", ""))
    capture_cta_policy = policy.get("cta_policy_flag")
    capture_nvls = policy.get("nvls_ctas")
    capture_cga = policy.get("cga_cluster_size", 0)
    if capture_max_ctas is not None and (
        not isinstance(capture_max_ctas, int) or capture_max_ctas <= 0
    ):
        raise ValueError("captured max CTAs must be positive or default")
    if capture_priority not in {"normal", "high"}:
        raise ValueError("captured stream priority must be normal or high")
    capture_cta_policy = parse_cta_policy_mode(capture_cta_policy)
    if capture_nvls is not None and (
        not isinstance(capture_nvls, int) or capture_nvls <= 0
    ):
        raise ValueError("captured NVLS CTAs must be positive or automatic")
    capture_cga = parse_cga_cluster_size(capture_cga)
    capture_cga = 0 if capture_cga is None else capture_cga

    by_compute: dict[str, list[TraceEvent]] = {}
    for compute, collective in overlap_pairs:
        by_compute.setdefault(compute.id, []).append(collective)
    replacements: dict[str, TraceEvent] = {}
    handled_compute_ids: set[str] = set()
    for compute in computes:
        if compute.id in handled_compute_ids:
            continue
        overlapping = by_compute.get(compute.id, [])
        if not overlapping:
            continue
        lookups = []
        for collective in overlapping:
            group_size = collective.group_size
            if group_size is None and collective.group_role is not None:
                group_size = parallelism.group_size(collective.group_role)
            signature = event_compute_identity(compute)
            if collective.message_bytes is None or group_size is None:
                continue
            lookup = calibration.interference(
                collective.message_bytes,
                group_size,
                str(trace.source.get("target")),
                capture_max_ctas,
                signature,
                capture_priority,
                capture_cta_policy,
                capture_nvls,
                capture_cga,
                collective.collective or "unknown",
            )
            if lookup is not None:
                lookups.append((collective, lookup))
        if not lookups:
            report["unmatched_compute_event_count"] += 1
            warnings.add(
                f"{compute.id}: no exact source-policy overlap calibration could de-slow captured compute timing"
            )
            continue
        if len(lookups) != 1:
            report["ambiguous_compute_event_count"] += 1
            warnings.add(
                f"{compute.id}: multiple captured collectives make compute de-slowing ambiguous"
            )
            continue
        collective, lookup = lookups[0]
        if lookup.compute_event_count is not None:
            region_metadata = compute_region_metadata(compute)
            if region_metadata is not None:
                region = list(compute_region_members(computes, compute))
                if any(event.id in handled_compute_ids for event in region):
                    report["ambiguous_compute_event_count"] += 1
                    warnings.add(
                        f"{compute.id}: compute region overlaps an already rebased region"
                    )
                    continue
            else:
                signature = compute.metadata.get("kernel_signature")
                region = sorted(
                    (
                        event
                        for event in computes
                        if event.rank == compute.rank
                        and event.device == compute.device
                        and event.stream == compute.stream
                        and event.metadata.get("kernel_signature") == signature
                        and event.observed_start_us is not None
                        and compute.observed_start_us is not None
                        and event.observed_start_us >= compute.observed_start_us
                        and event.id not in handled_compute_ids
                    ),
                    key=lambda event: (event.observed_start_us, event.id),
                )[: lookup.compute_event_count]
            if len(region) != lookup.compute_event_count:
                report["unmatched_compute_event_count"] += 1
                warnings.add(
                    f"{compute.id}: calibrated compute region expected "
                    f"{lookup.compute_event_count} events but capture did not contain them"
                )
                continue
            observed_region_duration = sum(event.duration_us for event in region)
            if observed_region_duration <= 0.0:
                report["unmatched_compute_event_count"] += 1
                continue
            region_id = (
                str(region_metadata["instance_id"])
                if region_metadata is not None
                else f"{collective.id}::compute-region::{compute.id}"
            )
            for region_index, region_event in enumerate(region):
                isolated_duration = (
                    lookup.compute_alone_us
                    * region_event.duration_us
                    / observed_region_duration
                )
                metadata = dict(region_event.metadata)
                metadata["captured_overlap_rebase"] = {
                    "status": "rebased-calibrated-compute-region",
                    "source_collective_event_id": collective.id,
                    "region_id": region_id,
                    "region_event_index": region_index,
                    "region_event_count": lookup.compute_event_count,
                    "observed_region_duration_us": observed_region_duration,
                    "isolated_region_duration_us": lookup.compute_alone_us,
                    "observed_duration_us": region_event.duration_us,
                    "isolated_duration_us": isolated_duration,
                    "source_compute_slowdown": lookup.compute_slowdown,
                    "source_overlap_regime": lookup.regime,
                    "source": lookup.source,
                    "capture_policy": policy,
                }
                replacements[region_event.id] = replace(
                    region_event,
                    duration_us=isolated_duration,
                    metadata=metadata,
                )
                handled_compute_ids.add(region_event.id)
            report["rebased_compute_event_count"] += len(region)
            report["rebased_compute_region_count"] += 1
            continue
        isolated_duration = compute.duration_us / lookup.compute_slowdown
        metadata = dict(compute.metadata)
        metadata["captured_overlap_rebase"] = {
            "status": "rebased",
            "source_collective_event_id": collective.id,
            "observed_duration_us": compute.duration_us,
            "isolated_duration_us": isolated_duration,
            "source_compute_slowdown": lookup.compute_slowdown,
            "source_overlap_regime": lookup.regime,
            "source": lookup.source,
            "capture_policy": policy,
        }
        replacements[compute.id] = replace(
            compute, duration_us=isolated_duration, metadata=metadata
        )
        report["rebased_compute_event_count"] += 1
        handled_compute_ids.add(compute.id)
    events = tuple(replacements.get(event.id, event) for event in trace.events)
    report["status"] = (
        "rebased"
        if report["rebased_compute_event_count"]
        else "no-exact-source-policy-match"
    )
    metadata = dict(trace.metadata)
    metadata["captured_overlap_rebase"] = report
    return replace(trace, events=events, metadata=metadata), report


def _capture_completeness(trace: WorkloadTrace) -> dict[str, Any]:
    affected_events: list[dict[str, Any]] = []
    event_limitation_count = 0
    for event in trace.events:
        raw_limitations = event.metadata.get("capture_limitations", ())
        if isinstance(raw_limitations, str):
            raw_limitations = (raw_limitations,)
        limitations = sorted(
            {str(item).strip() for item in raw_limitations if str(item).strip()}
        )
        if not limitations:
            continue
        event_limitation_count += len(limitations)
        affected_events.append(
            {
                "event_id": event.id,
                "event_name": event.name,
                "limitations": limitations,
            }
        )

    graph_launches = []
    for launch in trace.metadata.get("cuda_graph_launches", ()):
        if not isinstance(launch, dict):
            continue
        graph_launches.append(dict(launch))
    supported_graph_dependency_statuses = {
        "single-stream-order-preserved",
        "native-graph-topology-observed",
    }
    unsupported_graph_launches = [
        launch
        for launch in graph_launches
        if launch.get("timing_replay_eligible") is not True
        or launch.get("dependency_status") not in supported_graph_dependency_statuses
    ]
    raw_trace_limitations = trace.metadata.get("capture_limitations", ())
    if isinstance(raw_trace_limitations, str):
        raw_trace_limitations = (raw_trace_limitations,)
    trace_limitations = sorted(
        {str(item).strip() for item in raw_trace_limitations if str(item).strip()}
    )
    capture_artifact_semantic_scan = trace.metadata.get(
        "capture_artifact_semantic_scan"
    )
    loaded_native_semantic_scan = trace.metadata.get("loaded_native_semantic_scan")
    model_child_process_audit = trace.metadata.get("model_child_process_audit")
    if isinstance(model_child_process_audit, dict):
        child_status = model_child_process_audit.get("status")
        if child_status not in {None, "no-child-process-operations-observed"}:
            limitation = "model child-process activity could escape the profiled worker"
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    model_background_thread_audit = trace.metadata.get("model_background_thread_audit")
    if isinstance(model_background_thread_audit, dict):
        thread_status = model_background_thread_audit.get("status")
        if thread_status not in {
            None,
            "no-background-thread-survived-entrypoint",
        }:
            limitation = (
                "model background-thread activity could escape the profiled entrypoint"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    model_ctypes_native_library_lifetime_audit = trace.metadata.get(
        "model_ctypes_native_library_lifetime_audit"
    )
    if isinstance(model_ctypes_native_library_lifetime_audit, dict):
        ctypes_status = model_ctypes_native_library_lifetime_audit.get("status")
        if ctypes_status in {
            "audit-unavailable",
            "audit-incomplete",
            "deepbind-observed",
        }:
            limitation = (
                "model ctypes native-library lifetime audit could miss a transient "
                "accelerator library"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    model_native_loader_audit = trace.metadata.get("model_native_loader_audit")
    distributed_model_native_loader_audit = trace.metadata.get(
        "distributed_model_native_loader_audit"
    )
    effective_native_loader_audit = (
        model_native_loader_audit
        if isinstance(model_native_loader_audit, dict)
        else distributed_model_native_loader_audit
    )
    if isinstance(effective_native_loader_audit, dict):
        loader_status = effective_native_loader_audit.get("status")
        if loader_status in {"probe-unavailable", "audit-incomplete"}:
            limitation = (
                "model native-loader audit could miss a transient accelerator library"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
        if int(effective_native_loader_audit.get("deepbind_call_count", 0)) > 0:
            limitation = "native RTLD_DEEPBIND load may bypass preload capture"
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
        if int(effective_native_loader_audit.get("new_namespace_call_count", 0)) > 0:
            limitation = "new native loader namespace may bypass preload capture"
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    model_native_symbol_resolution_audit = trace.metadata.get(
        "model_native_symbol_resolution_audit"
    )
    distributed_model_native_symbol_resolution_audit = trace.metadata.get(
        "distributed_model_native_symbol_resolution_audit"
    )
    effective_native_symbol_resolution_audit = (
        model_native_symbol_resolution_audit
        if isinstance(model_native_symbol_resolution_audit, dict)
        else distributed_model_native_symbol_resolution_audit
    )
    if isinstance(effective_native_symbol_resolution_audit, dict):
        resolver_status = effective_native_symbol_resolution_audit.get("status")
        if resolver_status in {"probe-unavailable", "audit-incomplete"}:
            limitation = (
                "model native symbol-resolution audit could miss indirect "
                "accelerator dispatch"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
        accelerator_resolution_count = int(
            effective_native_symbol_resolution_audit.get(
                "accelerator_resolution_count", 0
            )
        )
        entry_point_resolution_count = int(
            effective_native_symbol_resolution_audit.get(
                "cuda_entry_point_model_resolution_count", 0
            )
        )
        if accelerator_resolution_count > entry_point_resolution_count:
            limitation = (
                "native dlsym/dlvsym resolved a CUDA/NCCL API whose indirect "
                "calls are not proven capture-visible"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
        if entry_point_resolution_count > 0:
            limitation = (
                "CUDA cuGetProcAddress/cudaGetDriverEntryPoint returned an "
                "entry point whose indirect calls are not proven capture-visible"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    model_native_process_audit = trace.metadata.get("model_native_process_audit")
    distributed_model_native_process_audit = trace.metadata.get(
        "distributed_model_native_process_audit"
    )
    effective_native_process_audit = (
        model_native_process_audit
        if isinstance(model_native_process_audit, dict)
        else distributed_model_native_process_audit
    )
    if isinstance(effective_native_process_audit, dict):
        process_status = effective_native_process_audit.get("status")
        if process_status in {"probe-unavailable", "audit-incomplete"}:
            limitation = "model native process audit could miss child CUDA/NCCL work"
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
        if (
            int(effective_native_process_audit.get("process_may_have_started_count", 0))
            > 0
        ):
            limitation = (
                "native model code may have created a child process outside the "
                "parent profiler"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    model_native_thread_audit = trace.metadata.get("model_native_thread_audit")
    distributed_model_native_thread_audit = trace.metadata.get(
        "distributed_model_native_thread_audit"
    )
    effective_native_thread_audit = (
        model_native_thread_audit
        if isinstance(model_native_thread_audit, dict)
        else distributed_model_native_thread_audit
    )
    if isinstance(effective_native_thread_audit, dict):
        native_thread_status = effective_native_thread_audit.get("status")
        if native_thread_status in {"probe-unavailable", "audit-incomplete"}:
            limitation = (
                "model native thread audit could miss asynchronous CUDA/NCCL work"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
        if int(effective_native_thread_audit.get("unresolved_thread_count", 0)) > 0:
            limitation = (
                "native model code left a pthread unresolved at entrypoint return"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    model_native_task_boundary_audit = trace.metadata.get(
        "model_native_task_boundary_audit"
    )
    distributed_model_native_task_boundary_audit = trace.metadata.get(
        "distributed_model_native_task_boundary_audit"
    )
    effective_native_task_boundary_audit = (
        model_native_task_boundary_audit
        if isinstance(model_native_task_boundary_audit, dict)
        else distributed_model_native_task_boundary_audit
    )
    if isinstance(effective_native_task_boundary_audit, dict):
        native_task_status = effective_native_task_boundary_audit.get("status")
        if native_task_status == "audit-unavailable":
            limitation = (
                "Linux native-task boundary audit could miss clone/clone3/raw-syscall "
                "thread escape"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
        if (
            int(
                effective_native_task_boundary_audit.get(
                    "unaccounted_surviving_task_count", 0
                )
            )
            > 0
        ):
            limitation = (
                "native model code left a task from an unobserved creation path "
                "alive at entrypoint return"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    model_delegated_work_audit = trace.metadata.get("model_delegated_work_audit")
    if isinstance(model_delegated_work_audit, dict):
        delegated_status = model_delegated_work_audit.get("status")
        delegated_resolution = model_delegated_work_audit.get("device_resolution")
        completed_thread_pool_is_resolved = (
            delegated_status == "completed-thread-pool-work-observed"
            and isinstance(delegated_resolution, dict)
            and delegated_resolution.get("status") == "fully-reconstructed"
        )
        if (
            delegated_status not in {None, "no-delegated-work-observed"}
            and not completed_thread_pool_is_resolved
        ):
            limitation = "model delegated work could escape the recovered device DAG"
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    capture_repeatability = trace.metadata.get("capture_repeatability")
    if isinstance(capture_repeatability, dict):
        repeatability_status = capture_repeatability.get("status")
        if repeatability_status not in {None, "stable"}:
            limitation = (
                "repeated executable capture did not produce stable replay evidence"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    cuda_event_dependency_recovery = trace.metadata.get(
        "cuda_event_dependency_recovery"
    )
    if isinstance(cuda_event_dependency_recovery, dict):
        unresolved_wait_count = int(
            cuda_event_dependency_recovery.get("unresolved_wait_count", 0)
        )
        if unresolved_wait_count > 0:
            limitation = (
                f"{unresolved_wait_count} CUDA event wait(s) could not be mapped "
                "to exact producer-consumer kernel edges"
            )
            if limitation not in trace_limitations:
                trace_limitations.append(limitation)
    cuda_stream_priority_capture = trace.metadata.get("cuda_stream_priority_capture")
    distributed_cuda_stream_priority_capture = trace.metadata.get(
        "distributed_cuda_stream_priority_capture"
    )
    effective_cuda_stream_priority_capture = (
        cuda_stream_priority_capture
        if isinstance(cuda_stream_priority_capture, dict)
        else distributed_cuda_stream_priority_capture
    )
    direct_native_nccl_observation = trace.metadata.get(
        "direct_native_nccl_observation"
    )
    distributed_direct_native_nccl_observation = trace.metadata.get(
        "distributed_direct_native_nccl_observation"
    )
    direct_native_host_synchronization_observation = trace.metadata.get(
        "direct_native_host_synchronization_observation"
    )
    distributed_direct_native_host_synchronization_observation = trace.metadata.get(
        "distributed_direct_native_host_synchronization_observation"
    )
    direct_native_host_synchronization_recovery = trace.metadata.get(
        "direct_native_host_synchronization_recovery"
    )
    distributed_direct_native_host_synchronization_recovery = trace.metadata.get(
        "distributed_direct_native_host_synchronization_recovery"
    )
    direct_native_context_selection_observation = trace.metadata.get(
        "direct_native_context_selection_observation"
    )
    distributed_direct_native_context_selection_observation = trace.metadata.get(
        "distributed_direct_native_context_selection_observation"
    )
    direct_native_context_selection_recovery = trace.metadata.get(
        "direct_native_context_selection_recovery"
    )
    distributed_direct_native_context_selection_recovery = trace.metadata.get(
        "distributed_direct_native_context_selection_recovery"
    )
    direct_native_stream_lifecycle_observation = trace.metadata.get(
        "direct_native_stream_lifecycle_observation"
    )
    distributed_direct_native_stream_lifecycle_observation = trace.metadata.get(
        "distributed_direct_native_stream_lifecycle_observation"
    )
    direct_native_stream_lifecycle_recovery = trace.metadata.get(
        "direct_native_stream_lifecycle_recovery"
    )
    distributed_direct_native_stream_lifecycle_recovery = trace.metadata.get(
        "distributed_direct_native_stream_lifecycle_recovery"
    )
    trace_limitations.sort()
    incomplete = bool(
        affected_events or unsupported_graph_launches or trace_limitations
    )
    return {
        "schema_version": "0.1",
        "status": (
            "incomplete-unsupported-semantics"
            if incomplete
            else "no-known-capture-limitations"
        ),
        "timing_recommendation_eligible": not incomplete,
        "affected_event_count": len(affected_events),
        "limitation_count": event_limitation_count + len(trace_limitations),
        "event_limitation_count": event_limitation_count,
        "affected_events": affected_events,
        "trace_limitations": trace_limitations,
        "cuda_graph_launches": graph_launches,
        "unsupported_cuda_graph_launch_count": len(unsupported_graph_launches),
        "cuda_event_dependency_recovery": (
            dict(cuda_event_dependency_recovery)
            if isinstance(cuda_event_dependency_recovery, dict)
            else None
        ),
        "cuda_stream_priority_capture": (
            dict(effective_cuda_stream_priority_capture)
            if isinstance(effective_cuda_stream_priority_capture, dict)
            else None
        ),
        "direct_native_nccl_observation": (
            dict(direct_native_nccl_observation)
            if isinstance(direct_native_nccl_observation, dict)
            else None
        ),
        "distributed_direct_native_nccl_observation": (
            dict(distributed_direct_native_nccl_observation)
            if isinstance(distributed_direct_native_nccl_observation, dict)
            else None
        ),
        "direct_native_host_synchronization_observation": (
            dict(direct_native_host_synchronization_observation)
            if isinstance(direct_native_host_synchronization_observation, dict)
            else None
        ),
        "distributed_direct_native_host_synchronization_observation": (
            dict(distributed_direct_native_host_synchronization_observation)
            if isinstance(
                distributed_direct_native_host_synchronization_observation,
                dict,
            )
            else None
        ),
        "direct_native_host_synchronization_recovery": (
            dict(direct_native_host_synchronization_recovery)
            if isinstance(direct_native_host_synchronization_recovery, dict)
            else None
        ),
        "distributed_direct_native_host_synchronization_recovery": (
            dict(distributed_direct_native_host_synchronization_recovery)
            if isinstance(
                distributed_direct_native_host_synchronization_recovery,
                dict,
            )
            else None
        ),
        "direct_native_context_selection_observation": (
            dict(direct_native_context_selection_observation)
            if isinstance(direct_native_context_selection_observation, dict)
            else None
        ),
        "distributed_direct_native_context_selection_observation": (
            dict(distributed_direct_native_context_selection_observation)
            if isinstance(distributed_direct_native_context_selection_observation, dict)
            else None
        ),
        "direct_native_context_selection_recovery": (
            dict(direct_native_context_selection_recovery)
            if isinstance(direct_native_context_selection_recovery, dict)
            else None
        ),
        "distributed_direct_native_context_selection_recovery": (
            dict(distributed_direct_native_context_selection_recovery)
            if isinstance(distributed_direct_native_context_selection_recovery, dict)
            else None
        ),
        "direct_native_stream_lifecycle_observation": (
            dict(direct_native_stream_lifecycle_observation)
            if isinstance(direct_native_stream_lifecycle_observation, dict)
            else None
        ),
        "distributed_direct_native_stream_lifecycle_observation": (
            dict(distributed_direct_native_stream_lifecycle_observation)
            if isinstance(
                distributed_direct_native_stream_lifecycle_observation, dict
            )
            else None
        ),
        "direct_native_stream_lifecycle_recovery": (
            dict(direct_native_stream_lifecycle_recovery)
            if isinstance(direct_native_stream_lifecycle_recovery, dict)
            else None
        ),
        "distributed_direct_native_stream_lifecycle_recovery": (
            dict(distributed_direct_native_stream_lifecycle_recovery)
            if isinstance(distributed_direct_native_stream_lifecycle_recovery, dict)
            else None
        ),
        "model_child_process_audit": (
            dict(model_child_process_audit)
            if isinstance(model_child_process_audit, dict)
            else None
        ),
        "model_background_thread_audit": (
            dict(model_background_thread_audit)
            if isinstance(model_background_thread_audit, dict)
            else None
        ),
        "model_ctypes_native_library_lifetime_audit": (
            dict(model_ctypes_native_library_lifetime_audit)
            if isinstance(model_ctypes_native_library_lifetime_audit, dict)
            else None
        ),
        "model_native_loader_audit": (
            dict(model_native_loader_audit)
            if isinstance(model_native_loader_audit, dict)
            else None
        ),
        "distributed_model_native_loader_audit": (
            dict(distributed_model_native_loader_audit)
            if isinstance(distributed_model_native_loader_audit, dict)
            else None
        ),
        "model_native_symbol_resolution_audit": (
            dict(model_native_symbol_resolution_audit)
            if isinstance(model_native_symbol_resolution_audit, dict)
            else None
        ),
        "distributed_model_native_symbol_resolution_audit": (
            dict(distributed_model_native_symbol_resolution_audit)
            if isinstance(distributed_model_native_symbol_resolution_audit, dict)
            else None
        ),
        "model_native_process_audit": (
            dict(model_native_process_audit)
            if isinstance(model_native_process_audit, dict)
            else None
        ),
        "distributed_model_native_process_audit": (
            dict(distributed_model_native_process_audit)
            if isinstance(distributed_model_native_process_audit, dict)
            else None
        ),
        "model_native_thread_audit": (
            dict(model_native_thread_audit)
            if isinstance(model_native_thread_audit, dict)
            else None
        ),
        "distributed_model_native_thread_audit": (
            dict(distributed_model_native_thread_audit)
            if isinstance(distributed_model_native_thread_audit, dict)
            else None
        ),
        "model_native_task_boundary_audit": (
            dict(model_native_task_boundary_audit)
            if isinstance(model_native_task_boundary_audit, dict)
            else None
        ),
        "distributed_model_native_task_boundary_audit": (
            dict(distributed_model_native_task_boundary_audit)
            if isinstance(distributed_model_native_task_boundary_audit, dict)
            else None
        ),
        "capture_artifact_semantic_scan": (
            dict(capture_artifact_semantic_scan)
            if isinstance(capture_artifact_semantic_scan, dict)
            else None
        ),
        "loaded_native_semantic_scan": (
            dict(loaded_native_semantic_scan)
            if isinstance(loaded_native_semantic_scan, dict)
            else None
        ),
        "model_delegated_work_audit": (
            dict(model_delegated_work_audit)
            if isinstance(model_delegated_work_audit, dict)
            else None
        ),
        "capture_repeatability": (
            dict(capture_repeatability)
            if isinstance(capture_repeatability, dict)
            else None
        ),
        "interpretation": (
            "At least one captured event has semantics that the replay model cannot "
            "reconstruct; the timeline is diagnostic and must not be used as a "
            "timing recommendation."
            if incomplete
            else "No explicit limitation was found in the captured evidence. This "
            "does not prove universal API coverage."
        ),
    }


def simulate(
    trace: WorkloadTrace,
    gpus: int,
    parallelism: Parallelism,
    topology: Topology,
    target: Architecture,
    max_ctas: int | None = None,
    calibration: OverlapCalibrationModel | None = None,
    cta_policy: dict[str, int] | None = None,
    comm_stream_priority: str = "normal",
    compute_transfer: ComputeTransferCalibration | None = None,
    nccl_cta_policy: int = 0,
    nccl_nvls_ctas: int | None = None,
    nccl_cga_cluster_size: int | None = None,
    network_calibration: NetworkContentionCalibration | None = None,
    materialize_timeline: bool = True,
) -> dict[str, Any]:
    warnings: set[str] = set()
    capture_completeness = _capture_completeness(trace)
    if not capture_completeness["timing_recommendation_eligible"]:
        warnings.add(
            "capture contains unsupported or incomplete execution semantics; "
            "timing recommendation abstains"
        )
    trace_hardware_domain = assess_hardware_identity(trace.source, target)
    resource_domains = [
        _kernel_resource_hardware_domain(event, trace.source, target)
        for event in trace.events
        if isinstance(event.metadata.get("kernel_resource"), dict)
    ]
    if any(domain["status"] != "exact-product-match" for domain in resource_domains):
        warnings.add(
            "captured kernel grid and occupancy observations are outside the "
            "exact target hardware domain; target-SM placement bounds were withheld"
        )
    occupancy_profile_evidence = trace.metadata.get(
        "kernel_occupancy_profile", trace.metadata.get("ncu_occupancy_profile")
    )
    if isinstance(occupancy_profile_evidence, dict) and occupancy_profile_evidence.get(
        "resource_mismatched_event_ids"
    ):
        warnings.add(
            "the supplied kernel occupancy profile had same-name kernels with "
            "non-matching launch resources; those events were not annotated"
        )
    effective_calibration = calibration
    calibration_environment = None
    cta_policy = {} if cta_policy is None else dict(cta_policy)
    invalid_roles = set(cta_policy) - {"tp", "pp", "dp", "ep"}
    if invalid_roles:
        raise ValueError(
            f"invalid CTA policy roles: {', '.join(sorted(invalid_roles))}"
        )
    if any(value <= 0 for value in cta_policy.values()):
        raise ValueError("CTA policy values must be positive")
    if comm_stream_priority not in {"normal", "high"}:
        raise ValueError("NCCL stream priority must be normal or high")
    nccl_cta_policy = parse_cta_policy_mode(nccl_cta_policy)
    if nccl_nvls_ctas is not None and nccl_nvls_ctas <= 0:
        raise ValueError("NCCL NVLS CTAs must be positive")
    nccl_cga_cluster_size = parse_cga_cluster_size(nccl_cga_cluster_size)
    effective_cga_cluster_size = (
        0 if nccl_cga_cluster_size is None else nccl_cga_cluster_size
    )
    if (
        trace.source.get("target") not in (None, target.name)
        and compute_transfer is None
    ):
        warnings.add(
            "compute durations were observed on a different target and no transfer model was applied"
        )
    if (max_ctas is not None or cta_policy) and calibration is None:
        warnings.add(
            "max_ctas is recorded but no validated CTA slowdown calibration was supplied"
        )
    if effective_cga_cluster_size != 0:
        warnings.add(
            "non-default NCCL CGA cluster size requires exact mode-matched timing and interference calibration; configuration acceptance does not establish its performance effect"
        )
    if calibration is not None and calibration.target != target.name:
        warnings.add(
            f"calibration target {calibration.target} does not match requested target {target.name}"
        )
    if calibration is not None:
        effective_calibration, calibration_environment = select_calibration_environment(
            calibration, trace.source
        )
        for assessment in calibration_environment["profiles"]:
            status = assessment["status"]
            if status == "mismatch":
                warnings.add(
                    "excluded overlap calibration with mismatched software versions: "
                    + assessment["source_path"]
                )
            elif status == "unspecified-calibration-versions":
                warnings.add(
                    "overlap calibration has no exact CUDA/NCCL/PyTorch version keys: "
                    + assessment["source_path"]
                )
            elif status == "unverifiable-trace-versions":
                warnings.add(
                    "captured trace lacks versions required to verify overlap calibration: "
                    + assessment["source_path"]
                )

    nvls_evidence = nvls_runtime_evidence(
        effective_calibration,
        target.name,
        nccl_cta_policy,
        nccl_nvls_ctas,
        effective_cga_cluster_size,
    )
    if nccl_nvls_ctas is not None:
        if nvls_evidence["status"] == "inactive":
            warnings.add(
                "NVLS CTA configuration was accepted but calibration evidence records "
                "that NVLS execution was inactive"
            )
        elif nvls_evidence["status"] != "active":
            warnings.add(
                "NVLS runtime eligibility is unverified; configuration acceptance "
                "does not prove NVLS execution"
            )

    rebased_trace, capture_rebase = _rebase_captured_overlap(
        trace, parallelism, effective_calibration, warnings
    )
    expanded_trace = expand_pipeline(rebased_trace, parallelism, gpus=gpus)
    prepared = _prepare_events(
        expanded_trace,
        parallelism,
        topology,
        target,
        max_ctas,
        cta_policy,
        effective_calibration,
        comm_stream_priority,
        warnings,
        compute_transfer,
        nccl_cta_policy,
        nccl_nvls_ctas,
        effective_cga_cluster_size,
    )
    timeline, matched_pairs, admission_assessments, network_contention = _schedule(
        prepared,
        target,
        topology,
        trace.source,
        effective_calibration,
        network_calibration,
        warnings,
    )
    step_time_us = max((event.end_us for event in timeline), default=0.0)
    overlap_region_predictions = _calibrated_overlap_region_predictions(
        timeline, matched_pairs, effective_calibration, target
    )
    duration_source_counts: dict[str, int] = {}
    for event in timeline:
        duration_source_counts[event.duration_source] = (
            duration_source_counts.get(event.duration_source, 0) + 1
        )
    critical_path = (
        _realized_critical_path(timeline, step_time_us)
        if materialize_timeline
        else {
            "schema": "realized-schedule-critical-path-v1",
            "status": "omitted-summary-mode",
            "step_time_us": step_time_us,
            "interpretation": (
                "Candidate search requested summary-only simulation; rerun a "
                "selected candidate with full detail to materialize its causal chain."
            ),
        }
    )
    critical_path_indexes = {
        event_id: index
        for index, event_id in enumerate(critical_path.get("event_ids", ()))
    }
    compute_work_us = sum(
        event.isolated_duration_us for event in timeline if event.kind == "compute"
    )
    collective_work_us = sum(
        event.isolated_duration_us for event in timeline if event.kind == "collective"
    )
    collective_events = [event for event in timeline if event.kind == "collective"]
    calibrated_collective_prefixes = (
        "measured-nccl-calibration",
        "interpolated-nccl-calibration",
        "ensemble-nccl-calibration",
    )
    if effective_cga_cluster_size == 0:
        cga_timing_status = "default-calibration-domain"
    elif collective_events and all(
        event.duration_source.startswith(calibrated_collective_prefixes)
        for event in collective_events
    ):
        cga_timing_status = "mode-matched-isolated-timing"
    else:
        cga_timing_status = "unmatched"
    prediction = {
        "schema_version": "0.1",
        "model": "trace-driven-calibrated-resource-progress-v0",
        "requested_gpus": gpus,
        "target": asdict(target),
        "parallelism": parallelism.to_dict(),
        "pipeline": expanded_trace.metadata.get("pipeline_expansion"),
        "topology": topology.to_dict(),
        "policy": {
            "max_ctas": max_ctas,
            "per_group_role_max_ctas": cta_policy,
            "comm_stream_priority": comm_stream_priority,
            "cta_to_sm_ratio": None if max_ctas is None else max_ctas / target.sm_count,
            "warning": "max_ctas is a channel/CTA budget, not an exclusive SM reservation",
            "sm_footprint_model": "full-grid-residency-bounds-v1",
            "exclusive_sm_allocation": None,
            "sm_footprint_interpretation": (
                "per-event bounds describe a fully resident CTA grid; achieved "
                "placement, partial residency, and slowdown require runtime evidence"
            ),
            "nccl_cta_policy": cta_policy_name(nccl_cta_policy),
            "nccl_cta_policy_flag": nccl_cta_policy,
            "nccl_nvls_ctas": nccl_nvls_ctas,
            "nccl_cga_cluster_size": nccl_cga_cluster_size,
            "nccl_cga_effective_calibration_key": effective_cga_cluster_size,
            "nccl_cga_timing_calibration_status": cga_timing_status,
            "nvls_runtime_evidence": nvls_evidence,
        },
        "calibration": None if calibration is None else calibration.to_summary(),
        "calibration_environment": calibration_environment,
        "capture_rebase": capture_rebase,
        "capture_completeness": capture_completeness,
        "resource_evidence": {
            "trace_hardware_domain": trace_hardware_domain,
            "kernel_occupancy_profile": expanded_trace.metadata.get(
                "kernel_occupancy_profile"
            ),
            "ncu_occupancy_profile": expanded_trace.metadata.get(
                "ncu_occupancy_profile"
            ),
        },
        "compute_transfer": (
            None if compute_transfer is None else compute_transfer.to_summary()
        ),
        "network_contention_calibration": (
            None if network_calibration is None else network_calibration.to_summary()
        ),
        "interference_matches": matched_pairs,
        "overlap_region_predictions": overlap_region_predictions,
        "admission_assessments": admission_assessments,
        "network_contention": network_contention,
        "duration_source_counts": dict(sorted(duration_source_counts.items())),
        "duration_evidence": (
            _duration_evidence(timeline)
            if materialize_timeline
            else {
                "schema": "duration-evidence-v1",
                "status": "omitted-summary-mode",
                "source_counts": dict(sorted(duration_source_counts.items())),
                "interpretation": (
                    "Per-event evidence is omitted during candidate search; "
                    "duration sources still participate in the claim boundary."
                ),
            }
        ),
        "critical_path": critical_path,
        "summary": {
            "step_time_us": step_time_us,
            "compute_work_us": compute_work_us,
            "collective_work_us": collective_work_us,
            "event_count": len(timeline),
            "unresolved_admission_pair_count": len(admission_assessments),
            "interference_domain_status": (
                "out-of-domain"
                if admission_assessments
                else "in-domain-or-not-overlapping"
            ),
            "network_contention_region_count": len(network_contention),
            "network_contention_model": (
                "exact-calibration-with-fluid-fallback-v1"
                if network_calibration is not None
                else "fluid-fair-share-whole-event-v0"
            ),
        },
        "warnings": sorted(warnings),
        "timeline_materialization": (
            "full" if materialize_timeline else "omitted-summary-mode"
        ),
        "timeline": (
            [
                {
                    **event.to_dict(),
                    "on_critical_path": event.id in critical_path_indexes,
                    "critical_path_index": critical_path_indexes.get(event.id),
                }
                for event in sorted(
                    timeline, key=lambda item: (item.start_us, item.id)
                )
            ]
            if materialize_timeline
            else []
        ),
    }
    return attach_uncertainty(prediction)
