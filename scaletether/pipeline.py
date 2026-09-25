from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .config import Parallelism
from .compute_region import validate_compute_regions
from .schema import TraceEvent, WorkloadTrace


MEGATRON_CORE_SCHEDULE_COMMIT = "f8e1ac64b0587ff7002a18fbaa5ecdeaeb8491be"


def _clone_compute_region_instance(
    metadata: dict[str, Any], schedule_identity: str
) -> None:
    raw = metadata.get("compute_region")
    if raw is not None:
        region = dict(raw)
        region["instance_id"] = f"{region['instance_id']}@{schedule_identity}"
        metadata["compute_region"] = region
    overlap = metadata.get("overlap_region_marker")
    if isinstance(overlap, dict):
        overlap = dict(overlap)
        overlap["instance_id"] = f"{overlap['instance_id']}@{schedule_identity}"
        metadata["overlap_region_marker"] = overlap


@dataclass(frozen=True)
class PipelineRankExpansion:
    rank_events: tuple[tuple[TraceEvent, ...], ...]
    template_send_count: int
    p2p_pair_count: int
    mapping_semantics: str
    rank_layout_source: str
    stage_ranks: tuple[tuple[int, ...], ...]
    rank_coordinates: tuple[tuple[int, int, int], ...]


def _stage_rank_layout(
    configuration: dict[str, Any],
    parallelism: Parallelism,
    ranks: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    dict[tuple[int, int, int], int],
    tuple[tuple[int, int, int], ...],
    str,
]:
    """Return a complete, invertible stage/DP/TP-to-rank placement.

    ``stage_ranks[stage][dp_replica * tp + tp_lane]`` is deliberately a
    compact JSON form. Requiring a permutation of every target rank prevents
    an apparently valid pipeline trace from silently dropping or duplicating
    workers.
    """

    expected_ranks = parallelism.tp * parallelism.pp * parallelism.dp
    if ranks != expected_ranks:
        raise ValueError("pipeline rank layout requires ranks=tp*pp*dp")
    raw_stage_ranks = configuration.get("stage_ranks")
    if raw_stage_ranks is None:
        stage_ranks = tuple(
            tuple(
                tp_lane + parallelism.tp * (stage + parallelism.pp * dp_replica)
                for dp_replica in range(parallelism.dp)
                for tp_lane in range(parallelism.tp)
            )
            for stage in range(parallelism.pp)
        )
        source = "assumed-contiguous-stage-major"
    else:
        if configuration.get("stage_devices") is not None:
            raise ValueError(
                "pipeline stage_ranks and stage_devices are mutually exclusive"
            )
        if (
            not isinstance(raw_stage_ranks, list)
            or len(raw_stage_ranks) != parallelism.pp
        ):
            raise ValueError(
                "pipeline stage_ranks must list one rank row for every PP stage"
            )
        row_size = parallelism.tp * parallelism.dp
        rows: list[tuple[int, ...]] = []
        for stage, raw_row in enumerate(raw_stage_ranks):
            if not isinstance(raw_row, list) or len(raw_row) != row_size:
                raise ValueError(
                    "pipeline stage_ranks rows must contain tp*dp ranks in "
                    f"DP-major, TP-minor order; stage={stage}, expected={row_size}"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_row
            ):
                raise ValueError("pipeline stage_ranks must contain integer ranks")
            rows.append(tuple(raw_row))
        stage_ranks = tuple(rows)
        flattened = [rank for row in stage_ranks for rank in row]
        if sorted(flattened) != list(range(ranks)):
            raise ValueError(
                "pipeline stage_ranks must be an exact permutation of target "
                f"ranks 0..{ranks - 1}"
            )
        source = "explicit-stage-ranks"

    rank_by_coordinates: dict[tuple[int, int, int], int] = {}
    coordinates_by_rank: list[tuple[int, int, int] | None] = [None] * ranks
    for stage, row in enumerate(stage_ranks):
        for dp_replica in range(parallelism.dp):
            for tp_lane in range(parallelism.tp):
                rank = row[dp_replica * parallelism.tp + tp_lane]
                coordinates = (tp_lane, stage, dp_replica)
                rank_by_coordinates[coordinates] = rank
                coordinates_by_rank[rank] = coordinates
    if any(coordinates is None for coordinates in coordinates_by_rank):
        raise ValueError("pipeline stage_ranks did not assign every target rank")
    return (
        stage_ranks,
        rank_by_coordinates,
        tuple(
            coordinates
            for coordinates in coordinates_by_rank
            if coordinates is not None
        ),
        source,
    )


def _phase_key(event: TraceEvent) -> tuple[int, str] | None:
    stage = event.metadata.get("pipeline_stage")
    phase = event.metadata.get("pipeline_phase")
    if stage is None and phase is None:
        return None
    if not isinstance(stage, int) or phase not in {"forward", "backward"}:
        raise ValueError(
            f"event {event.id!r} must have integer pipeline_stage and forward/backward pipeline_phase"
        )
    return stage, phase


def _megatron_interleaved_schedule_table(
    microbatches: int,
    virtual_stages: int,
    group_size: int,
) -> tuple[tuple[int, int], ...]:
    """Mirror Megatron-Core's microbatch/model-chunk lookup table."""

    table: list[tuple[int, int]] = []
    for group_start in range(0, microbatches, group_size):
        group_end = min(group_start + group_size, microbatches)
        table.extend(
            (microbatch, model_chunk)
            for model_chunk in range(virtual_stages)
            for microbatch in range(group_start, group_end)
        )
    return tuple(table)


def _pipeline_boolean(
    configuration: dict[str, Any], name: str, default: bool = False
) -> bool:
    value = configuration.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"pipeline {name} must be Boolean")
    return value


def _expand_interleaved_pipeline(
    trace: WorkloadTrace,
    configuration: dict[str, Any],
    parallelism: Parallelism,
    microbatches: int,
    stage_devices: tuple[int, ...],
    stage_lane_devices: tuple[tuple[int, ...], ...],
    stage_device_source: str,
    rank_layout_source: str,
    configured_stage_ranks: tuple[tuple[int, ...], ...] | None,
    mapping_semantics: str,
) -> WorkloadTrace:
    """Expand Megatron-style virtual-stage interleaved 1F1B dependencies."""

    if parallelism.pp < 2:
        raise ValueError("interleaved_1f1b requires pipeline parallelism pp >= 2")
    raw_virtual_stages = configuration.get("virtual_stages")
    if (
        isinstance(raw_virtual_stages, bool)
        or not isinstance(raw_virtual_stages, int)
        or raw_virtual_stages < 2
    ):
        raise ValueError(
            "interleaved_1f1b requires integer pipeline virtual_stages >= 2"
        )
    virtual_stages = raw_virtual_stages
    raw_group_size = configuration.get(
        "microbatch_group_size_per_virtual_stage", parallelism.pp
    )
    if (
        isinstance(raw_group_size, bool)
        or not isinstance(raw_group_size, int)
        or not parallelism.pp <= raw_group_size <= microbatches
    ):
        raise ValueError(
            "interleaved microbatch group size must be an integer in "
            f"[pp={parallelism.pp}, microbatches={microbatches}]"
        )
    group_size = raw_group_size
    remainder = microbatches % group_size
    if 0 < remainder < parallelism.pp:
        raise ValueError(
            "interleaved final microbatch group must be empty or contain at "
            f"least pp={parallelism.pp} microbatches; remainder={remainder}"
        )
    if any(event.collective == "recv" for event in trace.events):
        raise ValueError(
            "interleaved_1f1b requires a send-only template; Recv events are "
            "synthesized during rank expansion"
        )
    overlap_p2p_comm = _pipeline_boolean(configuration, "overlap_p2p_comm")
    overlap_warmup_flush = _pipeline_boolean(
        configuration, "overlap_p2p_comm_warmup_flush"
    )
    deallocate_pipeline_outputs = _pipeline_boolean(
        configuration, "deallocate_pipeline_outputs"
    )
    if overlap_warmup_flush and not overlap_p2p_comm:
        raise ValueError(
            "pipeline overlap_p2p_comm_warmup_flush requires overlap_p2p_comm=true"
        )

    groups: dict[tuple[int, int, str], list[TraceEvent]] = {}
    for event in trace.events:
        phase_key = _phase_key(event)
        if phase_key is None:
            raise ValueError(
                "an interleaved pipeline template cannot mix events without "
                "pipeline_stage/pipeline_phase"
            )
        stage, phase = phase_key
        chunk = event.metadata.get("pipeline_model_chunk")
        if (
            isinstance(chunk, bool)
            or not isinstance(chunk, int)
            or not 0 <= chunk < virtual_stages
        ):
            raise ValueError(
                f"event {event.id!r} pipeline_model_chunk must be in "
                f"[0, {virtual_stages})"
            )
        groups.setdefault((stage, chunk, phase), []).append(event)

    required = {
        (stage, chunk, phase)
        for stage in range(parallelism.pp)
        for chunk in range(virtual_stages)
        for phase in ("forward", "backward")
    }
    if set(groups) != required:
        missing = sorted(required - set(groups))
        extra = sorted(set(groups) - required)
        raise ValueError(
            "interleaved pipeline template stage/chunk mismatch; "
            f"missing={missing}, extra={extra}"
        )

    entries: dict[tuple[int, int, str], list[str]] = {}
    exits: dict[tuple[int, int, str], list[str]] = {}
    overlap_release_frontiers: dict[tuple[int, int, str], list[str]] = {}
    virtual_stage_count = parallelism.pp * virtual_stages
    for key, events in groups.items():
        ids = {event.id for event in events}
        referenced: set[str] = set()
        for event in events:
            cross_group = set(event.dependencies) - ids
            if cross_group:
                raise ValueError(
                    f"interleaved template event {event.id!r} has cross-phase "
                    f"dependencies {sorted(cross_group)}"
                )
            referenced.update(event.dependencies)
        entries[key] = sorted(event.id for event in events if not event.dependencies)
        exits[key] = sorted(ids - referenced)
        if not entries[key] or not exits[key]:
            raise ValueError(
                f"interleaved pipeline phase {key} has no entry or exit event"
            )
        stage, chunk, phase = key
        virtual_stage = chunk * parallelism.pp + stage
        crosses_boundary = (
            phase == "forward" and virtual_stage < virtual_stage_count - 1
        ) or (phase == "backward" and virtual_stage > 0)
        send_exits = [
            event
            for event in events
            if event.id in exits[key] and event.collective == "send"
        ]
        if crosses_boundary and len(send_exits) != len(exits[key]):
            raise ValueError(
                f"interleaved pipeline phase {key} requires explicit Send exits"
            )
        if not crosses_boundary and any(event.collective == "send" for event in events):
            raise ValueError(f"interleaved terminal pipeline phase {key} cannot Send")
        if send_exits:
            events_by_id = {event.id: event for event in events}
            release_dependencies: set[str] = set()
            for send in send_exits:
                if overlap_p2p_comm and not send.dependencies:
                    raise ValueError(
                        f"interleaved overlap Send {send.id!r} requires an "
                        "explicit local launch frontier"
                    )
                if overlap_p2p_comm and any(
                    events_by_id[dependency].collective == "send"
                    for dependency in send.dependencies
                ):
                    raise ValueError(
                        f"interleaved overlap Send {send.id!r} cannot use "
                        "another Send as its launch frontier"
                    )
                release_dependencies.update(send.dependencies)
            overlap_release_frontiers[key] = (
                sorted(release_dependencies) if release_dependencies else exits[key]
            )
        else:
            overlap_release_frontiers[key] = exits[key]

    schedule_table = _megatron_interleaved_schedule_table(
        microbatches, virtual_stages, group_size
    )
    total_virtual_microbatches = len(schedule_table)
    previous_operation: dict[
        tuple[int, int, str, int], tuple[int, str, int] | None
    ] = {}
    operation_regions: dict[tuple[int, int, str, int], str] = {}
    asynchronous_operations: dict[tuple[int, int, str, int], bool] = {}
    local_orders: dict[int, list[tuple[int, str, int]]] = {}
    for stage in range(parallelism.pp):
        warmup = min(
            (parallelism.pp - stage - 1) * 2 + (virtual_stages - 1) * group_size,
            total_virtual_microbatches,
        )
        forward_order = [
            (chunk, "forward", microbatch) for microbatch, chunk in schedule_table
        ]
        backward_order = [
            (virtual_stages - chunk - 1, "backward", microbatch)
            for microbatch, chunk in schedule_table
        ]
        order = list(forward_order[:warmup])
        for index in range(warmup, total_virtual_microbatches):
            order.append(forward_order[index])
            order.append(backward_order[index - warmup])
        if warmup:
            order.extend(backward_order[-warmup:])
        local_orders[stage] = order
        previous: tuple[int, str, int] | None = None
        steady_operations = 2 * (total_virtual_microbatches - warmup)
        for operation_index, (chunk, phase, microbatch) in enumerate(order):
            operation_key = (stage, chunk, phase, microbatch)
            previous_operation[operation_key] = previous
            if operation_index < warmup:
                region = "warmup"
            elif operation_index < warmup + steady_operations:
                region = "steady"
            else:
                region = "cooldown"
            operation_regions[operation_key] = region
            phase_key = (stage, chunk, phase)
            has_send = overlap_release_frontiers[phase_key] != exits[phase_key]
            asynchronous_operations[operation_key] = (
                overlap_p2p_comm
                and has_send
                and (overlap_warmup_flush or region == "steady")
                and not (deallocate_pipeline_outputs and phase == "forward")
            )
            previous = (chunk, phase, microbatch)

    def duplicated_id(event_id: str, microbatch: int) -> str:
        return f"{event_id}@mb{microbatch}"

    expanded: list[TraceEvent] = []
    for stage in range(parallelism.pp):
        for chunk in range(virtual_stages):
            virtual_stage = chunk * parallelism.pp + stage
            for phase in ("forward", "backward"):
                key = (stage, chunk, phase)
                for microbatch in range(microbatches):
                    for event in groups[key]:
                        dependencies = [
                            duplicated_id(dependency, microbatch)
                            for dependency in event.dependencies
                        ]
                        if event.id in entries[key]:
                            previous = previous_operation[
                                (stage, chunk, phase, microbatch)
                            ]
                            if previous is not None:
                                previous_chunk, previous_phase, previous_microbatch = (
                                    previous
                                )
                                previous_operation_key = (
                                    stage,
                                    previous_chunk,
                                    previous_phase,
                                    previous_microbatch,
                                )
                                current_operation_key = (
                                    stage,
                                    chunk,
                                    phase,
                                    microbatch,
                                )
                                release_previous = asynchronous_operations[
                                    previous_operation_key
                                ] and (
                                    overlap_warmup_flush
                                    or operation_regions[current_operation_key]
                                    == "steady"
                                )
                                previous_frontier = (
                                    overlap_release_frontiers[
                                        (stage, previous_chunk, previous_phase)
                                    ]
                                    if release_previous
                                    else exits[(stage, previous_chunk, previous_phase)]
                                )
                                dependencies.extend(
                                    duplicated_id(exit_id, previous_microbatch)
                                    for exit_id in previous_frontier
                                )
                            if phase == "forward" and virtual_stage > 0:
                                previous_virtual_stage = virtual_stage - 1
                                previous_chunk, previous_stage = divmod(
                                    previous_virtual_stage, parallelism.pp
                                )
                                dependencies.extend(
                                    duplicated_id(exit_id, microbatch)
                                    for exit_id in exits[
                                        (previous_stage, previous_chunk, "forward")
                                    ]
                                )
                            if phase == "backward":
                                forward_operation_key = (
                                    stage,
                                    chunk,
                                    "forward",
                                    microbatch,
                                )
                                forward_frontier = (
                                    overlap_release_frontiers[(stage, chunk, "forward")]
                                    if asynchronous_operations[forward_operation_key]
                                    else exits[(stage, chunk, "forward")]
                                )
                                dependencies.extend(
                                    duplicated_id(exit_id, microbatch)
                                    for exit_id in forward_frontier
                                )
                                if virtual_stage < virtual_stage_count - 1:
                                    next_chunk, next_stage = divmod(
                                        virtual_stage + 1, parallelism.pp
                                    )
                                    dependencies.extend(
                                        duplicated_id(exit_id, microbatch)
                                        for exit_id in exits[
                                            (next_stage, next_chunk, "backward")
                                        ]
                                    )

                        metadata = dict(event.metadata)
                        metadata.update(
                            {
                                "pipeline_microbatch": microbatch,
                                "pipeline_virtual_stage": virtual_stage,
                                "pipeline_schedule_region": operation_regions[
                                    (stage, chunk, phase, microbatch)
                                ],
                            }
                        )
                        _clone_compute_region_instance(
                            metadata,
                            f"stage{stage}:chunk{chunk}:{phase}:mb{microbatch}",
                        )
                        raw_tp_lane = metadata.get("pipeline_tp_lane", 0)
                        if (
                            isinstance(raw_tp_lane, bool)
                            or not isinstance(raw_tp_lane, int)
                            or not 0 <= raw_tp_lane < parallelism.tp
                        ):
                            raise ValueError(
                                f"event {event.id!r} pipeline_tp_lane must be in "
                                f"[0, {parallelism.tp})"
                            )
                        metadata["pipeline_tp_lane"] = raw_tp_lane
                        physical_device = stage_lane_devices[stage][raw_tp_lane]
                        metadata["pipeline_physical_device"] = physical_device
                        group_role = event.group_role
                        if event.collective == "send":
                            if group_role not in (None, "pp"):
                                raise ValueError(
                                    f"pipeline P2P event {event.id!r} must use "
                                    "group_role='pp'"
                                )
                            destination_virtual_stage = (
                                virtual_stage + 1
                                if phase == "forward"
                                else virtual_stage - 1
                            )
                            destination_chunk, destination_stage = divmod(
                                destination_virtual_stage, parallelism.pp
                            )
                            metadata.update(
                                {
                                    "p2p_route_source": (
                                        "pipeline-virtual-stage-map-v1"
                                    ),
                                    "p2p_source_stage": stage,
                                    "p2p_destination_stage": destination_stage,
                                    "p2p_source_model_chunk": chunk,
                                    "p2p_destination_model_chunk": (destination_chunk),
                                    "p2p_source_virtual_stage": virtual_stage,
                                    "p2p_destination_virtual_stage": (
                                        destination_virtual_stage
                                    ),
                                    "p2p_source_device": physical_device,
                                    "p2p_destination_device": (
                                        stage_lane_devices[destination_stage][
                                            raw_tp_lane
                                        ]
                                    ),
                                    "pipeline_p2p_mode": (
                                        "asynchronous"
                                        if asynchronous_operations[
                                            (stage, chunk, phase, microbatch)
                                        ]
                                        else "blocking"
                                    ),
                                    "pipeline_p2p_release_frontier": [
                                        duplicated_id(frontier_id, microbatch)
                                        for frontier_id in overlap_release_frontiers[
                                            key
                                        ]
                                    ],
                                }
                            )
                            group_role = "pp"
                        expanded.append(
                            replace(
                                event,
                                id=duplicated_id(event.id, microbatch),
                                device=physical_device,
                                dependencies=tuple(dict.fromkeys(dependencies)),
                                observed_start_us=None,
                                group_role=group_role,
                                metadata=metadata,
                            )
                        )

    result = WorkloadTrace(
        events=tuple(expanded),
        source=trace.source,
        metadata={
            **trace.metadata,
            "pipeline_expanded": True,
            "pipeline_expansion": {
                "schedule": "interleaved_1f1b",
                "microbatches": microbatches,
                "virtual_stages": virtual_stages,
                "microbatch_group_size_per_virtual_stage": group_size,
                "overlap_p2p_comm": overlap_p2p_comm,
                "overlap_p2p_comm_warmup_flush": overlap_warmup_flush,
                "deallocate_pipeline_outputs": deallocate_pipeline_outputs,
                "schedule_table": [list(item) for item in schedule_table],
                "stage_device_bases": list(stage_devices),
                "stage_device_lanes": [
                    list(lane_devices) for lane_devices in stage_lane_devices
                ],
                "stage_device_source": stage_device_source,
                "stage_rank_layout_source": rank_layout_source,
                "stage_ranks": (
                    None
                    if configured_stage_ranks is None
                    else [list(row) for row in configured_stage_ranks]
                ),
                "mapping_semantics": mapping_semantics,
                "schedule_semantics": (
                    "Megatron-Core conventional interleaved 1F1B lookup table; "
                    "virtual stage = model_chunk*pp + physical_stage"
                ),
                "p2p_overlap_semantics": (
                    "source compute is released at each explicit Send launch "
                    "frontier; destination compute still depends on transfer "
                    "completion; steady-state only unless warmup/flush overlap "
                    "is enabled"
                    if overlap_p2p_comm
                    else "blocking P2P completion orders the next local operation"
                ),
                "schedule_reference": {
                    "project": "NVIDIA/Megatron-LM",
                    "commit": MEGATRON_CORE_SCHEDULE_COMMIT,
                    "functions": [
                        "get_schedule_table",
                        "get_pp_rank_microbatches",
                        "convert_schedule_table_to_order",
                        "forward_backward_pipelining_with_interleaving",
                    ],
                },
            },
        },
    )
    result.validate()
    validate_compute_regions(result.events)
    return result


def expand_pipeline(
    trace: WorkloadTrace,
    parallelism: Parallelism,
    gpus: int | None = None,
) -> WorkloadTrace:
    validate_compute_regions(trace.events)
    configuration = trace.metadata.get("pipeline")
    if configuration is None:
        return trace
    if not isinstance(configuration, dict):
        raise ValueError("trace metadata.pipeline must be a mapping")
    schedule = str(configuration.get("schedule", "gpipe")).lower()
    if schedule not in {"gpipe", "1f1b", "interleaved_1f1b"}:
        raise ValueError("pipeline schedule must be gpipe, 1f1b, or interleaved_1f1b")
    microbatches = int(configuration.get("microbatches", 0))
    if microbatches <= 0:
        raise ValueError("pipeline microbatches must be positive")
    overlap_options = (
        "overlap_p2p_comm",
        "overlap_p2p_comm_warmup_flush",
        "deallocate_pipeline_outputs",
    )
    for option in overlap_options:
        if option in configuration and not isinstance(configuration[option], bool):
            raise ValueError(f"pipeline {option} must be Boolean")
    if schedule != "interleaved_1f1b" and any(
        configuration.get(option, False) for option in overlap_options
    ):
        raise ValueError(
            "pipeline P2P overlap/deallocation controls are currently supported "
            "only for interleaved_1f1b"
        )

    raw_stage_devices = configuration.get("stage_devices")
    raw_stage_ranks = configuration.get("stage_ranks")
    configured_stage_ranks: tuple[tuple[int, ...], ...] | None = None
    if raw_stage_ranks is not None:
        expected_ranks = parallelism.tp * parallelism.pp * parallelism.dp
        stage_ranks, _, _, rank_layout_source = _stage_rank_layout(
            configuration, parallelism, expected_ranks
        )
        configured_stage_ranks = stage_ranks
        stage_lane_devices = tuple(row[: parallelism.tp] for row in stage_ranks)
        stage_devices = tuple(row[0] for row in stage_lane_devices)
        stage_device_source = "explicit-stage-ranks-first-dp-replica"
        mapping_semantics = (
            "stage_ranks[stage][dp*tp+lane] gives the global rank/device; "
            "the analytical template represents DP replica zero"
        )
    elif raw_stage_devices is None:
        stage_lane_devices = tuple(
            tuple(stage * parallelism.tp + tp_lane for tp_lane in range(parallelism.tp))
            for stage in range(parallelism.pp)
        )
        stage_devices = tuple(row[0] for row in stage_lane_devices)
        stage_device_source = "assumed-contiguous-stage-major"
        rank_layout_source = "assumed-contiguous-stage-major"
        mapping_semantics = (
            "each base is TP lane zero; lane i maps to base+i within "
            "the first DP replica"
        )
    else:
        if (
            not isinstance(raw_stage_devices, list)
            or len(raw_stage_devices) != parallelism.pp
        ):
            raise ValueError(
                "pipeline stage_devices must list one base device for every PP stage"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_stage_devices
        ):
            raise ValueError(
                "pipeline stage_devices must contain integer device indices"
            )
        stage_devices = tuple(raw_stage_devices)
        stage_lane_devices = tuple(
            tuple(base + tp_lane for tp_lane in range(parallelism.tp))
            for base in stage_devices
        )
        stage_device_source = "explicit-stage-devices"
        rank_layout_source = "not-specified"
        mapping_semantics = (
            "each base is TP lane zero; lane i maps to base+i within "
            "the first DP replica"
        )
    if any(device < 0 for device in stage_devices):
        raise ValueError("pipeline stage_devices must be non-negative")
    occupied: set[int] = set()
    for lane_devices in stage_lane_devices:
        devices = set(lane_devices)
        if len(devices) != parallelism.tp:
            raise ValueError("pipeline stage TP lane devices must be unique")
        if occupied & devices:
            raise ValueError(
                "pipeline stage device ranges overlap under the requested TP width"
            )
        occupied.update(devices)
    if gpus is not None:
        if gpus <= 0:
            raise ValueError("pipeline GPU capacity must be positive")
        if raw_stage_ranks is not None and gpus != expected_ranks:
            raise ValueError(
                "pipeline stage_ranks require GPU capacity equal to tp*pp*dp; "
                f"gpus={gpus}, expected={expected_ranks}"
            )
        outside = sorted(device for device in occupied if device >= gpus)
        if outside:
            raise ValueError(
                "pipeline stage device ranges exceed the requested GPU capacity; "
                f"outside={outside}, gpus={gpus}"
            )

    if schedule == "interleaved_1f1b":
        return _expand_interleaved_pipeline(
            trace=trace,
            configuration=configuration,
            parallelism=parallelism,
            microbatches=microbatches,
            stage_devices=stage_devices,
            stage_lane_devices=stage_lane_devices,
            stage_device_source=stage_device_source,
            rank_layout_source=rank_layout_source,
            configured_stage_ranks=configured_stage_ranks,
            mapping_semantics=mapping_semantics,
        )

    groups: dict[tuple[int, str], list[TraceEvent]] = {}
    for event in trace.events:
        key = _phase_key(event)
        if key is None:
            raise ValueError(
                "a pipeline-template trace cannot mix events without pipeline_stage/pipeline_phase"
            )
        groups.setdefault(key, []).append(event)
    required = {
        (stage, phase)
        for stage in range(parallelism.pp)
        for phase in ("forward", "backward")
    }
    if set(groups) != required:
        missing = sorted(required - set(groups))
        extra = sorted(set(groups) - required)
        raise ValueError(
            f"pipeline template stage mismatch; missing={missing}, extra={extra}"
        )

    entries: dict[tuple[int, str], list[str]] = {}
    exits: dict[tuple[int, str], list[str]] = {}
    for key, events in groups.items():
        ids = {event.id for event in events}
        referenced: set[str] = set()
        for event in events:
            cross_group = set(event.dependencies) - ids
            if cross_group:
                raise ValueError(
                    f"pipeline template event {event.id!r} has cross-phase dependencies {sorted(cross_group)}"
                )
            referenced.update(event.dependencies)
        entries[key] = sorted(event.id for event in events if not event.dependencies)
        exits[key] = sorted(ids - referenced)
        if not entries[key] or not exits[key]:
            raise ValueError(f"pipeline phase {key} has no entry or exit event")

    def duplicated_id(event_id: str, microbatch: int) -> str:
        return f"{event_id}@mb{microbatch}"

    local_orders: dict[int, list[tuple[str, int]]] = {}
    for stage in range(parallelism.pp):
        if schedule == "gpipe":
            local_orders[stage] = [
                ("forward", microbatch) for microbatch in range(microbatches)
            ] + [
                ("backward", microbatch) for microbatch in reversed(range(microbatches))
            ]
        else:
            warmup = min(parallelism.pp - stage - 1, microbatches)
            order = [("forward", microbatch) for microbatch in range(warmup)]
            remaining = microbatches - warmup
            for index in range(remaining):
                order.append(("forward", warmup + index))
                order.append(("backward", index))
            order.extend(
                ("backward", microbatch)
                for microbatch in range(remaining, microbatches)
            )
            local_orders[stage] = order
    previous_operation: dict[tuple[int, str, int], tuple[str, int] | None] = {}
    for stage, order in local_orders.items():
        previous = None
        for phase, microbatch in order:
            previous_operation[(stage, phase, microbatch)] = previous
            previous = (phase, microbatch)

    expanded: list[TraceEvent] = []
    for stage in range(parallelism.pp):
        for phase in ("forward", "backward"):
            key = (stage, phase)
            for microbatch in range(microbatches):
                for event in groups[key]:
                    dependencies = [
                        duplicated_id(dependency, microbatch)
                        for dependency in event.dependencies
                    ]
                    if event.id in entries[key]:
                        previous = previous_operation[(stage, phase, microbatch)]
                        if previous is not None:
                            previous_phase, previous_microbatch = previous
                            dependencies.extend(
                                duplicated_id(exit_id, previous_microbatch)
                                for exit_id in exits[(stage, previous_phase)]
                            )
                        if phase == "forward":
                            if stage > 0:
                                dependencies.extend(
                                    duplicated_id(exit_id, microbatch)
                                    for exit_id in exits[(stage - 1, "forward")]
                                )
                        else:
                            # Backward also needs its own saved activation and the
                            # gradient produced by the downstream stage.
                            dependencies.extend(
                                duplicated_id(exit_id, microbatch)
                                for exit_id in exits[(stage, "forward")]
                            )
                            if stage < parallelism.pp - 1:
                                dependencies.extend(
                                    duplicated_id(exit_id, microbatch)
                                    for exit_id in exits[(stage + 1, "backward")]
                                )
                    metadata: dict[str, Any] = dict(event.metadata)
                    metadata["pipeline_microbatch"] = microbatch
                    _clone_compute_region_instance(
                        metadata,
                        f"stage{stage}:{phase}:mb{microbatch}",
                    )
                    raw_tp_lane = metadata.get("pipeline_tp_lane", 0)
                    if (
                        isinstance(raw_tp_lane, bool)
                        or not isinstance(raw_tp_lane, int)
                        or not 0 <= raw_tp_lane < parallelism.tp
                    ):
                        raise ValueError(
                            f"event {event.id!r} pipeline_tp_lane must be in "
                            f"[0, {parallelism.tp})"
                        )
                    metadata["pipeline_tp_lane"] = raw_tp_lane
                    physical_device = stage_lane_devices[stage][raw_tp_lane]
                    metadata["pipeline_physical_device"] = physical_device
                    group_role = event.group_role
                    if event.collective in {"send", "recv"}:
                        if group_role not in (None, "pp"):
                            raise ValueError(
                                f"pipeline P2P event {event.id!r} must use group_role='pp'"
                            )
                        if event.collective == "send":
                            source_stage = stage
                            destination_stage = (
                                stage + 1 if phase == "forward" else stage - 1
                            )
                        else:
                            source_stage = (
                                stage - 1 if phase == "forward" else stage + 1
                            )
                            destination_stage = stage
                        if not (
                            0 <= source_stage < parallelism.pp
                            and 0 <= destination_stage < parallelism.pp
                        ):
                            raise ValueError(
                                f"pipeline P2P event {event.id!r} has no adjacent "
                                f"stage in phase {phase!r}"
                            )
                        metadata.update(
                            {
                                "p2p_route_source": "pipeline-stage-map-v1",
                                "p2p_source_stage": source_stage,
                                "p2p_destination_stage": destination_stage,
                                "p2p_source_device": (
                                    stage_lane_devices[source_stage][raw_tp_lane]
                                ),
                                "p2p_destination_device": (
                                    stage_lane_devices[destination_stage][raw_tp_lane]
                                ),
                            }
                        )
                        group_role = "pp"
                    expanded.append(
                        replace(
                            event,
                            id=duplicated_id(event.id, microbatch),
                            device=physical_device,
                            dependencies=tuple(dict.fromkeys(dependencies)),
                            observed_start_us=None,
                            group_role=group_role,
                            metadata=metadata,
                        )
                    )
    result = WorkloadTrace(
        events=tuple(expanded),
        source=trace.source,
        metadata={
            **trace.metadata,
            "pipeline_expanded": True,
            "pipeline_expansion": {
                "schedule": schedule,
                "microbatches": microbatches,
                "stage_device_bases": list(stage_devices),
                "stage_device_lanes": [
                    list(lane_devices) for lane_devices in stage_lane_devices
                ],
                "stage_device_source": stage_device_source,
                "stage_rank_layout_source": rank_layout_source,
                "stage_ranks": (
                    None
                    if configured_stage_ranks is None
                    else [list(row) for row in configured_stage_ranks]
                ),
                "mapping_semantics": mapping_semantics,
            },
        },
    )
    result.validate()
    validate_compute_regions(result.events)
    return result


def expand_pipeline_ranks(
    trace: WorkloadTrace,
    parallelism: Parallelism,
    ranks: int,
) -> PipelineRankExpansion:
    """Compile a send-only pipeline template into rank-local Chakra DAGs.

    Cross-stage dependencies cannot appear in one rank's Chakra file.  Each
    template Send is therefore paired with a synthesized Recv on the adjacent
    destination rank, and destination dependencies are rewritten to that local
    Recv.  Explicit Recv templates and implicit transfers without a source Send
    are rejected because their pairing cannot be inferred safely.
    """

    expected_ranks = parallelism.tp * parallelism.pp * parallelism.dp
    if ranks != expected_ranks:
        raise ValueError("pipeline rank expansion requires ranks=tp*pp*dp")
    if any(event.collective == "recv" for event in trace.events):
        raise ValueError(
            "pipeline Chakra expansion requires a send-only template; "
            "receiver nodes are synthesized with exact peers and tags"
        )
    configuration = trace.metadata.get("pipeline")
    if not isinstance(configuration, dict):
        raise ValueError("pipeline rank expansion requires metadata.pipeline")
    (
        stage_ranks,
        rank_by_coordinates,
        coordinates_by_rank,
        rank_layout_source,
    ) = _stage_rank_layout(configuration, parallelism, ranks)
    expanded = expand_pipeline(trace, parallelism, gpus=ranks)
    expansion_metadata = expanded.metadata.get("pipeline_expansion", {})
    stage_bases = expansion_metadata.get("stage_device_bases")
    expected_bases = [stage * parallelism.tp for stage in range(parallelism.pp)]
    if rank_layout_source != "explicit-stage-ranks" and stage_bases != expected_bases:
        raise ValueError(
            "pipeline Chakra rank layout requires contiguous stage-major "
            f"placement {expected_bases} or an explicit stage_ranks map; "
            f"got stage_devices={stage_bases}"
        )

    events_by_id = {event.id: event for event in expanded.events}
    event_stages: dict[str, int] = {}
    for event in expanded.events:
        stage = event.metadata.get("pipeline_stage")
        if isinstance(stage, bool) or not isinstance(stage, int):
            raise ValueError(
                f"expanded pipeline event {event.id!r} lacks an integer stage"
            )
        event_stages[event.id] = stage

    sends = [event for event in expanded.events if event.collective == "send"]
    if not sends:
        raise ValueError(
            "pipeline Chakra expansion requires explicit Send exits for "
            "cross-stage activation and gradient transfers"
        )
    send_indices = {event.id: index for index, event in enumerate(sends)}
    interleaved = expansion_metadata.get("schedule") == "interleaved_1f1b"
    for event in sends:
        source_stage = event.metadata.get("p2p_source_stage")
        destination_stage = event.metadata.get("p2p_destination_stage")
        invalid = (
            isinstance(source_stage, bool)
            or not isinstance(source_stage, int)
            or isinstance(destination_stage, bool)
            or not isinstance(destination_stage, int)
            or source_stage != event_stages[event.id]
        )
        if interleaved:
            source_virtual_stage = event.metadata.get("p2p_source_virtual_stage")
            destination_virtual_stage = event.metadata.get(
                "p2p_destination_virtual_stage"
            )
            source_chunk = event.metadata.get("p2p_source_model_chunk")
            destination_chunk = event.metadata.get("p2p_destination_model_chunk")
            invalid = invalid or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    source_virtual_stage,
                    destination_virtual_stage,
                    source_chunk,
                    destination_chunk,
                )
            )
            if not invalid:
                assert isinstance(source_virtual_stage, int)
                assert isinstance(destination_virtual_stage, int)
                assert isinstance(source_stage, int)
                assert isinstance(destination_stage, int)
                assert isinstance(source_chunk, int)
                assert isinstance(destination_chunk, int)
                invalid = (
                    abs(source_virtual_stage - destination_virtual_stage) != 1
                    or divmod(source_virtual_stage, parallelism.pp)
                    != (source_chunk, source_stage)
                    or divmod(destination_virtual_stage, parallelism.pp)
                    != (destination_chunk, destination_stage)
                )
        else:
            invalid = invalid or abs(source_stage - destination_stage) != 1
        if invalid:
            raise ValueError(
                f"pipeline Send {event.id!r} lacks an exact adjacent virtual-stage route"
            )

    pair_count = len(sends) * parallelism.tp * parallelism.dp
    if pair_count > 2_147_483_647:
        raise ValueError("pipeline P2P pair count exceeds Chakra int32 tag capacity")

    rank_events: list[tuple[TraceEvent, ...]] = []
    for rank in range(ranks):
        tp_lane, stage, dp_replica = coordinates_by_rank[rank]
        local_template_events = [
            event for event in expanded.events if event_stages[event.id] == stage
        ]
        incoming_sends = [
            event
            for event in sends
            if event.metadata.get("p2p_destination_stage") == stage
        ]
        incoming_receive_ids = {
            event.id: f"scaletether-recv::{event.id}" for event in incoming_sends
        }
        existing_ids = {event.id for event in local_template_events}
        collisions = existing_ids & set(incoming_receive_ids.values())
        if collisions:
            raise ValueError(
                "pipeline synthesized Recv ids collide with template ids: "
                + ", ".join(sorted(collisions))
            )

        local_events: list[TraceEvent] = []
        consumed_receive_ids: set[str] = set()
        for event in local_template_events:
            dependencies: list[str] = []
            for dependency in event.dependencies:
                dependency_stage = event_stages[dependency]
                if dependency_stage == stage:
                    dependencies.append(dependency)
                    continue
                source_event = events_by_id[dependency]
                receive_id = incoming_receive_ids.get(dependency)
                if (
                    source_event.collective != "send"
                    or receive_id is None
                    or source_event.metadata.get("p2p_destination_stage") != stage
                ):
                    raise ValueError(
                        f"pipeline cross-stage dependency {dependency!r} -> "
                        f"{event.id!r} has no exact source Send"
                    )
                dependencies.append(receive_id)
                consumed_receive_ids.add(receive_id)

            metadata = dict(event.metadata)
            metadata.update(
                {
                    "pipeline_tp_lane": tp_lane,
                    "pipeline_dp_replica": dp_replica,
                    "pipeline_physical_device": rank,
                    "pipeline_rank_expansion": "chakra-send-recv-v1",
                    "pipeline_rank_layout_source": rank_layout_source,
                }
            )
            if event.collective == "send":
                destination_stage = int(metadata["p2p_destination_stage"])
                destination_rank = rank_by_coordinates[
                    (tp_lane, destination_stage, dp_replica)
                ]
                tag = 1 + (
                    (dp_replica * parallelism.tp + tp_lane) * len(sends)
                    + send_indices[event.id]
                )
                metadata.update(
                    {
                        "p2p_source_rank": rank,
                        "p2p_destination_rank": destination_rank,
                        "p2p_source_device": rank,
                        "p2p_destination_device": destination_rank,
                        "p2p_tag": tag,
                        "p2p_pair_id": f"{rank}->{destination_rank}:tag-{tag}",
                    }
                )
            local_events.append(
                replace(
                    event,
                    rank=rank,
                    device=rank,
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    metadata=metadata,
                )
            )

        for send in incoming_sends:
            source_stage = int(send.metadata["p2p_source_stage"])
            source_rank = rank_by_coordinates[(tp_lane, source_stage, dp_replica)]
            tag = 1 + (
                (dp_replica * parallelism.tp + tp_lane) * len(sends)
                + send_indices[send.id]
            )
            receive_id = incoming_receive_ids[send.id]
            metadata = dict(send.metadata)
            metadata.update(
                {
                    "pipeline_stage": stage,
                    "pipeline_tp_lane": tp_lane,
                    "pipeline_dp_replica": dp_replica,
                    "pipeline_physical_device": rank,
                    "pipeline_rank_expansion": "chakra-send-recv-v1",
                    "pipeline_rank_layout_source": rank_layout_source,
                    "p2p_source_rank": source_rank,
                    "p2p_destination_rank": rank,
                    "p2p_source_device": source_rank,
                    "p2p_destination_device": rank,
                    "p2p_tag": tag,
                    "p2p_pair_id": f"{source_rank}->{rank}:tag-{tag}",
                    "synthesized_from_send": send.id,
                }
            )
            destination_chunk = send.metadata.get("p2p_destination_model_chunk")
            if destination_chunk is not None:
                metadata["pipeline_model_chunk"] = destination_chunk
            destination_virtual_stage = send.metadata.get(
                "p2p_destination_virtual_stage"
            )
            if destination_virtual_stage is not None:
                metadata["pipeline_virtual_stage"] = destination_virtual_stage
            local_events.append(
                TraceEvent(
                    id=receive_id,
                    name=f"receive for {send.name}",
                    kind="collective",
                    duration_us=0.0,
                    stream=send.stream,
                    rank=rank,
                    device=rank,
                    collective="recv",
                    message_bytes=send.message_bytes,
                    group_role="pp",
                    group_size=parallelism.pp,
                    metadata=metadata,
                )
            )

        expected_receive_ids = set(incoming_receive_ids.values())
        if consumed_receive_ids != expected_receive_ids:
            unused = sorted(expected_receive_ids - consumed_receive_ids)
            raise ValueError(
                "pipeline Send exits are not consumed by destination-stage "
                f"dependencies: {unused}"
            )
        local_trace = WorkloadTrace(
            events=tuple(local_events),
            source=expanded.source,
            metadata=expanded.metadata,
        )
        local_trace.validate()
        validate_compute_regions(local_trace.events)
        rank_events.append(local_trace.events)

    return PipelineRankExpansion(
        rank_events=tuple(rank_events),
        template_send_count=len(sends),
        p2p_pair_count=pair_count,
        mapping_semantics=(
            "stage_ranks[stage][dp_replica*tp+tp_lane] selects each rank; "
            "each template Send has one synthesized destination-rank Recv"
        ),
        rank_layout_source=rank_layout_source,
        stage_ranks=stage_ranks,
        rank_coordinates=coordinates_by_rank,
    )
