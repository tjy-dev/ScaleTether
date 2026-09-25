from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .config import Parallelism
from .nccl_policy import parse_cga_cluster_size
from .pipeline import expand_pipeline_ranks
from .schema import TraceEvent, WorkloadTrace


CHAKRA_COMMIT = "21585d8f2d65603d791d2121e3d4632351120262"


class ChakraUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ChakraExport:
    prefix: str
    rank_files: tuple[str, ...]
    communicator_file: str
    quantized_events: int
    pipeline_expanded: bool = False
    p2p_pair_count: int = 0
    pipeline_rank_layout_source: str | None = None
    pipeline_stage_ranks: tuple[tuple[int, ...], ...] = ()
    local_chain_compaction: dict[str, Any] | None = None
    semantic_validation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "rank_files": list(self.rank_files),
            "communicator_file": self.communicator_file,
            "quantized_events": self.quantized_events,
            "pipeline_expanded": self.pipeline_expanded,
            "p2p_pair_count": self.p2p_pair_count,
            "pipeline_rank_layout_source": self.pipeline_rank_layout_source,
            "pipeline_stage_ranks": [list(row) for row in self.pipeline_stage_ranks],
            "local_chain_compaction": self.local_chain_compaction,
            "semantic_validation": self.semantic_validation,
            "chakra_commit": CHAKRA_COMMIT,
        }


def _encoded_duration_micros(event: TraceEvent) -> int:
    return max(1, math.ceil(event.duration_us))


def _materialize_counterfactual_p2p_tags(
    trace: WorkloadTrace,
    events_by_rank: dict[int, tuple[TraceEvent, ...]],
) -> tuple[dict[int, tuple[TraceEvent, ...]], dict[str, Any] | None]:
    """Lower exact framework P2P pairs to deterministic Chakra tags.

    The Transformer PP compiler predates the backend's explicit tag field but
    already freezes exact source/destination ranks, phase, and microbatch on
    both members of every pair.  Chakra needs an integer tag.  We derive it
    only for a declared semantic counterfactual, only when all P2P events omit
    tags, and only when every route key has exactly one send and one receive.
    Thus this is an auditable representation lowering, not a graph rewrite or
    an inference for arbitrary traces.
    """

    p2p = [
        event
        for events in events_by_rank.values()
        for event in events
        if event.collective in {"send", "recv"}
    ]
    if not p2p or all(isinstance(event.metadata.get("p2p_tag"), int) for event in p2p):
        return events_by_rank, None
    if trace.source.get("kind") != "framework-semantic-counterfactual":
        return events_by_rank, None
    if any(event.metadata.get("p2p_tag") is not None for event in p2p):
        raise ValueError("counterfactual P2P tags must be either complete or wholly absent")

    pairs: dict[tuple[int, int, str, int], dict[str, TraceEvent]] = {}
    for event in p2p:
        metadata = event.metadata
        source = metadata.get("p2p_source_rank")
        destination = metadata.get("p2p_destination_rank")
        phase = metadata.get("pipeline_phase")
        microbatch = metadata.get("pipeline_microbatch")
        if (
            isinstance(source, bool)
            or not isinstance(source, int)
            or isinstance(destination, bool)
            or not isinstance(destination, int)
            or not isinstance(phase, str)
            or isinstance(microbatch, bool)
            or not isinstance(microbatch, int)
        ):
            raise ValueError(
                "framework counterfactual P2P tag lowering requires exact route, "
                "phase, and microbatch metadata"
            )
        key = (source, destination, phase, microbatch)
        members = pairs.setdefault(key, {})
        if event.collective in members:
            raise ValueError("framework counterfactual P2P route is not one-to-one")
        members[event.collective] = event
    if any(set(members) != {"send", "recv"} for members in pairs.values()):
        raise ValueError("framework counterfactual P2P route lacks a matched send/recv")

    tags = {key: index for index, key in enumerate(sorted(pairs), start=1)}
    event_tags = {
        event.id: tags[key]
        for key, members in pairs.items()
        for event in members.values()
    }
    lowered: dict[int, tuple[TraceEvent, ...]] = {}
    for rank, events in events_by_rank.items():
        lowered[rank] = tuple(
            replace(
                event,
                metadata={
                    **event.metadata,
                    "p2p_tag": event_tags[event.id],
                    "p2p_tag_source": "canonical-framework-counterfactual-route-v1",
                },
            )
            if event.id in event_tags
            else event
            for event in events
        )
    return lowered, {
        "schema": "scaletether-chakra-counterfactual-p2p-tag-lowering-v1",
        "status": "exact-paired-route-materialization",
        "pair_count": len(pairs),
        "event_count": len(p2p),
        "ordering": "lexicographic-source-destination-phase-microbatch",
        "changes_dependencies_or_payloads": False,
    }


def _lower_chakra_p2p_peer_dependencies(
    events_by_rank: dict[int, tuple[TraceEvent, ...]],
) -> tuple[dict[int, tuple[TraceEvent, ...]], dict[str, Any] | None]:
    """Represent exact cross-rank Send->Recv edges with Chakra P2P pairing.

    Chakra execution traces have rank-local ``data_deps``.  Cross-rank P2P
    ordering is instead represented by the matching ``comm_src``, ``comm_dst``,
    and ``comm_tag`` attributes on SEND/RECV nodes.  A framework-generated PP
    graph retains an explicit Send->Recv dependency so its pre-export DAG can
    be validated as one global graph.  Remove only that redundant edge, and
    only after proving an exact one-to-one route/tag match.  Every other
    cross-rank dependency remains an export error.
    """

    if not any(
        event.collective in {"send", "recv"}
        for events in events_by_rank.values()
        for event in events
    ):
        # Ordinary TP/DP replication intentionally retains the same logical
        # event IDs on every rank. Global uniqueness is needed only when this
        # pass must reconcile a cross-rank P2P dependency.
        return events_by_rank, None

    for rank, events in events_by_rank.items():
        ids = [event.id for event in events]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Chakra P2P dependency lowering requires unique IDs within rank {rank}"
            )

    pair_members: dict[tuple[int, int, int], dict[str, TraceEvent]] = {}
    for events in events_by_rank.values():
        for event in events:
            if event.collective not in {"send", "recv"}:
                continue
            source = event.metadata.get("p2p_source_rank")
            destination = event.metadata.get("p2p_destination_rank")
            tag = event.metadata.get("p2p_tag")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (source, destination, tag)
            ):
                raise ValueError(
                    f"P2P event {event.id!r} requires integer p2p_source_rank, "
                    "p2p_destination_rank, and p2p_tag for Chakra export"
                )
            assert isinstance(source, int)
            assert isinstance(destination, int)
            assert isinstance(tag, int)
            members = pair_members.setdefault((source, destination, tag), {})
            if event.collective in members:
                raise ValueError("Chakra P2P route/tag is not one-to-one")
            members[event.collective] = event
    if any(set(members) != {"send", "recv"} for members in pair_members.values()):
        raise ValueError("Chakra P2P route/tag lacks a matched send/recv")

    removable: dict[tuple[int, str], str] = {}
    for members in pair_members.values():
        send = members["send"]
        recv = members["recv"]
        if send.rank == recv.rank:
            raise ValueError("Chakra P2P pair must cross ranks")
        if send.id in recv.dependencies:
            assert recv.rank is not None
            removable[(recv.rank, recv.id)] = send.id

    lowered: dict[int, tuple[TraceEvent, ...]] = {}
    for rank, events in events_by_rank.items():
        rank_ids = {event.id for event in events}
        projected = []
        for event in events:
            cross_rank = [
                dependency
                for dependency in event.dependencies
                if dependency not in rank_ids
            ]
            allowed = removable.get((rank, event.id))
            if any(dependency != allowed for dependency in cross_rank):
                raise ValueError(
                    f"event {event.id!r} has a cross-rank dependency that is not "
                    "its exact matched P2P send"
                )
            projected.append(
                replace(
                    event,
                    dependencies=tuple(
                        dependency
                        for dependency in event.dependencies
                        if dependency != allowed
                    ),
                )
                if allowed is not None
                else event
            )
        lowered[rank] = tuple(projected)
    if not removable:
        return events_by_rank, None
    return lowered, {
        "schema": "scaletether-chakra-p2p-peer-dependency-lowering-v1",
        "status": "exact-send-recv-edges-represented-by-chakra-p2p-pairing",
        "pair_count": len(pair_members),
        "replaced_cross_rank_edge_count": len(removable),
        "preserves_logical_happens_before": True,
        "changes_payloads": False,
    }


def _compact_local_chains(
    events: tuple[TraceEvent, ...],
) -> tuple[tuple[TraceEvent, ...], dict[str, int]]:
    """Collapse only provably linear local compute/memory chains.

    Chakra maps both compute and memory events to COMP_NODE. A chain is eligible
    only when every interior event has exactly one predecessor and one consumer,
    all members share rank/device/stream, and no collective or synchronization
    phase boundary in source order is crossed. Summing the already-quantized
    Chakra durations preserves the backend's local and total GPU cycle inputs;
    the retained phase boundaries protect compute/communication interleaving.
    """

    by_id = {event.id: event for event in events}
    event_index = {event.id: index for index, event in enumerate(events)}
    boundary_prefix = [0]
    for event in events:
        boundary_prefix.append(
            boundary_prefix[-1] + (event.kind in {"collective", "synchronization"})
        )
    consumers: dict[str, list[str]] = {event.id: [] for event in events}
    for event in events:
        for dependency in event.dependencies:
            consumers[dependency].append(event.id)

    eligible_kinds = {"compute", "memory"}

    def crosses_phase_boundary(predecessor: TraceEvent, successor: TraceEvent) -> bool:
        first = min(event_index[predecessor.id], event_index[successor.id])
        last = max(event_index[predecessor.id], event_index[successor.id])
        return boundary_prefix[last] != boundary_prefix[first + 1]

    def linked(predecessor: TraceEvent, successor: TraceEvent) -> bool:
        return (
            predecessor.kind in eligible_kinds
            and successor.kind in eligible_kinds
            and predecessor.rank == successor.rank
            and predecessor.device == successor.device
            and predecessor.stream == successor.stream
            and successor.dependencies == (predecessor.id,)
            and consumers[predecessor.id] == [successor.id]
            and not crosses_phase_boundary(predecessor, successor)
        )

    successor_by_id: dict[str, str] = {}
    predecessor_by_id: dict[str, str] = {}
    for event in events:
        if len(event.dependencies) != 1:
            continue
        predecessor = by_id[event.dependencies[0]]
        if linked(predecessor, event):
            successor_by_id[predecessor.id] = event.id
            predecessor_by_id[event.id] = predecessor.id

    chains: list[tuple[TraceEvent, ...]] = []
    member_to_chain: dict[str, tuple[TraceEvent, ...]] = {}
    for event in events:
        if event.id in predecessor_by_id or event.id not in successor_by_id:
            continue
        chain = [event]
        while chain[-1].id in successor_by_id:
            chain.append(by_id[successor_by_id[chain[-1].id]])
        if len(chain) < 2:
            continue
        frozen = tuple(chain)
        chains.append(frozen)
        for member in frozen:
            member_to_chain[member.id] = frozen

    replacement_id = {member.id: chain[0].id for chain in chains for member in chain}
    projected: list[TraceEvent] = []
    emitted_chains: set[str] = set()
    for event in events:
        chain = member_to_chain.get(event.id)
        if chain is None:
            projected.append(
                TraceEvent(
                    **{
                        **event.__dict__,
                        "dependencies": tuple(
                            replacement_id.get(dependency, dependency)
                            for dependency in event.dependencies
                        ),
                    }
                )
            )
            continue
        chain_id = chain[0].id
        if chain_id in emitted_chains:
            continue
        emitted_chains.add(chain_id)
        kind_counts = {
            kind: sum(member.kind == kind for member in chain)
            for kind in sorted(eligible_kinds)
        }
        duration_micros = sum(_encoded_duration_micros(member) for member in chain)
        projected.append(
            TraceEvent(
                id=chain_id,
                name=(f"scaletether:coalesced-local-chain:{chain[0].id}..{chain[-1].id}"),
                kind="compute",
                duration_us=float(duration_micros),
                stream=chain[0].stream,
                rank=chain[0].rank,
                device=chain[0].device,
                dependencies=tuple(
                    replacement_id.get(dependency, dependency)
                    for dependency in chain[0].dependencies
                ),
                observed_start_us=chain[0].observed_start_us,
                metadata={
                    "chakra_local_chain_compaction": {
                        "schema": "scaletether-chakra-local-chain-v1",
                        "source_event_count": len(chain),
                        "first_event_id": chain[0].id,
                        "last_event_id": chain[-1].id,
                        "source_kind_counts": kind_counts,
                        "encoded_duration_micros": duration_micros,
                    }
                },
            )
        )

    source_duration = sum(
        _encoded_duration_micros(event)
        for event in events
        if event.kind in eligible_kinds
    )
    projected_duration = sum(
        _encoded_duration_micros(event)
        for event in projected
        if event.kind in eligible_kinds
    )
    if source_duration != projected_duration:
        raise RuntimeError("local-chain compaction changed encoded local duration")
    projected_trace = WorkloadTrace(events=tuple(projected))
    projected_trace.validate()
    return projected_trace.events, {
        "source_nodes": len(events),
        "projected_nodes": len(projected),
        "collapsed_nodes": len(events) - len(projected),
        "coalesced_chains": len(chains),
        "source_collective_nodes": sum(event.kind == "collective" for event in events),
        "projected_collective_nodes": sum(
            event.kind == "collective" for event in projected
        ),
        "source_local_duration_micros": source_duration,
        "projected_local_duration_micros": projected_duration,
    }


def _chakra_modules() -> tuple[Any, Any, Any]:
    try:
        from chakra.schema.protobuf import et_def_pb2
        from chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage
    except (ImportError, RuntimeError) as error:
        raise ChakraUnavailable(
            "Chakra export requires `python -m pip install -e '.[chakra]'` from the repository root"
        ) from error
    return et_def_pb2, encodeMessage, decodeMessage


def _validate_encoded_rank(
    proto: Any,
    decode_message: Any,
    path: Path,
    expected_metadata: Any,
    expected_nodes: list[Any],
) -> tuple[list[Any], str]:
    """Decode one just-written ET file and require exact protobuf equality."""

    with path.open("rb") as handle:
        actual_metadata = proto.GlobalMetadata()
        if not decode_message(handle, actual_metadata):
            raise RuntimeError(f"Chakra semantic validation found no metadata: {path}")
        if actual_metadata != expected_metadata:
            raise RuntimeError(f"Chakra metadata changed during encoding: {path}")
        actual_nodes = []
        for expected in expected_nodes:
            actual = proto.Node()
            if not decode_message(handle, actual):
                raise RuntimeError(
                    f"Chakra file ended before node {expected.id}: {path}"
                )
            if actual != expected:
                raise RuntimeError(
                    f"Chakra node {expected.id} changed during encoding: {path}"
                )
            actual_nodes.append(actual)
        trailing = proto.Node()
        if decode_message(handle, trailing):
            raise RuntimeError(
                f"Chakra file contains an unexpected trailing node {trailing.id}: {path}"
            )

    digest = hashlib.sha256()
    metadata_bytes = actual_metadata.SerializeToString(deterministic=True)
    digest.update(len(metadata_bytes).to_bytes(8, "big"))
    digest.update(metadata_bytes)
    for node in actual_nodes:
        encoded = node.SerializeToString(deterministic=True)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return actual_nodes, digest.hexdigest()


def _rank_coordinates(rank: int, parallelism: Parallelism) -> tuple[int, int, int]:
    tp = rank % parallelism.tp
    rank //= parallelism.tp
    pp = rank % parallelism.pp
    dp = rank // parallelism.pp
    return tp, pp, dp


def _communicators(
    ranks: int,
    parallelism: Parallelism,
    rank_coordinates: tuple[tuple[int, int, int], ...] | None = None,
) -> tuple[dict[str, list[int]], dict[tuple[int, str], str]]:
    groups: dict[str, list[int]] = {}
    rank_roles: dict[tuple[int, str], str] = {}
    next_id = 1
    for role in ("tp", "pp", "dp"):
        role_groups: dict[tuple[int, int], list[int]] = {}
        for rank in range(ranks):
            tp, pp, dp = (
                _rank_coordinates(rank, parallelism)
                if rank_coordinates is None
                else rank_coordinates[rank]
            )
            key = (pp, dp) if role == "tp" else ((tp, dp) if role == "pp" else (tp, pp))
            role_groups.setdefault(key, []).append(rank)
        for members in role_groups.values():
            group_id = str(next_id)
            next_id += 1
            groups[group_id] = members
            for rank in members:
                rank_roles[(rank, role)] = group_id
    return groups, rank_roles


def _node_type_and_collective(proto: Any, event: TraceEvent) -> tuple[int, int | None]:
    if event.kind != "collective":
        return proto.COMP_NODE, None
    mapping = {
        "all_reduce": proto.ALL_REDUCE,
        "all_gather": proto.ALL_GATHER,
        "reduce_scatter": proto.REDUCE_SCATTER,
        "broadcast": proto.BROADCAST,
        "all_to_all": proto.ALL_TO_ALL,
        "reduce": proto.REDUCE,
        "gather": proto.GATHER,
        "scatter": proto.SCATTER,
        "barrier": proto.BARRIER,
    }
    if event.collective == "send":
        return proto.COMM_SEND_NODE, None
    if event.collective == "recv":
        return proto.COMM_RECV_NODE, None
    if event.collective not in mapping:
        raise ValueError(
            f"event {event.id!r} has no Chakra mapping for collective {event.collective!r}"
        )
    return proto.COMM_COLL_NODE, mapping[event.collective]


def _expand_physical_logical_collectives(
    events: tuple[TraceEvent, ...],
) -> tuple[TraceEvent, ...]:
    """Lower compound physical NCCL events to Chakra logical operations.

    ASTRA/Chakra consumes network operations, while capture retains one event
    per physical NCCL kernel.  A fused SendRecv kernel is therefore projected
    to parallel Send and Recv nodes only at export.  The physical duration is
    carried by the first logical node exactly once; additional logical nodes
    carry zero local duration.  Dependencies on the physical event become a
    join over every projected logical operation.
    """

    expansions: dict[str, tuple[str, ...]] = {}
    for event in events:
        raw = event.metadata.get("logical_collective_operations")
        if raw is None:
            continue
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                f"event {event.id!r} has malformed logical_collective_operations"
            )
        expansions[event.id] = tuple(
            f"{event.id}#logical-{index}" for index in range(len(raw))
        )

    projected: list[TraceEvent] = []
    for event in events:
        raw = event.metadata.get("logical_collective_operations")
        if raw is None:
            projected.append(event)
            continue
        assert isinstance(raw, list)
        for index, operation in enumerate(raw):
            if (
                not isinstance(operation, dict)
                or operation.get("schema") != "logical-collective-operation-v1"
                or operation.get("collective") not in {"send", "recv"}
                or isinstance(operation.get("message_bytes"), bool)
                or not isinstance(operation.get("message_bytes"), int)
                or int(operation["message_bytes"]) <= 0
                or isinstance(operation.get("group_size"), bool)
                or not isinstance(operation.get("group_size"), int)
                or int(operation["group_size"]) <= 0
            ):
                raise ValueError(
                    f"event {event.id!r} contains an invalid logical collective"
                )
            metadata = {**event.metadata, **operation}
            metadata["physical_event_id"] = event.id
            metadata["physical_duration_owner"] = index == 0
            metadata["physical_duration_us"] = event.duration_us
            projected.append(
                replace(
                    event,
                    id=expansions[event.id][index],
                    name=f"{event.name} [{operation['collective']} logical]",
                    duration_us=event.duration_us if index == 0 else 0.0,
                    collective=str(operation["collective"]),
                    message_bytes=int(operation["message_bytes"]),
                    group_size=int(operation["group_size"]),
                    group_role="pp",
                    metadata=metadata,
                )
            )

    rewritten: list[TraceEvent] = []
    for event in projected:
        dependencies: list[str] = []
        for dependency in event.dependencies:
            dependencies.extend(expansions.get(dependency, (dependency,)))
        rewritten.append(replace(event, dependencies=tuple(dict.fromkeys(dependencies))))
    ids = [event.id for event in rewritten]
    if len(ids) != len(set(ids)):
        raise ValueError("logical collective projection produced duplicate event IDs")
    return tuple(rewritten)


def export_chakra(
    trace: WorkloadTrace,
    output_prefix: Path,
    ranks: int,
    parallelism: Parallelism,
    max_ctas: int | None = None,
    cta_policy: dict[str, int] | None = None,
    comm_stream_priority: str = "normal",
    nccl_cta_policy: int = 0,
    nccl_nvls_ctas: int | None = None,
    nccl_cga_cluster_size: int | None = None,
    compact_local_chains: bool = False,
) -> ChakraExport:
    cta_policy = {} if cta_policy is None else cta_policy
    nccl_cga_cluster_size = parse_cga_cluster_size(nccl_cga_cluster_size)
    if ranks != parallelism.tp * parallelism.pp * parallelism.dp:
        raise ValueError("Chakra export ranks must equal tp*pp*dp")
    pipeline_expanded = trace.metadata.get("pipeline") is not None
    p2p_pair_count = 0
    pipeline_rank_layout_source = None
    pipeline_stage_ranks: tuple[tuple[int, ...], ...] = ()
    rank_coordinates = None
    if pipeline_expanded:
        if compact_local_chains:
            raise ValueError(
                "Chakra local-chain compaction does not support pipeline-expanded "
                "traces because P2P scheduling boundaries require full fidelity"
            )
        pipeline_ranks = expand_pipeline_ranks(trace, parallelism, ranks)
        events_by_rank = {
            rank: events for rank, events in enumerate(pipeline_ranks.rank_events)
        }
        p2p_pair_count = pipeline_ranks.p2p_pair_count
        pipeline_rank_layout_source = pipeline_ranks.rank_layout_source
        pipeline_stage_ranks = pipeline_ranks.stage_ranks
        rank_coordinates = pipeline_ranks.rank_coordinates
    else:
        source_ranks = {event.rank for event in trace.events}
        if source_ranks == {0}:
            events_by_rank = {rank: trace.events for rank in range(ranks)}
        elif source_ranks == set(range(ranks)):
            events_by_rank = {
                rank: tuple(event for event in trace.events if event.rank == rank)
                for rank in range(ranks)
            }
        else:
            raise ValueError(
                "multi-rank Chakra export requires captured ranks to equal target ranks; "
                "use a framework adapter before counterfactual rank expansion"
            )

    # Captured fused SendRecv kernels carry their exact logical operations in
    # metadata.  Project those operations before validating route/tag
    # uniqueness: the physical owner can legitimately expose a placeholder
    # top-level tag, while each recorded logical member has its own reconciled
    # cross-rank identity.
    events_by_rank = {
        rank: _expand_physical_logical_collectives(events)
        for rank, events in events_by_rank.items()
    }
    events_by_rank, p2p_tag_lowering = _materialize_counterfactual_p2p_tags(
        trace, events_by_rank
    )
    events_by_rank, p2p_dependency_lowering = (
        _lower_chakra_p2p_peer_dependencies(events_by_rank)
    )

    proto, encode_message, decode_message = _chakra_modules()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    groups, rank_roles = _communicators(
        ranks, parallelism, rank_coordinates=rank_coordinates
    )
    communicator_path = output_prefix.with_name(
        output_prefix.name + ".comm-groups.json"
    )
    communicator_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    rank_files = []
    quantized = 0
    compaction_totals = {
        "source_nodes": 0,
        "projected_nodes": 0,
        "collapsed_nodes": 0,
        "coalesced_chains": 0,
        "source_collective_nodes": 0,
        "projected_collective_nodes": 0,
        "source_local_duration_micros": 0,
        "projected_local_duration_micros": 0,
    }
    semantic_rank_digests: list[str] = []
    semantic_node_count = 0
    semantic_data_edge_count = 0
    semantic_collective_node_count = 0
    semantic_p2p_node_count = 0
    semantic_communication_bytes = 0

    for rank in range(ranks):
        rank_events = events_by_rank[rank]
        if compact_local_chains:
            rank_events, rank_compaction = _compact_local_chains(rank_events)
            for key, value in rank_compaction.items():
                compaction_totals[key] += value
        node_ids = {event.id: index + 1 for index, event in enumerate(rank_events)}
        rank_path = output_prefix.with_name(f"{output_prefix.name}.{rank}.et")
        expected_nodes = []
        metadata = proto.GlobalMetadata(version="1.0.0")
        metadata.attr.extend(
            [
                proto.AttributeProto(
                    name="scaletether.schema_version", string_val=trace.schema_version
                ),
                proto.AttributeProto(
                    name="scaletether.chakra_commit", string_val=CHAKRA_COMMIT
                ),
            ]
        )
        with rank_path.open("wb") as handle:
            encode_message(handle, metadata)
            for event in rank_events:
                node_type, collective_type = _node_type_and_collective(proto, event)
                node = proto.Node(
                    id=node_ids[event.id],
                    name=event.name,
                    type=node_type,
                    start_time_micros=max(0, int(event.observed_start_us or 0)),
                    duration_micros=(
                        0
                        if event.metadata.get("physical_duration_owner") is False
                        else max(1, math.ceil(event.duration_us))
                    ),
                )
                if event.duration_us != float(node.duration_micros):
                    quantized += 1
                unknown_dependencies = set(event.dependencies) - set(node_ids)
                if unknown_dependencies:
                    raise ValueError(
                        f"event {event.id!r} has cross-rank Chakra dependencies "
                        f"{sorted(unknown_dependencies)}"
                    )
                node.data_deps.extend(
                    node_ids[dependency] for dependency in event.dependencies
                )
                node.attr.extend(
                    [
                        proto.AttributeProto(name="is_cpu_op", bool_val=False),
                        proto.AttributeProto(
                            name="scaletether.event_id", string_val=event.id
                        ),
                        proto.AttributeProto(
                            name="scaletether.stream", string_val=event.stream
                        ),
                    ]
                )
                if isinstance(event.metadata.get("physical_event_id"), str):
                    node.attr.extend(
                        [
                            proto.AttributeProto(
                                name="scaletether.physical_event_id",
                                string_val=str(event.metadata["physical_event_id"]),
                            ),
                            proto.AttributeProto(
                                name="scaletether.physical_duration_owner",
                                bool_val=bool(
                                    event.metadata.get("physical_duration_owner")
                                ),
                            ),
                        ]
                    )
                event_max_ctas = (
                    cta_policy.get(event.group_role, max_ctas)
                    if event.group_role is not None
                    else max_ctas
                )
                if event_max_ctas is not None:
                    node.attr.append(
                        proto.AttributeProto(
                            name="scaletether.max_ctas", int64_val=event_max_ctas
                        )
                    )
                if event.kind == "collective":
                    node.attr.extend(
                        [
                            proto.AttributeProto(
                                name="scaletether.comm_stream_priority",
                                string_val=comm_stream_priority,
                            ),
                            proto.AttributeProto(
                                name="scaletether.nccl_cta_policy",
                                int64_val=nccl_cta_policy,
                            ),
                            proto.AttributeProto(
                                name="scaletether.zero_cta_eligible",
                                bool_val=bool(
                                    event.metadata.get(
                                        "symmetric_registered_buffers", False
                                    )
                                    and event.collective in {"all_gather", "all_to_all"}
                                ),
                            ),
                        ]
                    )
                    if nccl_nvls_ctas is not None:
                        node.attr.append(
                            proto.AttributeProto(
                                name="scaletether.nccl_nvls_ctas",
                                int64_val=nccl_nvls_ctas,
                            )
                        )
                    if nccl_cga_cluster_size is not None:
                        node.attr.append(
                            proto.AttributeProto(
                                name="scaletether.nccl_cga_cluster_size",
                                int64_val=nccl_cga_cluster_size,
                            )
                        )
                if event.kind == "memory":
                    node.attr.append(
                        proto.AttributeProto(
                            name="scaletether.original_kind", string_val="memory"
                        )
                    )
                if event.kind == "collective":
                    if event.message_bytes is None and event.collective != "barrier":
                        raise ValueError(
                            f"collective event {event.id!r} lacks message_bytes for Chakra"
                        )
                    node.attr.append(
                        proto.AttributeProto(
                            name="comm_size",
                            int64_val=event.message_bytes or 0,
                        )
                    )
                    if collective_type is not None:
                        node.attr.append(
                            proto.AttributeProto(
                                name="comm_type", int64_val=collective_type
                            )
                        )
                    if event.collective in {"send", "recv"}:
                        source_rank = event.metadata.get("p2p_source_rank")
                        destination_rank = event.metadata.get("p2p_destination_rank")
                        tag = event.metadata.get("p2p_tag")
                        if any(
                            isinstance(value, bool) or not isinstance(value, int)
                            for value in (source_rank, destination_rank, tag)
                        ):
                            raise ValueError(
                                f"P2P event {event.id!r} requires integer "
                                "p2p_source_rank, p2p_destination_rank, and p2p_tag "
                                "for Chakra export"
                            )
                        assert isinstance(source_rank, int)
                        assert isinstance(destination_rank, int)
                        assert isinstance(tag, int)
                        if (
                            not 0 <= source_rank < ranks
                            or not 0 <= destination_rank < ranks
                            or source_rank == destination_rank
                            or not 0 <= tag <= 2_147_483_647
                        ):
                            raise ValueError(
                                f"P2P event {event.id!r} has invalid Chakra "
                                f"route/tag {source_rank}->{destination_rank} tag={tag}"
                            )
                        if (event.collective == "send" and source_rank != rank) or (
                            event.collective == "recv" and destination_rank != rank
                        ):
                            raise ValueError(
                                f"P2P event {event.id!r} is placed on rank {rank} "
                                f"but routes {source_rank}->{destination_rank}"
                            )
                        node.attr.extend(
                            [
                                proto.AttributeProto(
                                    name="comm_src", int32_val=source_rank
                                ),
                                proto.AttributeProto(
                                    name="comm_dst", int32_val=destination_rank
                                ),
                                proto.AttributeProto(name="comm_tag", int32_val=tag),
                                proto.AttributeProto(
                                    name="scaletether.p2p_pair_id",
                                    string_val=str(
                                        event.metadata.get(
                                            "p2p_pair_id",
                                            f"{source_rank}->{destination_rank}:tag-{tag}",
                                        )
                                    ),
                                ),
                            ]
                        )
                        pipeline_p2p_mode = event.metadata.get("pipeline_p2p_mode")
                        if pipeline_p2p_mode is not None:
                            if pipeline_p2p_mode not in {
                                "blocking",
                                "asynchronous",
                            }:
                                raise ValueError(
                                    f"P2P event {event.id!r} has invalid "
                                    "pipeline_p2p_mode"
                                )
                            node.attr.append(
                                proto.AttributeProto(
                                    name="scaletether.pipeline_p2p_mode",
                                    string_val=pipeline_p2p_mode,
                                )
                            )
                    if event.group_role is not None:
                        try:
                            group_id = rank_roles[(rank, event.group_role)]
                        except KeyError as error:
                            raise ValueError(
                                f"unsupported group role {event.group_role!r} for Chakra export"
                            ) from error
                        node.attr.extend(
                            [
                                proto.AttributeProto(
                                    name="pg_name", string_val=group_id
                                ),
                                proto.AttributeProto(
                                    name="scaletether.group_role",
                                    string_val=event.group_role,
                                ),
                            ]
                        )
                expected_nodes.append(node)
                encode_message(handle, node)
        decoded_nodes, rank_digest = _validate_encoded_rank(
            proto,
            decode_message,
            rank_path,
            metadata,
            expected_nodes,
        )
        semantic_rank_digests.append(rank_digest)
        semantic_node_count += len(decoded_nodes)
        for node in decoded_nodes:
            semantic_data_edge_count += len(node.data_deps)
            if node.type in {
                proto.COMM_COLL_NODE,
                proto.COMM_SEND_NODE,
                proto.COMM_RECV_NODE,
            }:
                semantic_collective_node_count += 1
                attributes = {attribute.name: attribute for attribute in node.attr}
                semantic_communication_bytes += attributes["comm_size"].int64_val
            if node.type in {proto.COMM_SEND_NODE, proto.COMM_RECV_NODE}:
                semantic_p2p_node_count += 1
        rank_files.append(str(rank_path))

    aggregate_digest = hashlib.sha256()
    for rank, rank_digest in enumerate(semantic_rank_digests):
        aggregate_digest.update(rank.to_bytes(8, "big"))
        aggregate_digest.update(bytes.fromhex(rank_digest))

    return ChakraExport(
        prefix=str(output_prefix),
        rank_files=tuple(rank_files),
        communicator_file=str(communicator_path),
        quantized_events=quantized,
        pipeline_expanded=pipeline_expanded,
        p2p_pair_count=p2p_pair_count,
        pipeline_rank_layout_source=pipeline_rank_layout_source,
        pipeline_stage_ranks=pipeline_stage_ranks,
        local_chain_compaction=(
            {
                "schema": "scaletether-chakra-local-chain-compaction-v1",
                "mode": ("linear-compute-memory-phase-boundary-exact-encoded-duration"),
                **compaction_totals,
                "encoded_local_duration_preserved": (
                    compaction_totals["source_local_duration_micros"]
                    == compaction_totals["projected_local_duration_micros"]
                ),
                "collective_nodes_preserved": (
                    compaction_totals["source_collective_nodes"]
                    == compaction_totals["projected_collective_nodes"]
                ),
                "wall_time_equivalence": "not-claimed",
                "warning": (
                    "local-chain projection preserves encoded local-duration and "
                    "collective-node totals, but coarser backend scheduling may "
                    "change compute/communication overlap and wall cycles"
                ),
            }
            if compact_local_chains
            else None
        ),
        semantic_validation={
            "schema": "scaletether-chakra-semantic-validation-v1",
            "status": "exact-protobuf-roundtrip",
            "rank_files_validated": len(semantic_rank_digests),
            "node_count": semantic_node_count,
            "data_dependency_edge_count": semantic_data_edge_count,
            "collective_node_count": semantic_collective_node_count,
            "p2p_node_count": semantic_p2p_node_count,
            "encoded_communication_bytes": semantic_communication_bytes,
            "rank_semantic_sha256": semantic_rank_digests,
            "aggregate_semantic_sha256": aggregate_digest.hexdigest(),
            "counterfactual_p2p_tag_lowering": p2p_tag_lowering,
            "p2p_peer_dependency_lowering": p2p_dependency_lowering,
            "claim": (
                "exact exporter-intent/protobuf equality; backend timing accuracy "
                "is not implied"
            ),
        },
    )
