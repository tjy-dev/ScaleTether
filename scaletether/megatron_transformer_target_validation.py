"""Fail-closed physical-target gates for the frozen Transformer Level 2 study."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .megatron_transformer_family import COMMIT, validate_matrix
from .megatron_transformer_rules import (
    TransformerRuleError,
    derive_pinned_ddp_contract,
)


SCHEMA = "scaletether-transformer-physical-target-validation-v1"
COMPONENT_PATHS = (
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
)


def _fail(failures: list[str], condition: bool, label: str) -> None:
    if not condition:
        failures.append(label)


def _validated_domain(matrix: dict[str, Any]) -> dict[str, Any]:
    """Return the exact family domain after ``validate_matrix`` admits it.

    ``validate_matrix`` compares the complete domain mapping against the
    versioned preregistration.  Keeping the admitted mapping here avoids a
    second, drifting set of model constants in the physical-target gate.
    """

    domain = matrix.get("domain")
    if not isinstance(domain, dict):  # defensive; validate_matrix rejects it
        raise ValueError("validated family domain is unavailable")
    return domain


def _phase(event: Any) -> str | None:
    if not isinstance(event, dict):
        return None
    marker = event.get("metadata", {}).get("framework_phase_marker", {})
    return marker.get("name") if isinstance(marker, dict) else None


def physical_pp_logical_collectives(workload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact logical PP/TP calls proven against native NCCL.

    A single NCCL ``SendRecv`` kernel can implement two ordered P2P calls, and
    Kineto may omit or duplicate a physical activity.  The merged trace keeps
    each rank's complete declared-step framework ledger plus the independent
    native reconciliation.  Use that ledger only when every in-step call was
    reconciled; otherwise fail closed rather than infer logical multiplicity
    from physical kernel count.
    """

    metadata = workload.get("metadata")
    rank_metadata = (
        metadata.get("rank_metadata") if isinstance(metadata, dict) else None
    )
    if not isinstance(rank_metadata, list) or len(rank_metadata) != 4:
        raise ValueError("PP workload lacks four rank metadata records")
    rows: list[dict[str, Any]] = []
    observed_ranks: set[int] = set()
    for record in rank_metadata:
        if not isinstance(record, dict):
            raise ValueError("PP rank metadata record is malformed")
        rank = record.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in range(4):
            raise ValueError("PP rank metadata has an invalid rank")
        if rank in observed_ranks:
            raise ValueError("PP rank metadata contains a duplicate rank")
        observed_ranks.add(rank)
        scope = record.get("declared_step_collective_scope")
        binding = record.get("direct_native_nccl_binding")
        observations = record.get("collective_observations")
        if (
            not isinstance(scope, dict)
            or scope.get("status") != "complete"
            or scope.get("unresolved_count") != 0
            or not isinstance(binding, dict)
            or binding.get("status") != "framework-owned-fully-reconciled"
            or not isinstance(observations, list)
        ):
            raise ValueError(
                "PP logical collectives lack complete native reconciliation"
            )
        inside = [
            observation
            for observation in observations
            if isinstance(observation, dict)
            and observation.get("capture_scope") == "inside-declared-step"
        ]
        if (
            scope.get("inside_count") != len(inside)
            or binding.get("operation_count") != len(inside)
            or binding.get("framework_reconciliation_count") != len(inside)
        ):
            raise ValueError("PP logical/native collective cardinality is inconsistent")
        indices: set[int] = set()
        for observation in inside:
            index = observation.get("index")
            collective = observation.get("collective")
            role = observation.get("group_role_hint")
            message_bytes = observation.get("message_bytes")
            members = observation.get("process_group_ranks")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index in indices
                or collective not in {"all_reduce", "send", "recv"}
                or role not in {"tp", "pp"}
                or isinstance(message_bytes, bool)
                or not isinstance(message_bytes, int)
                or message_bytes <= 0
                or not isinstance(members, list)
                or not members
                or any(
                    isinstance(member, bool) or not isinstance(member, int)
                    for member in members
                )
                or len(set(members)) != len(members)
            ):
                raise ValueError("PP logical collective record is malformed")
            indices.add(index)
            row = {
                "rank": rank,
                "index": index,
                "collective": collective,
                "group_role": role,
                "message_bytes": message_bytes,
                "process_group_ranks": list(members),
                "framework_marker_stack": observation.get("framework_marker_stack"),
            }
            if collective in {"send", "recv"}:
                source = observation.get("p2p_source_rank")
                destination = observation.get("p2p_destination_rank")
                if (
                    isinstance(source, bool)
                    or not isinstance(source, int)
                    or isinstance(destination, bool)
                    or not isinstance(destination, int)
                    or source == destination
                ):
                    raise ValueError("PP logical P2P route is malformed")
                row.update(
                    p2p_source_rank=source,
                    p2p_destination_rank=destination,
                )
            rows.append(row)
    if observed_ranks != {0, 1, 2, 3}:
        raise ValueError("PP rank metadata coverage is incomplete")
    return rows


def _pp_logical_event_bindings(
    workload: dict[str, Any], logical: list[dict[str, Any]]
) -> dict[tuple[int, int], str]:
    """Bind each logical framework call to one immutable physical event ID."""

    events = workload.get("events")
    metadata = workload.get("metadata")
    rank_metadata = (
        metadata.get("rank_metadata") if isinstance(metadata, dict) else None
    )
    if not isinstance(events, list) or not isinstance(rank_metadata, list):
        raise ValueError("PP send-readiness binding lacks events or rank metadata")
    event_ids = {
        event.get("id")
        for event in events
        if isinstance(event, dict) and isinstance(event.get("id"), str)
    }
    expected = {(row["rank"], row["index"]) for row in logical}
    mappings: dict[tuple[int, int], str] = {}
    for record in rank_metadata:
        if not isinstance(record, dict) or not isinstance(record.get("rank"), int):
            raise ValueError("PP send-readiness rank metadata is malformed")
        rank = record["rank"]
        binding = record.get("direct_native_nccl_binding")
        if not isinstance(binding, dict):
            raise ValueError("PP send-readiness lacks native binding")
        for field in ("bindings", "general_fallback_bindings"):
            rows = binding.get(field, [])
            if not isinstance(rows, list):
                raise ValueError("PP send-readiness binding rows are malformed")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("PP send-readiness binding row is malformed")
                event_id = row.get("event_id")
                sequences = row.get("framework_collective_sequences")
                if not isinstance(event_id, str) or not isinstance(sequences, list):
                    continue
                merged_id = (
                    event_id
                    if event_id.startswith("rank-")
                    else f"rank-{rank}:{event_id}"
                )
                for sequence in sequences:
                    if isinstance(sequence, bool) or not isinstance(sequence, int):
                        raise ValueError("PP send-readiness sequence is malformed")
                    key = (rank, sequence)
                    if key in mappings and mappings[key] != merged_id:
                        raise ValueError(
                            "PP logical send maps to multiple physical events"
                        )
                    mappings[key] = merged_id
    for event in events:
        if not isinstance(event, dict) or event.get("kind") != "collective":
            continue
        rank = event.get("rank")
        event_id = event.get("id")
        event_metadata = event.get("metadata")
        sequence = (
            event_metadata.get("collective_sequence")
            if isinstance(event_metadata, dict)
            else None
        )
        if (
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and isinstance(event_id, str)
            and isinstance(sequence, int)
            and not isinstance(sequence, bool)
        ):
            mappings.setdefault((rank, sequence), event_id)
    unknown_event = any(event_id not in event_ids for event_id in mappings.values())
    if set(mappings) != expected or unknown_event:
        raise ValueError("PP logical/event binding is incomplete for send readiness")
    return mappings


def _pp_send_producer_readiness(
    workload: dict[str, Any], logical: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require observed compute-to-send readiness without manufacturing edges."""

    events = workload.get("events")
    if not isinstance(events, list):
        raise ValueError("PP send readiness lacks events")
    by_id = {
        event["id"]: event
        for event in events
        if isinstance(event, dict) and isinstance(event.get("id"), str)
    }
    if len(by_id) != len(events):
        raise ValueError("PP send readiness requires unique event IDs")
    visiting: set[str] = set()
    ancestors: dict[str, frozenset[str]] = {}

    def visit(identifier: str) -> frozenset[str]:
        if identifier in ancestors:
            return ancestors[identifier]
        if identifier in visiting:
            raise ValueError("PP send-readiness dependency cycle")
        visiting.add(identifier)
        event = by_id[identifier]
        dependencies = event.get("dependencies", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(dependency, str) or dependency not in by_id
            for dependency in dependencies
        ):
            raise ValueError("PP send-readiness dependency is unresolved")
        reached: set[str] = set(dependencies)
        for dependency in dependencies:
            reached.update(visit(dependency))
        visiting.remove(identifier)
        ancestors[identifier] = frozenset(reached)
        return ancestors[identifier]

    bindings = _pp_logical_event_bindings(workload, logical)
    producer_groups: dict[tuple[int, str], list[set[str]]] = {}
    for rank in range(4):
        stage = rank // 2
        for phase, suffix in (
            ("forward", "mlp.linear_fc2"),
            ("backward", "self_attention.linear_qkv"),
        ):
            marker_name = (
                f"megatron_transformer_{phase}_component:layers.{stage}.{suffix}"
            )
            groups: list[set[str]] = []
            group_by_instance: dict[str, set[str]] = {}
            for event in events:
                if not isinstance(event, dict) or event.get("rank") != rank:
                    continue
                marker = event.get("metadata", {}).get("framework_phase_marker", {})
                if not isinstance(marker, dict) or marker.get("name") != marker_name:
                    continue
                instance = marker.get("instance_id")
                if not isinstance(instance, str) or not instance:
                    instance = f"event:{event['id']}"
                if instance not in group_by_instance:
                    group_by_instance[instance] = set()
                    groups.append(group_by_instance[instance])
                group_by_instance[instance].add(event["id"])
            producer_groups[(rank, phase)] = groups

    sends_by_rank_phase: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for operation in logical:
        if operation.get("group_role") != "pp" or operation.get("collective") != "send":
            continue
        source = operation["p2p_source_rank"]
        destination = operation["p2p_destination_rank"]
        source_stage, destination_stage = source // 2, destination // 2
        if (source_stage, destination_stage) == (0, 1):
            phase = "forward"
        elif (source_stage, destination_stage) == (1, 0):
            phase = "backward"
        else:
            raise ValueError("PP send readiness found a non-stage-crossing route")
        sends_by_rank_phase.setdefault((operation["rank"], phase), []).append(operation)

    missing: list[dict[str, Any]] = []
    checked = 0
    for key, sends in sorted(sends_by_rank_phase.items()):
        sends.sort(key=lambda row: row["index"])
        groups = producer_groups.get(key, [])
        for occurrence, operation in enumerate(sends):
            checked += 1
            event_id = bindings[(operation["rank"], operation["index"])]
            producer_ids = groups[occurrence] if occurrence < len(groups) else set()
            if not producer_ids or not (producer_ids & set(visit(event_id))):
                missing.append(
                    {
                        "rank": operation["rank"],
                        "phase": key[1],
                        "occurrence": occurrence,
                        "logical_index": operation["index"],
                        "physical_event_id": event_id,
                        "expected_producer_event_ids": sorted(producer_ids),
                    }
                )
    return {
        "schema": "scaletether-physical-pp-send-producer-readiness-v1",
        "passed": not missing and checked > 0,
        "checked_send_count": checked,
        "missing_send_count": len(missing),
        "missing": missing,
        "evidence": "explicit-transitive-device-dependency-only",
        "inferred_edges_added": 0,
    }


def _validate_manifest(
    stage: Any,
    *,
    stage_index: int,
    microbatches: int,
    failures: list[str],
) -> tuple[set[str], int]:
    if (
        not isinstance(stage, dict)
        or stage.get("pipeline_parallel_rank") != stage_index
    ):
        failures.append(f"stage-{stage_index}-identity")
        return set(), 0
    manifest = stage.get("parameter_manifest")
    if not isinstance(manifest, dict):
        failures.append(f"stage-{stage_index}-manifest")
        return set(), 0
    parameters = manifest.get("parameters")
    if (
        manifest.get("schema") != "ordered-megatron-stage-parameter-manifest-v1"
        or manifest.get("ordering") != "module.named_parameters-before-schedule"
        or not isinstance(parameters, list)
        or not parameters
    ):
        failures.append(f"stage-{stage_index}-manifest")
        return set(), 0
    names: set[str] = set()
    total = 0
    for order, row in enumerate(parameters):
        if not isinstance(row, dict):
            failures.append(f"stage-{stage_index}-manifest-record")
            continue
        name = row.get("name")
        local_name = row.get("local_name")
        shape = row.get("shape")
        numel = row.get("numel")
        allowed_name = isinstance(name, str) and (
            name.startswith(f"layers.{stage_index}.")
            or (stage_index == 1 and name.startswith("final_layernorm."))
        )
        allowed_local = isinstance(local_name, str) and (
            local_name.startswith("layers.0.")
            or (stage_index == 1 and local_name.startswith("final_layernorm."))
        )
        valid = (
            row.get("order") == order
            and allowed_name
            and allowed_local
            and name not in names
            and isinstance(shape, list)
            and bool(shape)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in shape
            )
            and isinstance(numel, int)
            and not isinstance(numel, bool)
            and numel > 0
            and row.get("dtype") == "float32"
            and row.get("requires_grad") is True
        )
        if not valid:
            failures.append(f"stage-{stage_index}-manifest-record")
            continue
        product = 1
        for value in shape:
            product *= value
        if product != numel:
            failures.append(f"stage-{stage_index}-manifest-numel")
        names.add(name)
        total += numel
    if (
        manifest.get("parameter_count") != len(parameters)
        or manifest.get("total_numel") != total
    ):
        failures.append(f"stage-{stage_index}-manifest-totals")

    readiness = stage.get("gradient_readiness")
    if not isinstance(readiness, dict):
        failures.append(f"stage-{stage_index}-readiness")
        return names, total
    occurrences = readiness.get("occurrences")
    valid_occurrences = (
        isinstance(occurrences, list)
        and bool(occurrences)
        and all(isinstance(name, str) and name in names for name in occurrences)
    )
    if (
        readiness.get("schema") != "megatron-stage-gradient-readiness-occurrences-v1"
        or readiness.get("observation") != "unmeasured-complete-warmup-step"
        or not valid_occurrences
        or readiness.get("occurrence_count")
        != (len(occurrences) if isinstance(occurrences, list) else -1)
        or readiness.get("unique_parameter_count") != len(names)
        or (set(occurrences) if isinstance(occurrences, list) else set()) != names
        or readiness.get("measured_step_repeatability_required") is not True
    ):
        failures.append(f"stage-{stage_index}-readiness")
    # Do not assume whether post-accumulate hooks fire once per optimizer step
    # or once per microbatch. Record the observed multiplicities, but require
    # the entrypoint's exact warmup/measured equality before capture completes.
    if isinstance(occurrences, list) and len(occurrences) < len(names):
        failures.append(f"stage-{stage_index}-readiness-cardinality")
    if microbatches <= 0:  # defensive connection to the target contract
        failures.append("microbatch-contract")
    return names, total


def _validate_full_manifest(
    measurement: dict[str, Any], failures: list[str]
) -> tuple[set[str], int]:
    manifest = measurement.get("parameter_manifest")
    if not isinstance(manifest, dict):
        failures.append("parameter-manifest")
        return set(), 0
    parameters = manifest.get("parameters")
    if (
        manifest.get("schema") != "ordered-megatron-parameter-manifest-v1"
        or manifest.get("ordering") != "module.named_parameters-before-ddp-wrap"
        or not isinstance(parameters, list)
        or not parameters
    ):
        failures.append("parameter-manifest")
        return set(), 0
    names: set[str] = set()
    total = 0
    for order, row in enumerate(parameters):
        if not isinstance(row, dict):
            failures.append("parameter-manifest-record")
            continue
        name = row.get("name")
        shape = row.get("shape")
        numel = row.get("numel")
        valid = (
            row.get("order") == order
            and isinstance(name, str)
            and (
                name.startswith("layers.0.")
                or name.startswith("layers.1.")
                or name.startswith("final_layernorm.")
            )
            and name not in names
            and isinstance(shape, list)
            and bool(shape)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in shape
            )
            and isinstance(numel, int)
            and not isinstance(numel, bool)
            and numel > 0
            and row.get("dtype") == "float32"
            and row.get("requires_grad") is True
        )
        if not valid:
            failures.append("parameter-manifest-record")
            continue
        product = 1
        for value in shape:
            product *= value
        if product != numel:
            failures.append("parameter-manifest-numel")
        names.add(name)
        total += numel
    if (
        manifest.get("parameter_count") != len(parameters)
        or manifest.get("total_numel") != total
    ):
        failures.append("parameter-manifest-totals")
    readiness = measurement.get("gradient_readiness")
    readiness_names = (
        readiness.get("parameter_names") if isinstance(readiness, dict) else None
    )
    if (
        not isinstance(readiness, dict)
        or readiness.get("schema") != "megatron-parameter-gradient-readiness-v1"
        or readiness.get("observation") != "unmeasured-complete-warmup-step"
        or not isinstance(readiness_names, list)
        or len(readiness_names) != len(names)
        or set(readiness_names) != names
        or readiness.get("parameter_count") != len(names)
        or readiness.get("measured_step_repeatability_required") is not True
    ):
        failures.append("gradient-readiness")
    return names, total


def validate_physical_tp_dp_target(
    workload: dict[str, Any], matrix: dict[str, Any], target_id: str
) -> dict[str, Any]:
    """Admit one physical TP or DP target before structural comparison."""
    failures: list[str] = []
    try:
        targets = validate_matrix(matrix)
        target = targets[target_id]
        domain = _validated_domain(matrix)
    except (KeyError, ValueError) as error:
        return {
            "schema": SCHEMA,
            "passed": False,
            "failures": [f"matrix-or-target:{error}"],
            "claim": "no-physical-transformer-target-claim",
        }
    if target.dimension not in {"tp", "dp"} or target.pp != 1:
        failures.append("not-frozen-tp-dp-target")

    source = workload.get("source")
    if not isinstance(source, dict):
        source = {}
    _fail(failures, source.get("rank_count") == 4, "rank-count")
    _fail(
        failures,
        source.get("target") == "h100" and source.get("observed_targets") == ["h100"],
        "hardware",
    )
    measurement = workload.get("metadata", {}).get("framework_measurement")
    if not isinstance(measurement, dict):
        measurement = {}
        failures.append("rank-identical-framework-measurement")
    expected_top = {
        "schema": "megatron-core-transformer-block-measurement-v1",
        "framework": "megatron-core",
        "framework_commit": COMMIT,
        "tensor_parallel_size": target.tp,
        "pipeline_parallel_size": 1,
        "data_parallel_size": target.dp,
    }
    for key, expected in expected_top.items():
        _fail(failures, measurement.get(key) == expected, f"measurement-{key}")
    model = measurement.get("model")
    if not isinstance(model, dict):
        model = {}
    expected_model = {
        "kind": "dense-causal-transformer-block",
        "num_layers": domain["num_layers"],
        "hidden_size": domain["hidden_size"],
        "ffn_hidden_size": domain["ffn_hidden_size"],
        "attention_heads": domain["attention_heads"],
        "sequence_length": target.sequence_length,
        "micro_batch_size": domain.get("micro_batch_size", 1),
        "global_batch_size": target.global_batch_size,
        "parameter_dtype": domain["parameter_dtype"],
        "activation_dtype": domain["parameter_dtype"],
        "attention_mask": domain["attention_mask"],
        "dropout": domain["dropout"],
    }
    for key, expected in expected_model.items():
        _fail(failures, model.get(key) == expected, f"model-{key}")
    layer = measurement.get("layer")
    _fail(
        failures,
        layer
        == {
            "kind": "megatron-core-local-dense-transformer-layer-v1",
            "self_attention": True,
            "dense_mlp": True,
            "sequence_parallel": False,
            "transformer_engine": domain["transformer_engine"],
        },
        "layer-contract",
    )
    names, parameter_numel = _validate_full_manifest(measurement, failures)
    all_components = [
        f"layers.{layer_index}{'.' if path else ''}{path}"
        for layer_index in range(domain["num_layers"])
        for path in COMPONENT_PATHS
    ] + ["final_layernorm"]
    _fail(
        failures,
        measurement.get("component_markers")
        == {
            "schema": "megatron-transformer-component-markers-v1",
            "phases": ["forward", "backward"],
            "components": all_components,
        },
        "component-contract",
    )
    data_parallel = measurement.get("data_parallel")
    expected_dp = {
        "kind": "torch.nn.parallel.DistributedDataParallel"
        if target.dp == 2
        else "none",
        "gradient_sync": target.dp == 2,
    }
    _fail(failures, data_parallel == expected_dp, "data-parallel-contract")

    events = workload.get("events")
    if not isinstance(events, list) or not events:
        failures.append("events")
        events = []
    ranks = {event.get("rank") for event in events if isinstance(event, dict)}
    _fail(failures, ranks == {0, 1, 2, 3}, "rank-coverage")
    phases_by_rank: dict[int, set[str]] = {rank: set() for rank in range(4)}
    softmax_by_rank = Counter()
    for event in events:
        if not isinstance(event, dict) or event.get("rank") not in phases_by_rank:
            continue
        rank = event["rank"]
        phase = _phase(event)
        if phase:
            phases_by_rank[rank].add(phase)
        metadata = event.get("metadata")
        launch = (
            metadata.get("kernel_launch_payload")
            if isinstance(metadata, dict)
            else None
        )
        framework_operator = (
            launch.get("framework_operator") if isinstance(launch, dict) else None
        )
        operator = (
            framework_operator.get("name", "")
            if isinstance(framework_operator, dict)
            else ""
        )
        if isinstance(operator, str) and "softmax" in operator.lower():
            softmax_by_rank[rank] += 1
    for rank in range(4):
        phases = phases_by_rank[rank]
        for outer in ("megatron_transformer_optimizer",):
            _fail(failures, outer in phases, f"rank-{rank}-phase-{outer}")
        for phase in ("forward", "backward"):
            for component in all_components:
                _fail(
                    failures,
                    f"megatron_transformer_{phase}_component:{component}" in phases,
                    f"rank-{rank}-component-{phase}-{component}",
                )
        _fail(failures, softmax_by_rank[rank] > 0, f"rank-{rank}-softmax")

    collectives = [
        event
        for event in events
        if isinstance(event, dict) and event.get("kind") == "collective"
    ]
    tp_counts = Counter()
    dp_syncs: dict[int, list[dict[str, Any]]] = {rank: [] for rank in range(4)}
    expected_tp_groups = {tuple(range(4))} if target.tp == 4 else {(0, 1), (2, 3)}
    observed_tp_groups: set[tuple[int, ...]] = set()
    observed_dp_groups: set[tuple[int, ...]] = set()
    for event in collectives:
        rank = event.get("rank")
        metadata = event.get("metadata", {})
        members = (
            metadata.get("process_group_ranks") if isinstance(metadata, dict) else None
        )
        if isinstance(members, str):
            try:
                members = json.loads(members)
            except json.JSONDecodeError:
                members = None
        membership_valid = (
            isinstance(members, list)
            and bool(members)
            and all(
                isinstance(member, int) and not isinstance(member, bool)
                for member in members
            )
            and len(set(members)) == len(members)
        )
        group = tuple(members) if membership_valid else ()
        _fail(
            failures,
            membership_valid,
            f"rank-{rank}-collective-process-group-membership",
        )
        if event.get("group_role") == "tp" and event.get("collective") == "all_reduce":
            tp_counts[rank] += 1
            if group:
                observed_tp_groups.add(group)
        if event.get("group_role") == "dp":
            if rank in dp_syncs:
                dp_syncs[rank].append(event)
            if group:
                observed_dp_groups.add(group)
    for rank in range(4):
        _fail(failures, tp_counts[rank] == 8, f"rank-{rank}-tp-collective-count")
    _fail(
        failures, observed_tp_groups == expected_tp_groups, "tp-communicator-membership"
    )
    if target.dp == 2:
        _fail(
            failures,
            observed_dp_groups == {(0, 2), (1, 3)},
            "dp-communicator-membership",
        )
        try:
            ddp_contract = derive_pinned_ddp_contract(measurement)
            expected_dp_sequence = [
                (event["operation"], event["payload_bytes"])
                for event in ddp_contract["forward_metadata_broadcasts"]
            ] + [
                (event["operation"], event["payload_bytes"])
                for event in ddp_contract["gradient_buckets"]
            ]
        except TransformerRuleError as error:
            failures.append(f"ddp-contract:{error}")
            expected_dp_sequence = []
        for rank in range(4):
            rank_events = dp_syncs[rank]
            if rank_events and all(
                isinstance(event.get("observed_start_us"), (int, float))
                and not isinstance(event.get("observed_start_us"), bool)
                for event in rank_events
            ):
                rank_events = sorted(
                    rank_events, key=lambda event: event["observed_start_us"]
                )
            observed_sequence = [
                (event.get("collective"), event.get("message_bytes"))
                for event in rank_events
            ]
            _fail(
                failures,
                len(observed_sequence) == len(expected_dp_sequence),
                f"rank-{rank}-dp-sync-count",
            )
            _fail(
                failures,
                observed_sequence == expected_dp_sequence,
                f"rank-{rank}-dp-sync-bytes",
            )
    else:
        _fail(
            failures,
            all(not values for values in dp_syncs.values()) and not observed_dp_groups,
            "unexpected-dp-sync",
        )

    unique_failures = sorted(set(failures))
    return {
        "schema": SCHEMA,
        "passed": not unique_failures,
        "failures": unique_failures,
        "target_id": target_id,
        "event_count": len(events),
        "collective_count": len(collectives),
        "parameter_count": len(names),
        "parameter_numel_per_tp_rank": parameter_numel,
        "claim": (
            "physical-h100-transformer-tp-dp-target-admitted-for-structural-scoring"
            if not unique_failures
            else "no-physical-transformer-target-claim"
        ),
    }


def validate_physical_pp_target(
    workload: dict[str, Any], matrix: dict[str, Any], target_id: str
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        targets = validate_matrix(matrix)
        target = targets[target_id]
        domain = _validated_domain(matrix)
    except (KeyError, ValueError) as error:
        return {
            "schema": SCHEMA,
            "passed": False,
            "failures": [f"matrix-or-target:{error}"],
            "claim": "no-physical-transformer-target-claim",
        }
    if target.dimension != "pp" or (target.tp, target.pp, target.dp) != (2, 2, 1):
        failures.append("not-frozen-pp-target")

    source = workload.get("source")
    if not isinstance(source, dict):
        source = {}
    _fail(failures, source.get("rank_count") == 4, "rank-count")
    _fail(
        failures,
        source.get("target") == "h100" and source.get("observed_targets") == ["h100"],
        "hardware",
    )
    measurement = workload.get("metadata", {}).get("framework_measurement")
    if not isinstance(measurement, dict):
        measurement = {}
        failures.append("rank-identical-framework-measurement")
    expected_top = {
        "schema": "megatron-core-transformer-pipeline-measurement-v1",
        "framework": "megatron-core",
        "framework_commit": COMMIT,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 1,
    }
    for key, expected in expected_top.items():
        _fail(failures, measurement.get(key) == expected, f"measurement-{key}")
    model = measurement.get("model")
    if not isinstance(model, dict):
        model = {}
    expected_model = {
        "kind": "dense-causal-transformer-block",
        "num_layers": domain["num_layers"],
        "layers_per_stage": domain["num_layers"] // target.pp,
        "hidden_size": domain["hidden_size"],
        "ffn_hidden_size": domain["ffn_hidden_size"],
        "attention_heads": domain["attention_heads"],
        "sequence_length": target.sequence_length,
        "micro_batch_size": domain.get("micro_batch_size", 1),
        "global_batch_size": target.global_batch_size,
        "parameter_dtype": domain["parameter_dtype"],
        "activation_dtype": domain["parameter_dtype"],
        "attention_mask": domain["attention_mask"],
        "dropout": domain["dropout"],
    }
    for key, expected in expected_model.items():
        _fail(failures, model.get(key) == expected, f"model-{key}")
    pipeline = measurement.get("pipeline")
    _fail(
        failures,
        pipeline
        == {
            "kind": "megatron-core-non-interleaved-1f1b",
            "microbatches": target.pipeline_microbatches,
            "layers_per_stage": domain["num_layers"] // target.pp,
        },
        "pipeline-contract",
    )

    stage_contract = measurement.get("pipeline_stage_evidence")
    stages = stage_contract.get("stages") if isinstance(stage_contract, dict) else None
    if (
        not isinstance(stage_contract, dict)
        or stage_contract.get("schema")
        != "megatron-transformer-pipeline-stage-evidence-v1"
        or stage_contract.get("ordering") != "pipeline-parallel-rank"
        or not isinstance(stages, list)
        or len(stages) != 2
    ):
        failures.append("pipeline-stage-evidence")
        stages = [{}, {}]
    stage_parameter_counts = []
    stage_parameter_numel = []
    for stage_index, stage in enumerate(stages):
        names, total = _validate_manifest(
            stage,
            stage_index=stage_index,
            microbatches=target.pipeline_microbatches,
            failures=failures,
        )
        stage_parameter_counts.append(len(names))
        stage_parameter_numel.append(total)

    all_components = [
        f"layers.{layer}{'.' if path else ''}{path}"
        for layer in range(domain["num_layers"])
        for path in COMPONENT_PATHS
    ] + ["final_layernorm"]
    _fail(
        failures,
        measurement.get("component_markers")
        == {
            "schema": "megatron-transformer-component-markers-v1",
            "phases": ["forward", "backward"],
            "components": all_components,
        },
        "component-contract",
    )

    events = workload.get("events")
    if not isinstance(events, list) or not events:
        failures.append("events")
        events = []
    ranks = {event.get("rank") for event in events if isinstance(event, dict)}
    _fail(failures, ranks == {0, 1, 2, 3}, "rank-coverage")
    softmax_by_rank = Counter()
    phase_names_by_rank: dict[int, set[str]] = {rank: set() for rank in range(4)}
    for event in events:
        if not isinstance(event, dict) or event.get("rank") not in phase_names_by_rank:
            continue
        rank = event["rank"]
        phase = _phase(event)
        if phase:
            phase_names_by_rank[rank].add(phase)
        metadata = event.get("metadata")
        launch = (
            metadata.get("kernel_launch_payload")
            if isinstance(metadata, dict)
            else None
        )
        framework_operator = (
            launch.get("framework_operator") if isinstance(launch, dict) else None
        )
        operator = (
            framework_operator.get("name", "")
            if isinstance(framework_operator, dict)
            else ""
        )
        if isinstance(operator, str) and "softmax" in operator.lower():
            softmax_by_rank[rank] += 1
    for rank in range(4):
        stage = rank // 2
        phases = phase_names_by_rank[rank]
        for outer in ("megatron_transformer_optimizer",):
            _fail(failures, outer in phases, f"rank-{rank}-phase-{outer}")
        expected_components = [
            f"layers.{stage}{'.' if path else ''}{path}" for path in COMPONENT_PATHS
        ]
        if stage == 1:
            expected_components.append("final_layernorm")
        for phase in ("forward", "backward"):
            for component in expected_components:
                marker = f"megatron_transformer_{phase}_component:{component}"
                _fail(
                    failures,
                    marker in phases,
                    f"rank-{rank}-component-{phase}-{component}",
                )
        other_stage_prefix = (
            f"megatron_transformer_forward_component:layers.{1 - stage}"
        )
        _fail(
            failures,
            not any(name.startswith(other_stage_prefix) for name in phases),
            f"rank-{rank}-foreign-stage-component",
        )
        _fail(failures, softmax_by_rank[rank] > 0, f"rank-{rank}-softmax")
        has_loss = "megatron_transformer_terminal_loss" in phases
        _fail(
            failures, has_loss == (stage == 1), f"rank-{rank}-terminal-loss-placement"
        )

    collectives = [
        event
        for event in events
        if isinstance(event, dict) and event.get("kind") == "collective"
    ]
    try:
        logical_collectives = physical_pp_logical_collectives(workload)
    except ValueError as error:
        failures.append(f"logical-collective-reconciliation:{error}")
        logical_collectives = []
    send_readiness: dict[str, Any] | None = None
    if logical_collectives:
        try:
            send_readiness = _pp_send_producer_readiness(
                workload, logical_collectives
            )
        except ValueError as error:
            send_readiness = {
                "schema": "scaletether-physical-pp-send-producer-readiness-v1",
                "passed": False,
                "checked_send_count": 0,
                "missing_send_count": None,
                "error": str(error),
                "evidence": "explicit-transitive-device-dependency-only",
                "inferred_edges_added": 0,
            }
        if send_readiness.get("passed") is not True:
            failures.append("pp-send-producer-readiness")
    # Both admitted family versions use float32 activations.  Derive every
    # shape term from the validated domain so a future version cannot silently
    # inherit the legacy hidden-size constant.
    activation_bytes = {"float32": 4}[domain["parameter_dtype"]]
    payload = (
        target.sequence_length
        * domain.get("micro_batch_size", 1)
        * domain["hidden_size"]
        * activation_bytes
    )
    p2p: Counter[tuple[str, int, int, int]] = Counter()
    tp_counts = Counter()
    for event in logical_collectives:
        rank = event.get("rank")
        if event.get("group_role") == "tp" and event.get("collective") == "all_reduce":
            tp_counts[rank] += 1
        if event.get("group_role") == "pp" and event.get("collective") in {
            "send",
            "recv",
        }:
            p2p[
                (
                    event.get("collective"),
                    event.get("p2p_source_rank"),
                    event.get("p2p_destination_rank"),
                    event.get("message_bytes"),
                )
            ] += 1
    expected_tp = 4 * target.pipeline_microbatches
    for rank in range(4):
        _fail(
            failures, tp_counts[rank] == expected_tp, f"rank-{rank}-tp-collective-count"
        )
    for lane in range(2):
        source_rank, destination_rank = lane, lane + 2
        forward_send = p2p[("send", source_rank, destination_rank, payload)]
        forward_recv = p2p[("recv", source_rank, destination_rank, payload)]
        backward_send = p2p[("send", destination_rank, source_rank, payload)]
        backward_recv = p2p[("recv", destination_rank, source_rank, payload)]
        expected = target.pipeline_microbatches
        _fail(
            failures,
            (forward_send, forward_recv, backward_send, backward_recv)
            == (expected, expected, expected, expected),
            f"lane-{lane}-p2p-match",
        )
    recognized_p2p = sum(p2p.values())
    _fail(
        failures,
        recognized_p2p == 4 * target.pipeline_microbatches * 2,
        "unexpected-p2p-operation",
    )

    unique_failures = sorted(set(failures))
    return {
        "schema": SCHEMA,
        "passed": not unique_failures,
        "failures": unique_failures,
        "target_id": target_id,
        "event_count": len(events),
        "collective_count": len(collectives),
        "pipeline_payload_bytes": payload,
        "stage_parameter_counts": stage_parameter_counts,
        "stage_parameter_numel": stage_parameter_numel,
        "pp_send_producer_readiness": send_readiness,
        "decision": (
            "unsupported-capture"
            if "pp-send-producer-readiness" in unique_failures
            else "admit" if not unique_failures else "reject"
        ),
        "unsupported_capture": (
            {
                "schema": "scaletether-unsupported-capture-decision-v1",
                "reason_code": "missing-pp-send-producer-readiness",
                "action": "UPGRADE_CAPTURE",
                "inferred_edges_added": 0,
            }
            if "pp-send-producer-readiness" in unique_failures
            else None
        ),
        "claim": (
            "physical-h100-transformer-pp-target-admitted-for-structural-scoring"
            if not unique_failures
            else "no-physical-transformer-target-claim"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("target_id")
    parser.add_argument("workload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    validator = (
        validate_physical_pp_target
        if args.target_id.startswith("PP-")
        else validate_physical_tp_dp_target
    )
    result = validator(workload, matrix, args.target_id)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
