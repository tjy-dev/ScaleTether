from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import Parallelism, architecture, load_topology
from .pipeline import (
    MEGATRON_CORE_SCHEDULE_COMMIT,
    expand_pipeline,
    expand_pipeline_ranks,
)
from .schema import TraceEvent, WorkloadTrace


CONTRACT_SCHEMA = "scaletether-megatron-counterfactual-v1"


try:
    _MALLOC_TRIM = ctypes.CDLL(None).malloc_trim
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int
except (AttributeError, OSError):  # Non-glibc platforms retain normal GC behavior.
    _MALLOC_TRIM = None


class FrameworkCounterfactualError(ValueError):
    """The requested framework rewrite is not justified by its contract."""


def _release_candidate_memory() -> None:
    """Release discarded candidate graphs; trimming is an optional RSS hint."""

    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


@dataclass(frozen=True)
class FrameworkCounterfactualCompilation:
    trace: WorkloadTrace
    applied: bool
    summary: dict[str, Any]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FrameworkCounterfactualError(f"{name} must be a positive integer")
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrameworkCounterfactualError(f"{name} must be a mapping")
    return value


def _event_ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise FrameworkCounterfactualError(f"{name} must be a non-empty list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise FrameworkCounterfactualError(
            f"{name} must contain unique non-empty event ids"
        )
    return result


def _captured_group_size(event: TraceEvent) -> int | None:
    value = event.group_size
    if value is None:
        value = event.metadata.get("observed_group_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _profile_events(
    source_events: dict[str, TraceEvent],
    event_ids: tuple[str, ...],
    phase: str,
    expected_tp: int,
) -> tuple[TraceEvent, ...]:
    missing = [event_id for event_id in event_ids if event_id not in source_events]
    if missing:
        raise FrameworkCounterfactualError(
            f"{phase} profile references missing measured events: {missing}"
        )
    events = tuple(source_events[event_id] for event_id in event_ids)
    for event in events:
        if event.kind not in {"compute", "memory", "synchronization", "collective"}:
            raise FrameworkCounterfactualError(
                f"profile event {event.id!r} has unsupported kind {event.kind!r}"
            )
        if event.kind == "collective":
            if event.group_role != "tp" or event.collective in {"send", "recv"}:
                raise FrameworkCounterfactualError(
                    f"profile collective {event.id!r} must be a TP collective"
                )
            if event.message_bytes is None:
                raise FrameworkCounterfactualError(
                    f"profile collective {event.id!r} lacks exact message bytes"
                )
            if _captured_group_size(event) != expected_tp:
                raise FrameworkCounterfactualError(
                    f"profile collective {event.id!r} group size does not match "
                    f"target TP={expected_tp}"
                )
    return events


def _clone_layer_phase(
    measured: tuple[TraceEvent, ...],
    *,
    stage: int,
    logical_layer: int,
    phase: str,
    predecessors: tuple[str, ...] = (),
    semantic_region: str | None = None,
) -> tuple[list[TraceEvent], tuple[str, ...]]:
    measured_ids = {event.id for event in measured}
    region = semantic_region or f"layer{logical_layer}-{phase}"
    id_map = {
        event.id: f"semantic-s{stage}-{region}-op{index}"
        for index, event in enumerate(measured)
    }
    internally_referenced = {
        dependency
        for event in measured
        for dependency in event.dependencies
        if dependency in measured_ids
    }
    result: list[TraceEvent] = []
    for operation_index, event in enumerate(measured):
        event_id = id_map[event.id]
        internal_dependencies = tuple(
            id_map[dependency]
            for dependency in event.dependencies
            if dependency in measured_ids
        )
        dependencies = internal_dependencies or predecessors
        metadata = dict(event.metadata)
        metadata.update(
            {
                "framework_counterfactual": CONTRACT_SCHEMA,
                "framework": "megatron-core",
                "pipeline_stage": stage,
                "pipeline_phase": phase,
                "pipeline_tp_lane": 0,
                "logical_layer": logical_layer,
                "measured_primitive_event_id": event.id,
                "measurement_transfer": "exact-target-tp-profile",
                "discarded_external_phase_dependencies": [
                    dependency
                    for dependency in event.dependencies
                    if dependency not in measured_ids
                ],
            }
        )
        if semantic_region is not None:
            metadata["framework_subphase"] = semantic_region
        result.append(
            replace(
                event,
                id=event_id,
                name=f"layer {logical_layer} {phase}: {event.name}",
                rank=0,
                device=0,
                dependencies=dependencies,
                observed_start_us=None,
                metadata=metadata,
            )
        )
    exits = tuple(
        id_map[event.id] for event in measured if event.id not in internally_referenced
    )
    return result, exits


def _insert_pp_loss_scale_backward(
    events: list[TraceEvent],
    exits: tuple[str, ...],
    *,
    rule: dict[str, Any],
    stage: int,
) -> tuple[list[TraceEvent], tuple[str, ...]]:
    """Insert the pinned scheduler's scalar-loss autograd nodes.

    The measured terminal-loss backward primitive is required to be one
    direct linear chain. Megatron inserts ``*= cp_group_size`` and
    ``/= num_microbatches`` between the first and second nodes of that chain.
    Their backward kernels therefore occur at that exact boundary.
    """

    if len(events) < 2:
        raise FrameworkCounterfactualError(
            "terminal-loss backward primitive is too short for loss scaling"
        )
    first = events[0]
    if any(event.dependencies != (events[index - 1].id,) for index, event in enumerate(events[1:], 1)):
        raise FrameworkCounterfactualError(
            "terminal-loss backward primitive is not one direct linear chain"
        )
    kernels = rule.get("loss_scale_backward_kernels")
    if not isinstance(kernels, list) or len(kernels) != 2:
        raise FrameworkCounterfactualError(
            "PP scheduler rule lacks two loss-scale backward kernels"
        )
    inserted: list[TraceEvent] = []
    predecessor = first.id
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict):
            raise FrameworkCounterfactualError("invalid loss-scale kernel rule")
        event_id = f"semantic-s{stage}-terminal-loss-scale-backward-op{index}"
        inserted.append(
            TraceEvent(
                id=event_id,
                name=str(kernel["name"]),
                kind="compute",
                duration_us=float(kernel["duration_us"]),
                stream=first.stream,
                dependencies=(predecessor,),
                metadata={
                    "framework_counterfactual": CONTRACT_SCHEMA,
                    "framework_rule": rule["schema"],
                    "framework_subphase": "terminal-loss-backward-scale",
                    "pipeline_stage": stage,
                    "pipeline_phase": "backward",
                    "pipeline_tp_lane": 0,
                    "kernel_signature": kernel["kernel_signature"],
                    "kernel_resource": rule["kernel_resource"],
                    "provenance": rule["provenance"],
                },
            )
        )
        predecessor = event_id
    remainder = [
        replace(events[1], dependencies=(predecessor,)),
        *events[2:],
    ]
    return [first, *inserted, *remainder], exits


def _parameter_gradient_ready_chains(
    events: list[TraceEvent],
) -> list[tuple[TraceEvent, tuple[str, ...]]]:
    """Return the two terminal GEMMs and their bounded decomposition chains.

    cuBLAS may implement one ``aten::mm`` as a main GEMM followed by a
    split-K reduction.  Kineto gives both kernels the same framework operator.
    We accept only two disjoint, direct, linear chains with that exact operator
    identity.  Anything more complicated is outside the calibrated rule.
    """

    def is_parameter_gradient_ready(event: TraceEvent) -> bool:
        if event.kind != "compute":
            return False
        launch = event.metadata.get("kernel_launch_payload")
        if not isinstance(launch, dict):
            return False
        operator = launch.get("framework_operator")
        if not isinstance(operator, dict) or operator.get("name") != "aten::mm":
            return False
        dimensions = operator.get("input_dims")
        strides = operator.get("input_strides")
        return (
            isinstance(dimensions, list)
            and len(dimensions) == 2
            and all(isinstance(shape, list) and len(shape) == 2 for shape in dimensions)
            and dimensions[0][0] == 256
            and dimensions[1][1] == 256
            and dimensions[0][1] == dimensions[1][0]
            and strides == [[1, 256], [256, 1]]
        )

    candidates = [event for event in events if is_parameter_gradient_ready(event)]
    by_id = {event.id: event for event in candidates}
    operators = {
        event.id: event.metadata["kernel_launch_payload"]["framework_operator"]
        for event in candidates
    }
    predecessors: dict[str, list[str]] = {event.id: [] for event in candidates}
    successors: dict[str, list[str]] = {event.id: [] for event in candidates}
    for event in candidates:
        launch = event.metadata["kernel_launch_payload"]
        launch_name = str(launch.get("name", ""))
        is_split_k_reduction = (
            "cublasLt::splitKreduce_kernel<" in event.name
            or "cublasLt::splitKreduce_kernel<" in launch_name
        )
        for dependency in event.dependencies:
            if (
                is_split_k_reduction
                and dependency in by_id
                and operators[dependency] == operators[event.id]
            ):
                predecessors[event.id].append(dependency)
                successors[dependency].append(event.id)

    if any(len(value) > 1 for value in (*predecessors.values(), *successors.values())):
        raise FrameworkCounterfactualError(
            "parameter-gradient GEMM decomposition is not a set of linear chains"
        )
    roots = [event for event in candidates if not predecessors[event.id]]
    if len(roots) != 2:
        raise FrameworkCounterfactualError(
            "calibrated DDP rule requires exactly two parameter-gradient-ready "
            "GEMM chains per local MLP layer"
        )

    selected: list[tuple[TraceEvent, tuple[str, ...]]] = []
    covered: set[str] = set()
    for root in roots:
        chain: list[str] = []
        current = root.id
        while True:
            if current in covered:
                raise FrameworkCounterfactualError(
                    "parameter-gradient GEMM decomposition contains a cycle or join"
                )
            covered.add(current)
            chain.append(current)
            if not successors[current]:
                selected.append((by_id[current], tuple(chain)))
                break
            current = successors[current][0]
    if covered != set(by_id):
        raise FrameworkCounterfactualError(
            "parameter-gradient GEMM decomposition is disconnected or cyclic"
        )
    order = {event.id: index for index, event in enumerate(events)}
    return sorted(selected, key=lambda item: order[item[0].id])


def _insert_ddp_gradient_normalization(
    events: list[TraceEvent],
    exits: tuple[str, ...],
    *,
    rule: dict[str, Any],
    stage: int,
    logical_layer: int,
) -> tuple[list[TraceEvent], tuple[str, ...]]:
    """Insert the pinned PyTorch DDP per-parameter normalization hooks.

    The ordering is framework behavior observed in the v11 discovery run.  A
    hook follows each parameter-gradient GEMM.  The final hook also joins any
    remaining phase exit (the TP gradient collective in this bounded MLP), so
    the later DP synchronization cannot start before either branch completes.
    """

    expected_schema = "megatron-pytorch-ddp-gradient-normalization-h100-v1"
    if (
        rule.get("schema") != expected_schema
        or rule.get("status") != "calibrated-framework-rule"
        or rule.get("target") != "h100"
        or rule.get("device_name") != "NVIDIA H100"
        or rule.get("compute_capability") != "9.0"
        or rule.get("torch_version") != "2.9.1+cu128"
        or rule.get("cuda_version") != "12.8"
        or rule.get("nccl_version") != "2.27.5"
        or rule.get("local_parameter_shape") != [256, 256]
        or rule.get("local_parameter_shapes") != [[256, 256], [256, 256]]
        or rule.get("parameter_count_per_layer") != 2
    ):
        raise FrameworkCounterfactualError(
            "DP gradient-normalization rule is outside its calibrated H100 domain"
        )
    ready_signature = str(rule.get("parameter_gradient_ready_kernel_signature", ""))
    ready_operator = _mapping(
        rule.get("parameter_gradient_ready_operator"),
        "normalization parameter_gradient_ready_operator",
    )
    kernel_signature = str(rule.get("kernel_signature", ""))
    launch_signature = str(rule.get("kernel_launch_signature", ""))
    kernel_name = str(rule.get("kernel_name", ""))
    duration_us = rule.get("duration_us")
    resource = _mapping(rule.get("kernel_resource"), "normalization kernel_resource")
    provenance = _mapping(rule.get("provenance"), "normalization provenance")
    if (
        ready_signature != "torch-kernel-v2:2aff6db44315bc8b32053c93"
        or ready_operator.get("name") != "aten::mm"
        or ready_operator.get("output_shape") != [256, 256]
        or kernel_signature != "torch-kernel-v2:824d23297edf28a28779e7dd"
        or launch_signature != "torch-kernel-launch-v1:0a42fb668482efff7a70739b"
        or not kernel_name
        or not isinstance(duration_us, (int, float))
        or isinstance(duration_us, bool)
        or duration_us <= 0
        or resource.get("grid") != [64, 1, 1]
        or resource.get("block") != [128, 1, 1]
        or provenance.get("discovery_job") != 8237716
        or provenance.get("discovery_status")
        != "prospective-structural-negative"
    ):
        raise FrameworkCounterfactualError(
            "DP gradient-normalization rule evidence is incomplete or modified"
        )

    candidates = _parameter_gradient_ready_chains(events)

    result = list(events)
    for index, (candidate, decomposition) in enumerate(candidates):
        event_id = (
            f"semantic-s{stage}-layer{logical_layer}-backward-"
            f"ddp-normalize-param{index}"
        )
        is_last = index == len(candidates) - 1
        dependencies = (
            tuple(dict.fromkeys((candidate.id, *exits)))
            if is_last
            else (candidate.id,)
        )
        if not is_last:
            result = [
                replace(
                    event,
                    dependencies=tuple(
                        event_id if dependency == candidate.id else dependency
                        for dependency in event.dependencies
                    ),
                )
                for event in result
            ]
        metadata = {
            "framework_counterfactual": CONTRACT_SCHEMA,
            "framework": "megatron-core",
            "pipeline_stage": stage,
            "pipeline_phase": "backward",
            "pipeline_tp_lane": 0,
            "logical_layer": logical_layer,
            "parameter_index": index,
            "parameter_gradient_ready_decomposition": list(decomposition),
            "framework_rule": expected_schema,
            "measurement_transfer": "calibrated-framework-rule",
            "kernel_signature": kernel_signature,
            "kernel_launch_signature": launch_signature,
            "kernel_signature_payload": {
                "schema": "torch-kernel-v2",
                "launch_signature": launch_signature,
                "environment": {
                    key: rule[key]
                    for key in (
                        "target",
                        "device_name",
                        "compute_capability",
                        "torch_version",
                        "cuda_version",
                        "nccl_version",
                    )
                },
            },
            "kernel_resource": dict(resource),
            "framework_operator": {
                "name": "aten::mul",
                "input_dims": [[256, 256], [], [256, 256]],
                "input_types": ["float", "double", "float"],
            },
            "duration_evidence": dict(
                _mapping(rule.get("duration_evidence"), "normalization duration_evidence")
            ),
            "provenance": dict(provenance),
        }
        result.append(
            TraceEvent(
                id=event_id,
                name=kernel_name,
                kind="compute",
                duration_us=float(duration_us),
                stream=candidate.stream,
                dependencies=dependencies,
                sm_fraction=64 / 132,
                metadata=metadata,
            )
        )
        if is_last:
            exits = (event_id,)
    return result, exits


def _prefix_rank_events(
    rank_events: tuple[tuple[TraceEvent, ...], ...],
    rank_coordinates: tuple[tuple[int, int, int], ...],
) -> list[TraceEvent]:
    flattened: list[TraceEvent] = []
    for rank, events in enumerate(rank_events):
        tp_lane, pp_stage, dp_replica = rank_coordinates[rank]
        local_ids = {event.id for event in events}
        for event in events:
            metadata = dict(event.metadata)
            if event.kind == "collective" and event.group_role == "tp":
                microbatch = metadata.get("pipeline_microbatch")
                metadata["collective_instance_id"] = (
                    f"semantic-tp:pp{pp_stage}:dp{dp_replica}:mb{microbatch}:{event.id}"
                )
            flattened.append(
                replace(
                    event,
                    id=f"rank{rank}::{event.id}",
                    dependencies=tuple(
                        f"rank{rank}::{dependency}"
                        if dependency in local_ids
                        else dependency
                        for dependency in event.dependencies
                    ),
                    metadata=metadata,
                )
            )
    return flattened


def compile_framework_counterfactual(
    trace: WorkloadTrace,
    parallelism: Parallelism,
    gpus: int,
    target: str,
) -> FrameworkCounterfactualCompilation:
    """Compile an explicit Megatron semantic contract into a target-rank DAG.

    Absence of a contract is a no-op. Presence is fail-closed: the compiler
    never guesses an unmeasured TP shape or a framework schedule.
    """

    raw_contract = trace.metadata.get("framework_counterfactual")
    if raw_contract is None:
        return FrameworkCounterfactualCompilation(
            trace=trace,
            applied=False,
            summary={"status": "not-requested"},
        )
    contract = _mapping(raw_contract, "metadata.framework_counterfactual")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise FrameworkCounterfactualError(
            "unsupported framework counterfactual schema"
        )
    if contract.get("framework") != "megatron-core":
        raise FrameworkCounterfactualError(
            "the v1 framework counterfactual compiler supports only Megatron-Core"
        )
    if contract.get("framework_commit") != MEGATRON_CORE_SCHEDULE_COMMIT:
        raise FrameworkCounterfactualError(
            "Megatron-Core schedule revision is not the validated v1 revision"
        )
    if gpus != parallelism.tp * parallelism.pp * parallelism.dp:
        raise FrameworkCounterfactualError("target GPUs must equal tp*pp*dp")
    if parallelism.ep != 1:
        raise FrameworkCounterfactualError(
            "the v1 Megatron counterfactual contract has no expert-parallel "
            "profile; target EP must equal 1"
        )

    model = _mapping(contract.get("model"), "framework counterfactual model")
    num_layers = _positive_integer(model.get("num_layers"), "model.num_layers")
    micro_batch_size = _positive_integer(
        model.get("micro_batch_size"), "model.micro_batch_size"
    )
    global_batch_size = _positive_integer(
        model.get("global_batch_size"), "model.global_batch_size"
    )
    if num_layers % parallelism.pp:
        raise FrameworkCounterfactualError(
            "v1 requires num_layers to be exactly divisible by target PP"
        )
    batch_denominator = micro_batch_size * parallelism.dp
    if global_batch_size % batch_denominator:
        raise FrameworkCounterfactualError(
            "global batch must be divisible by micro_batch_size*target_dp"
        )
    microbatches = global_batch_size // batch_denominator

    schedule = _mapping(contract.get("schedule"), "framework schedule")
    schedule_name = str(schedule.get("type", "1f1b")).lower()
    if schedule_name not in {"gpipe", "1f1b"}:
        raise FrameworkCounterfactualError(
            "v1 supports only Megatron GPipe and non-interleaved 1F1B"
        )

    profiles = _mapping(contract.get("tp_profiles"), "tp_profiles")
    profile = _mapping(
        profiles.get(str(parallelism.tp)),
        f"tp_profiles[{parallelism.tp}]",
    )
    if str(profile.get("target", "")).lower() != target.lower():
        raise FrameworkCounterfactualError(
            f"TP profile target does not match requested target {target!r}"
        )
    if str(trace.source.get("target", "")).lower() != target.lower():
        raise FrameworkCounterfactualError(
            "measured primitive trace target does not match requested target"
        )
    measurement = _mapping(profile.get("measurement"), "TP profile measurement")
    if measurement.get("status") != "measured":
        raise FrameworkCounterfactualError("TP profile is not marked measured")
    _positive_integer(measurement.get("sample_count"), "measurement.sample_count")

    raw_adapter = contract.get("adapter")
    applicability_certificate: dict[str, Any]
    if raw_adapter is None:
        applicability_certificate = {
            "status": "contract-only-not-adapter-certified",
            "claim": "synthetic-or-manual-contract; no measured-rank symmetry claim",
        }
    else:
        adapter = _mapping(raw_adapter, "framework counterfactual adapter")
        from .megatron_adapter import (
            SOURCE_SEMANTIC_BINDING_SCHEMA,
            megatron_source_semantic_sha256,
        )

        applicability = _mapping(
            contract.get("applicability"), "framework applicability certificate"
        )
        rank_symmetry = _mapping(
            measurement.get("rank_symmetry"), "measurement.rank_symmetry"
        )
        source_parallelism = _mapping(
            applicability.get("source_parallelism"),
            "applicability.source_parallelism",
        )
        fixed_dimensions = _mapping(
            applicability.get("fixed_dimensions"),
            "applicability.fixed_dimensions",
        )
        measured_semantic_sha256 = measurement.get("source_semantic_sha256")
        adapter_semantic_sha256 = adapter.get("source_semantic_sha256")
        current_semantic_sha256 = megatron_source_semantic_sha256(trace)
        if (
            adapter.get("source_semantic_binding_schema")
            != SOURCE_SEMANTIC_BINDING_SCHEMA
            or not isinstance(adapter_semantic_sha256, str)
            or adapter_semantic_sha256 != measured_semantic_sha256
            or adapter_semantic_sha256 != current_semantic_sha256
            or adapter.get("source_workload_sha256")
            != measurement.get("source_workload_sha256")
            or adapter.get("provenance_sha256")
            != measurement.get("megatron_provenance_sha256")
        ):
            raise FrameworkCounterfactualError(
                "adapter certificate is not bound to the current measured source "
                "semantics"
            )
        if (
            adapter.get("schema") != "megatron-core-mlp-adapter-v1"
            or applicability.get("schema") != "megatron-core-mlp-applicability-v1"
            or applicability.get("status") != "certified-for-bounded-rewrite"
            or source_parallelism != {"tp": parallelism.tp, "pp": 1, "dp": 1, "ep": 1}
            or fixed_dimensions.get("framework_commit") != MEGATRON_CORE_SCHEDULE_COMMIT
            or str(fixed_dimensions.get("target", "")).lower() != target.lower()
            or fixed_dimensions.get("tp") != parallelism.tp
            or fixed_dimensions.get("model_family") != "bias-free-column-gelu-row-mlp"
            or any(
                fixed_dimensions.get(key) != model.get(key)
                for key in (
                    "hidden_size",
                    "ffn_hidden_size",
                    "sequence_length",
                    "micro_batch_size",
                )
            )
            or applicability.get("rewritable_dimensions") != ["pp", "dp"]
            or rank_symmetry.get("schema") != "megatron-tp-rank-structural-symmetry-v1"
            or rank_symmetry.get("status") != "exact-structural-match"
            or rank_symmetry.get("measured_ranks") != list(range(parallelism.tp))
            or rank_symmetry.get("representative_rank")
            != measurement.get("representative_rank")
            or adapter.get("representative_rank")
            != measurement.get("representative_rank")
            or adapter.get("rank_symmetry") != rank_symmetry
        ):
            raise FrameworkCounterfactualError(
                "adapter applicability or measured TP-rank symmetry certificate "
                "does not cover the requested rewrite"
            )
        applicability_certificate = {
            "status": "adapter-certified-for-bounded-rewrite",
            "schema": applicability["schema"],
            "source_parallelism": source_parallelism,
            "fixed_dimensions": fixed_dimensions,
            "rewritable_dimensions": applicability.get("rewritable_dimensions"),
            "rank_symmetry": rank_symmetry,
            "source_semantic_binding": {
                "schema": SOURCE_SEMANTIC_BINDING_SCHEMA,
                "sha256": current_semantic_sha256,
                "status": "verified-before-rewrite",
            },
            "timing_scope": applicability.get("timing_scope"),
        }

    source_events = {event.id: event for event in trace.events}
    forward = _profile_events(
        source_events,
        _event_ids(profile.get("forward_event_ids"), "forward_event_ids"),
        "forward",
        parallelism.tp,
    )
    backward = _profile_events(
        source_events,
        _event_ids(profile.get("backward_event_ids"), "backward_event_ids"),
        "backward",
        parallelism.tp,
    )
    raw_optimizer_ids = profile.get("optimizer_event_ids")
    optimizer = (
        ()
        if raw_optimizer_ids is None
        else _profile_events(
            source_events,
            _event_ids(raw_optimizer_ids, "optimizer_event_ids"),
            "optimizer",
            parallelism.tp,
        )
    )
    if any(event.kind == "collective" for event in optimizer):
        raise FrameworkCounterfactualError(
            "v1 optimizer profile cannot contain collectives; DP synchronization "
            "is declared separately"
        )
    raw_terminal_loss_ids = profile.get("terminal_loss_event_ids")
    terminal_loss = (
        ()
        if raw_terminal_loss_ids is None
        else _profile_events(
            source_events,
            _event_ids(raw_terminal_loss_ids, "terminal_loss_event_ids"),
            "terminal_loss",
            parallelism.tp,
        )
    )
    if terminal_loss:
        terminal_loss_contract = _mapping(
            profile.get("terminal_loss"), "terminal_loss"
        )
        if terminal_loss_contract != {
            "kind": "torch.nn.functional.mse_loss",
            "reduction": "mean",
            "placement": "last-pipeline-stage-only",
            "measurement_transfer": "exact-target-tp-profile",
        }:
            raise FrameworkCounterfactualError(
                "terminal loss profile is outside the exact bounded contract"
            )
    raw_terminal_loss_backward_ids = profile.get(
        "terminal_loss_backward_event_ids"
    )
    terminal_loss_backward = (
        ()
        if raw_terminal_loss_backward_ids is None
        else _profile_events(
            source_events,
            _event_ids(
                raw_terminal_loss_backward_ids,
                "terminal_loss_backward_event_ids",
            ),
            "terminal_loss_backward",
            parallelism.tp,
        )
    )
    pp_backward_scheduler: dict[str, Any] | None = None
    if terminal_loss:
        pp_backward_scheduler = _mapping(
            profile.get("pp_backward_scheduler"), "pp_backward_scheduler"
        )
        scheduler_schema = pp_backward_scheduler.get("schema")
        legacy_two_microbatch_rule = (
            scheduler_schema == "megatron-pp-backward-scheduler-h100-v1"
            and pp_backward_scheduler.get("status") == "observed-framework-rule"
            and pp_backward_scheduler.get("microbatches") == 2
            and microbatches == 2
        )
        bounded_schedule_rule = (
            scheduler_schema == "megatron-pp-backward-scheduler-h100-v2"
            and pp_backward_scheduler.get("status")
            == "source-pinned-rule-with-h100-anchor"
            and pp_backward_scheduler.get("supported_microbatches") == [2, 4]
            and microbatches
            in pp_backward_scheduler.get("supported_microbatches", [])
        )
        if (
            not (legacy_two_microbatch_rule or bounded_schedule_rule)
            or pp_backward_scheduler.get("target") != "h100"
            or pp_backward_scheduler.get("framework_commit")
            != MEGATRON_CORE_SCHEDULE_COMMIT
            or _mapping(
                pp_backward_scheduler.get("provenance"),
                "pp_backward_scheduler.provenance",
            ).get("discovery_job")
            != 8240676
        ):
            raise FrameworkCounterfactualError(
                "PP backward scheduler rule is outside the pinned v24 domain"
            )
    activation_bytes = _positive_integer(
        profile.get("pipeline_activation_bytes_per_tp_rank"),
        "pipeline_activation_bytes_per_tp_rank",
    )

    gradient_sync = profile.get("gradient_sync")
    ddp_forward_sync: dict[str, Any] | None = None
    normalization_compute: dict[str, Any] | None = None
    gradient_collective: str | None = None
    gradient_bytes_per_layer: int | None = None
    if parallelism.dp > 1:
        gradient_sync = _mapping(gradient_sync, "gradient_sync")
        gradient_collective = gradient_sync.get("collective")
        if gradient_collective not in {"all_reduce", "reduce_scatter"}:
            raise FrameworkCounterfactualError(
                "gradient_sync.collective must be all_reduce or reduce_scatter"
            )
        gradient_bytes_per_layer = _positive_integer(
            gradient_sync.get("bytes_per_layer_per_rank"),
            "gradient_sync.bytes_per_layer_per_rank",
        )
        if raw_adapter is not None:
            normalization_compute = _mapping(
                gradient_sync.get("normalization_compute"),
                "gradient_sync.normalization_compute",
            )
            ddp_forward_sync = _mapping(
                profile.get("ddp_forward_sync"), "ddp_forward_sync"
            )
            if (
                ddp_forward_sync.get("schema")
                != "megatron-pytorch-ddp-forward-sync-h100-v1"
                or ddp_forward_sync.get("status") != "observed-framework-rule"
                or ddp_forward_sync.get("target") != "h100"
                or ddp_forward_sync.get("torch_version") != "2.9.1+cu128"
                or ddp_forward_sync.get("cuda_version") != "12.8"
                or ddp_forward_sync.get("nccl_version") != "2.27.5"
                or ddp_forward_sync.get("collective") != "broadcast"
                or ddp_forward_sync.get("message_bytes") != [12, 4]
                or ddp_forward_sync.get("phase") != "forward"
                or _mapping(
                    ddp_forward_sync.get("provenance"),
                    "ddp_forward_sync.provenance",
                ).get("discovery_job")
                != 8238772
            ):
                raise FrameworkCounterfactualError(
                    "ddp_forward_sync is not the exact pinned H100 framework rule"
                )

    layers_per_stage = num_layers // parallelism.pp
    template_events: list[TraceEvent] = []
    for stage in range(parallelism.pp):
        stage_layers = range(stage * layers_per_stage, (stage + 1) * layers_per_stage)
        forward_events: list[TraceEvent] = []
        forward_exits: tuple[str, ...] = ()
        for layer in stage_layers:
            cloned, forward_exits = _clone_layer_phase(
                forward,
                stage=stage,
                logical_layer=layer,
                phase="forward",
                predecessors=forward_exits,
            )
            forward_events.extend(cloned)
        if stage == parallelism.pp - 1 and terminal_loss:
            cloned_loss, forward_exits = _clone_layer_phase(
                terminal_loss,
                stage=stage,
                logical_layer=num_layers - 1,
                phase="forward",
                predecessors=forward_exits,
                semantic_region="terminal-loss",
            )
            forward_events.extend(cloned_loss)
        template_events.extend(forward_events)
        if stage < parallelism.pp - 1:
            template_events.append(
                TraceEvent(
                    id=f"semantic-s{stage}-forward-send",
                    name=f"stage {stage} activation send",
                    kind="collective",
                    duration_us=0.0,
                    stream="communication",
                    dependencies=forward_exits,
                    collective="send",
                    message_bytes=activation_bytes,
                    group_role="pp",
                    group_size=parallelism.pp,
                    metadata={
                        "framework_counterfactual": CONTRACT_SCHEMA,
                        "pipeline_stage": stage,
                        "pipeline_phase": "forward",
                        "pipeline_tp_lane": 0,
                        "message_bytes_semantics": "payload",
                    },
                )
            )

        backward_events: list[TraceEvent] = []
        backward_exits: tuple[str, ...] = ()
        if stage == parallelism.pp - 1 and terminal_loss_backward:
            assert pp_backward_scheduler is not None
            cloned_loss_backward, backward_exits = _clone_layer_phase(
                terminal_loss_backward,
                stage=stage,
                logical_layer=num_layers - 1,
                phase="backward",
                predecessors=backward_exits,
                semantic_region="terminal-loss-backward",
            )
            cloned_loss_backward, backward_exits = _insert_pp_loss_scale_backward(
                cloned_loss_backward,
                backward_exits,
                rule=pp_backward_scheduler,
                stage=stage,
            )
            backward_events.extend(cloned_loss_backward)
        for layer in reversed(tuple(stage_layers)):
            cloned, backward_exits = _clone_layer_phase(
                backward,
                stage=stage,
                logical_layer=layer,
                phase="backward",
                predecessors=backward_exits,
            )
            if parallelism.dp > 1 and normalization_compute is not None:
                cloned, backward_exits = _insert_ddp_gradient_normalization(
                    cloned,
                    backward_exits,
                    rule=normalization_compute,
                    stage=stage,
                    logical_layer=layer,
                )
            backward_events.extend(cloned)
        template_events.extend(backward_events)
        if stage > 0:
            template_events.append(
                TraceEvent(
                    id=f"semantic-s{stage}-backward-send",
                    name=f"stage {stage} gradient send",
                    kind="collective",
                    duration_us=0.0,
                    stream="communication",
                    dependencies=backward_exits,
                    collective="send",
                    message_bytes=activation_bytes,
                    group_role="pp",
                    group_size=parallelism.pp,
                    metadata={
                        "framework_counterfactual": CONTRACT_SCHEMA,
                        "pipeline_stage": stage,
                        "pipeline_phase": "backward",
                        "pipeline_tp_lane": 0,
                        "message_bytes_semantics": "payload",
                    },
                )
            )

    template = WorkloadTrace(
        events=tuple(template_events),
        source=trace.source,
        metadata={
            "pipeline": {"schedule": schedule_name, "microbatches": microbatches},
            "framework_counterfactual": contract,
        },
    )
    if parallelism.pp == 1:
        scheduled = expand_pipeline(template, parallelism, gpus=gpus)
        rank_coordinates = tuple(
            (tp_lane, 0, dp_replica)
            for dp_replica in range(parallelism.dp)
            for tp_lane in range(parallelism.tp)
        )
        rank_events = tuple(
            tuple(
                replace(
                    event,
                    rank=rank,
                    device=rank,
                    metadata={
                        **event.metadata,
                        "pipeline_tp_lane": rank_coordinates[rank][0],
                        "pipeline_dp_replica": rank_coordinates[rank][2],
                        "pipeline_physical_device": rank,
                        "pipeline_rank_expansion": "single-stage-rank-replication-v1",
                    },
                )
                for event in scheduled.events
            )
            for rank in range(gpus)
        )
    else:
        expanded = expand_pipeline_ranks(template, parallelism, gpus)
        rank_events = expanded.rank_events
        rank_coordinates = expanded.rank_coordinates
    compiled_events = _prefix_rank_events(rank_events, rank_coordinates)

    if pp_backward_scheduler is not None:
        accumulation = _mapping(
            pp_backward_scheduler.get("gradient_accumulation_kernel"),
            "gradient_accumulation_kernel",
        )
        accumulation_count = _positive_integer(
            accumulation.get("count_per_accumulating_microbatch"),
            "gradient_accumulation_kernel.count_per_accumulating_microbatch",
        )
        additions: list[TraceEvent] = []
        for rank in range(gpus):
            for microbatch in range(1, microbatches):
                phase_events = [
                    event
                    for event in compiled_events
                    if event.rank == rank
                    and event.kind == "compute"
                    and event.metadata.get("pipeline_phase") == "backward"
                    and event.metadata.get("pipeline_microbatch") == microbatch
                ]
                phase_ids = {event.id for event in phase_events}
                referenced = {
                    dependency
                    for event in phase_events
                    for dependency in event.dependencies
                    if dependency in phase_ids
                }
                predecessors = tuple(
                    event.id for event in phase_events if event.id not in referenced
                )
                if not predecessors:
                    raise FrameworkCounterfactualError(
                        "accumulating backward phase has no measured compute exit"
                    )
                tp_lane, pp_stage, _ = rank_coordinates[rank]
                for index in range(accumulation_count):
                    event_id = (
                        f"rank{rank}::semantic-pp-gradient-accumulation-{index}"
                        f"@mb{microbatch}"
                    )
                    additions.append(
                        TraceEvent(
                            id=event_id,
                            name=str(accumulation["name"]),
                            kind="compute",
                            duration_us=float(accumulation["duration_us"]),
                            stream=phase_events[-1].stream,
                            rank=rank,
                            device=rank,
                            dependencies=predecessors,
                            metadata={
                                "framework_counterfactual": CONTRACT_SCHEMA,
                                "framework_rule": pp_backward_scheduler["schema"],
                                "pipeline_phase": "backward",
                                "pipeline_stage": pp_stage,
                                "pipeline_tp_lane": tp_lane,
                                "pipeline_microbatch": microbatch,
                                "framework_subphase": "parameter-gradient-accumulation",
                                "kernel_signature": accumulation["kernel_signature"],
                                "kernel_resource": pp_backward_scheduler[
                                    "kernel_resource"
                                ],
                                "provenance": pp_backward_scheduler["provenance"],
                            },
                        )
                    )
                    predecessors = (event_id,)
        compiled_events.extend(additions)

    if parallelism.dp > 1 and ddp_forward_sync is not None:
        with_forward_sync: list[TraceEvent] = []
        seen_forward_calls: set[tuple[int, int]] = set()
        for event in compiled_events:
            microbatch = event.metadata.get("pipeline_microbatch", 0)
            forward_call = (event.rank, int(microbatch))
            is_first_forward_event = (
                event.metadata.get("pipeline_phase") == "forward"
                and forward_call not in seen_forward_calls
            )
            if is_first_forward_event:
                seen_forward_calls.add(forward_call)
                tp_lane, pp_stage, _ = rank_coordinates[event.rank]
                first_id = (
                    f"rank{event.rank}::semantic-ddp-forward-broadcast-0"
                    f"@mb{microbatch}"
                )
                second_id = (
                    f"rank{event.rank}::semantic-ddp-forward-broadcast-1"
                    f"@mb{microbatch}"
                )
                memory_ids = [
                    f"rank{event.rank}::semantic-ddp-forward-memory-{index}"
                    f"@mb{microbatch}"
                    for index in range(4)
                ]
                common_metadata = {
                    "framework_counterfactual": CONTRACT_SCHEMA,
                    "framework_rule": ddp_forward_sync["schema"],
                    "pipeline_phase": "forward",
                    "pipeline_stage": pp_stage,
                    "pipeline_tp_lane": tp_lane,
                    "pipeline_microbatch": microbatch,
                    "message_bytes_semantics": "payload",
                    "provenance": ddp_forward_sync["provenance"],
                }
                memory_metadata = {
                    **common_metadata,
                    "framework_rule": "megatron-pytorch-ddp-memory-ops-h100-v1",
                    "byte_claim": "unknown",
                }
                with_forward_sync.extend(
                    (
                        TraceEvent(
                            id=memory_ids[0],
                            name="Memcpy HtoD (Pageable -> Device)",
                            kind="memory",
                            duration_us=0.0,
                            stream="communication-dp-control",
                            rank=event.rank,
                            device=event.rank,
                            dependencies=event.dependencies,
                            metadata={**memory_metadata, "framework_rule_index": 0},
                        ),
                        TraceEvent(
                            id=first_id,
                            name="PyTorch DDP pre-forward state broadcast (12 bytes)",
                            kind="collective",
                            duration_us=0.0,
                            stream="communication-dp",
                            rank=event.rank,
                            device=event.rank,
                            dependencies=(memory_ids[0],),
                            collective="broadcast",
                            message_bytes=12,
                            group_role="dp",
                            group_size=parallelism.dp,
                            metadata={**common_metadata, "framework_rule_index": 0},
                        ),
                        TraceEvent(
                            id=memory_ids[1],
                            name="Memcpy DtoH (Device -> Pageable)",
                            kind="memory",
                            duration_us=0.0,
                            stream="communication-dp-control",
                            rank=event.rank,
                            device=event.rank,
                            dependencies=(first_id,),
                            metadata={**memory_metadata, "framework_rule_index": 1},
                        ),
                        TraceEvent(
                            id=memory_ids[2],
                            name="Memcpy HtoD (Pageable -> Device)",
                            kind="memory",
                            duration_us=0.0,
                            stream="communication-dp-control",
                            rank=event.rank,
                            device=event.rank,
                            dependencies=(memory_ids[1],),
                            metadata={**memory_metadata, "framework_rule_index": 2},
                        ),
                        TraceEvent(
                            id=second_id,
                            name="PyTorch DDP pre-forward state broadcast (4 bytes)",
                            kind="collective",
                            duration_us=0.0,
                            stream="communication-dp",
                            rank=event.rank,
                            device=event.rank,
                            dependencies=(memory_ids[2],),
                            collective="broadcast",
                            message_bytes=4,
                            group_role="dp",
                            group_size=parallelism.dp,
                            metadata={**common_metadata, "framework_rule_index": 1},
                        ),
                        TraceEvent(
                            id=memory_ids[3],
                            name="Memcpy DtoH (Device -> Pageable)",
                            kind="memory",
                            duration_us=0.0,
                            stream="communication-dp-control",
                            rank=event.rank,
                            device=event.rank,
                            dependencies=(second_id,),
                            metadata={**memory_metadata, "framework_rule_index": 3},
                        ),
                    )
                )
            if is_first_forward_event:
                event = replace(
                    event, dependencies=(memory_ids[3],)
                )
            with_forward_sync.append(event)
        compiled_events = with_forward_sync

    by_rank: dict[int, list[TraceEvent]] = {}
    for event in compiled_events:
        by_rank.setdefault(event.rank, []).append(event)
    for rank in range(gpus):
        local = by_rank[rank]
        referenced = {
            dependency for event in local for dependency in event.dependencies
        }
        predecessors = tuple(event.id for event in local if event.id not in referenced)
        tp_lane, pp_stage, _ = rank_coordinates[rank]
        if parallelism.dp > 1:
            assert gradient_collective is not None
            assert gradient_bytes_per_layer is not None
            backward_memory_tail: str | None = None
            if ddp_forward_sync is not None:
                backward_memory_0 = TraceEvent(
                    id=f"rank{rank}::semantic-ddp-backward-memory-0",
                    name="Memcpy DtoD (Device -> Device)",
                    kind="memory",
                    duration_us=0.0,
                    stream="communication-dp-control",
                    rank=rank,
                    device=rank,
                    dependencies=predecessors,
                    metadata={
                        "framework_counterfactual": CONTRACT_SCHEMA,
                        "framework_rule": "megatron-pytorch-ddp-memory-ops-h100-v1",
                        "pipeline_phase": "backward",
                        "pipeline_stage": pp_stage,
                        "pipeline_tp_lane": tp_lane,
                        "byte_claim": "unknown",
                    },
                )
                backward_memory_1 = replace(
                    backward_memory_0,
                    id=f"rank{rank}::semantic-ddp-backward-memory-1",
                    dependencies=(backward_memory_0.id,),
                )
                compiled_events.extend((backward_memory_0, backward_memory_1))
                backward_memory_tail = backward_memory_1.id
            gradient_event = TraceEvent(
                id=f"rank{rank}::semantic-dp-gradient-sync",
                name=f"stage {pp_stage} data-parallel gradient synchronization",
                kind="collective",
                duration_us=0.0,
                stream="communication-dp",
                rank=rank,
                device=rank,
                dependencies=predecessors,
                collective=gradient_collective,
                message_bytes=gradient_bytes_per_layer * layers_per_stage,
                group_role="dp",
                group_size=parallelism.dp,
                metadata={
                    "framework_counterfactual": CONTRACT_SCHEMA,
                    "gradient_accumulation_boundary": "after-all-microbatches",
                    "pipeline_stage": pp_stage,
                    "pipeline_tp_lane": tp_lane,
                    "collective_instance_id": (
                        f"semantic-dp:tp{tp_lane}:pp{pp_stage}:gradient-sync"
                    ),
                    "message_bytes_semantics": str(
                        gradient_sync.get("message_bytes_semantics", "payload")
                    ),
                },
            )
            compiled_events.append(gradient_event)
            predecessors = (
                (gradient_event.id, backward_memory_tail)
                if backward_memory_tail is not None
                else (gradient_event.id,)
            )
        if optimizer:
            stage_start = pp_stage * layers_per_stage
            for logical_layer in range(stage_start, stage_start + layers_per_stage):
                cloned, exits = _clone_layer_phase(
                    optimizer,
                    stage=pp_stage,
                    logical_layer=logical_layer,
                    phase="optimizer",
                    predecessors=predecessors,
                )
                local_ids = {event.id for event in cloned}
                for event in cloned:
                    metadata = dict(event.metadata)
                    metadata.update(
                        {
                            "pipeline_tp_lane": tp_lane,
                            "pipeline_dp_replica": rank_coordinates[rank][2],
                            "pipeline_physical_device": rank,
                            "iteration_tail": "optimizer-after-gradient-sync-v1",
                        }
                    )
                    compiled_events.append(
                        replace(
                            event,
                            id=f"rank{rank}::{event.id}",
                            rank=rank,
                            device=rank,
                            dependencies=tuple(
                                f"rank{rank}::{dependency}"
                                if dependency in local_ids
                                else dependency
                                for dependency in event.dependencies
                            ),
                            metadata=metadata,
                        )
                    )
                predecessors = tuple(f"rank{rank}::{event_id}" for event_id in exits)

    summary = {
        "status": "compiled",
        "schema": CONTRACT_SCHEMA,
        "framework": "megatron-core",
        "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
        "target": target,
        "parallelism": parallelism.to_dict(),
        "gpus": gpus,
        "layers": num_layers,
        "model_dimensions": {
            "hidden_size": model.get("hidden_size"),
            "ffn_hidden_size": model.get("ffn_hidden_size"),
            "sequence_length": model.get("sequence_length"),
            "micro_batch_size": micro_batch_size,
        },
        "layers_per_stage": layers_per_stage,
        "microbatches": microbatches,
        "schedule": schedule_name,
        "source_measured_event_ids": sorted(
            set(profile["forward_event_ids"])
            | set(profile["backward_event_ids"])
            | set(profile.get("optimizer_event_ids", ()))
            | set(profile.get("terminal_loss_event_ids", ()))
            | set(profile.get("terminal_loss_backward_event_ids", ()))
        ),
        "rank_event_count": len(compiled_events),
        "dp_gradient_normalization": (
            {
                "status": "not-required",
                "reason": "target-dp-is-one",
            }
            if parallelism.dp == 1
            else (
                {
                    "status": "applied-calibrated-framework-rule",
                    "schema": normalization_compute["schema"],
                    "discovery_job": normalization_compute["provenance"][
                        "discovery_job"
                    ],
                    "kernels_per_layer": normalization_compute[
                        "parameter_count_per_layer"
                    ],
                }
                if normalization_compute is not None
                else {
                    "status": "contract-only-no-adapter-rule",
                    "claim": "synthetic/manual contract only",
                }
            )
        ),
        "dp_forward_sync": (
            {
                "status": "not-required",
                "reason": "target-dp-is-one",
            }
            if parallelism.dp == 1
            else {
                "status": "applied-observed-framework-rule",
                "schema": ddp_forward_sync["schema"],
                "discovery_job": ddp_forward_sync["provenance"]["discovery_job"],
                "message_bytes": ddp_forward_sync["message_bytes"],
            }
            if ddp_forward_sync is not None
            else {
                "status": "contract-only-no-adapter-rule",
                "claim": "synthetic/manual contract only",
            }
        ),
        "applicability_certificate": applicability_certificate,
        "limitations": [
            "exact measured TP profile only; no TP interpolation",
            "non-interleaved pipeline schedules only",
            "uniform layer allocation across PP stages",
            "one post-accumulation DP gradient collective per stage/rank",
            (
                "DP gradient normalization is not required"
                if parallelism.dp == 1
                else (
                    "two calibrated PyTorch DDP normalization kernels per local "
                    "logical layer"
                    if normalization_compute is not None
                    else "manual contract has no adapter-certified DP compute rule"
                )
            ),
            (
                "optimizer tail omitted because optimizer_event_ids were not supplied"
                if not optimizer
                else "optimizer primitive is replicated once per local logical layer"
            ),
        ],
    }
    compiled = WorkloadTrace(
        events=tuple(compiled_events),
        source={
            **trace.source,
            "kind": "framework-semantic-counterfactual",
            "counterfactual_target": target,
        },
        metadata={
            **trace.metadata,
            "framework_counterfactual_compilation": summary,
        },
    )
    compiled.validate()
    return FrameworkCounterfactualCompilation(
        trace=compiled,
        applied=True,
        summary=summary,
    )


def _parse_candidate(text: str) -> Parallelism:
    values: dict[str, int] = {}
    for assignment in text.split(","):
        if "=" not in assignment:
            raise FrameworkCounterfactualError(
                f"invalid candidate assignment {assignment!r}"
            )
        name, raw_value = assignment.split("=", 1)
        name = name.strip().lower()
        if name in values:
            raise FrameworkCounterfactualError(
                f"duplicate candidate dimension {name!r}"
            )
        values[name] = int(raw_value)
    missing = {"tp", "pp", "dp"} - values.keys()
    if missing:
        raise FrameworkCounterfactualError(
            "candidate is missing " + ", ".join(sorted(missing))
        )
    gpus = values["tp"] * values["pp"] * values["dp"]
    return Parallelism.parse(text, gpus)


def _candidate_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        unknown = set(value) - {"tp", "pp", "dp", "ep"}
        if unknown:
            raise FrameworkCounterfactualError(
                f"candidate has unknown dimensions {sorted(unknown)}"
            )
        return ",".join(
            f"{name}={value[name]}"
            for name in ("tp", "pp", "dp", "ep")
            if name in value
        )
    raise FrameworkCounterfactualError(
        "candidate-file entries must be strings or dimension mappings"
    )


def _evaluate_search_candidate(
    trace: WorkloadTrace,
    topology_path: Path,
    target: Any,
    parallelism: Parallelism,
) -> dict[str, Any]:
    """Compile one candidate in an isolated frame and retain only its summary."""

    gpus = parallelism.tp * parallelism.pp * parallelism.dp
    topology = load_topology(topology_path, gpus)
    if topology.target is not None and topology.target != target.name:
        raise FrameworkCounterfactualError(
            "topology target does not match search target"
        )
    compilation = compile_framework_counterfactual(
        trace, parallelism, gpus, target.name
    )
    if not compilation.applied:
        raise FrameworkCounterfactualError(
            "workload has no framework counterfactual contract"
        )
    from .simulator import simulate

    prediction = simulate(
        compilation.trace,
        gpus,
        parallelism,
        topology,
        target,
        materialize_timeline=False,
    )
    return {
        "status": "compiled-and-simulated",
        "compilation": compilation.summary,
        "prediction_summary": prediction["summary"],
        "prediction_claim": prediction["prediction_claim"],
        "warnings": prediction["warnings"],
    }


def search_main(argv: list[str] | None = None) -> int:
    """Compile and simulate many semantic candidates without training reruns."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Megatron semantic TP/PP/DP candidates from one measured "
            "primitive workload; this command never executes the training model"
        )
    )
    parser.add_argument("--workload-trace", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--framework-adapter",
        choices=("auto", "none"),
        default="auto",
        help="automatically bind an exact supported framework measurement",
    )
    parser.add_argument(
        "--megatron-provenance",
        type=Path,
        help=(
            "pinned Megatron PROVENANCE.json; otherwise use "
            "SCALETETHER_MEGATRON_PROVENANCE or MEGATRON_ROOT"
        ),
    )
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    raw_candidates: list[object] = list(args.candidate)
    if args.candidate_file is not None:
        loaded = json.loads(args.candidate_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise FrameworkCounterfactualError("candidate file must contain a list")
        raw_candidates.extend(loaded)
    if not raw_candidates:
        raise FrameworkCounterfactualError("at least one candidate is required")

    candidates = [_parse_candidate(_candidate_text(item)) for item in raw_candidates]
    identities = [json.dumps(item.to_dict(), sort_keys=True) for item in candidates]
    if len(set(identities)) != len(identities):
        raise FrameworkCounterfactualError("candidate list contains duplicates")

    trace = WorkloadTrace.load(args.workload_trace)
    automatic_framework_adapter: dict[str, Any] | None = None
    has_contract = isinstance(trace.metadata.get("framework_counterfactual"), dict)
    measurement = trace.metadata.get("framework_measurement")
    if (
        not has_contract
        and measurement is not None
        and args.framework_adapter == "auto"
    ):
        from .megatron_adapter import (
            MEASUREMENT_SCHEMA,
            prepare_megatron_counterfactual,
            resolve_megatron_provenance_path,
        )

        if not isinstance(measurement, dict) or (
            measurement.get("schema") != MEASUREMENT_SCHEMA
            or measurement.get("framework") != "megatron-core"
        ):
            raise FrameworkCounterfactualError(
                "automatic search does not support this framework measurement "
                "declaration"
            )
        provenance = resolve_megatron_provenance_path(args.megatron_provenance)
        if provenance is None:
            raise FrameworkCounterfactualError(
                "automatic Megatron search preparation requires "
                "--megatron-provenance, SCALETETHER_MEGATRON_PROVENANCE, or "
                "MEGATRON_ROOT/PROVENANCE.json"
            )
        trace = prepare_megatron_counterfactual(
            trace,
            workload_path=args.workload_trace,
            provenance_path=provenance,
        )
        automatic_framework_adapter = {
            "schema": "scaletether-automatic-framework-search-adapter-v1",
            "status": "prepared",
            "adapter": "megatron-core-mlp-v1",
            "source_workload_sha256": hashlib.sha256(
                args.workload_trace.read_bytes()
            ).hexdigest(),
            "provenance": str(provenance),
            "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
            "preparation": trace.metadata["framework_counterfactual_preparation"],
        }
    target = architecture(args.target.strip().lower())
    results: list[dict[str, Any]] = []
    compiled_count = 0
    for index, parallelism in enumerate(candidates):
        gpus = parallelism.tp * parallelism.pp * parallelism.dp
        record: dict[str, Any] = {
            "index": index,
            "parallelism": parallelism.to_dict(),
            "gpus": gpus,
        }
        try:
            record.update(
                _evaluate_search_candidate(trace, args.topology, target, parallelism)
            )
            compiled_count += 1
        except ValueError as error:
            record.update(
                {
                    "status": "abstained",
                    "reason": str(error),
                    "error_type": type(error).__name__,
                }
            )
        results.append(record)
        # Generated traces and prediction documents contain large mutually
        # referential metadata graphs. Release each discarded candidate before
        # compiling the next one so thousand-candidate searches do not retain
        # every prior graph until process exit.
        _release_candidate_memory()

    payload = args.workload_trace.read_bytes()
    document = {
        "schema": "scaletether-framework-counterfactual-search-v1",
        "workload_trace": {
            "path": str(args.workload_trace.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "topology": str(args.topology.resolve()),
        "target": target.name,
        "candidate_count": len(results),
        "compiled_count": compiled_count,
        "abstained_count": len(results) - compiled_count,
        "executed_training_candidate_count": 0,
        "framework_adapter": args.framework_adapter,
        "automatic_framework_adapter": automatic_framework_adapter,
        "timing_interpretation": (
            "each candidate retains its simulator prediction_claim; analytical "
            "communication fallbacks are not calibrated timing evidence"
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"compiled={compiled_count} abstained={len(results) - compiled_count}")
    print("executed_training_candidates=0")
    return 0 if compiled_count else 2


if __name__ == "__main__":
    raise SystemExit(search_main())
