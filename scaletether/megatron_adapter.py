from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .framework_counterfactual import CONTRACT_SCHEMA
from .pipeline import MEGATRON_CORE_SCHEDULE_COMMIT
from .schema import TraceEvent, WorkloadTrace


MEASUREMENT_SCHEMA = "megatron-core-mlp-measurement-v1"
SOURCE_SEMANTIC_BINDING_SCHEMA = "megatron-source-semantic-binding-v1"


# Prospective job 8238772 captured this PyTorch DDP pre-forward synchronization
# on every TP=2/DP=2 H100 rank.  It is a pinned framework rule, not an
# inference from the DP=1 source.
_H100_DDP_FORWARD_SYNC_EVIDENCE = {
    "schema": "megatron-pytorch-ddp-forward-sync-h100-v1",
    "status": "observed-framework-rule",
    "target": "h100",
    "device_name": "NVIDIA H100",
    "compute_capability": "9.0",
    "torch_version": "2.9.1+cu128",
    "cuda_version": "12.8",
    "nccl_version": "2.27.5",
    "collective": "broadcast",
    "message_bytes": [12, 4],
    "message_bytes_semantics": "payload",
    "phase": "forward",
    "provenance": {
        "discovery_job": 8238772,
        "discovery_status": "prospective-validation-precondition-negative",
        "claim": "explicit bounded rule; not inferred from the DP=1 source",
    },
}


# This rule is discovery evidence from prospective H100 job 8237716, not a
# property inferred from a DP=1 anchor.  It is deliberately narrow: the kernel
# launch identity is valid only for the pinned software stack and 256x256 local
# float32 parameter gradients.  The counterfactual compiler validates every
# field again before it may use the rule.
_H100_DDP_NORMALIZATION_EVIDENCE = {
    "schema": "megatron-pytorch-ddp-gradient-normalization-h100-v1",
    "status": "calibrated-framework-rule",
    "target": "h100",
    "device_name": "NVIDIA H100",
    "compute_capability": "9.0",
    "torch_version": "2.9.1+cu128",
    "cuda_version": "12.8",
    "nccl_version": "2.27.5",
    "local_parameter_shape": [256, 256],
    "parameter_count_per_layer": 2,
    "parameter_gradient_ready_kernel_signature": (
        "torch-kernel-v2:2aff6db44315bc8b32053c93"
    ),
    "parameter_gradient_ready_operator": {
        "name": "aten::mm",
        "output_shape": [256, 256],
        "derivation": "[256,K] x [K,256] local parameter-gradient GEMM",
    },
    "kernel_name": (
        "void at::native::vectorized_elementwise_kernel<4, "
        "at::native::AUnaryFunctor<float, float, float, "
        "at::native::binary_internal::MulFunctor<float> >, "
        "std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, "
        "float, float, at::native::binary_internal::MulFunctor<float> >, "
        "std::array<char*, 2ul>)"
    ),
    "kernel_signature": "torch-kernel-v2:824d23297edf28a28779e7dd",
    "kernel_launch_signature": "torch-kernel-launch-v1:0a42fb668482efff7a70739b",
    "duration_us": 1.2640380859375,
    "duration_evidence": {
        "method": "same-qualified-signature-median-v1",
        "sample_count": 2,
        "minimum_duration_us": 1.248046875,
        "maximum_duration_us": 1.280029296875,
        "status": "stable-by-range-gate",
    },
    "kernel_resource": {
        "grid": [64, 1, 1],
        "block": [128, 1, 1],
        "registers_per_thread": 32,
        "shared_memory_bytes": 0,
        "grid_blocks": 64,
        "device_sm_count": 132,
    },
    "provenance": {
        "discovery_job": 8237716,
        "discovery_result": "megatron-heldout-dp-h100-v11",
        "discovery_status": "prospective-structural-negative",
        "report": "docs/experiments/8237716-megatron-heldout-dp-h100-v11.md",
        "heldout_rank0_workload": (
            "results/8237716-megatron-heldout-dp-h100-v11/"
            "heldout-tp2-dp2/workload.rank0.json"
        ),
        "claim": "explicit calibrated rule; not inferred from the DP1 source",
    },
}


# Prospective PP v24 reached the frozen phase-inventory gate after every
# communication and memory gate passed. It exposed two deterministic pieces
# of the pinned non-interleaved scheduler that are absent from a single
# PP=1 module primitive: scalar-loss scaling on the last stage and parameter
# gradient accumulation after the first microbatch. This is an explicit,
# versioned framework rule, not an inference from arbitrary training code.
_H100_PP_BACKWARD_SCHEDULER_EVIDENCE = {
    "schema": "megatron-pp-backward-scheduler-h100-v2",
    "status": "source-pinned-rule-with-h100-anchor",
    "target": "h100",
    "device_name": "NVIDIA H100",
    "compute_capability": "9.0",
    "torch_version": "2.9.1+cu128",
    "cuda_version": "12.8",
    "nccl_version": "2.27.5",
    "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
    # The scheduler source makes both transformations independent of the
    # number of microbatches: loss scaling occurs once on the last stage and
    # parameter-gradient accumulation occurs after every backward microbatch
    # except the first.  Two microbatches are physically anchored below; four
    # is the only source-derived extension admitted by the frozen Level-2
    # protocol and remains subject to prospective held-out H100 validation.
    "supported_microbatches": [2, 4],
    "physical_anchor_microbatches": [2],
    "loss_scale_backward_kernels": [
        {
            "name": (
                "void at::native::vectorized_elementwise_kernel<4, "
                "at::native::BUnaryFunctor<float, float, float, "
                "at::native::binary_internal::MulFunctor<float> >, "
                "std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<float, "
                "float, float, at::native::binary_internal::MulFunctor<float> >, "
                "std::array<char*, 2ul>)"
            ),
            "kernel_signature": "torch-kernel-v2:5c4e3315ec9a577eb30e9af0",
            "duration_us": 1.152587890625,
        },
        {
            "name": (
                "void at::native::vectorized_elementwise_kernel<4, "
                "at::native::AUnaryFunctor<float, float, float, "
                "at::native::binary_internal::MulFunctor<float> >, "
                "std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, "
                "float, float, at::native::binary_internal::MulFunctor<float> >, "
                "std::array<char*, 2ul>)"
            ),
            "kernel_signature": "torch-kernel-v2:2a2b92b8d714af27616961f8",
            "duration_us": 1.152099609375,
        },
    ],
    "gradient_accumulation_kernel": {
        "name": (
            "void at::native::vectorized_elementwise_kernel<4, "
            "at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >"
            "(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)"
        ),
        "kernel_signature": "torch-kernel-v2:33c920cc466da5737c5a8612",
        "duration_us": 1.2640380859375,
        "count_per_accumulating_microbatch": 2,
    },
    "kernel_resource": {
        "grid": [1, 1, 1],
        "block": [128, 1, 1],
        "registers_per_thread": 32,
        "shared_memory_bytes": 0,
        "grid_blocks": 1,
        "device_sm_count": 132,
    },
    "provenance": {
        "discovery_job": 8240676,
        "discovery_result": "megatron-heldout-pp-h100-v24",
        "discovery_status": "prospective-structural-negative",
        "report": "docs/experiments/8240676-megatron-heldout-pp-h100-v24.md",
        "claim": "bounded pinned-scheduler rule; no cross-version timing claim",
        "four_microbatch_status": (
            "source-derived-prospective-candidate-not-yet-physically-validated"
        ),
    },
}

_RUNTIME_DOMAIN_KEYS = (
    "device_name",
    "compute_capability",
    "torch_version",
    "cuda_version",
    "nccl_version",
)


class MegatronAdapterError(ValueError):
    """Captured Megatron evidence cannot justify the requested contract."""


def _uniform_runtime_domain(trace: WorkloadTrace) -> dict[str, Any]:
    """Return runtime identity only when every captured rank agrees exactly.

    A single-rank capture records these fields directly in ``source``.  The
    distributed merger deliberately retains each original source under
    ``source.rank_sources``.  Runtime metadata is not promoted to the merged
    trace metadata, so consulting that metadata loses the physical evidence
    and caused the v12 operational negative.  Do not infer a common domain
    from one rank: a missing or heterogeneous value leaves the rule
    inapplicable.
    """

    raw_rank_sources = trace.source.get("rank_sources")
    if raw_rank_sources is None:
        sources: tuple[dict[str, Any], ...] = (trace.source,)
    elif isinstance(raw_rank_sources, list) and raw_rank_sources and all(
        isinstance(source, dict) for source in raw_rank_sources
    ):
        sources = tuple(raw_rank_sources)
    else:
        return {key: None for key in _RUNTIME_DOMAIN_KEYS}

    domain: dict[str, Any] = {"target": trace.source.get("target")}
    for key in _RUNTIME_DOMAIN_KEYS:
        values = {source.get(key) for source in sources}
        domain[key] = values.pop() if len(values) == 1 else None
    return domain


def resolve_megatron_provenance_path(explicit: Path | None) -> Path | None:
    """Resolve pinned Megatron provenance without ignoring bad configuration."""

    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Megatron provenance does not exist: {candidate}")
        return candidate
    configured = os.environ.get("SCALETETHER_MEGATRON_PROVENANCE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"SCALETETHER_MEGATRON_PROVENANCE does not name a file: {candidate}"
            )
        return candidate
    root = os.environ.get("MEGATRON_ROOT")
    if root:
        candidate = (Path(root).expanduser() / "PROVENANCE.json").resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"MEGATRON_ROOT has no PROVENANCE.json: {candidate}"
            )
        return candidate
    return None


def exact_megatron_mlp_measurement_contract(
    *,
    tensor_parallel_size: int,
    data_parallel_size: int,
    num_layers: int,
    pipeline_parallel_size: int | None = None,
    pipeline_microbatches: int | None = None,
    sequence_length: int = 64,
    warmup_steps: int | None = None,
    terminal_loss: bool = False,
) -> dict[str, Any]:
    """Return the exact measurement declaration supported by held-out v1 gates."""

    measurement: dict[str, Any] = {
        "schema": MEASUREMENT_SCHEMA,
        "framework": "megatron-core",
        "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
        "environment_constraints": {"NO_VCS_VERSION": "1"},
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": data_parallel_size,
        "model": {
            "num_layers": num_layers,
            "hidden_size": 256,
            "ffn_hidden_size": 512,
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
        "data_parallel": {
            "kind": (
                "torch.nn.parallel.DistributedDataParallel"
                if data_parallel_size > 1
                else "none"
            ),
            "gradient_sync": data_parallel_size > 1,
        },
        "phase_markers": {
            "forward": "megatron_mlp_forward",
            "backward": "megatron_mlp_backward",
            "optimizer": "megatron_mlp_optimizer",
        },
    }
    if terminal_loss:
        measurement["terminal_loss"] = {
            "kind": "torch.nn.functional.mse_loss",
            "reduction": "mean",
            "placement": "last-pipeline-stage-only",
        }
        measurement["phase_markers"]["terminal_loss"] = (
            "megatron_mlp_terminal_loss"
        )
        measurement["phase_markers"]["terminal_loss_backward"] = (
            "megatron_mlp_terminal_loss_backward"
        )
    if warmup_steps is not None:
        if isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        measurement["measurement_protocol"] = {
            "schema": "steady-state-optimizer-step-v1",
            "warmup_steps": warmup_steps,
            "warmup_scope": "forward-backward-optimizer",
            "quiescence_before_measured_step": True,
        }
    if pipeline_parallel_size is not None:
        if pipeline_microbatches is None:
            raise ValueError(
                "pipeline_microbatches is required with pipeline_parallel_size"
            )
        measurement["pipeline_parallel_size"] = pipeline_parallel_size
        measurement["pipeline"] = {
            "kind": "megatron-core-non-interleaved-1f1b",
            "microbatches": pipeline_microbatches,
            "layers_per_stage": 1,
        }
    elif pipeline_microbatches is not None:
        raise ValueError(
            "pipeline_parallel_size is required with pipeline_microbatches"
        )
    return measurement


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def megatron_source_semantic_document(trace: WorkloadTrace) -> dict[str, Any]:
    """Return the capture evidence covered by an adapter certificate.

    The prepared trace gains counterfactual metadata after this document is
    computed, so binding the complete ``WorkloadTrace`` would be recursive.
    Instead, bind every measured event, the physical source identity, the
    framework declaration, and the capture limitations that control adapter
    admission.  The compiler recomputes this document before every rewrite.
    """

    raw_limitations = trace.metadata.get("capture_limitations", ())
    if isinstance(raw_limitations, str):
        raw_limitations = (raw_limitations,)
    limitations = [str(item) for item in raw_limitations if str(item).strip()]
    return {
        "schema": SOURCE_SEMANTIC_BINDING_SCHEMA,
        "workload_schema_version": trace.schema_version,
        "source": trace.source,
        "framework_measurement": trace.metadata.get("framework_measurement"),
        "capture_limitations": limitations,
        "events": [event.to_dict() for event in trace.events],
    }


def megatron_source_semantic_sha256(trace: WorkloadTrace) -> str:
    payload = json.dumps(
        megatron_source_semantic_document(trace),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MegatronAdapterError(f"{name} must be a positive integer")
    return value


def _captured_group_size(event: TraceEvent) -> int | None:
    """Return the measured size retained before role-based generalization."""

    value = event.group_size
    if value is None:
        value = event.metadata.get("observed_group_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _measurement(trace: WorkloadTrace) -> dict[str, Any]:
    raw = trace.metadata.get("framework_measurement")
    if not isinstance(raw, dict):
        raise MegatronAdapterError(
            "capture has no consistent framework_measurement declaration"
        )
    if raw.get("schema") != MEASUREMENT_SCHEMA:
        raise MegatronAdapterError("unsupported Megatron measurement schema")
    if raw.get("framework") != "megatron-core":
        raise MegatronAdapterError("measurement is not Megatron-Core")
    if raw.get("framework_commit") != MEGATRON_CORE_SCHEDULE_COMMIT:
        raise MegatronAdapterError("measurement uses an unvalidated Megatron commit")
    constraints = raw.get("environment_constraints")
    if not isinstance(constraints, dict) or constraints.get("NO_VCS_VERSION") != "1":
        raise MegatronAdapterError(
            "measurement does not prove NO_VCS_VERSION=1 process isolation"
        )
    return raw


def _phase_events(
    trace: WorkloadTrace,
    *,
    rank: int,
    phase_name: str,
) -> tuple[TraceEvent, ...]:
    events = tuple(
        event
        for event in trace.events
        if event.rank == rank
        and isinstance(event.metadata.get("framework_phase_marker"), dict)
        and event.metadata["framework_phase_marker"].get("name") == phase_name
    )
    if not events:
        raise MegatronAdapterError(
            f"rank {rank} has no events for framework phase {phase_name!r}"
        )
    instances = {
        event.metadata["framework_phase_marker"].get("instance_id") for event in events
    }
    if len(instances) != 1:
        raise MegatronAdapterError(
            f"framework phase {phase_name!r} does not resolve to one rank-local instance"
        )
    if not any(event.kind == "compute" for event in events):
        raise MegatronAdapterError(
            f"framework phase {phase_name!r} contains no captured compute"
        )
    return events


def _kernel_identity(event: TraceEvent) -> str:
    code = event.metadata.get("kernel_code_identity")
    if isinstance(code, dict):
        for key in (
            "qualified_signature_sha256",
            "code_object_sha256",
            "symbol_sha256",
        ):
            value = code.get(key)
            if isinstance(value, str) and value:
                return f"{key}:{value}"
    signature = event.metadata.get("kernel_signature")
    if isinstance(signature, str) and signature:
        return f"kernel_signature:{signature}"
    return f"name:{event.name}"


def _event_structure(event: TraceEvent) -> tuple[object, ...]:
    if event.kind == "compute":
        operation = _kernel_identity(event)
    elif event.kind == "collective":
        operation = event.collective
    else:
        operation = event.name
    return (
        event.kind,
        operation,
        event.collective,
        event.message_bytes,
        event.group_role,
        _captured_group_size(event) if event.kind == "collective" else None,
    )


def _counter_rows(counter: Counter[object]) -> list[dict[str, object]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(
            (
                (json.dumps(key, separators=(",", ":")), count)
                for key, count in counter.items()
            ),
            key=lambda item: item[0],
        )
    ]


def _phase_structure_fingerprint(events: tuple[TraceEvent, ...]) -> dict[str, Any]:
    """Return a rank-independent structural summary, excluding durations.

    Event IDs, devices, raw stream handles, and observed timestamps are
    intentionally excluded. The certificate covers operation multiplicity,
    in-phase dependency edges, external-frontier shape, and each stream-local
    operation sequence. This is strong enough to reject an asymmetric TP lane
    without claiming that one lane's measured duration represents all lanes.
    """

    by_id = {event.id: event for event in events}
    if len(by_id) != len(events):
        raise MegatronAdapterError("framework phase contains duplicate event ids")
    structures = {event.id: _event_structure(event) for event in events}
    nodes: Counter[object] = Counter(structures.values())
    edges: Counter[object] = Counter()
    external_frontier: Counter[object] = Counter()
    referenced: set[str] = set()
    for event in events:
        destination = structures[event.id]
        for dependency in event.dependencies:
            if dependency in by_id:
                edges[(structures[dependency], destination)] += 1
                referenced.add(dependency)
            else:
                external_frontier[destination] += 1
    entries = Counter(
        structures[event.id]
        for event in events
        if not any(dependency in by_id for dependency in event.dependencies)
    )
    exits = Counter(
        structures[event.id] for event in events if event.id not in referenced
    )
    streams: dict[str, list[tuple[object, ...]]] = {}
    for event in events:
        streams.setdefault(event.stream, []).append(structures[event.id])
    stream_sequences = sorted(
        json.dumps(sequence, separators=(",", ":")) for sequence in streams.values()
    )
    document: dict[str, Any] = {
        "event_count": len(events),
        "node_multiset": _counter_rows(nodes),
        "dependency_edge_multiset": _counter_rows(edges),
        "external_frontier_multiset": _counter_rows(external_frontier),
        "entry_multiset": _counter_rows(entries),
        "exit_multiset": _counter_rows(exits),
        "stream_local_sequences": stream_sequences,
        "excluded_fields": [
            "duration_us",
            "observed_start_us",
            "rank",
            "device",
            "raw_stream_handle",
            "event_id",
        ],
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    document["sha256"] = hashlib.sha256(payload).hexdigest()
    return document


def _certify_tp_rank_symmetry(
    trace: WorkloadTrace,
    *,
    tp: int,
    representative_rank: int,
    phase_names: dict[str, str],
) -> tuple[dict[str, Any], dict[int, dict[str, tuple[TraceEvent, ...]]]]:
    observed_ranks = {event.rank for event in trace.events}
    expected_ranks = set(range(tp))
    if observed_ranks != expected_ranks:
        raise MegatronAdapterError(
            "DP=1/PP=1 source must contain exactly one measured rank per TP lane; "
            f"expected={sorted(expected_ranks)}, observed={sorted(observed_ranks)}"
        )
    if representative_rank not in expected_ranks:
        raise MegatronAdapterError("representative rank is outside measured TP lanes")

    phases_by_rank: dict[int, dict[str, tuple[TraceEvent, ...]]] = {}
    fingerprints: dict[int, dict[str, dict[str, Any]]] = {}
    for rank in sorted(expected_ranks):
        phases_by_rank[rank] = {}
        fingerprints[rank] = {}
        for phase, marker_name in phase_names.items():
            events = _phase_events(trace, rank=rank, phase_name=marker_name)
            phases_by_rank[rank][phase] = events
            fingerprints[rank][phase] = _phase_structure_fingerprint(events)

    representative = fingerprints[representative_rank]
    for rank in sorted(expected_ranks - {representative_rank}):
        for phase in phase_names:
            if fingerprints[rank][phase]["sha256"] != representative[phase]["sha256"]:
                raise MegatronAdapterError(
                    "measured TP rank structure is asymmetric: "
                    f"rank={rank}, representative={representative_rank}, phase={phase}"
                )

    return (
        {
            "schema": "megatron-tp-rank-structural-symmetry-v1",
            "status": "exact-structural-match",
            "scope": "all-measured-TP-lanes-forward-backward-optimizer",
            "representative_rank": representative_rank,
            "measured_ranks": sorted(expected_ranks),
            "phase_fingerprints": {
                phase: representative[phase]["sha256"] for phase in phase_names
            },
            "timing_claim": (
                "none; duration fields are excluded and representative-rank timing "
                "is retained explicitly"
            ),
        },
        phases_by_rank,
    )


def prepare_megatron_counterfactual(
    trace: WorkloadTrace,
    *,
    workload_path: Path,
    provenance_path: Path,
    representative_rank: int = 0,
    schedule: str = "1f1b",
    gradient_collective: str = "all_reduce",
) -> WorkloadTrace:
    measurement = _measurement(trace)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("commit") != MEGATRON_CORE_SCHEDULE_COMMIT:
        raise MegatronAdapterError("Megatron provenance commit does not match adapter")
    target = str(trace.source.get("target", "")).strip().lower()
    if not target:
        raise MegatronAdapterError("captured workload has no exact target accelerator")
    if schedule not in {"gpipe", "1f1b"}:
        raise MegatronAdapterError("adapter supports only gpipe or 1f1b")
    if gradient_collective not in {"all_reduce", "reduce_scatter"}:
        raise MegatronAdapterError(
            "gradient collective must be all_reduce or reduce_scatter"
        )
    raw_limitations = trace.metadata.get("capture_limitations", ())
    if isinstance(raw_limitations, str):
        raw_limitations = (raw_limitations,)
    limitations = [str(item) for item in raw_limitations if str(item).strip()]
    if limitations:
        raise MegatronAdapterError(
            "captured workload is incomplete: " + "; ".join(limitations)
        )

    tp = _positive_integer(
        measurement.get("tensor_parallel_size"), "tensor_parallel_size"
    )
    dp = _positive_integer(measurement.get("data_parallel_size"), "data_parallel_size")
    if dp != 1:
        raise MegatronAdapterError(
            "v1 adapter requires a DP=1 source measurement; DP is synthesized "
            "from the declared gradient contract"
        )
    model = measurement.get("model")
    if not isinstance(model, dict):
        raise MegatronAdapterError("measurement model must be a mapping")
    num_layers = _positive_integer(model.get("num_layers"), "model.num_layers")
    hidden = _positive_integer(model.get("hidden_size"), "model.hidden_size")
    ffn_hidden = _positive_integer(
        model.get("ffn_hidden_size"), "model.ffn_hidden_size"
    )
    sequence = _positive_integer(model.get("sequence_length"), "model.sequence_length")
    micro_batch = _positive_integer(
        model.get("micro_batch_size"), "model.micro_batch_size"
    )
    global_batch = _positive_integer(
        model.get("global_batch_size"), "model.global_batch_size"
    )
    activation_element_bytes = _positive_integer(
        model.get("activation_element_bytes"), "model.activation_element_bytes"
    )
    parameter_element_bytes = _positive_integer(
        model.get("parameter_element_bytes"), "model.parameter_element_bytes"
    )
    if (
        model.get("parameter_dtype") != "float32"
        or model.get("activation_dtype") != "float32"
        or parameter_element_bytes != 4
        or activation_element_bytes != 4
    ):
        raise MegatronAdapterError(
            "v1 supports only float32 parameters and activations with four-byte elements"
        )
    if model.get("bias") is not False:
        raise MegatronAdapterError(
            "v1 MLP parameter-volume formula requires bias=false"
        )
    if ffn_hidden % tp:
        raise MegatronAdapterError("ffn_hidden_size must be divisible by captured TP")

    layer = measurement.get("layer")
    expected_layer = {
        "kind": "megatron-core-column-gelu-row-mlp-v1",
        "column_gather_output": False,
        "row_input_is_parallel": True,
        "sequence_parallel": False,
    }
    if layer != expected_layer:
        raise MegatronAdapterError(
            "measurement layer is outside the exact bounded column/GELU/row MLP contract"
        )
    optimizer_declaration = measurement.get("optimizer")
    if optimizer_declaration != {"kind": "torch.optim.AdamW"}:
        raise MegatronAdapterError(
            "measurement optimizer is outside the exact bounded AdamW contract"
        )
    data_parallel = measurement.get("data_parallel")
    if data_parallel != {"kind": "none", "gradient_sync": False}:
        raise MegatronAdapterError(
            "DP=1 source must declare no data-parallel wrapper or gradient sync"
        )

    phase_markers = measurement.get("phase_markers")
    if not isinstance(phase_markers, dict):
        raise MegatronAdapterError("measurement phase_markers must be a mapping")
    phase_names = {
        phase: str(phase_markers.get(phase, ""))
        for phase in ("forward", "backward", "optimizer")
    }
    if any(not value for value in phase_names.values()):
        raise MegatronAdapterError(
            "measurement must declare non-empty forward/backward/optimizer markers"
        )
    terminal_loss_declaration = measurement.get("terminal_loss")
    terminal_loss_enabled = terminal_loss_declaration is not None
    if terminal_loss_enabled:
        if terminal_loss_declaration != {
            "kind": "torch.nn.functional.mse_loss",
            "reduction": "mean",
            "placement": "last-pipeline-stage-only",
        }:
            raise MegatronAdapterError(
                "terminal loss is outside the exact bounded MSE-mean contract"
            )
        terminal_loss_marker = phase_markers.get("terminal_loss")
        if not isinstance(terminal_loss_marker, str) or not terminal_loss_marker:
            raise MegatronAdapterError(
                "terminal loss declaration requires a non-empty phase marker"
            )
        phase_names["terminal_loss"] = terminal_loss_marker
        terminal_loss_backward_marker = phase_markers.get(
            "terminal_loss_backward"
        )
        if (
            not isinstance(terminal_loss_backward_marker, str)
            or not terminal_loss_backward_marker
        ):
            raise MegatronAdapterError(
                "terminal loss declaration requires a backward phase marker"
            )
        phase_names["terminal_loss_backward"] = terminal_loss_backward_marker
    rank_symmetry, phases_by_rank = _certify_tp_rank_symmetry(
        trace,
        tp=tp,
        representative_rank=representative_rank,
        phase_names=phase_names,
    )
    forward = phases_by_rank[representative_rank]["forward"]
    backward = phases_by_rank[representative_rank]["backward"]
    optimizer = phases_by_rank[representative_rank]["optimizer"]
    terminal_loss = (
        phases_by_rank[representative_rank]["terminal_loss"]
        if terminal_loss_enabled
        else ()
    )
    terminal_loss_backward = (
        phases_by_rank[representative_rank]["terminal_loss_backward"]
        if terminal_loss_enabled
        else ()
    )
    source_semantic_sha256 = megatron_source_semantic_sha256(trace)
    phase_order = {
        "forward": 0,
        "terminal_loss": 1,
        "terminal_loss_backward": 2,
        "backward": 3,
        "optimizer": 4,
    }
    phase_by_event_id = {
        event.id: phase
        for rank_phases in phases_by_rank.values()
        for phase, events in rank_phases.items()
        for event in events
    }
    unassigned = sorted(
        event.id for event in trace.events if event.id not in phase_by_event_id
    )
    if unassigned:
        preview = unassigned[:8]
        raise MegatronAdapterError(
            "captured source contains device events outside the declared framework "
            f"phases: {preview}"
        )
    for event in trace.events:
        event_phase = phase_by_event_id[event.id]
        for dependency in event.dependencies:
            dependency_phase = phase_by_event_id.get(dependency)
            if (
                dependency_phase is None
                or phase_order[dependency_phase] > phase_order[event_phase]
            ):
                raise MegatronAdapterError(
                    "captured phase dependency violates forward/backward/optimizer order: "
                    f"{dependency!r} -> {event.id!r}"
                )
    if any(event.kind == "collective" for event in optimizer):
        raise MegatronAdapterError(
            "optimizer phase unexpectedly contains a collective; gradient sync "
            "must remain a separate contract operation"
        )
    if tp > 1:
        for phase_name, events in (("forward", forward), ("backward", backward)):
            collectives = [event for event in events if event.kind == "collective"]
            if not collectives:
                raise MegatronAdapterError(
                    f"captured TP>1 {phase_name} phase has no collective"
                )
            for event in collectives:
                if event.group_role != "tp" or _captured_group_size(event) != tp:
                    raise MegatronAdapterError(
                        f"{phase_name} collective {event.id!r} lacks exact TP role/size"
                    )

    runtime_domain = _uniform_runtime_domain(trace)
    activation_bytes = sequence * micro_batch * hidden * activation_element_bytes
    local_parameter_elements_per_layer = 2 * hidden * ffn_hidden // tp
    gradient_bytes_per_layer = (
        local_parameter_elements_per_layer * parameter_element_bytes
    )
    profile = {
        "target": target,
        "measurement": {
            "status": "measured",
            "sample_count": 1,
            "method": "captured-pinned-megatron-phase-dag+tp-symmetry-v2",
            "representative_rank": representative_rank,
            "rank_symmetry": rank_symmetry,
            "source_workload_sha256": _sha256(workload_path),
            "source_semantic_sha256": source_semantic_sha256,
            "megatron_provenance_sha256": _sha256(provenance_path),
        },
        "forward_event_ids": [event.id for event in forward],
        "backward_event_ids": [event.id for event in backward],
        "optimizer_event_ids": [event.id for event in optimizer],
        "pipeline_activation_bytes_per_tp_rank": activation_bytes,
        "gradient_sync": {
            "collective": gradient_collective,
            "bytes_per_layer_per_rank": gradient_bytes_per_layer,
            "message_bytes_semantics": "payload",
            "derivation": "two bias-free MLP weight matrices sharded over TP",
        },
    }
    if terminal_loss:
        profile["terminal_loss_event_ids"] = [
            event.id for event in terminal_loss
        ]
        profile["terminal_loss"] = {
            **terminal_loss_declaration,
            "measurement_transfer": "exact-target-tp-profile",
        }
        profile["terminal_loss_backward_event_ids"] = [
            event.id for event in terminal_loss_backward
        ]
        scheduler_domain = {
            key: _H100_PP_BACKWARD_SCHEDULER_EVIDENCE[key]
            for key in runtime_domain
        }
        if runtime_domain == scheduler_domain:
            profile["pp_backward_scheduler"] = (
                _H100_PP_BACKWARD_SCHEDULER_EVIDENCE
            )
    local_parameter_shapes = [
        [ffn_hidden // tp, hidden],
        [hidden, ffn_hidden // tp],
    ]
    forward_sync_domain = {
        key: _H100_DDP_FORWARD_SYNC_EVIDENCE[key] for key in runtime_domain
    }
    if runtime_domain == forward_sync_domain:
        profile["ddp_forward_sync"] = _H100_DDP_FORWARD_SYNC_EVIDENCE
    evidence_domain = {
        key: _H100_DDP_NORMALIZATION_EVIDENCE[key]
        for key in runtime_domain
    }
    if (
        runtime_domain == evidence_domain
        and local_parameter_shapes
        == [_H100_DDP_NORMALIZATION_EVIDENCE["local_parameter_shape"]] * 2
    ):
        profile["gradient_sync"]["normalization_compute"] = {
            **_H100_DDP_NORMALIZATION_EVIDENCE,
            "local_parameter_shapes": local_parameter_shapes,
        }
    contract = {
        "schema": CONTRACT_SCHEMA,
        "framework": "megatron-core",
        "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
        "adapter": {
            "schema": "megatron-core-mlp-adapter-v1",
            "measurement_schema": MEASUREMENT_SCHEMA,
            "source_workload": str(workload_path.resolve()),
            "source_workload_sha256": _sha256(workload_path),
            "source_semantic_binding_schema": SOURCE_SEMANTIC_BINDING_SCHEMA,
            "source_semantic_sha256": source_semantic_sha256,
            "provenance": str(provenance_path.resolve()),
            "provenance_sha256": _sha256(provenance_path),
            "representative_rank": representative_rank,
            "scope": "column-parallel GELU row-parallel dense MLP",
            "rank_symmetry": rank_symmetry,
        },
        "applicability": {
            "schema": "megatron-core-mlp-applicability-v1",
            "status": "certified-for-bounded-rewrite",
            "source_parallelism": {"tp": tp, "pp": 1, "dp": 1, "ep": 1},
            "fixed_dimensions": {
                "framework_commit": MEGATRON_CORE_SCHEDULE_COMMIT,
                "target": target,
                "tp": tp,
                "model_family": "bias-free-column-gelu-row-mlp",
                "hidden_size": hidden,
                "ffn_hidden_size": ffn_hidden,
                "sequence_length": sequence,
                "micro_batch_size": micro_batch,
            },
            "rewritable_dimensions": ["pp", "dp"],
            "required_conditions": [
                "all captured device events belong to declared phases",
                "uniform logical layers",
                "num_layers divisible by target pp",
                "global batch divisible by micro_batch_size*target_dp",
                "gpipe or pinned non-interleaved 1f1b schedule",
                "target TP has an exact measured profile",
                "all measured TP lanes have the certified phase structure",
                "expert parallelism remains one",
                (
                    "target DP greater than one requires an exact calibrated "
                    "gradient-normalization compute rule"
                ),
            ],
            "timing_scope": (
                "representative-rank measured primitive durations only; structural "
                "symmetry is certified but timing symmetry is not"
            ),
        },
        "model": {
            "num_layers": num_layers,
            "hidden_size": hidden,
            "ffn_hidden_size": ffn_hidden,
            "sequence_length": sequence,
            "micro_batch_size": micro_batch,
            "global_batch_size": global_batch,
        },
        "schedule": {"type": schedule},
        "tp_profiles": {str(tp): profile},
    }
    metadata = dict(trace.metadata)
    metadata["framework_counterfactual"] = contract
    metadata["framework_counterfactual_preparation"] = {
        "schema": "megatron-core-counterfactual-preparation-v1",
        "status": "ready",
        "target": target,
        "tp": tp,
        "representative_rank": representative_rank,
        "phase_event_counts": {
            "forward": len(forward),
            "backward": len(backward),
            "optimizer": len(optimizer),
        },
        "rank_symmetry": rank_symmetry,
        "applicability": contract["applicability"],
        "activation_bytes_per_tp_rank": activation_bytes,
        "gradient_bytes_per_layer_per_rank": gradient_bytes_per_layer,
    }
    result = replace(trace, metadata=metadata)
    result.validate()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bind a captured pinned Megatron MLP step to a semantic contract"
    )
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--megatron-provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--representative-rank", type=int, default=0)
    parser.add_argument("--schedule", choices=("gpipe", "1f1b"), default="1f1b")
    parser.add_argument(
        "--gradient-collective",
        choices=("all_reduce", "reduce_scatter"),
        default="all_reduce",
    )
    args = parser.parse_args(argv)
    if args.representative_rank < 0:
        raise MegatronAdapterError("representative rank must be non-negative")
    trace = WorkloadTrace.load(args.workload)
    result = prepare_megatron_counterfactual(
        trace,
        workload_path=args.workload,
        provenance_path=args.megatron_provenance,
        representative_rank=args.representative_rank,
        schedule=args.schedule,
        gradient_collective=args.gradient_collective,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.dump(args.output)
    preparation = result.metadata["framework_counterfactual_preparation"]
    print(f"output={args.output}")
    print(
        "megatron_counterfactual=ready "
        f"tp={preparation['tp']} phases={preparation['phase_event_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
