"""Explicit framework-rule plans for the frozen Megatron Transformer family."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .megatron_transformer_family import COMMIT as PINNED_MEGATRON_COMMIT
from .megatron_transformer_family import admit_generation


SCHEMA = "scaletether-megatron-transformer-rule-plan-v1"
ELEMENT_BYTES = 4
DDP_FIRST_BUCKET_CAP_BYTES = 1 * 1024 * 1024
DDP_BUCKET_CAP_BYTES = 25 * 1024 * 1024
PINNED_TORCH_VERSION = "2.9.1+cu128"
TP_RULE_SCHEMA = "megatron-local-dense-transformer-tp-width-v1"
TP_MEASUREMENT_SCHEMA = "megatron-core-transformer-block-measurement-v1"


class TransformerRuleError(ValueError):
    """The admitted target cannot be resolved by the explicit frozen rule."""

    def __init__(self, message: str, *, code: str = "transformer-rule-rejected") -> None:
        super().__init__(message)
        self.code = code


def _manifest(measurement: dict[str, Any]) -> dict[str, Any]:
    manifest = measurement.get("parameter_manifest")
    if not isinstance(manifest, dict):
        raise TransformerRuleError("ordered parameter manifest is missing")
    return manifest


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransformerRuleError(f"{label} must be a positive integer")
    return value


def validate_megatron_tp_applicability(measurement: dict[str, Any]) -> None:
    """Bind the reusable TP rule to its validated Megatron implementation.

    Shape compatibility alone is insufficient: a future Megatron revision may
    move collectives or change parameter layouts while retaining the same model
    dimensions.  Rejecting unknown revisions keeps the framework semantics in
    this module from silently becoming an unvalidated guess.
    """

    expected = {
        "schema": TP_MEASUREMENT_SCHEMA,
        "framework": "megatron-core",
        "framework_commit": PINNED_MEGATRON_COMMIT,
    }
    for key, value in expected.items():
        if measurement.get(key) != value:
            raise TransformerRuleError(
                f"TP rewrite requires {key}={value!r}; "
                f"observed {measurement.get(key)!r}"
            )


@dataclass(frozen=True)
class MegatronTpSpec:
    """All structural inputs to a dense Megatron TP width rewrite.

    Every field is either observed in the source framework measurement or is
    the explicitly requested target width.  In particular, this object never
    contains a target trace, target duration, or fitted end-to-end result.
    """

    source_tp: int
    target_tp: int
    hidden_size: int
    ffn_hidden_size: int
    attention_heads: int
    num_layers: int
    sequence_length: int
    micro_batch_size: int
    element_bytes: int


def _tp_spec(
    measurement: dict[str, Any],
    *,
    target_tp: int,
    sequence_length: int | None = None,
) -> MegatronTpSpec:
    validate_megatron_tp_applicability(measurement)
    model = measurement.get("model")
    if not isinstance(model, dict):
        raise TransformerRuleError("framework measurement lacks model dimensions")
    layer = measurement.get("layer")
    if (
        model.get("kind") != "dense-causal-transformer-block"
        or not isinstance(layer, dict)
        or layer.get("kind") != "megatron-core-local-dense-transformer-layer-v1"
        or layer.get("self_attention") is not True
        or layer.get("dense_mlp") is not True
        or layer.get("sequence_parallel") is not False
        or layer.get("transformer_engine") is not False
    ):
        raise TransformerRuleError(
            "TP rewrite supports only the declared local dense Transformer "
            "without sequence parallelism or Transformer Engine"
        )
    source_tp = _positive_integer(
        measurement.get("tensor_parallel_size"), "source tensor parallel size"
    )
    target_tp = _positive_integer(target_tp, "target tensor parallel size")
    if measurement.get("pipeline_parallel_size") != 1:
        raise TransformerRuleError("TP rewrite requires source PP=1")
    if measurement.get("data_parallel_size") != 1:
        raise TransformerRuleError("TP rewrite requires source DP=1")
    hidden = _positive_integer(model.get("hidden_size"), "hidden size")
    ffn = _positive_integer(model.get("ffn_hidden_size"), "FFN hidden size")
    heads = _positive_integer(model.get("attention_heads"), "attention heads")
    layers = _positive_integer(model.get("num_layers"), "number of layers")
    measured_sequence = _positive_integer(
        model.get("sequence_length"), "sequence length"
    )
    sequence = (
        measured_sequence
        if sequence_length is None
        else _positive_integer(sequence_length, "sequence length")
    )
    if sequence != measured_sequence:
        raise TransformerRuleError(
            "requested sequence length differs from the signed source measurement"
        )
    micro_batch = _positive_integer(model.get("micro_batch_size"), "micro batch size")
    if (
        model.get("parameter_dtype") != "float32"
        or model.get("activation_dtype") != "float32"
    ):
        raise TransformerRuleError("TP rewrite currently requires float32 tensors")
    # This rule is the ordinary MHA layout.  Grouped-query attention changes
    # the QKV partition formula and must have its own explicit rule.
    query_groups = model.get("num_query_groups", heads)
    if query_groups != heads:
        raise TransformerRuleError("grouped-query attention is not supported")
    if hidden % heads:
        raise TransformerRuleError("hidden size must be divisible by attention heads")
    for width, label in ((source_tp, "source"), (target_tp, "target")):
        if heads % width or ffn % width:
            raise TransformerRuleError(
                f"{label} TP={width} does not evenly partition attention heads "
                "and FFN hidden size",
                code="tp-divisibility",
            )
    return MegatronTpSpec(
        source_tp=source_tp,
        target_tp=target_tp,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        attention_heads=heads,
        num_layers=layers,
        sequence_length=sequence,
        micro_batch_size=micro_batch,
        element_bytes=ELEMENT_BYTES,
    )


def derive_megatron_tp_rule(
    measurement: dict[str, Any],
    *,
    target_tp: int,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Derive a TP structural rule without consulting target execution data."""

    spec = _tp_spec(measurement, target_tp=target_tp, sequence_length=sequence_length)
    hidden = spec.hidden_size
    ffn = spec.ffn_hidden_size
    source_tp = spec.source_tp
    target_tp = spec.target_tp
    activation_bytes = (
        spec.sequence_length * spec.micro_batch_size * hidden * spec.element_bytes
    )
    return {
        "schema": TP_RULE_SCHEMA,
        "rule": f"megatron-local-transformer-tp{source_tp}-to-tp{target_tp}-v1",
        "source_tp": source_tp,
        "target_tp": target_tp,
        "source_tp_group": list(range(source_tp)),
        "tp_groups": [list(range(target_tp))],
        "model": {
            "hidden_size": hidden,
            "ffn_hidden_size": ffn,
            "attention_heads": spec.attention_heads,
            "num_layers": spec.num_layers,
            "sequence_length": spec.sequence_length,
            "micro_batch_size": spec.micro_batch_size,
            "element_bytes": spec.element_bytes,
        },
        "local_attention_heads": {
            "source": spec.attention_heads // source_tp,
            "target": spec.attention_heads // target_tp,
        },
        "local_tensor_shapes": {
            "attention_qkv_weight": {
                "source": [3 * hidden // source_tp, hidden],
                "target": [3 * hidden // target_tp, hidden],
            },
            "attention_qkv_bias": {
                "source": [3 * hidden // source_tp],
                "target": [3 * hidden // target_tp],
            },
            "attention_projection_weight": {
                "source": [hidden, hidden // source_tp],
                "target": [hidden, hidden // target_tp],
            },
            "mlp_fc1_weight": {
                "source": [ffn // source_tp, hidden],
                "target": [ffn // target_tp, hidden],
            },
            "mlp_fc1_bias": {
                "source": [ffn // source_tp],
                "target": [ffn // target_tp],
            },
            "mlp_fc2_weight": {
                "source": [hidden, ffn // source_tp],
                "target": [hidden, ffn // target_tp],
            },
        },
        "replicated_parameter_shapes": {
            "layernorm_weight": [hidden],
            "attention_projection_bias": [hidden],
            "mlp_fc2_bias": [hidden],
        },
        "collective_rule": {
            "operation": "all_reduce",
            "payload_bytes": activation_bytes,
            "per_layer": [
                "attention-projection-forward",
                "mlp-fc2-forward",
                "mlp-fc1-input-gradient",
                "attention-qkv-input-gradient",
            ],
            "layers": spec.num_layers,
            "per_layer_operation_count": 4,
            "per_rank_operation_count": 4 * spec.num_layers,
        },
        "provenance": {
            "source": "signed-framework-measurement-and-pinned-megatron-rule",
            "target_runtime_observation": False,
            "target_timing_observation": False,
        },
    }


def derive_megatron_tp_parameter_manifest(
    measurement: dict[str, Any], *, target_tp: int
) -> dict[str, Any]:
    """Materialize target-local parameter shapes from a signed source manifest."""

    rule = derive_megatron_tp_rule(measurement, target_tp=target_tp)
    source = measurement.get("parameter_manifest")
    parameters = source.get("parameters") if isinstance(source, dict) else None
    if (
        not isinstance(source, dict)
        or source.get("schema") != "ordered-megatron-parameter-manifest-v1"
        or not isinstance(parameters, list)
        or not parameters
    ):
        raise TransformerRuleError("signed source parameter manifest is incomplete")
    shapes = rule["local_tensor_shapes"]
    hidden = rule["model"]["hidden_size"]

    def contract(name: str) -> tuple[list[int], list[int], str]:
        suffixes = (
            ("self_attention.linear_qkv.weight", "attention_qkv_weight"),
            ("self_attention.linear_qkv.bias", "attention_qkv_bias"),
            ("self_attention.linear_proj.weight", "attention_projection_weight"),
            ("mlp.linear_fc1.weight", "mlp_fc1_weight"),
            ("mlp.linear_fc1.bias", "mlp_fc1_bias"),
            ("mlp.linear_fc2.weight", "mlp_fc2_weight"),
        )
        for suffix, key in suffixes:
            if name.endswith(suffix):
                return shapes[key]["source"], shapes[key]["target"], key
        if name.endswith(("self_attention.linear_proj.bias", "mlp.linear_fc2.bias")):
            return [hidden], [hidden], "replicated-bias"
        if name.endswith(
            (
                "layernorm.weight",
                "layernorm.bias",
                "layer_norm_weight",
                "layer_norm_bias",
                "final_layernorm.weight",
                "final_layernorm.bias",
            )
        ):
            return [hidden], [hidden], "replicated-layernorm-affine"
        raise TransformerRuleError(
            f"parameter {name!r} is outside the dense Megatron TP shape rule"
        )

    target_parameters: list[dict[str, Any]] = []
    observed_names: set[str] = set()
    observed_total_numel = 0
    for order, row in enumerate(parameters):
        if not isinstance(row, dict):
            raise TransformerRuleError("source parameter manifest row is malformed")
        name = row.get("name")
        observed_shape = row.get("shape")
        if (
            row.get("order") != order
            or not isinstance(name, str)
            or name in observed_names
            or not isinstance(observed_shape, list)
            or row.get("dtype") != "float32"
            or row.get("requires_grad") is not True
        ):
            raise TransformerRuleError("source parameter manifest row is malformed")
        expected_source, target_shape, shape_rule = contract(name)
        if observed_shape != expected_source:
            raise TransformerRuleError(
                f"source parameter {name!r} shape differs from its TP rule"
            )
        observed_numel = 1
        for dimension in observed_shape:
            observed_numel *= dimension
        if row.get("numel") != observed_numel:
            raise TransformerRuleError(
                f"source parameter {name!r} numel differs from its shape"
            )
        observed_names.add(name)
        observed_total_numel += observed_numel
        target_numel = 1
        for dimension in target_shape:
            target_numel *= dimension
        target_parameters.append(
            {
                "order": order,
                "name": name,
                "shape": target_shape,
                "numel": target_numel,
                "dtype": "float32",
                "requires_grad": True,
                "tp_shape_rule": shape_rule,
            }
        )
    if (
        source.get("parameter_count") != len(parameters)
        or source.get("total_numel") != observed_total_numel
    ):
        raise TransformerRuleError("source parameter manifest totals are inconsistent")
    target_manifest = {
        "schema": "derived-megatron-tp-parameter-manifest-v1",
        "ordering": source.get("ordering"),
        "source_tp": rule["source_tp"],
        "target_tp": rule["target_tp"],
        "source_manifest_sha256": _canonical_sha256(source),
        "parameter_count": len(target_parameters),
        "total_numel": sum(row["numel"] for row in target_parameters),
        "parameters": target_parameters,
        "target_runtime_observation": False,
    }
    return {
        **target_manifest,
        "canonical_sha256": _canonical_sha256(target_manifest),
    }


def compare_megatron_tp_parameter_manifest(
    expected: dict[str, Any], physical_measurement: dict[str, Any]
) -> dict[str, Any]:
    """Compare a frozen derived manifest with a subsequently opened target."""

    actual = physical_measurement.get("parameter_manifest")
    rows = actual.get("parameters") if isinstance(actual, dict) else None
    expected_rows = expected.get("parameters")
    if not isinstance(rows, list) or not isinstance(expected_rows, list):
        raise TransformerRuleError("physical target parameter manifest is incomplete")
    normalized_actual = [
        {
            key: row.get(key)
            for key in ("order", "name", "shape", "numel", "dtype", "requires_grad")
        }
        for row in rows
        if isinstance(row, dict)
    ]
    normalized_expected = [
        {
            key: row.get(key)
            for key in ("order", "name", "shape", "numel", "dtype", "requires_grad")
        }
        for row in expected_rows
        if isinstance(row, dict)
    ]
    failures: list[str] = []
    if physical_measurement.get("tensor_parallel_size") != expected.get("target_tp"):
        failures.append("target-tp")
    if normalized_actual != normalized_expected:
        failures.append("parameter-shapes-or-order")
    if actual.get("parameter_count") != expected.get("parameter_count"):
        failures.append("parameter-count")
    if actual.get("total_numel") != expected.get("total_numel"):
        failures.append("parameter-numel")
    return {
        "schema": "scaletether-megatron-tp-parameter-manifest-comparison-v1",
        "passed": not failures,
        "failures": failures,
        "target_tp": expected.get("target_tp"),
        "expected_manifest_sha256": expected.get("canonical_sha256"),
        "claim": (
            "target-local-parameter-shapes-match-derived-framework-rule"
            if not failures
            else "no-target-parameter-shape-validity-claim"
        ),
    }


def _tp_plan(
    sequence: int, measurement: dict[str, Any], *, target_tp: int
) -> dict[str, Any]:
    """Compatibility-shaped plan backed by the generic TP-width rule."""

    generic = derive_megatron_tp_rule(
        measurement, target_tp=target_tp, sequence_length=sequence
    )
    # Preserve the byte-for-byte legacy rule shape used by sealed studies.
    return {
        "rule": generic["rule"],
        "tp_groups": generic["tp_groups"],
        "local_attention_heads": generic["local_attention_heads"],
        "local_tensor_shapes": generic["local_tensor_shapes"],
        "replicated_parameter_shapes": generic["replicated_parameter_shapes"],
        "collective_rule": {
            key: value
            for key, value in generic["collective_rule"].items()
            if key != "per_layer_operation_count"
        },
    }


def _model_dimensions(measurement: dict[str, Any]) -> tuple[int, int, int]:
    """Legacy frozen-family helper retained for DP/PP rule compatibility."""

    model = measurement.get("model")
    if not isinstance(model, dict):
        raise TransformerRuleError("framework measurement lacks model dimensions")
    values = tuple(
        model.get(key) for key in ("hidden_size", "ffn_hidden_size", "attention_heads")
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise TransformerRuleError("model dimensions must be positive integers")
    hidden, ffn, heads = values
    if hidden % 4 or ffn % 4 or heads % 4:
        raise TransformerRuleError("TP4 requires model dimensions divisible by four")
    return hidden, ffn, heads


def derive_pinned_ddp_contract(measurement: dict[str, Any]) -> dict[str, Any]:
    """Derive the exact bounded PyTorch DDP communication contract.

    The frozen family uses PyTorch 2.9.1's reducer after an unmeasured warmup
    step.  Parameters are accumulated in observed gradient-readiness order;
    the parameter that crosses a cap remains in that bucket.  This helper is
    shared by generation and physical admission so neither side can silently
    use a different byte or bucket model.
    """

    manifest = _manifest(measurement)
    parameters = manifest.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        raise TransformerRuleError("parameter manifest has no ordered parameters")
    by_name: dict[str, dict[str, Any]] = {}
    total_numel = 0
    for order, parameter in enumerate(parameters):
        if not isinstance(parameter, dict):
            raise TransformerRuleError("parameter manifest entry is malformed")
        name = parameter.get("name")
        numel = parameter.get("numel")
        if (
            not isinstance(name, str)
            or not name
            or name in by_name
            or parameter.get("order") != order
            or isinstance(numel, bool)
            or not isinstance(numel, int)
            or numel <= 0
            or parameter.get("dtype") != "float32"
            or parameter.get("requires_grad") is not True
        ):
            raise TransformerRuleError("parameter manifest entry violates DDP contract")
        by_name[name] = parameter
        total_numel += numel
    if (
        manifest.get("parameter_count") != len(parameters)
        or manifest.get("total_numel") != total_numel
    ):
        raise TransformerRuleError("parameter manifest totals are inconsistent")
    readiness = measurement.get("gradient_readiness")
    if not isinstance(readiness, dict) or not isinstance(
        readiness.get("parameter_names"), list
    ):
        raise TransformerRuleError("gradient readiness order is missing")
    readiness_names = readiness["parameter_names"]
    if (
        not readiness_names
        or len(readiness_names) != len(set(readiness_names))
        or set(readiness_names) != set(by_name)
        or readiness.get("parameter_count") != len(parameters)
    ):
        raise TransformerRuleError(
            "gradient readiness is not an exact permutation of the parameter manifest"
        )

    buckets: list[dict[str, Any]] = []
    names: list[str] = []
    payload_bytes = 0
    cap = DDP_FIRST_BUCKET_CAP_BYTES
    for name in readiness_names:
        names.append(name)
        payload_bytes += by_name[name]["numel"] * ELEMENT_BYTES
        if payload_bytes >= cap:
            buckets.append(
                {
                    "index": len(buckets),
                    "cap_bytes": cap,
                    "payload_bytes": payload_bytes,
                    "parameter_names": list(names),
                    "parameter_names_sha256": _canonical_sha256(names),
                    "last_ready_parameter": name,
                    "operation": "all_reduce",
                }
            )
            names = []
            payload_bytes = 0
            cap = DDP_BUCKET_CAP_BYTES
    if names:
        buckets.append(
            {
                "index": len(buckets),
                "cap_bytes": cap,
                "payload_bytes": payload_bytes,
                "parameter_names": list(names),
                "parameter_names_sha256": _canonical_sha256(names),
                "last_ready_parameter": names[-1],
                "operation": "all_reduce",
            }
        )
    gradient_bytes = total_numel * ELEMENT_BYTES
    if (
        not buckets
        or sum(bucket["payload_bytes"] for bucket in buckets) != gradient_bytes
    ):
        raise TransformerRuleError(
            "derived DDP buckets do not cover the manifest exactly"
        )

    broadcasts = [
        {
            "index": 0,
            "purpose": "parameter-index-metadata",
            "payload_bytes": (len(parameters) + 1) * ELEMENT_BYTES,
            "operation": "broadcast",
        },
        {
            "index": 1,
            "purpose": "bucket-assignment-metadata",
            "payload_bytes": len(buckets) * ELEMENT_BYTES,
            "operation": "broadcast",
        },
    ]
    return {
        "schema": "torch-ddp-warm-reducer-communication-contract-v1",
        "torch_version": PINNED_TORCH_VERSION,
        "first_bucket_cap_bytes": DDP_FIRST_BUCKET_CAP_BYTES,
        "subsequent_bucket_cap_bytes": DDP_BUCKET_CAP_BYTES,
        "parameter_order_sha256": _canonical_sha256(parameters),
        "gradient_readiness_order_sha256": _canonical_sha256(readiness_names),
        "parameter_count": len(parameters),
        "gradient_bytes": gradient_bytes,
        "forward_metadata_broadcasts": broadcasts,
        "gradient_buckets": buckets,
        "per_rank_operation_count": len(broadcasts) + len(buckets),
    }


def _dp_plan(measurement: dict[str, Any]) -> dict[str, Any]:
    contract = derive_pinned_ddp_contract(measurement)
    return {
        "rule": "torch-ddp-tp2-dp1-to-dp2-warm-reducer-v1",
        "tp_groups": [[0, 1], [2, 3]],
        "dp_groups": [[0, 2], [1, 3]],
        "replica_mapping": {"0": [0, 2], "1": [1, 3]},
        "ddp_contract": contract,
    }


def _pp_plan(
    sequence: int, microbatches: int, measurement: dict[str, Any]
) -> dict[str, Any]:
    hidden, _ffn, _heads = _model_dimensions(measurement)
    activation_bytes = sequence * hidden * ELEMENT_BYTES
    lane_templates = []
    for lane, (first, second) in enumerate(((0, 2), (1, 3))):
        operations = []
        for microbatch in range(microbatches):
            operations.extend(
                [
                    {
                        "phase": "forward",
                        "microbatch": microbatch,
                        "event_rank": first,
                        "source_rank": first,
                        "destination_rank": second,
                        "operation": "send",
                    },
                    {
                        "phase": "forward",
                        "microbatch": microbatch,
                        "event_rank": second,
                        "source_rank": first,
                        "destination_rank": second,
                        "operation": "recv",
                    },
                    {
                        "phase": "backward",
                        "microbatch": microbatch,
                        "event_rank": second,
                        "source_rank": second,
                        "destination_rank": first,
                        "operation": "send",
                    },
                    {
                        "phase": "backward",
                        "microbatch": microbatch,
                        "event_rank": first,
                        "source_rank": second,
                        "destination_rank": first,
                        "operation": "recv",
                    },
                ]
            )
        lane_templates.append(
            {"tp_lane": lane, "global_ranks": [first, second], "operations": operations}
        )
    return {
        "rule": "megatron-noninterleaved-1f1b-pp1-to-pp2-v1",
        "tp_groups": [[0, 1], [2, 3]],
        "pp_groups": [[0, 2], [1, 3]],
        "layer_placement": {"stage_0": [0], "stage_1": [1]},
        "terminal_stage": 1,
        "microbatches": microbatches,
        "p2p": {
            "payload_bytes": activation_bytes,
            "tp_lane_templates": lane_templates,
            "per_rank_operation_count": 2 * microbatches,
            "global_operation_count": 8 * microbatches,
        },
        "tp_collectives": {
            "operation": "all_reduce",
            "payload_bytes": activation_bytes,
            "per_rank_operation_count": 4 * microbatches,
        },
    }


def build_rule_plan(
    matrix: dict[str, Any],
    measurement: dict[str, Any],
    target_id: str,
    *,
    target_exists: bool,
) -> dict[str, Any]:
    admission = admit_generation(
        matrix, measurement, target_id, target_exists=target_exists
    )
    dimension = admission["transformation_dimension"]
    if dimension == "tp":
        rule = _tp_plan(
            admission["sequence_length"],
            measurement,
            target_tp=admission["target_parallelism"]["tp"],
        )
    elif dimension == "dp":
        rule = _dp_plan(measurement)
    elif dimension == "pp":
        rule = _pp_plan(
            admission["sequence_length"],
            admission["pipeline_microbatches"],
            measurement,
        )
    else:  # guarded by family admission; retain fail-closed behavior here.
        raise TransformerRuleError("unsupported transformation dimension")
    return {
        "schema": SCHEMA,
        "target_id": target_id,
        "admission": admission,
        "framework_rule": rule,
        "framework_rule_sha256": _canonical_sha256(rule),
        "provenance": {
            "source": "framework-observed-anchor-measurement",
            "transformation": "explicit-version-pinned-framework-rule",
            "target_runtime_observation": False,
        },
        "graph_status": "unresolved-until-event-graph-compiler-and-validator-pass",
        "timing_status": "unresolved-until-structural-admission-and-calibration-pass",
        "claim": "framework-rule-plan-only-no-generated-workload-claim",
    }
