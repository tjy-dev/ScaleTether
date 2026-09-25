"""Fail-closed admission for the preregistered Megatron Transformer family."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


SCHEMA = "scaletether-level2-h100-transformer-family-v1"
CURRENT_SCHEMA = "scaletether-level2-h100-transformer-family-v7"
COMMIT = "f8e1ac64b0587ff7002a18fbaa5ecdeaeb8491be"
LEGACY_FROZEN_STATUSES = {
    "frozen-before-heldout-transformer-target-execution",
    "frozen-before-study-v3-anchor-or-heldout-execution",
    "frozen-before-study-v4-anchor-or-heldout-execution",
    "frozen-before-study-v5-anchor-or-heldout-execution",
    "frozen-before-study-v6-anchor-or-heldout-execution",
    "frozen-before-study-v7-anchor-or-heldout-execution",
}
CURRENT_FROZEN_STATUSES = {
    "scaletether-level2-h100-transformer-family-v6": {
        "frozen-before-v12-calibration",
        "frozen-before-v13-calibration",
        "frozen-before-v14-five-launch-anchor-acquisition",
    },
    CURRENT_SCHEMA: {"frozen-before-v15-allocation-conditioned-heldouts"},
}
LEGACY_DOMAIN = {
    "attention_heads": 8,
    "attention_mask": "causal",
    "dropout": 0.0,
    "ffn_hidden_size": 512,
    "framework": "megatron-core",
    "framework_commit": COMMIT,
    "hidden_size": 256,
    "num_layers": 2,
    "parameter_dtype": "float32",
    "target": "h100",
    "transformer_engine": False,
}
CURRENT_DOMAIN = {
    "attention_heads": 16,
    "attention_mask": "causal",
    "dropout": 0.0,
    "ffn_hidden_size": 4096,
    "framework": "megatron-core",
    "framework_commit": COMMIT,
    "hidden_size": 1024,
    "micro_batch_size": 1,
    "num_layers": 2,
    "parameter_dtype": "float32",
    "target": "h100",
    "transformer_engine": False,
}


class TransformerFamilyAdmissionError(ValueError):
    """The source or requested target is outside the frozen family."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TransformerFamilyAdmissionError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransformerFamilyAdmissionError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class TargetSpec:
    identifier: str
    sequence_length: int
    tp: int
    pp: int
    dp: int
    pipeline_microbatches: int
    global_batch_size: int

    @property
    def dimension(self) -> str:
        return self.identifier.split("-", 1)[0].lower()


def validate_matrix(document: dict[str, Any]) -> dict[str, TargetSpec]:
    schema = document.get("schema")
    if schema == SCHEMA:
        statuses = LEGACY_FROZEN_STATUSES
        expected_domain = LEGACY_DOMAIN
    elif schema in CURRENT_FROZEN_STATUSES:
        statuses = CURRENT_FROZEN_STATUSES[schema]
        expected_domain = CURRENT_DOMAIN
    else:
        raise TransformerFamilyAdmissionError("unsupported family schema")
    if document.get("status") not in statuses:
        raise TransformerFamilyAdmissionError("family matrix is not frozen")
    domain = _mapping(document.get("domain"), "domain")
    if domain != expected_domain:
        raise TransformerFamilyAdmissionError(
            "family domain differs from preregistration"
        )
    rows = document.get("targets")
    if not isinstance(rows, list) or len(rows) != 6:
        raise TransformerFamilyAdmissionError(
            "family requires exactly six frozen targets"
        )
    targets: dict[str, TargetSpec] = {}
    for row in rows:
        row = _mapping(row, "target")
        identifier = row.get("id")
        parallel = _mapping(row.get("parallelism"), "target parallelism")
        if not isinstance(identifier, str) or identifier in targets:
            raise TransformerFamilyAdmissionError("target ids must be unique strings")
        spec = TargetSpec(
            identifier=identifier,
            sequence_length=_positive_int(
                row.get("sequence_length"), "sequence length"
            ),
            tp=_positive_int(parallel.get("tp"), "TP"),
            pp=_positive_int(parallel.get("pp"), "PP"),
            dp=_positive_int(parallel.get("dp"), "DP"),
            pipeline_microbatches=_positive_int(
                row.get("pipeline_microbatches"), "pipeline microbatches"
            ),
            global_batch_size=_positive_int(
                row.get("global_batch_size"), "global batch"
            ),
        )
        if spec.global_batch_size != spec.dp * spec.pipeline_microbatches:
            raise TransformerFamilyAdmissionError(
                "target violates global-batch invariant"
            )
        if spec.dimension not in {"tp", "dp", "pp"}:
            raise TransformerFamilyAdmissionError(
                "target has an unknown transformation dimension"
            )
        targets[identifier] = spec
    if {spec.dimension for spec in targets.values()} != {"tp", "dp", "pp"}:
        raise TransformerFamilyAdmissionError("matrix must cover TP, DP, and PP")
    return targets


def validate_anchor_measurement(
    measurement: dict[str, Any],
    sequence_length: int,
    *,
    domain: dict[str, Any] | None = None,
) -> None:
    admitted_domain = LEGACY_DOMAIN if domain is None else domain
    if admitted_domain not in (LEGACY_DOMAIN, CURRENT_DOMAIN):
        raise TransformerFamilyAdmissionError("anchor domain is not admitted")
    expected_top = {
        "schema": "megatron-core-transformer-block-measurement-v1",
        "framework": "megatron-core",
        "framework_commit": COMMIT,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }
    for key, expected in expected_top.items():
        if measurement.get(key) != expected:
            raise TransformerFamilyAdmissionError(f"anchor {key} mismatch")
    model = _mapping(measurement.get("model"), "anchor model")
    expected_model = {
        "kind": "dense-causal-transformer-block",
        "num_layers": admitted_domain["num_layers"],
        "hidden_size": admitted_domain["hidden_size"],
        "ffn_hidden_size": admitted_domain["ffn_hidden_size"],
        "attention_heads": admitted_domain["attention_heads"],
        "sequence_length": sequence_length,
        "micro_batch_size": admitted_domain.get("micro_batch_size", 1),
        "global_batch_size": 1,
        "parameter_dtype": admitted_domain["parameter_dtype"],
        "activation_dtype": admitted_domain["parameter_dtype"],
        "attention_mask": admitted_domain["attention_mask"],
        "dropout": admitted_domain["dropout"],
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise TransformerFamilyAdmissionError(f"anchor model {key} mismatch")
    layer = _mapping(measurement.get("layer"), "anchor layer")
    expected_layer = {
        "kind": "megatron-core-local-dense-transformer-layer-v1",
        "self_attention": True,
        "dense_mlp": True,
        "sequence_parallel": False,
        "transformer_engine": admitted_domain["transformer_engine"],
    }
    for key, expected in expected_layer.items():
        if layer.get(key) != expected:
            raise TransformerFamilyAdmissionError(f"anchor layer {key} mismatch")
    components = [
        f"layers.{layer_index}{'.' if path else ''}{path}"
        for layer_index in range(2)
        for path in (
            "self_attention.linear_qkv",
            "self_attention.linear_proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        )
    ] + ["final_layernorm"]
    if measurement.get("component_markers") != {
        "schema": "megatron-transformer-component-markers-v1",
        "phases": ["forward", "backward"],
        "components": components,
    }:
        raise TransformerFamilyAdmissionError("component marker contract mismatch")
    manifest = _mapping(measurement.get("parameter_manifest"), "parameter manifest")
    parameters = manifest.get("parameters")
    if (
        manifest.get("schema") != "ordered-megatron-parameter-manifest-v1"
        or manifest.get("ordering") != "module.named_parameters-before-ddp-wrap"
        or not isinstance(parameters, list)
        or not parameters
        or manifest.get("parameter_count") != len(parameters)
    ):
        raise TransformerFamilyAdmissionError("parameter manifest contract mismatch")
    total = 0
    names: set[str] = set()
    for order, parameter in enumerate(parameters):
        parameter = _mapping(parameter, "parameter record")
        name = parameter.get("name")
        shape = parameter.get("shape")
        numel = parameter.get("numel")
        if (
            parameter.get("order") != order
            or not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
            or isinstance(numel, bool)
            or not isinstance(numel, int)
            or numel <= 0
            or parameter.get("dtype") != "float32"
            or parameter.get("requires_grad") is not True
        ):
            raise TransformerFamilyAdmissionError("parameter manifest record mismatch")
        product = 1
        for value in shape:
            product *= value
        if product != numel:
            raise TransformerFamilyAdmissionError("parameter manifest numel mismatch")
        names.add(name)
        total += numel
    if manifest.get("total_numel") != total:
        raise TransformerFamilyAdmissionError("parameter manifest total mismatch")
    readiness = _mapping(measurement.get("gradient_readiness"), "gradient readiness")
    readiness_names = readiness.get("parameter_names")
    if (
        readiness.get("schema") != "megatron-parameter-gradient-readiness-v1"
        or readiness.get("observation") != "unmeasured-complete-warmup-step"
        or readiness.get("parameter_count") != len(parameters)
        or not isinstance(readiness_names, list)
        or len(readiness_names) != len(parameters)
        or set(readiness_names) != names
        or readiness.get("measured_step_repeatability_required") is not True
    ):
        raise TransformerFamilyAdmissionError("gradient readiness contract mismatch")


def admit_generation(
    matrix: dict[str, Any],
    measurement: dict[str, Any],
    target_id: str,
    *,
    target_exists: bool,
) -> dict[str, Any]:
    """Admit a target to its compiler; this is not graph-validity evidence."""
    if target_exists:
        raise TransformerFamilyAdmissionError("held-out target already exists")
    targets = validate_matrix(matrix)
    try:
        target = targets[target_id]
    except KeyError as error:
        raise TransformerFamilyAdmissionError("target is not preregistered") from error
    validate_anchor_measurement(
        measurement,
        target.sequence_length,
        domain=_mapping(matrix.get("domain"), "domain"),
    )
    matrix_sha256 = hashlib.sha256(
        json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    parameter_manifest = _mapping(
        measurement["parameter_manifest"], "parameter manifest"
    )
    parameter_manifest_sha256 = hashlib.sha256(
        json.dumps(parameter_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "scaletether-transformer-family-generation-admission-v1",
        "admitted": True,
        "target_id": target.identifier,
        "transformation_dimension": target.dimension,
        "source_parallelism": {"tp": 2, "pp": 1, "dp": 1},
        "target_parallelism": {"tp": target.tp, "pp": target.pp, "dp": target.dp},
        "sequence_length": target.sequence_length,
        "pipeline_microbatches": target.pipeline_microbatches,
        "global_batch_size": target.global_batch_size,
        "matrix_canonical_sha256": matrix_sha256,
        "parameter_manifest_canonical_sha256": parameter_manifest_sha256,
        "target_training_executed": False,
        "graph_status": "unresolved-requires-dimension-compiler-and-structural-gate",
        "timing_status": "unresolved-structural-admission-must-pass-first",
        "claim": "compiler-eligibility-only-no-generated-workload-claim",
    }
