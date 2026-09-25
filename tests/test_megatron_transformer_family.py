from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scaletether.megatron_transformer_family import (
    TransformerFamilyAdmissionError,
    admit_generation,
    validate_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads(
    (
        ROOT / "tests/fixtures/level2-h100-transformer-family-v1.json"
    ).read_text()
)
V15_MATRIX = json.loads(
    (
        ROOT / "tests/fixtures/level2-h100-transformer-family-v15.json"
    ).read_text()
)


def measurement(sequence: int = 256) -> dict:
    components = [
        f"layers.{layer}{'.' if path else ''}{path}"
        for layer in range(2)
        for path in (
            "self_attention.linear_qkv",
            "self_attention.linear_proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        )
    ] + ["final_layernorm"]
    return {
        "schema": "megatron-core-transformer-block-measurement-v1",
        "framework": "megatron-core",
        "framework_commit": "f8e1ac64b0587ff7002a18fbaa5ecdeaeb8491be",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "model": {
            "kind": "dense-causal-transformer-block",
            "num_layers": 2,
            "hidden_size": 256,
            "ffn_hidden_size": 512,
            "attention_heads": 8,
            "sequence_length": sequence,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "parameter_dtype": "float32",
            "activation_dtype": "float32",
            "attention_mask": "causal",
            "dropout": 0.0,
        },
        "layer": {
            "kind": "megatron-core-local-dense-transformer-layer-v1",
            "self_attention": True,
            "dense_mlp": True,
            "sequence_parallel": False,
            "transformer_engine": False,
        },
        "parameter_manifest": {
            "schema": "ordered-megatron-parameter-manifest-v1",
            "ordering": "module.named_parameters-before-ddp-wrap",
            "parameter_count": 1,
            "total_numel": 8,
            "parameters": [
                {
                    "order": 0,
                    "name": "layers.0.self_attention.linear_qkv.weight",
                    "shape": [2, 4],
                    "numel": 8,
                    "dtype": "float32",
                    "requires_grad": True,
                }
            ],
        },
        "component_markers": {
            "schema": "megatron-transformer-component-markers-v1",
            "phases": ["forward", "backward"],
            "components": components,
        },
        "gradient_readiness": {
            "schema": "megatron-parameter-gradient-readiness-v1",
            "observation": "unmeasured-complete-warmup-step",
            "parameter_names": ["layers.0.self_attention.linear_qkv.weight"],
            "parameter_count": 1,
            "measured_step_repeatability_required": True,
        },
    }


def test_all_frozen_targets_are_eligible_from_matching_shape_anchor() -> None:
    targets = validate_matrix(MATRIX)
    for identifier, target in targets.items():
        report = admit_generation(
            MATRIX, measurement(target.sequence_length), identifier, target_exists=False
        )
        assert report["admitted"] is True
        assert report["target_training_executed"] is False
        assert report["graph_status"].startswith("unresolved")


def test_study_v5_preregistered_freeze_status_is_accepted() -> None:
    matrix = copy.deepcopy(MATRIX)
    matrix["status"] = "frozen-before-study-v5-anchor-or-heldout-execution"
    assert set(validate_matrix(matrix)) == {
        "TP-A",
        "TP-B",
        "DP-A",
        "DP-B",
        "PP-A",
        "PP-B",
    }


def test_study_v6_preregistered_freeze_status_is_accepted() -> None:
    matrix = copy.deepcopy(MATRIX)
    matrix["status"] = "frozen-before-study-v6-anchor-or-heldout-execution"
    assert set(validate_matrix(matrix)) == {
        "TP-A",
        "TP-B",
        "DP-A",
        "DP-B",
        "PP-A",
        "PP-B",
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["model"].update(global_batch_size=2), "global_batch_size"),
        (
            lambda value: value["model"].update(parameter_dtype="float16"),
            "parameter_dtype",
        ),
        (
            lambda value: value["layer"].update(transformer_engine=True),
            "transformer_engine",
        ),
        (lambda value: value["layer"].update(self_attention=False), "self_attention"),
        (lambda value: value.update(framework_commit="wrong"), "framework_commit"),
        (
            lambda value: value["parameter_manifest"]["parameters"][0].update(numel=7),
            "numel",
        ),
        (lambda value: value.pop("component_markers"), "component marker"),
        (lambda value: value.pop("gradient_readiness"), "gradient readiness"),
    ],
)
def test_source_family_mismatch_abstains(mutation, match: str) -> None:
    source = measurement()
    mutation(source)
    with pytest.raises(TransformerFamilyAdmissionError, match=match):
        admit_generation(MATRIX, source, "TP-A", target_exists=False)


def test_target_leakage_and_unregistered_target_abstain() -> None:
    with pytest.raises(TransformerFamilyAdmissionError, match="already exists"):
        admit_generation(MATRIX, measurement(), "TP-A", target_exists=True)
    with pytest.raises(TransformerFamilyAdmissionError, match="not preregistered"):
        admit_generation(MATRIX, measurement(), "TP-C", target_exists=False)


def test_tampered_matrix_abstains() -> None:
    matrix = copy.deepcopy(MATRIX)
    matrix["targets"][0]["global_batch_size"] = 4
    with pytest.raises(TransformerFamilyAdmissionError, match="global-batch"):
        validate_matrix(matrix)


def test_stronger_pre_anchor_frozen_status_is_accepted() -> None:
    matrix = copy.deepcopy(MATRIX)
    matrix["status"] = "frozen-before-study-v3-anchor-or-heldout-execution"
    assert set(validate_matrix(matrix)) == {
        "TP-A",
        "TP-B",
        "DP-A",
        "DP-B",
        "PP-A",
        "PP-B",
    }


def test_unknown_status_is_not_treated_as_frozen() -> None:
    matrix = copy.deepcopy(MATRIX)
    matrix["status"] = "draft"
    with pytest.raises(TransformerFamilyAdmissionError, match="not frozen"):
        validate_matrix(matrix)


def test_v15_versioned_domain_is_admitted_without_weakening_legacy_domain() -> None:
    targets = validate_matrix(V15_MATRIX)
    for identifier, target in targets.items():
        source = measurement(target.sequence_length)
        source["model"].update(
            hidden_size=1024,
            ffn_hidden_size=4096,
            attention_heads=16,
        )
        assert admit_generation(V15_MATRIX, source, identifier, target_exists=False)[
            "admitted"
        ]

    changed = copy.deepcopy(V15_MATRIX)
    changed["domain"]["hidden_size"] = 2048
    with pytest.raises(TransformerFamilyAdmissionError, match="domain"):
        validate_matrix(changed)
