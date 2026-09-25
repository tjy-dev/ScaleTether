from __future__ import annotations

import json
from pathlib import Path

import pytest

from scaletether.megatron_transformer_family import TransformerFamilyAdmissionError
from scaletether.megatron_transformer_rules import build_rule_plan
from tests.test_megatron_transformer_family import V15_MATRIX, measurement


ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads(
    (
        ROOT / "tests/fixtures/level2-h100-transformer-family-v1.json"
    ).read_text()
)


def test_tp_plan_derives_local_shapes_and_collective_bytes() -> None:
    plan = build_rule_plan(MATRIX, measurement(256), "TP-A", target_exists=False)
    rule = plan["framework_rule"]
    assert rule["local_attention_heads"] == {"source": 4, "target": 2}
    assert rule["local_tensor_shapes"]["attention_qkv_weight"]["target"] == [192, 256]
    assert rule["collective_rule"] == {
        "operation": "all_reduce",
        "payload_bytes": 262144,
        "per_layer": [
            "attention-projection-forward",
            "mlp-fc2-forward",
            "mlp-fc1-input-gradient",
            "attention-qkv-input-gradient",
        ],
        "layers": 2,
        "per_rank_operation_count": 8,
    }


def test_dp_plan_uses_exact_manifest_and_orthogonal_groups() -> None:
    source = measurement(320)
    plan = build_rule_plan(MATRIX, source, "DP-B", target_exists=False)
    rule = plan["framework_rule"]
    assert rule["tp_groups"] == [[0, 1], [2, 3]]
    assert rule["dp_groups"] == [[0, 2], [1, 3]]
    contract = rule["ddp_contract"]
    assert [
        event["payload_bytes"] for event in contract["forward_metadata_broadcasts"]
    ] == [8, 4]
    assert [bucket["payload_bytes"] for bucket in contract["gradient_buckets"]] == [32]
    assert contract["per_rank_operation_count"] == 3
    assert contract["gradient_buckets"][0]["last_ready_parameter"] == (
        "layers.0.self_attention.linear_qkv.weight"
    )


def test_dp_plan_derives_first_and_continuation_buckets() -> None:
    source = measurement(256)
    first = source["parameter_manifest"]["parameters"][0]
    first["shape"] = [262_144]
    first["numel"] = 262_144
    second = {
        **first,
        "order": 1,
        "name": "layers.0.self_attention.linear_qkv.bias",
        "shape": [8],
        "numel": 8,
    }
    source["parameter_manifest"]["parameters"].append(second)
    source["parameter_manifest"]["parameter_count"] = 2
    source["parameter_manifest"]["total_numel"] = 262_152
    source["gradient_readiness"]["parameter_names"].append(second["name"])
    source["gradient_readiness"]["parameter_count"] = 2
    contract = build_rule_plan(MATRIX, source, "DP-A", target_exists=False)[
        "framework_rule"
    ]["ddp_contract"]
    assert [bucket["payload_bytes"] for bucket in contract["gradient_buckets"]] == [
        1_048_576,
        32,
    ]
    assert [
        event["payload_bytes"] for event in contract["forward_metadata_broadcasts"]
    ] == [
        12,
        8,
    ]


def test_dp_plan_rejects_non_permutation_readiness() -> None:
    source = measurement(256)
    source["gradient_readiness"]["parameter_names"] = ["unknown.weight"]
    with pytest.raises(TransformerFamilyAdmissionError, match="gradient readiness"):
        build_rule_plan(MATRIX, source, "DP-A", target_exists=False)


@pytest.mark.parametrize(
    "target,microbatches,payload,global_operations",
    [("PP-A", 2, 262144, 16), ("PP-B", 4, 327680, 32)],
)
def test_pp_plan_derives_schedule_routes_and_payloads(
    target: str, microbatches: int, payload: int, global_operations: int
) -> None:
    sequence = 256 if target == "PP-A" else 320
    plan = build_rule_plan(MATRIX, measurement(sequence), target, target_exists=False)
    rule = plan["framework_rule"]
    assert rule["pp_groups"] == [[0, 2], [1, 3]]
    assert rule["layer_placement"] == {"stage_0": [0], "stage_1": [1]}
    assert rule["microbatches"] == microbatches
    assert rule["p2p"]["payload_bytes"] == payload
    assert rule["p2p"]["global_operation_count"] == global_operations
    assert rule["tp_collectives"]["per_rank_operation_count"] == 4 * microbatches
    lanes = rule["p2p"]["tp_lane_templates"]
    assert [lane["global_ranks"] for lane in lanes] == [[0, 2], [1, 3]]
    first = lanes[0]["operations"][:4]
    assert [(event["event_rank"], event["operation"]) for event in first] == [
        (0, "send"),
        (2, "recv"),
        (2, "send"),
        (0, "recv"),
    ]


def test_rule_plan_never_claims_target_observation_or_graph_validity() -> None:
    plan = build_rule_plan(MATRIX, measurement(), "TP-A", target_exists=False)
    assert plan["provenance"]["target_runtime_observation"] is False
    assert plan["graph_status"].startswith("unresolved")
    assert plan["timing_status"].startswith("unresolved")
    assert plan["claim"] == "framework-rule-plan-only-no-generated-workload-claim"


def test_v15_plan_derives_shapes_and_payloads_from_frozen_model_domain() -> None:
    source = measurement(1024)
    source["model"].update(
        hidden_size=1024,
        ffn_hidden_size=4096,
        attention_heads=16,
    )
    tp = build_rule_plan(V15_MATRIX, source, "TP-A", target_exists=False)[
        "framework_rule"
    ]
    assert tp["local_attention_heads"] == {"source": 8, "target": 4}
    assert tp["local_tensor_shapes"]["attention_qkv_weight"] == {
        "source": [1536, 1024],
        "target": [768, 1024],
    }
    assert tp["local_tensor_shapes"]["mlp_fc1_weight"] == {
        "source": [2048, 1024],
        "target": [1024, 1024],
    }
    assert tp["collective_rule"]["payload_bytes"] == 4_194_304
    pp = build_rule_plan(V15_MATRIX, source, "PP-A", target_exists=False)[
        "framework_rule"
    ]
    assert pp["p2p"]["payload_bytes"] == 4_194_304
