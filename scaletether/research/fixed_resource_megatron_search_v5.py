"""Freeze and score prospective interior targets for the fixed-resource search."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median, stdev
from typing import Any

import scaletether.research.fixed_resource_megatron_search_v4 as v4


PROTOCOL_SHA256 = "4a80952d8a2cf27df23a558dddc0bdf68eb1c70ba6d9a8a697f1cc3a3f4f2d4e"
V4_HELPER_SHA256 = "7e1d50a4357dccc6763e3736cc303a6b4fc610869051142cfe9736e49199cc70"
FREEZE_SCHEMA = "scaletether-fixed-resource-megatron-search-freeze-v5"
SCORE_SCHEMA = "scaletether-fixed-resource-megatron-search-score-v5"
TARGETS = ((768, 768), (1024, 1024), (1280, 1280), (1024, 1536))
TPS = (1, 2, 4)


def _v4_helper_sha256() -> str:
    return v4.sha256(Path(v4.__file__).resolve())


def _verified_v4_freeze(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _v4_helper_sha256() != V4_HELPER_SHA256 or v4.sha256(path) != expected_sha256:
        raise v4.SearchError("V4 helper or freeze hash mismatch")
    document = v4.load_json(path)
    core = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("schema") != v4.FREEZE_SCHEMA
        or document.get("artifact_sha256") != v4.canonical_sha256(core)
        or document.get("all_gates_accepted") is not True
        or document.get("target_observations_consumed") is not False
    ):
        raise v4.SearchError("invalid V4 freeze")
    return document


def _corner_logs(document: dict[str, Any]) -> dict[tuple[int, int, int], float]:
    corners: dict[tuple[int, int, int], float] = {}
    for gate in document["contrast_gates"]:
        key = (
            int(gate["hidden_size"]),
            int(gate["sequence_length"]),
            int(gate["tensor_parallel_size"]),
        )
        if gate.get("accepted") is not True or key in corners:
            raise v4.SearchError("invalid V4 contrast gate")
        corners[key] = math.log(float(gate["centered_ratio"]))
    expected = {
        (hidden, sequence, tp)
        for hidden, sequence in v4.CALIBRATION
        for tp in (2, 4)
    }
    if set(corners) != expected:
        raise v4.SearchError("V4 contrast matrix is incomplete")
    return corners


def _raw_bilinear(
    corners: dict[tuple[int, int, int], float], hidden: int, sequence: int, tp: int
) -> float:
    x = (hidden - 512) / (1536 - 512)
    y = (sequence - 512) / (2048 - 512)
    return math.exp(
        corners[(512, 512, tp)] * (1 - x) * (1 - y)
        + corners[(1536, 512, tp)] * x * (1 - y)
        + corners[(512, 2048, tp)] * (1 - x) * y
        + corners[(1536, 2048, tp)] * x * y
    )


def _nearest(
    corners: dict[tuple[int, int, int], float], hidden: int, sequence: int, tp: int
) -> float:
    candidates = []
    for corner_h, corner_s in v4.CALIBRATION:
        distance = (
            math.log(hidden / corner_h) / math.log(3)
        ) ** 2 + (math.log(sequence / corner_s) / math.log(4)) ** 2
        candidates.append((distance, corner_h, corner_s))
    _, corner_h, corner_s = min(candidates)
    return math.exp(corners[(corner_h, corner_s, tp)])


def _selection(ratios: dict[int, float]) -> int:
    return min(TPS, key=ratios.__getitem__)


def freeze(v4_freeze_path: Path, v4_freeze_sha256: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise v4.SearchError("fresh V5 freeze output is required")
    source = _verified_v4_freeze(v4_freeze_path, v4_freeze_sha256)
    corners = _corner_logs(source)
    global_ratios = {
        tp: math.exp(
            median(
                corners[(hidden, sequence, tp)]
                for hidden, sequence in v4.CALIBRATION
            )
        )
        for tp in (2, 4)
    }
    predictions = []
    for task_id, (hidden, sequence) in enumerate(TARGETS, 1):
        primary = {
            1: 1.0,
            2: math.exp(
                v4._interpolate(
                    {(h, s): corners[(h, s, 2)] for h, s in v4.CALIBRATION},
                    hidden,
                    sequence,
                )
            ),
            4: math.exp(
                v4._interpolate(
                    {(h, s): corners[(h, s, 4)] for h, s in v4.CALIBRATION},
                    hidden,
                    sequence,
                )
            ),
        }
        raw = {
            1: 1.0,
            2: _raw_bilinear(corners, hidden, sequence, 2),
            4: _raw_bilinear(corners, hidden, sequence, 4),
        }
        nearest = {
            1: 1.0,
            2: _nearest(corners, hidden, sequence, 2),
            4: _nearest(corners, hidden, sequence, 4),
        }
        global_ratio = {1: 1.0, **global_ratios}
        methods = {
            "ScaleTether log-coordinate bilinear": primary,
            "raw-coordinate bilinear": raw,
            "nearest calibration corner": nearest,
            "global median ratio": global_ratio,
        }
        predictions.append(
            {
                "task_id": task_id,
                "hidden_size": hidden,
                "sequence_length": sequence,
                "methods": {
                    name: {
                        "predicted_ratios_to_tp1": {str(tp): value for tp, value in ratios.items()},
                        "selected_tp": _selection(ratios),
                    }
                    for name, ratios in methods.items()
                },
            }
        )
    core = {
        "schema": FREEZE_SCHEMA,
        "status": "stored-before-v5-target-submission",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": PROTOCOL_SHA256,
        "helper_sha256": v4.sha256(Path(__file__).resolve()),
        "v4_helper_sha256": V4_HELPER_SHA256,
        "v4_prediction_freeze_sha256": v4_freeze_sha256,
        "target_observations_consumed": False,
        "targets": predictions,
    }
    document = {**core, "artifact_sha256": v4.canonical_sha256(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _verify_target(
    root: Path, task_id: int, hidden: int, sequence: int, freeze_sha256: str
) -> tuple[str, dict[int, float], dict[int, list[float]]]:
    v4._verify_result_manifest(root)
    v4._verify_completion(root, "scaletether-fixed-resource-megatron-target-complete-v5")
    v4._verify_h100_hardware(root)
    if v4.sha256(root / "prediction-freeze.json") != freeze_sha256:
        raise v4.SearchError("target freeze changed")
    identity = v4.load_json(root / "runtime-identity.json")
    hostname = identity.get("hostname")
    if identity.get("schema") != "scaletether-fixed-resource-megatron-runtime-v5" or not hostname:
        raise v4.SearchError("invalid V5 runtime identity")
    medians: dict[int, float] = {}
    repetitions: dict[int, list[float]] = {}
    shift = (task_id - 1) % 3
    order = list(TPS[shift:] + TPS[:shift])
    for launch, tp in enumerate(order, 1):
        dp = 4 // tp
        endpoint = root / "transformer" / f"H{hidden}-S{sequence}-TP{tp}-DP{dp}"
        contract = v4.load_json(endpoint / "run-contract.json")
        if contract != {
            "schema": "scaletether-fixed-resource-megatron-run-v5",
            "phase": "prospective-interior-target",
            "task_id": task_id,
            "launch_order_index": launch,
            "hidden_size": hidden,
            "sequence_length": sequence,
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "micro_batch_size": tp,
            "global_batch_size": 4,
            "total_gpus": 4,
            "num_layers": 2,
            "warmup_seconds": 3,
            "retained_repetitions": 20,
            "prediction_freeze_sha256": freeze_sha256,
        }:
            raise v4.SearchError("V5 target contract mismatch")
        timing = v4.load_json(endpoint / "timing.json")
        values = [float(value) for value in timing.get("iteration_device_envelope_us", [])]
        if timing.get("repetition_count") != 20 or len(values) != 20:
            raise v4.SearchError("V5 timing repetitions are incomplete")
        repetitions[tp] = values
        medians[tp] = float(median(values))
    return str(hostname), medians, repetitions


def _metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    exact = within_one = 0
    regrets: list[float] = []
    errors: list[float] = []
    for row in rows:
        selected = int(row["methods"][method]["selected_tp"])
        fastest = int(row["physically_fastest_tp"])
        regret = 100 * (row["medians_us"][str(selected)] / row["medians_us"][str(fastest)] - 1)
        exact += selected == fastest
        within_one += regret <= 1.0
        regrets.append(regret)
        predicted = row["methods"][method]["predicted_ratios_to_tp1"]
        observed = row["observed_ratios_to_tp1"]
        errors.extend(100 * abs(predicted[str(tp)] / observed[str(tp)] - 1) for tp in (2, 4))
    return {
        "method": method,
        "exact_best": exact,
        "within_one_percent_of_best": within_one,
        "median_regret_percent": float(median(regrets)),
        "maximum_regret_percent": max(regrets),
        "median_absolute_ratio_error_percent": float(median(errors)),
        "maximum_absolute_ratio_error_percent": max(errors),
    }


def score(freeze_path: Path, target_roots: list[Path], output: Path) -> dict[str, Any]:
    freeze_document = v4.load_json(freeze_path)
    core = {key: value for key, value in freeze_document.items() if key != "artifact_sha256"}
    if (
        freeze_document.get("schema") != FREEZE_SCHEMA
        or freeze_document.get("artifact_sha256") != v4.canonical_sha256(core)
        or freeze_document.get("target_observations_consumed") is not False
        or len(target_roots) != len(TARGETS)
        or len({root.resolve() for root in target_roots}) != len(TARGETS)
    ):
        raise v4.SearchError("invalid V5 freeze or target roots")
    freeze_sha256 = v4.sha256(freeze_path)
    rows = []
    for task_id, (root, (hidden, sequence)) in enumerate(zip(target_roots, TARGETS), 1):
        hostname, medians, repetitions = _verify_target(
            root, task_id, hidden, sequence, freeze_sha256
        )
        observed = {1: 1.0, 2: medians[2] / medians[1], 4: medians[4] / medians[1]}
        fastest = min(TPS, key=medians.__getitem__)
        prediction = freeze_document["targets"][task_id - 1]
        cvs = {
            str(tp): 100 * stdev(repetitions[tp]) / (sum(repetitions[tp]) / len(repetitions[tp]))
            for tp in TPS
        }
        rows.append(
            {
                "task_id": task_id,
                "hidden_size": hidden,
                "sequence_length": sequence,
                "hostname": hostname,
                "physically_fastest_tp": fastest,
                "medians_us": {str(tp): value for tp, value in medians.items()},
                "observed_ratios_to_tp1": {str(tp): value for tp, value in observed.items()},
                "within_candidate_cv_percent": cvs,
                "methods": prediction["methods"],
            }
        )
    method_names = list(freeze_document["targets"][0]["methods"])
    baselines = [_metrics(rows, method) for method in method_names]
    result = {
        "schema": SCORE_SCHEMA,
        "prediction_freeze_sha256": freeze_sha256,
        "tasks": rows,
        "baselines": baselines,
        "summary": {
            "tasks": len(rows),
            "distinct_hosts": sorted({row["hostname"] for row in rows}),
            "physical_optimum_counts": {
                str(tp): sum(row["physically_fastest_tp"] == tp for row in rows)
                for tp in TPS
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--v4-freeze", type=Path, required=True)
    freeze_parser.add_argument("--v4-freeze-sha256", required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--freeze", type=Path, required=True)
    score_parser.add_argument("--target-root", action="append", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.v4_freeze, args.v4_freeze_sha256, args.output)
    else:
        score(args.freeze, args.target_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
