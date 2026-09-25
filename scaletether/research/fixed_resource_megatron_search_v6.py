"""Freeze and score a selective fixed-resource H100 search."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median, stdev
from typing import Any

import scaletether.research.fixed_resource_megatron_search_v4 as v4
import scaletether.research.fixed_resource_megatron_search_v5 as v5


FREEZE_SCHEMA = "scaletether-fixed-resource-megatron-selective-freeze-v6"
SCORE_SCHEMA = "scaletether-fixed-resource-megatron-selective-score-v6"
TARGETS = ((768, 1280), (1280, 768), (768, 1536), (1280, 1536), (1024, 768), (1024, 1280))
TPS = (1, 2, 4)


def _selection(ratios: dict[int, float]) -> int:
    return min(TPS, key=ratios.__getitem__)


def freeze(v4_freeze_path: Path, v4_freeze_sha256: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise v4.SearchError("fresh V6 freeze output is required")
    source = v5._verified_v4_freeze(v4_freeze_path, v4_freeze_sha256)
    corners = v5._corner_logs(source)
    targets = []
    for task_id, (hidden, sequence) in enumerate(TARGETS, 1):
        methods = {
            "log-coordinate bilinear": {
                1: 1.0,
                2: math.exp(v4._interpolate({(h, s): corners[(h, s, 2)] for h, s in v4.CALIBRATION}, hidden, sequence)),
                4: math.exp(v4._interpolate({(h, s): corners[(h, s, 4)] for h, s in v4.CALIBRATION}, hidden, sequence)),
            },
            "raw-coordinate bilinear": {
                1: 1.0,
                2: v5._raw_bilinear(corners, hidden, sequence, 2),
                4: v5._raw_bilinear(corners, hidden, sequence, 4),
            },
            "nearest calibration corner": {
                1: 1.0,
                2: v5._nearest(corners, hidden, sequence, 2),
                4: v5._nearest(corners, hidden, sequence, 4),
            },
        }
        selections = {name: _selection(ratios) for name, ratios in methods.items()}
        unanimous = len(set(selections.values())) == 1
        targets.append({
            "task_id": task_id,
            "hidden_size": hidden,
            "sequence_length": sequence,
            "action": "ESTIMATE" if unanimous else "MEASURE",
            "selected_tp": selections["log-coordinate bilinear"] if unanimous else None,
            "requested_tps": [] if unanimous else list(TPS),
            "methods": {name: {"selected_tp": selections[name], "predicted_ratios_to_tp1": {str(tp): value for tp, value in ratios.items()}} for name, ratios in methods.items()},
        })
    core = {
        "schema": FREEZE_SCHEMA,
        "status": "stored-before-v6-target-submission",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "v4_prediction_freeze_sha256": v4_freeze_sha256,
        "target_observations_consumed": False,
        "targets": targets,
    }
    document = {**core, "artifact_sha256": v4.canonical_sha256(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _verify_target(root: Path, task_id: int, hidden: int, sequence: int, freeze_sha256: str) -> tuple[str, dict[int, float], dict[int, float]]:
    v4._verify_result_manifest(root)
    v4._verify_completion(root, "scaletether-fixed-resource-megatron-selective-target-complete-v6")
    v4._verify_h100_hardware(root)
    if v4.sha256(root / "prediction-freeze.json") != freeze_sha256:
        raise v4.SearchError("target freeze changed")
    identity = v4.load_json(root / "runtime-identity.json")
    hostname = identity.get("hostname")
    if identity.get("schema") != "scaletether-fixed-resource-megatron-runtime-v6" or not hostname:
        raise v4.SearchError("invalid V6 runtime identity")
    medians: dict[int, float] = {}
    cvs: dict[int, float] = {}
    order = list(TPS[(task_id - 1) % 3:] + TPS[:(task_id - 1) % 3])
    for launch, tp in enumerate(order, 1):
        dp = 4 // tp
        endpoint = root / "transformer" / f"H{hidden}-S{sequence}-TP{tp}-DP{dp}"
        contract = v4.load_json(endpoint / "run-contract.json")
        expected = {"schema": "scaletether-fixed-resource-megatron-run-v6", "phase": "prospective-selective-target", "task_id": task_id, "launch_order_index": launch, "hidden_size": hidden, "sequence_length": sequence, "tensor_parallel_size": tp, "data_parallel_size": dp, "micro_batch_size": tp, "global_batch_size": 4, "total_gpus": 4, "num_layers": 2, "warmup_seconds": 3, "retained_repetitions": 20, "prediction_freeze_sha256": freeze_sha256}
        if contract != expected:
            raise v4.SearchError("V6 target contract mismatch")
        timing = v4.load_json(endpoint / "timing.json")
        values = [float(value) for value in timing.get("iteration_device_envelope_us", [])]
        if timing.get("repetition_count") != 20 or len(values) != 20:
            raise v4.SearchError("V6 timing repetitions are incomplete")
        medians[tp] = float(median(values))
        cvs[tp] = 100 * stdev(values) / (sum(values) / len(values))
    return str(hostname), medians, cvs


def score(freeze_path: Path, target_roots: list[Path], output: Path) -> dict[str, Any]:
    freeze_doc = v4.load_json(freeze_path)
    core = {key: value for key, value in freeze_doc.items() if key != "artifact_sha256"}
    if freeze_doc.get("schema") != FREEZE_SCHEMA or freeze_doc.get("artifact_sha256") != v4.canonical_sha256(core) or freeze_doc.get("target_observations_consumed") is not False or len(target_roots) != len(TARGETS):
        raise v4.SearchError("invalid V6 freeze or target roots")
    freeze_sha256 = v4.sha256(freeze_path)
    rows = []
    for task_id, (root, (hidden, sequence)) in enumerate(zip(target_roots, TARGETS), 1):
        hostname, medians, cvs = _verify_target(root, task_id, hidden, sequence, freeze_sha256)
        fastest = min(TPS, key=medians.__getitem__)
        frozen = freeze_doc["targets"][task_id - 1]
        blind_tp = int(frozen["methods"]["log-coordinate bilinear"]["selected_tp"])
        blind_regret = 100 * (medians[blind_tp] / medians[fastest] - 1)
        selected = frozen["selected_tp"]
        emitted_regret = None if selected is None else 100 * (medians[int(selected)] / medians[fastest] - 1)
        rows.append({"task_id": task_id, "hidden_size": hidden, "sequence_length": sequence, "hostname": hostname, "action": frozen["action"], "selected_tp": selected, "requested_tps": frozen["requested_tps"], "physically_fastest_tp": fastest, "medians_us": {str(k): v for k, v in medians.items()}, "within_candidate_cv_percent": {str(k): v for k, v in cvs.items()}, "blind_log_selected_tp": blind_tp, "blind_log_regret_percent": blind_regret, "emitted_regret_percent": emitted_regret})
    emitted = [row for row in rows if row["action"] == "ESTIMATE"]
    measured = [row for row in rows if row["action"] == "MEASURE"]
    unsafe = [row for row in rows if row["blind_log_regret_percent"] > 1.0]
    summary = {
        "tasks": len(rows),
        "estimate_count": len(emitted),
        "measure_count": len(measured),
        "estimate_coverage_percent": 100 * len(emitted) / len(rows),
        "emitted_exact_best": sum(row["selected_tp"] == row["physically_fastest_tp"] for row in emitted),
        "emitted_within_one_percent": sum(row["emitted_regret_percent"] <= 1.0 for row in emitted),
        "maximum_emitted_regret_percent": max((row["emitted_regret_percent"] for row in emitted), default=None),
        "blind_above_one_percent": len(unsafe),
        "blind_above_one_percent_withheld": sum(row["action"] == "MEASURE" for row in unsafe),
        "distinct_hosts": sorted({row["hostname"] for row in rows}),
    }
    summary["accepted"] = len(emitted) >= 2 and len(measured) >= 2 and summary["emitted_within_one_percent"] == len(emitted) and summary["blind_above_one_percent_withheld"] == len(unsafe)
    result = {"schema": SCORE_SCHEMA, "prediction_freeze_sha256": freeze_sha256, "summary": summary, "tasks": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze"); p.add_argument("--v4-freeze", type=Path, required=True); p.add_argument("--v4-freeze-sha256", required=True); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("score"); p.add_argument("--freeze", type=Path, required=True); p.add_argument("--target-root", action="append", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze": freeze(args.v4_freeze, args.v4_freeze_sha256, args.output)
    else: score(args.freeze, args.target_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
