"""Freeze and score an expanded selective fixed-resource H100 search."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import median, stdev
from typing import Any

import scaletether.research.fixed_resource_megatron_search_v4 as v4
import scaletether.research.fixed_resource_megatron_search_v5 as v5
import scaletether.research.fixed_resource_megatron_search_v6 as v6


FREEZE_SCHEMA = "scaletether-fixed-resource-megatron-selective-freeze-v7"
SCORE_SCHEMA = "scaletether-fixed-resource-megatron-selective-score-v7"
TPS = (1, 2, 4)
REPRESENTATIVE_COUNT = 12
STRESS_COUNT = 6
SAMPLING_SEED = 20260809
# The model uses 64-wide attention heads. Every sampled candidate must be
# legal for every TP degree, including TP4, before its prediction is frozen.
HIDDEN_GRID = tuple(
    hidden
    for hidden in range(640, 1409, 128)
    if (hidden // 64) % max(TPS) == 0
)
SEQUENCE_GRID = tuple(range(640, 1921, 128))
PREVIOUSLY_MEASURED = frozenset(v4.TARGETS) | frozenset(v5.TARGETS) | frozenset(v6.TARGETS) | frozenset(v4.CALIBRATION)


def _methods(corners: dict[tuple[int, int, int], float], hidden: int, sequence: int) -> dict[str, dict[int, float]]:
    return {
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


def _candidate(corners: dict[tuple[int, int, int], float], hidden: int, sequence: int) -> dict[str, Any]:
    methods = _methods(corners, hidden, sequence)
    selections = {name: v6._selection(ratios) for name, ratios in methods.items()}
    log_times = sorted(methods["log-coordinate bilinear"].values())
    return {
        "hidden_size": hidden,
        "sequence_length": sequence,
        "action": "ESTIMATE" if len(set(selections.values())) == 1 else "MEASURE",
        "selected_tp": selections["log-coordinate bilinear"] if len(set(selections.values())) == 1 else None,
        "requested_tps": [] if len(set(selections.values())) == 1 else list(TPS),
        "model_selection_count": len(set(selections.values())),
        "predicted_fastest_margin_percent": 100.0 * (log_times[1] / log_times[0] - 1.0),
        "methods": {
            name: {
                "selected_tp": selections[name],
                "predicted_ratios_to_tp1": {str(tp): value for tp, value in ratios.items()},
            }
            for name, ratios in methods.items()
        },
    }


def freeze(v4_freeze_path: Path, v4_freeze_sha256: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise v4.SearchError("fresh V7 freeze output is required")
    source = v5._verified_v4_freeze(v4_freeze_path, v4_freeze_sha256)
    corners = v5._corner_logs(source)
    candidates = [
        _candidate(corners, hidden, sequence)
        for hidden in HIDDEN_GRID
        for sequence in SEQUENCE_GRID
        if (hidden, sequence) not in PREVIOUSLY_MEASURED
    ]
    rng = random.Random(SAMPLING_SEED)
    representative = rng.sample(candidates, REPRESENTATIVE_COUNT)
    representative_coordinates = {
        (row["hidden_size"], row["sequence_length"]) for row in representative
    }
    remaining = [
        row for row in candidates
        if (row["hidden_size"], row["sequence_length"]) not in representative_coordinates
    ]
    stress = sorted(
        remaining,
        key=lambda row: (
            -int(row["model_selection_count"]),
            float(row["predicted_fastest_margin_percent"]),
            int(row["hidden_size"]),
            int(row["sequence_length"]),
        ),
    )[:STRESS_COUNT]
    targets = []
    for cohort, rows in (("representative", representative), ("stress", stress)):
        for row in rows:
            targets.append({"task_id": len(targets) + 1, "cohort": cohort, **row})
    core = {
        "schema": FREEZE_SCHEMA,
        "status": "stored-before-v7-target-submission",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "v4_prediction_freeze_sha256": v4_freeze_sha256,
        "target_observations_consumed": False,
        "sampling": {
            "representative_rule": "python-random-sample-over-declared-grid",
            "seed": SAMPLING_SEED,
            "stress_rule": "descending-model-selection-count-then-ascending-log-margin",
            "hidden_grid": list(HIDDEN_GRID),
            "sequence_grid": list(SEQUENCE_GRID),
            "previously_measured_coordinates": sorted([list(x) for x in PREVIOUSLY_MEASURED]),
        },
        "targets": targets,
    }
    document = {**core, "artifact_sha256": v4.canonical_sha256(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _verify_target(root: Path, frozen: dict[str, Any], freeze_sha256: str) -> tuple[str, dict[int, float], dict[int, float]]:
    task_id = int(frozen["task_id"]); hidden = int(frozen["hidden_size"]); sequence = int(frozen["sequence_length"])
    v4._verify_result_manifest(root)
    v4._verify_completion(root, "scaletether-fixed-resource-megatron-selective-target-complete-v7")
    v4._verify_h100_hardware(root)
    if v4.sha256(root / "prediction-freeze.json") != freeze_sha256:
        raise v4.SearchError("target freeze changed")
    identity = v4.load_json(root / "runtime-identity.json")
    hostname = identity.get("hostname")
    if identity.get("schema") != "scaletether-fixed-resource-megatron-runtime-v7" or not hostname:
        raise v4.SearchError("invalid V7 runtime identity")
    medians: dict[int, float] = {}; cvs: dict[int, float] = {}
    order = list(TPS[(task_id - 1) % 3:] + TPS[:(task_id - 1) % 3])
    for launch, tp in enumerate(order, 1):
        dp = 4 // tp; endpoint = root / "transformer" / f"H{hidden}-S{sequence}-TP{tp}-DP{dp}"
        expected = {"schema": "scaletether-fixed-resource-megatron-run-v7", "phase": "prospective-selective-expanded-target", "task_id": task_id, "cohort": frozen["cohort"], "launch_order_index": launch, "hidden_size": hidden, "sequence_length": sequence, "tensor_parallel_size": tp, "data_parallel_size": dp, "micro_batch_size": tp, "global_batch_size": 4, "total_gpus": 4, "num_layers": 2, "warmup_seconds": 3, "retained_repetitions": 20, "prediction_freeze_sha256": freeze_sha256}
        if v4.load_json(endpoint / "run-contract.json") != expected:
            raise v4.SearchError("V7 target contract mismatch")
        timing = v4.load_json(endpoint / "timing.json")
        values = [float(value) for value in timing.get("iteration_device_envelope_us", [])]
        if timing.get("repetition_count") != 20 or len(values) != 20:
            raise v4.SearchError("V7 timing repetitions are incomplete")
        medians[tp] = float(median(values)); cvs[tp] = 100 * stdev(values) / (sum(values) / len(values))
    return str(hostname), medians, cvs


def score(freeze_path: Path, target_roots: list[Path], output: Path) -> dict[str, Any]:
    freeze_doc = v4.load_json(freeze_path); core = {k: v for k, v in freeze_doc.items() if k != "artifact_sha256"}
    targets = freeze_doc.get("targets", [])
    if freeze_doc.get("schema") != FREEZE_SCHEMA or freeze_doc.get("artifact_sha256") != v4.canonical_sha256(core) or freeze_doc.get("target_observations_consumed") is not False or len(targets) != 18 or len(target_roots) != 18:
        raise v4.SearchError("invalid V7 freeze or target roots")
    freeze_sha256 = v4.sha256(freeze_path); rows = []
    for root, frozen in zip(target_roots, targets):
        hostname, medians, cvs = _verify_target(root, frozen, freeze_sha256)
        fastest = min(TPS, key=medians.__getitem__)
        blind_tp = int(frozen["methods"]["log-coordinate bilinear"]["selected_tp"])
        blind_regret = 100 * (medians[blind_tp] / medians[fastest] - 1)
        selected = frozen["selected_tp"]
        emitted_regret = None if selected is None else 100 * (medians[int(selected)] / medians[fastest] - 1)
        rows.append({"task_id": frozen["task_id"], "cohort": frozen["cohort"], "hidden_size": frozen["hidden_size"], "sequence_length": frozen["sequence_length"], "hostname": hostname, "action": frozen["action"], "selected_tp": selected, "requested_tps": frozen["requested_tps"], "physically_fastest_tp": fastest, "medians_us": {str(k): v for k, v in medians.items()}, "within_candidate_cv_percent": {str(k): v for k, v in cvs.items()}, "blind_log_selected_tp": blind_tp, "blind_log_regret_percent": blind_regret, "emitted_regret_percent": emitted_regret})
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        estimates = [r for r in items if r["action"] == "ESTIMATE"]; measures = [r for r in items if r["action"] == "MEASURE"]; harmful = [r for r in items if r["blind_log_regret_percent"] > 1.0]
        regrets = [float(r["emitted_regret_percent"]) for r in estimates]
        return {"tasks": len(items), "estimate_count": len(estimates), "measure_count": len(measures), "estimate_retention_percent": 100 * len(estimates) / len(items), "emitted_exact_best": sum(r["selected_tp"] == r["physically_fastest_tp"] for r in estimates), "emitted_within_one_percent": sum(float(r["emitted_regret_percent"]) <= 1.0 for r in estimates), "median_emitted_regret_percent": median(regrets) if regrets else None, "maximum_emitted_regret_percent": max(regrets) if regrets else None, "unconditional_above_one_percent": len(harmful), "unconditional_above_one_percent_withheld": sum(r["action"] == "MEASURE" for r in harmful), "conservative_measure_count": sum(r["action"] == "MEASURE" and r["blind_log_regret_percent"] <= 1.0 for r in items), "distinct_hosts": sorted({r["hostname"] for r in items})}
    summary = {"all": summarize(rows), "representative": summarize([r for r in rows if r["cohort"] == "representative"]), "stress": summarize([r for r in rows if r["cohort"] == "stress"])}
    summary["protocol_gates"] = {
        "at_least_three_hosts": len(summary["all"]["distinct_hosts"]) >= 3,
        "all_estimates_within_one_percent": summary["all"]["emitted_within_one_percent"] == summary["all"]["estimate_count"],
        "all_unconditional_above_one_percent_withheld": summary["all"]["unconditional_above_one_percent_withheld"] == summary["all"]["unconditional_above_one_percent"],
    }
    summary["accepted"] = all(summary["protocol_gates"].values())
    result = {"schema": SCORE_SCHEMA, "prediction_freeze_sha256": freeze_sha256, "summary": summary, "tasks": rows}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze"); p.add_argument("--v4-freeze", type=Path, required=True); p.add_argument("--v4-freeze-sha256", required=True); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("score"); p.add_argument("--freeze", type=Path, required=True); p.add_argument("--target-root", action="append", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze": freeze(args.v4_freeze, args.v4_freeze_sha256, args.output)
    else: score(args.freeze, args.target_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
