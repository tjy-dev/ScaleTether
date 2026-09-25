"""Score prospective Megatron layout timing brackets without refitting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_bracket(result_path: Path) -> dict[str, Any]:
    result = _read(result_path)
    if result.get("schema") != "scaletether-megatron-layout-timing-bracket-v1":
        raise ValueError(f"{result_path}: unexpected schema")
    freeze_path = result_path.parent / "prediction-freeze.json"
    freeze_bytes = freeze_path.read_bytes()
    if hashlib.sha256(freeze_bytes).hexdigest() != result.get("prediction_freeze_sha256"):
        raise ValueError(f"{result_path}: prediction freeze digest mismatch")
    freeze = json.loads(freeze_bytes)
    if freeze.get("target_observed") is not False:
        raise ValueError(f"{result_path}: freeze does not exclude the target")
    if int(freeze["created_unix_ns"]) >= int(result["target_started_unix_ns"]):
        raise ValueError(f"{result_path}: target did not start after prediction freeze")
    if result.get("measured_repetitions") != 20:
        raise ValueError(f"{result_path}: expected 20 retained repetitions")
    anchors = freeze.get("anchor_hidden_sizes")
    target = freeze.get("target_hidden_size")
    if (
        not isinstance(anchors, list)
        or len(anchors) != 2
        or not all(isinstance(value, int) and value > 0 for value in anchors)
        or anchors != sorted(anchors)
        or not isinstance(target, int)
        or not anchors[0] < target < anchors[1]
    ):
        raise ValueError(f"{result_path}: invalid frozen hidden-size bracket")
    records = result.get("records")
    required = {
        *(f"source_h{hidden}" for hidden in anchors),
        *(f"target_h{hidden}" for hidden in anchors),
        "heldout_source_left",
        "heldout_target",
        "heldout_source_right",
    }
    if not isinstance(records, dict) or set(records) != required:
        raise ValueError(f"{result_path}: unexpected endpoint set")
    for name, record in records.items():
        samples = record.get("samples_us") if isinstance(record, dict) else None
        if not isinstance(samples, list) or len(samples) != 20:
            raise ValueError(f"{result_path}: {name} does not have 20 samples")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError(f"{result_path}: {name} contains invalid timing samples")
    return result


def score(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 10:
        raise ValueError("exactly ten brackets are required (five each for TP2 and TP4)")
    results = [validate_bracket(path) for path in paths]
    groups: dict[int, list[dict[str, Any]]] = {2: [], 4: []}
    for result in results:
        groups[int(result["tensor_parallel_size"])].append(result)
    if any(len(group) != 5 for group in groups.values()):
        raise ValueError("expected five independent brackets for each TP size")
    summary: dict[str, Any] = {}
    for tp, group in groups.items():
        outer = sorted(int(item["outer_repetition"]) for item in group)
        if outer != [1, 2, 3, 4, 5]:
            raise ValueError(f"TP{tp}: outer repetitions must be exactly 1..5")
        errors = [float(item["prediction_absolute_percentage_error"]) for item in group]
        drifts = [float(item["source_control_drift_percent"]) for item in group]
        summary[f"tp{tp}"] = {
            "brackets": 5,
            "median_absolute_percentage_error": statistics.median(errors),
            "maximum_absolute_percentage_error": max(errors),
            "median_source_control_drift_percent": statistics.median(drifts),
            "maximum_source_control_drift_percent": max(drifts),
            "individual_errors_percent": errors,
            "individual_drifts_percent": drifts,
            "timing_gate": "passed" if max(errors) <= 5.0 else "failed",
            "source_control_gate": "passed" if max(drifts) <= 3.0 else "failed",
        }
    return {
        "schema": "scaletether-megatron-layout-timing-score-v1",
        "decision_rule": "all five brackets per TP must satisfy APE <=5% and drift <=3%",
        "groups": summary,
        "overall_gate": "passed" if all(
            group[gate] == "passed"
            for group in summary.values()
            for gate in ("timing_gate", "source_control_gate")
        ) else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.result)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
