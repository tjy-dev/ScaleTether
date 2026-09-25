"""Prospectively repair one v1 calibration cell and score untouched targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


V1_FREEZE_SCHEMA = "scaletether-fixed-resource-megatron-search-freeze-v1"
FREEZE_SCHEMA = "scaletether-fixed-resource-megatron-search-freeze-v2"
SCORE_SCHEMA = "scaletether-fixed-resource-megatron-search-score-v2"
TIMING_SCHEMA = "megatron-steady-state-repeated-timing-v1"
CANDIDATES = ((1, 4), (2, 2), (4, 1))
CALIBRATION = ((512, 512), (512, 2048), (1536, 512), (1536, 2048))
REPAIR_CELL = (512, 2048, 4)
TARGETS = (
    (768, 512),
    (768, 1024),
    (768, 2048),
    (1024, 512),
    (1024, 2048),
    (1280, 512),
    (1280, 1024),
    (1280, 2048),
)
STABILITY_GATE_PERCENT = 5.0
PROTOCOL_SHA256 = "396385f4f51c8cabf8a6a61d85f7745666bbb84eda1538584a983ed942dad7be"
MODEL_SHA256 = "501988c30882a65ca34ad0190e45011b347ef948fe4655a03ff0f704b6234491"


class SearchError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SearchError(f"unsafe or absent JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SearchError(f"JSON root is not an object: {path}")
    return value


def timing_median(path: Path, tp: int) -> float:
    document = load_json(path)
    samples = document.get("iteration_device_envelope_us")
    if (
        document.get("schema") != TIMING_SCHEMA
        or document.get("repetition_count") != 20
        or not isinstance(document.get("rank_records"), list)
        or len(document["rank_records"]) != 4
        or not isinstance(samples, list)
        or len(samples) != 20
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in samples
        )
    ):
        raise SearchError(f"invalid four-rank TP{tp} timing: {path}")
    return float(median(samples))


def endpoint_path(root: Path, hidden: int, sequence: int, tp: int) -> Path:
    return (
        root
        / "transformer"
        / f"H{hidden}-S{sequence}-TP{tp}-DP{4 // tp}"
        / "timing.json"
    )


def _verify_result_manifest(root: Path, expected_manifest_sha256: str | None = None) -> str:
    manifest = root / "sha256sums.txt"
    manifest_sha = sha256(manifest)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise SearchError(f"result manifest binding changed: {root}")
    entries: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            raise SearchError(f"malformed result manifest: {root}")
        expected, relative = line.split("  ", 1)
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in entries
        ):
            raise SearchError(f"unsafe result manifest entry: {root}")
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise SearchError(f"result file binding changed: {path}")
        entries.add(relative)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "sha256sums.txt"
    }
    if entries != actual:
        raise SearchError(f"result manifest is incomplete: {root}")
    return manifest_sha


def _verify_completion(root: Path, acquisition_schema: str) -> None:
    acquisition = root / "acquisition-complete.txt"
    completion = root / "completion.txt"
    if (
        not acquisition.is_file()
        or acquisition.is_symlink()
        or acquisition.read_text(encoding="utf-8") != f"schema={acquisition_schema}\n"
        or not completion.is_file()
        or completion.is_symlink()
    ):
        raise SearchError(f"invalid completion marker: {root}")
    fields = dict(
        line.split("=", 1)
        for line in completion.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    if fields.get("exit_status") != "0" or not fields.get("finished_at"):
        raise SearchError(f"unsuccessful result: {root}")


def _verify_h100_hardware(root: Path) -> None:
    hardware = root / "hardware.csv"
    lines = hardware.read_text(encoding="utf-8").splitlines() if hardware.is_file() else []
    if len(lines) != 4 or any("H100" not in line for line in lines):
        raise SearchError(f"result is not a four-H100 allocation: {root}")


def _verify_identities(root: Path) -> None:
    identities = load_json(root / "artifact-identities.json")
    helper_sha = sha256(Path(__file__).resolve())
    if identities != {
        "schema": "scaletether-fixed-resource-megatron-artifact-identities-v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "helper_sha256": helper_sha,
        "model_sha256": MODEL_SHA256,
    }:
        raise SearchError(f"invalid artifact identities: {root}")
    if (
        sha256(root / "protocol.md") != PROTOCOL_SHA256
        or sha256(root / "helper.py") != helper_sha
    ):
        raise SearchError(f"artifact identity binding changed: {root}")


def _verify_repair_root(root: Path, expected_outer: int) -> float:
    _verify_result_manifest(root)
    _verify_completion(root, "scaletether-fixed-resource-megatron-repair-complete-v2")
    _verify_h100_hardware(root)
    _verify_identities(root)
    contract_path = root / "transformer" / "H512-S2048-TP4-DP1" / "run-contract.json"
    contract = load_json(contract_path)
    expected = {
        "schema": "scaletether-fixed-resource-megatron-run-v2",
        "phase": "one-cell-repair",
        "outer_ordinal": expected_outer,
        "hidden_size": 512,
        "sequence_length": 2048,
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "micro_batch_size": 4,
        "global_batch_size": 4,
        "total_gpus": 4,
        "num_layers": 2,
        "warmup_seconds": 3,
        "retained_repetitions": 20,
        "replaces_only": "H512-S2048-TP4-DP1",
        "protocol_sha256": PROTOCOL_SHA256,
        "helper_sha256": contract.get("helper_sha256"),
        "model_sha256": MODEL_SHA256,
    }
    if contract != expected:
        raise SearchError(f"repair run contract mismatch: {contract_path}")
    identities = load_json(root / "artifact-identities.json")
    if contract["helper_sha256"] != identities["helper_sha256"]:
        raise SearchError(f"repair helper identity mismatch: {root}")
    return timing_median(endpoint_path(root, 512, 2048, 4), 4)


def _verify_target_root(
    root: Path,
    task_id: int,
    hidden: int,
    sequence: int,
    freeze_sha256: str,
) -> dict[int, float]:
    _verify_result_manifest(root)
    _verify_completion(root, "scaletether-fixed-resource-megatron-target-complete-v2")
    _verify_h100_hardware(root)
    copied_freeze = root / "prediction-freeze.json"
    if sha256(copied_freeze) != freeze_sha256:
        raise SearchError(f"target prediction-freeze binding changed: {root}")
    observed: dict[int, float] = {}
    shift = (task_id - 1) % len(CANDIDATES)
    expected_order = [CANDIDATES[(offset + shift) % len(CANDIDATES)][0] for offset in range(3)]
    for launch, tp in enumerate(expected_order, 1):
        dp = 4 // tp
        contract_path = root / "transformer" / f"H{hidden}-S{sequence}-TP{tp}-DP{dp}" / "run-contract.json"
        contract = load_json(contract_path)
        if contract != {
            "schema": "scaletether-fixed-resource-megatron-run-v2",
            "phase": "untouched-v2-target",
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
            raise SearchError(f"target run contract mismatch: {contract_path}")
        observed[tp] = timing_median(endpoint_path(root, hidden, sequence, tp), tp)
    return observed


def _bilinear(
    corners: dict[tuple[int, int], float], hidden: int, sequence: int
) -> float:
    if (
        set(corners) != set(CALIBRATION)
        or not 512 <= hidden <= 1536
        or not 512 <= sequence <= 2048
    ):
        raise SearchError("prediction lies outside the calibration rectangle")
    x = math.log(hidden / 512) / math.log(1536 / 512)
    y = math.log(sequence / 512) / math.log(2048 / 512)
    values = {key: math.log(value) for key, value in corners.items()}
    return math.exp(
        values[(512, 512)] * (1 - x) * (1 - y)
        + values[(1536, 512)] * x * (1 - y)
        + values[(512, 2048)] * (1 - x) * y
        + values[(1536, 2048)] * x * y
    )


def _verified_v1_freeze(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256(path) != expected_sha256:
        raise SearchError("v1 freeze hash mismatch")
    document = load_json(path)
    core = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("schema") != V1_FREEZE_SCHEMA
        or document.get("artifact_sha256") != canonical_sha256(core)
        or document.get("target_observations_consumed") is not False
    ):
        raise SearchError("invalid v1 prediction freeze")
    unstable = document.get("unstable_calibrations")
    if not isinstance(unstable, list) or len(unstable) != 1:
        raise SearchError("v1 must identify exactly one unstable calibration")
    item = unstable[0]
    identity = (
        item.get("hidden_size"),
        item.get("sequence_length"),
        item.get("tensor_parallel_size"),
    )
    if identity != REPAIR_CELL or float(item.get("maximum_relative_deviation_percent", 0)) <= STABILITY_GATE_PERCENT:
        raise SearchError("v1 instability is not the predeclared repair cell")
    predictions = document.get("predictions")
    if (
        not isinstance(predictions, list)
        or not predictions
        or any(
            item.get("timing_action") != "MEASURE" or item.get("selected_tp") is not None
            for item in predictions
        )
    ):
        raise SearchError("v1 predecessor did not abstain")
    bindings = document.get("calibration_bindings")
    if not isinstance(bindings, list) or len(bindings) != 3:
        raise SearchError("v1 calibration bindings are incomplete")
    for binding in bindings:
        root = Path(binding["root"])
        if (
            not (root / "acquisition-complete.txt").is_file()
            or _verify_result_manifest(root, binding["sha256sums_sha256"])
            != binding["sha256sums_sha256"]
        ):
            raise SearchError(f"v1 calibration binding changed: {root}")
    return document


def _v1_centers(document: dict[str, Any]) -> dict[tuple[int, int, int], float]:
    centers: dict[tuple[int, int, int], float] = {}
    for item in document.get("calibration_centers_us", []):
        key = (
            item.get("hidden_size"),
            item.get("sequence_length"),
            item.get("tensor_parallel_size"),
        )
        value = item.get("median_across_allocations_us")
        if (
            key in centers
            or key[:2] not in CALIBRATION
            or key[2] not in {tp for tp, _ in CANDIDATES}
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise SearchError("invalid v1 calibration center")
        centers[key] = float(value)
    if len(centers) != len(CALIBRATION) * len(CANDIDATES):
        raise SearchError("v1 calibration matrix is incomplete")
    return centers


def freeze(
    previous_freeze_path: Path,
    previous_freeze_sha256: str,
    repair_roots: list[Path],
    output: Path,
) -> dict[str, Any]:
    if len(repair_roots) != 3 or output.exists():
        raise SearchError("three repair roots and a fresh output are required")
    resolved_repair_roots = [root.resolve() for root in repair_roots]
    if len(set(resolved_repair_roots)) != 3:
        raise SearchError("repair roots must be distinct")
    previous = _verified_v1_freeze(previous_freeze_path, previous_freeze_sha256)
    centers = _v1_centers(previous)
    repair_medians: list[float] = []
    repair_bindings = []
    hidden, sequence, tp = REPAIR_CELL
    for ordinal, root in enumerate(repair_roots, 1):
        checksum = root / "sha256sums.txt"
        repair_bindings.append(
            {
                "outer_ordinal": ordinal,
                "root": str(root),
                "sha256sums_sha256": sha256(checksum),
            }
        )
        repair_medians.append(_verify_repair_root(root, ordinal))
    repair_center = float(median(repair_medians))
    repair_deviation = max(abs(value / repair_center - 1.0) for value in repair_medians) * 100.0
    accepted = repair_deviation <= STABILITY_GATE_PERCENT
    if accepted:
        centers[REPAIR_CELL] = repair_center
    action = "ESTIMATE" if accepted else "MEASURE"
    predictions = []
    for task_id, (target_hidden, target_sequence) in enumerate(TARGETS, 1):
        times = {
            candidate_tp: _bilinear(
                {
                    (h, s): centers[(h, s, candidate_tp)]
                    for h, s in CALIBRATION
                },
                target_hidden,
                target_sequence,
            )
            for candidate_tp, _ in CANDIDATES
        }
        predictions.append(
            {
                "task_id": task_id,
                "hidden_size": target_hidden,
                "sequence_length": target_sequence,
                "timing_action": action,
                "predicted_medians_us": {
                    str(candidate_tp): value for candidate_tp, value in times.items()
                },
                "selected_tp": min(times, key=times.get) if accepted else None,
                "required_measurement": None
                if accepted
                else {
                    "hidden_size": hidden,
                    "sequence_length": sequence,
                    "tensor_parallel_size": tp,
                    "maximum_relative_deviation_percent": repair_deviation,
                },
            }
        )
    core: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "status": "stored-before-v2-target-submission",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor": {
            "path": str(previous_freeze_path),
            "sha256": previous_freeze_sha256,
            "schema": V1_FREEZE_SCHEMA,
            "calibration_bindings": previous["calibration_bindings"],
        },
        "candidates": [
            {
                "tp": candidate_tp,
                "dp": dp,
                "micro_batch_size": candidate_tp,
                "global_batch_size": 4,
                "gpus": 4,
            }
            for candidate_tp, dp in CANDIDATES
        ],
        "repair": {
            "hidden_size": hidden,
            "sequence_length": sequence,
            "tensor_parallel_size": tp,
            "previous_center_us": _v1_centers(previous)[REPAIR_CELL],
            "new_outer_medians_us": repair_medians,
            "new_center_us": repair_center,
            "maximum_relative_deviation_percent": repair_deviation,
            "stability_gate_percent": STABILITY_GATE_PERCENT,
            "accepted": accepted,
            "bindings": repair_bindings,
        },
        "calibration_centers_us": [
            {
                "hidden_size": h,
                "sequence_length": s,
                "tensor_parallel_size": candidate_tp,
                "median_across_allocations_us": value,
                "source": "v2-repair" if (h, s, candidate_tp) == REPAIR_CELL and accepted else "v1-immutable",
            }
            for (h, s, candidate_tp), value in sorted(centers.items())
        ],
        "predictions": predictions,
        "target_observations_consumed": False,
    }
    document = {**core, "artifact_sha256": canonical_sha256(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def score(freeze_path: Path, target_roots: list[Path], output: Path) -> dict[str, Any]:
    freeze_document = load_json(freeze_path)
    core = {key: value for key, value in freeze_document.items() if key != "artifact_sha256"}
    if (
        freeze_document.get("schema") != FREEZE_SCHEMA
        or freeze_document.get("artifact_sha256") != canonical_sha256(core)
    ):
        raise SearchError("invalid v2 prediction freeze")
    if len(target_roots) != len(TARGETS):
        raise SearchError(f"{len(TARGETS)} target roots are required")
    if len({root.resolve() for root in target_roots}) != len(TARGETS):
        raise SearchError("target roots must be distinct")
    freeze_sha = sha256(freeze_path)
    rows = []
    target_bindings = []
    for task_id, root in enumerate(target_roots, 1):
        prediction = freeze_document["predictions"][task_id - 1]
        hidden, sequence = TARGETS[task_id - 1]
        target_bindings.append(
            {
                "task_id": task_id,
                "root": str(root),
                "sha256sums_sha256": sha256(root / "sha256sums.txt"),
            }
        )
        observed = _verify_target_root(root, task_id, hidden, sequence, freeze_sha)
        fastest = min(observed, key=observed.get)
        selected = prediction["selected_tp"]
        predicted = {
            int(key): float(value)
            for key, value in prediction["predicted_medians_us"].items()
        }
        signed = {
            tp: 100.0 * (predicted[tp] - observed[tp]) / observed[tp]
            for tp in observed
        }
        rows.append(
            {
                "task_id": task_id,
                "hidden_size": hidden,
                "sequence_length": sequence,
                "timing_action": prediction["timing_action"],
                "selected_tp": selected,
                "physically_fastest_tp": fastest,
                "correct_selection": selected == fastest if selected is not None else False,
                "regret_percent": None
                if selected is None
                else 100.0 * (observed[selected] / observed[fastest] - 1.0),
                "predicted_medians_us": {str(key): value for key, value in predicted.items()},
                "observed_medians_us": {str(key): value for key, value in observed.items()},
                "signed_errors_percent": {str(key): value for key, value in signed.items()},
                "absolute_errors_percent": {str(key): abs(value) for key, value in signed.items()},
            }
        )
    estimated = [row for row in rows if row["timing_action"] == "ESTIMATE"]
    absolute_errors = [
        value
        for row in estimated
        for value in row["absolute_errors_percent"].values()
    ]
    regrets = [float(row["regret_percent"]) for row in estimated]
    result = {
        "schema": SCORE_SCHEMA,
        "prediction_freeze_sha256": freeze_sha,
        "target_bindings": target_bindings,
        "tasks": rows,
        "summary": {
            "tasks": len(rows),
            "estimated_tasks": len(estimated),
            "measure_requests": len(rows) - len(estimated),
            "correct_selections": sum(row["correct_selection"] for row in estimated),
            "selection_accuracy_percent": None
            if not estimated
            else 100.0 * sum(row["correct_selection"] for row in estimated) / len(estimated),
            "median_regret_percent": median(regrets) if regrets else None,
            "maximum_regret_percent": max(regrets, default=None),
            "median_absolute_timing_error_percent": median(absolute_errors)
            if absolute_errors
            else None,
            "maximum_absolute_timing_error_percent": max(absolute_errors, default=None),
            "candidate_median_absolute_errors_percent": {
                str(tp): median(
                    row["absolute_errors_percent"][str(tp)] for row in estimated
                )
                if estimated
                else None
                for tp, _ in CANDIDATES
            },
            "physical_optimum_counts": {
                str(tp): sum(row["physically_fastest_tp"] == tp for row in rows)
                for tp, _ in CANDIDATES
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--previous-freeze", type=Path, required=True)
    freeze_parser.add_argument("--previous-freeze-sha256", required=True)
    freeze_parser.add_argument("--repair-root", type=Path, action="append", required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--freeze", type=Path, required=True)
    score_parser.add_argument("--target-root", type=Path, action="append", required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(
            args.previous_freeze,
            args.previous_freeze_sha256,
            args.repair_root,
            args.output,
        )
    else:
        score(args.freeze, args.target_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
