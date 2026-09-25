"""Freeze, account for, and score the expanded selective H100 study V8.

The prediction and action gate are unchanged from V7.  V8 expands the
target-blind domain and treats a physical allocation, rather than an
individual timing repetition, as the statistical unit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
from statistics import mean, median, stdev
from typing import Any, Callable

import scaletether.research.fixed_resource_megatron_search_v4 as v4
import scaletether.research.fixed_resource_megatron_search_v5 as v5
import scaletether.research.fixed_resource_megatron_search_v6 as v6
import scaletether.research.fixed_resource_megatron_search_v7 as v7


FREEZE_SCHEMA = "scaletether-fixed-resource-megatron-selective-freeze-v8"
ACCOUNTING_SCHEMA = "scaletether-fixed-resource-megatron-selective-accounting-v8"
SCORE_SCHEMA = "scaletether-fixed-resource-megatron-selective-score-v8"
TPS = (1, 2, 4)
OFF_GRID_COUNT = 77
OFF_GRID_SEED = 20260810
REPEATED_TARGET_COUNT = 10
REPEAT_ALLOCATIONS_PER_TARGET = 4
PRIMARY_TARGET_COUNT = 80
REPEAT_ALLOCATION_COUNT = REPEATED_TARGET_COUNT * REPEAT_ALLOCATIONS_PER_TARGET
CALIBRATION_H100_GPU_SECONDS = 5355.296
RISK_THRESHOLD_PERCENT = 1.0
MARGIN_THRESHOLDS_PERCENT = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

# These coordinates are part of the public V7 freeze.  Listing them here keeps
# V8 target selection independent of V7 outcome values.
V7_COORDINATES = frozenset(
    {
        (768, 1152), (1024, 1792), (768, 640), (768, 1408),
        (1280, 896), (1280, 1152), (1024, 896), (768, 896),
        (1280, 1408), (768, 1792), (1024, 1664), (1280, 1792),
        (1280, 1920), (1280, 1664), (1024, 1408), (1024, 1152),
        (1024, 1920), (768, 1664),
    }
)
ALL_PREVIOUS_COORDINATES = v7.PREVIOUSLY_MEASURED | V7_COORDINATES
PAPER_GRID = frozenset(
    (hidden, sequence)
    for hidden in v7.HIDDEN_GRID
    for sequence in v7.SEQUENCE_GRID
)
PAPER_GRID_REMAINDER = tuple(sorted(PAPER_GRID - ALL_PREVIOUS_COORDINATES))
OFF_GRID_POOL = tuple(
    (hidden, sequence)
    for hidden in v7.HIDDEN_GRID
    for sequence in range(544, 2017, 32)
    if sequence not in v7.SEQUENCE_GRID
    and (hidden, sequence) not in ALL_PREVIOUS_COORDINATES
)


def _methods(
    corners: dict[tuple[int, int, int], float], hidden: int, sequence: int
) -> dict[str, dict[int, float]]:
    return v7._methods(corners, hidden, sequence)


def _candidate(
    corners: dict[tuple[int, int, int], float], hidden: int, sequence: int
) -> dict[str, Any]:
    methods = _methods(corners, hidden, sequence)
    selections = {name: v6._selection(ratios) for name, ratios in methods.items()}
    log_times = sorted(methods["log-coordinate bilinear"].values())
    unanimous = len(set(selections.values())) == 1
    return {
        "hidden_size": hidden,
        "sequence_length": sequence,
        "action": "ESTIMATE" if unanimous else "MEASURE",
        "selected_tp": selections["log-coordinate bilinear"] if unanimous else None,
        "requested_tps": [] if unanimous else list(TPS),
        "model_selection_count": len(set(selections.values())),
        "predicted_fastest_margin_percent": 100.0 * (log_times[1] / log_times[0] - 1.0),
        "methods": {
            name: {
                "selected_tp": selections[name],
                "predicted_ratios_to_tp1": {
                    str(tp): value for tp, value in ratios.items()
                },
            }
            for name, ratios in methods.items()
        },
    }


def freeze(v4_freeze_path: Path, v4_freeze_sha256: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise v4.SearchError("fresh V8 freeze output is required")
    source = v5._verified_v4_freeze(v4_freeze_path, v4_freeze_sha256)
    corners = v5._corner_logs(source)
    rng = random.Random(OFF_GRID_SEED)
    off_grid_coordinates = rng.sample(list(OFF_GRID_POOL), OFF_GRID_COUNT)
    rows: list[dict[str, Any]] = []
    for cohort, coordinates in (
        ("paper-grid-remainder", PAPER_GRID_REMAINDER),
        ("seeded-off-grid-interior", off_grid_coordinates),
    ):
        for hidden, sequence in coordinates:
            rows.append({"cohort": cohort, **_candidate(corners, hidden, sequence)})
    if len(rows) != PRIMARY_TARGET_COUNT:
        raise v4.SearchError("V8 target construction did not produce 80 queries")
    targets = [{"task_id": index, **row} for index, row in enumerate(rows, 1)]
    repeated = sorted(
        targets,
        key=lambda row: (
            float(row["predicted_fastest_margin_percent"]),
            int(row["hidden_size"]),
            int(row["sequence_length"]),
        ),
    )[:REPEATED_TARGET_COUNT]
    repeat_allocations: list[dict[str, int]] = []
    for allocation_ordinal in range(2, 2 + REPEAT_ALLOCATIONS_PER_TARGET):
        for target in repeated:
            repeat_allocations.append(
                {
                    "repeat_task_id": len(repeat_allocations) + 1,
                    "source_task_id": int(target["task_id"]),
                    "allocation_ordinal": allocation_ordinal,
                }
            )
    core = {
        "schema": FREEZE_SCHEMA,
        "status": "stored-before-v8-target-submission",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "v4_prediction_freeze_sha256": v4_freeze_sha256,
        "v7_public_freeze_coordinates_only": [list(x) for x in sorted(V7_COORDINATES)],
        "target_observations_consumed": False,
        "gate": {
            "name": "unchanged-v6-v7-three-model-unanimity",
            "estimate_rule": "all-three-model-selections-equal",
            "risk_threshold_percent": RISK_THRESHOLD_PERCENT,
        },
        "sampling": {
            "paper_grid_rule": "all-unmeasured-admissible-paper-grid-coordinates",
            "paper_grid_remainder": [list(x) for x in PAPER_GRID_REMAINDER],
            "off_grid_rule": "python-random-sample-without-replacement",
            "off_grid_seed": OFF_GRID_SEED,
            "off_grid_pool_size": len(OFF_GRID_POOL),
            "off_grid_count": OFF_GRID_COUNT,
            "hidden_grid": list(v7.HIDDEN_GRID),
            "off_grid_sequence_pool": sorted({sequence for _, sequence in OFF_GRID_POOL}),
            "previously_measured_coordinates": [list(x) for x in sorted(ALL_PREVIOUS_COORDINATES)],
        },
        "repeated_allocation_design": {
            "selection_rule": "ten-smallest-frozen-log-model-margins-then-coordinate",
            "targets": REPEATED_TARGET_COUNT,
            "additional_allocations_per_target": REPEAT_ALLOCATIONS_PER_TARGET,
            "statistical_unit": "physical-node-allocation",
        },
        "risk_coverage_design": {
            "risk_definition": "fraction-of-estimated-allocation-units-above-one-percent-regret",
            "margin_thresholds_percent": list(MARGIN_THRESHOLDS_PERCENT),
            "baseline_gates": [
                "estimate-all-log-model",
                "log-raw-selection-agreement",
                "log-nearest-selection-agreement",
                "three-model-unanimity",
            ],
        },
        "targets": targets,
        "repeat_allocations": repeat_allocations,
    }
    document = {**core, "artifact_sha256": v4.canonical_sha256(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _parse_qacct_records(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for block in re.split(r"^=+\s*$", text, flags=re.MULTILINE):
        values: dict[str, str] = {}
        for line in block.splitlines():
            match = re.match(r"^(\S+)\s+(.*?)\s*$", line)
            if match:
                values[match.group(1)] = match.group(2)
        if "jobnumber" in values:
            records.append(values)
    return records


def _qacct_allocations(
    path: Path,
    expected_job_name: str,
    expected_count: int,
    kind: str,
    freeze_document: dict[str, Any],
) -> list[dict[str, Any]]:
    records = _parse_qacct_records(path.read_text(encoding="utf-8"))
    if len(records) != expected_count:
        raise v4.SearchError(f"expected {expected_count} {kind} qacct records")
    output: list[dict[str, Any]] = []
    for record in records:
        task_id = int(record.get("taskid", "0"))
        if (
            record.get("jobname") != expected_job_name
            or record.get("group") != "hp190122"
            or record.get("failed") != "0"
            or record.get("exit_status") != "0"
            or not (1 <= task_id <= expected_count)
            or "h_rt=300" not in record.get("hard_resources", "")
        ):
            raise v4.SearchError(f"invalid {kind} qacct record for task {task_id}")
        ru_wallclock = float(record["ru_wallclock"])
        if not (0.0 < ru_wallclock <= 360.0):
            raise v4.SearchError("implausible qacct ru_wallclock")
        row: dict[str, Any] = {
            "kind": kind,
            "scheduler_job_id": int(record["jobnumber"]),
            "scheduler_task_id": task_id,
            "hostname": record["hostname"],
            "ru_wallclock_seconds": ru_wallclock,
            "gpus": 4,
            "gpu_seconds": 4.0 * ru_wallclock,
        }
        if kind == "primary":
            row.update({"source_task_id": task_id, "allocation_ordinal": 1})
        else:
            mapping = freeze_document["repeat_allocations"][task_id - 1]
            row.update(
                {
                    "source_task_id": int(mapping["source_task_id"]),
                    "allocation_ordinal": int(mapping["allocation_ordinal"]),
                }
            )
        output.append(row)
    output.sort(key=lambda row: int(row["scheduler_task_id"]))
    if len({row["scheduler_task_id"] for row in output}) != expected_count:
        raise v4.SearchError(f"duplicate {kind} qacct task")
    return output


def accounting(
    freeze_path: Path, primary_qacct: Path, repeat_qacct: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise v4.SearchError("fresh V8 accounting output is required")
    freeze_document = v4.load_json(freeze_path)
    _verify_freeze_document(freeze_document)
    allocations = _qacct_allocations(
        primary_qacct, "scaletether_fixed4_target_v8", PRIMARY_TARGET_COUNT, "primary", freeze_document
    ) + _qacct_allocations(
        repeat_qacct, "scaletether_fixed4_repeat_v8", REPEAT_ALLOCATION_COUNT, "repeat", freeze_document
    )
    core = {
        "schema": ACCOUNTING_SCHEMA,
        "source": "TSUBAME-UGE-qacct-ru_wallclock",
        "prediction_freeze_sha256": v4.sha256(freeze_path),
        "primary_qacct_sha256": v4.sha256(primary_qacct),
        "repeat_qacct_sha256": v4.sha256(repeat_qacct),
        "calibration_h100_gpu_seconds": CALIBRATION_H100_GPU_SECONDS,
        "allocations": allocations,
    }
    document = {**core, "artifact_sha256": v4.canonical_sha256(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _verify_freeze_document(document: dict[str, Any]) -> None:
    core = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("schema") != FREEZE_SCHEMA
        or document.get("artifact_sha256") != v4.canonical_sha256(core)
        or document.get("target_observations_consumed") is not False
        or len(document.get("targets", [])) != PRIMARY_TARGET_COUNT
        or len(document.get("repeat_allocations", [])) != REPEAT_ALLOCATION_COUNT
    ):
        raise v4.SearchError("invalid V8 freeze")


def _verify_target(
    root: Path,
    frozen: dict[str, Any],
    freeze_sha256: str,
    allocation_ordinal: int,
    scheduler_task_id: int,
    kind: str,
) -> tuple[str, dict[int, float], dict[int, float]]:
    v4._verify_result_manifest(root)
    v4._verify_completion(root, f"scaletether-fixed-resource-megatron-selective-{kind}-complete-v8")
    v4._verify_h100_hardware(root)
    if v4.sha256(root / "prediction-freeze.json") != freeze_sha256:
        raise v4.SearchError("V8 target freeze changed")
    identity = v4.load_json(root / "runtime-identity.json")
    hostname = identity.get("hostname")
    if identity.get("schema") != "scaletether-fixed-resource-megatron-runtime-v8" or not hostname:
        raise v4.SearchError("invalid V8 runtime identity")
    task_id = int(frozen["task_id"])
    hidden = int(frozen["hidden_size"])
    sequence = int(frozen["sequence_length"])
    medians: dict[int, float] = {}
    cvs: dict[int, float] = {}
    shift = (task_id + allocation_ordinal - 2) % len(TPS)
    order = list(TPS[shift:] + TPS[:shift])
    for launch, tp in enumerate(order, 1):
        dp = 4 // tp
        endpoint = root / "transformer" / f"H{hidden}-S{sequence}-TP{tp}-DP{dp}"
        expected = {
            "schema": "scaletether-fixed-resource-megatron-run-v8",
            "phase": "expanded-selective-target" if kind == "target" else "low-margin-repeat-allocation",
            "task_id": task_id,
            "scheduler_task_id": scheduler_task_id,
            "allocation_ordinal": allocation_ordinal,
            "cohort": frozen["cohort"],
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
        }
        if v4.load_json(endpoint / "run-contract.json") != expected:
            raise v4.SearchError("V8 target contract mismatch")
        timing = v4.load_json(endpoint / "timing.json")
        values = [float(value) for value in timing.get("iteration_device_envelope_us", [])]
        if timing.get("repetition_count") != 20 or len(values) != 20:
            raise v4.SearchError("V8 timing repetitions are incomplete")
        medians[tp] = float(median(values))
        cvs[tp] = 100.0 * stdev(values) / mean(values)
    return str(hostname), medians, cvs


def _allocation_row(
    frozen: dict[str, Any], hostname: str, medians: dict[int, float], cvs: dict[int, float], allocation_ordinal: int
) -> dict[str, Any]:
    fastest = min(TPS, key=medians.__getitem__)
    blind_tp = int(frozen["methods"]["log-coordinate bilinear"]["selected_tp"])
    blind_regret = 100.0 * (medians[blind_tp] / medians[fastest] - 1.0)
    selected = frozen["selected_tp"]
    emitted_regret = None if selected is None else 100.0 * (medians[int(selected)] / medians[fastest] - 1.0)
    return {
        "task_id": int(frozen["task_id"]),
        "allocation_ordinal": allocation_ordinal,
        "cohort": frozen["cohort"],
        "hidden_size": int(frozen["hidden_size"]),
        "sequence_length": int(frozen["sequence_length"]),
        "hostname": hostname,
        "action": frozen["action"],
        "selected_tp": selected,
        "requested_tps": frozen["requested_tps"],
        "physically_fastest_tp": fastest,
        "medians_us": {str(key): value for key, value in medians.items()},
        "within_candidate_cv_percent": {str(key): value for key, value in cvs.items()},
        "blind_log_selected_tp": blind_tp,
        "blind_log_regret_percent": blind_regret,
        "emitted_regret_percent": emitted_regret,
        "predicted_fastest_margin_percent": frozen["predicted_fastest_margin_percent"],
        "method_selected_tps": {
            name: int(method["selected_tp"]) for name, method in frozen["methods"].items()
        },
    }


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _policy_metrics(
    rows: list[dict[str, Any]], name: str, accepts: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    retained = [row for row in rows if accepts(row)]
    regrets = [float(row["blind_log_regret_percent"]) for row in retained]
    violations = sum(regret > RISK_THRESHOLD_PERCENT for regret in regrets)
    return {
        "policy": name,
        "allocation_units": len(rows),
        "estimated_allocation_units": len(retained),
        "coverage_percent": 100.0 * len(retained) / len(rows) if rows else 0.0,
        "above_one_percent_regret": violations,
        "selective_risk_percent": 100.0 * violations / len(retained) if retained else None,
        "selective_risk_wilson_95_fraction": _wilson_interval(violations, len(retained)),
        "exact_best": sum(row["blind_log_selected_tp"] == row["physically_fastest_tp"] for row in retained),
        "mean_regret_percent": mean(regrets) if regrets else None,
        "maximum_regret_percent": max(regrets) if regrets else None,
    }


def _risk_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def agreement(row: dict[str, Any], names: tuple[str, ...]) -> bool:
        return len({row["method_selected_tps"][name] for name in names}) == 1

    policies = [
        _policy_metrics(rows, "estimate-all-log-model", lambda row: True),
        _policy_metrics(
            rows,
            "log-raw-selection-agreement",
            lambda row: agreement(row, ("log-coordinate bilinear", "raw-coordinate bilinear")),
        ),
        _policy_metrics(
            rows,
            "log-nearest-selection-agreement",
            lambda row: agreement(row, ("log-coordinate bilinear", "nearest calibration corner")),
        ),
        _policy_metrics(rows, "three-model-unanimity", lambda row: row["action"] == "ESTIMATE"),
    ]
    for threshold in MARGIN_THRESHOLDS_PERCENT:
        policies.append(
            _policy_metrics(
                rows,
                f"log-margin-at-least-{threshold:g}-percent",
                lambda row, threshold=threshold: float(row["predicted_fastest_margin_percent"]) >= threshold,
            )
        )
        policies.append(
            _policy_metrics(
                rows,
                f"unanimity-and-log-margin-at-least-{threshold:g}-percent",
                lambda row, threshold=threshold: row["action"] == "ESTIMATE"
                and float(row["predicted_fastest_margin_percent"]) >= threshold,
            )
        )
    return policies


def _verify_accounting(
    document: dict[str, Any], freeze_sha256: str
) -> dict[tuple[str, int], dict[str, Any]]:
    core = {key: value for key, value in document.items() if key != "artifact_sha256"}
    allocations = document.get("allocations", [])
    if (
        document.get("schema") != ACCOUNTING_SCHEMA
        or document.get("artifact_sha256") != v4.canonical_sha256(core)
        or document.get("prediction_freeze_sha256") != freeze_sha256
        or len(allocations) != PRIMARY_TARGET_COUNT + REPEAT_ALLOCATION_COUNT
    ):
        raise v4.SearchError("invalid V8 accounting")
    keyed = {(str(row["kind"]), int(row["scheduler_task_id"])): row for row in allocations}
    if len(keyed) != len(allocations):
        raise v4.SearchError("duplicate V8 accounting allocation")
    return keyed


def score(
    freeze_path: Path,
    target_roots: list[Path],
    repeat_roots: list[Path],
    accounting_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise v4.SearchError("fresh V8 score output is required")
    freeze_document = v4.load_json(freeze_path)
    _verify_freeze_document(freeze_document)
    if (
        len(target_roots) != PRIMARY_TARGET_COUNT
        or len(repeat_roots) != REPEAT_ALLOCATION_COUNT
        or len({root.resolve() for root in target_roots + repeat_roots})
        != PRIMARY_TARGET_COUNT + REPEAT_ALLOCATION_COUNT
    ):
        raise v4.SearchError("V8 requires 80 primary and 40 repeat roots")
    freeze_sha256 = v4.sha256(freeze_path)
    accounting_document = v4.load_json(accounting_path)
    accounting_rows = _verify_accounting(accounting_document, freeze_sha256)
    targets_by_id = {int(row["task_id"]): row for row in freeze_document["targets"]}
    primary_rows: list[dict[str, Any]] = []
    for scheduler_task_id, root in enumerate(target_roots, 1):
        frozen = targets_by_id[scheduler_task_id]
        host, medians, cvs = _verify_target(
            root, frozen, freeze_sha256, 1, scheduler_task_id, "target"
        )
        accounting_row = accounting_rows[("primary", scheduler_task_id)]
        if host != accounting_row["hostname"]:
            raise v4.SearchError("primary qacct hostname mismatch")
        primary_rows.append(_allocation_row(frozen, host, medians, cvs, 1))
    repeat_rows: list[dict[str, Any]] = []
    for scheduler_task_id, (root, mapping) in enumerate(
        zip(repeat_roots, freeze_document["repeat_allocations"]), 1
    ):
        frozen = targets_by_id[int(mapping["source_task_id"])]
        ordinal = int(mapping["allocation_ordinal"])
        host, medians, cvs = _verify_target(
            root, frozen, freeze_sha256, ordinal, scheduler_task_id, "repeat"
        )
        accounting_row = accounting_rows[("repeat", scheduler_task_id)]
        if (
            host != accounting_row["hostname"]
            or ordinal != int(accounting_row["allocation_ordinal"])
            or int(frozen["task_id"]) != int(accounting_row["source_task_id"])
        ):
            raise v4.SearchError("repeat qacct binding mismatch")
        repeat_rows.append(_allocation_row(frozen, host, medians, cvs, ordinal))
    estimates = [row for row in primary_rows if row["action"] == "ESTIMATE"]
    harmful = [row for row in primary_rows if row["blind_log_regret_percent"] > RISK_THRESHOLD_PERCENT]
    repeated_task_ids = {int(row["source_task_id"]) for row in freeze_document["repeat_allocations"]}
    repeated_units = [row for row in primary_rows + repeat_rows if row["task_id"] in repeated_task_ids]
    by_task: dict[int, list[dict[str, Any]]] = {}
    for row in repeated_units:
        by_task.setdefault(int(row["task_id"]), []).append(row)
    repeated_summary = {
        "targets": len(by_task),
        "allocation_units": len(repeated_units),
        "five_allocations_per_target": all(len(rows) == 5 for rows in by_task.values()),
        "physical_optimum_agreement_all_five": sum(
            len({row["physically_fastest_tp"] for row in rows}) == 1 for rows in by_task.values()
        ),
        "frozen_action_consistent_by_construction": True,
        "risk_coverage_by_allocation_unit": _risk_coverage(repeated_units),
    }
    direct_gpu_seconds = sum(
        float(row["gpu_seconds"])
        for row in accounting_document["allocations"]
        if row["kind"] == "primary"
    )
    requested_gpu_seconds = sum(
        float(accounting_rows[("primary", int(row["task_id"]))]["gpu_seconds"])
        for row in primary_rows
        if row["action"] == "MEASURE"
    )
    repeat_gpu_seconds = sum(
        float(row["gpu_seconds"])
        for row in accounting_document["allocations"]
        if row["kind"] == "repeat"
    )
    summary = {
        "primary_queries": len(primary_rows),
        "primary_allocation_units": len(primary_rows),
        "estimate_count": len(estimates),
        "measure_count": len(primary_rows) - len(estimates),
        "estimate_coverage_percent": 100.0 * len(estimates) / len(primary_rows),
        "emitted_exact_best": sum(row["selected_tp"] == row["physically_fastest_tp"] for row in estimates),
        "emitted_within_one_percent": sum(float(row["emitted_regret_percent"]) <= RISK_THRESHOLD_PERCENT for row in estimates),
        "maximum_emitted_regret_percent": max((float(row["emitted_regret_percent"]) for row in estimates), default=None),
        "unconditional_above_one_percent": len(harmful),
        "unconditional_above_one_percent_withheld": sum(row["action"] == "MEASURE" for row in harmful),
        "distinct_primary_hosts": sorted({row["hostname"] for row in primary_rows}),
        "risk_coverage_primary_allocation_units": _risk_coverage(primary_rows),
        "repeated_low_margin": repeated_summary,
        "gpu_second_accounting": {
            "source": "TSUBAME-UGE-qacct-ru_wallclock-times-four-H100s",
            "calibration_gpu_seconds": CALIBRATION_H100_GPU_SECONDS,
            "direct_primary_grid_gpu_seconds": direct_gpu_seconds,
            "selective_requested_measurement_gpu_seconds": requested_gpu_seconds,
            "selective_cold_start_gpu_seconds": CALIBRATION_H100_GPU_SECONDS + requested_gpu_seconds,
            "marginal_acquisition_reduction_percent": 100.0 * (1.0 - requested_gpu_seconds / direct_gpu_seconds),
            "repeat_validation_gpu_seconds_not_charged_to_policy": repeat_gpu_seconds,
            "total_experimental_gpu_seconds": direct_gpu_seconds + repeat_gpu_seconds,
        },
    }
    summary["protocol_gates"] = {
        "at_least_three_primary_hosts": len(summary["distinct_primary_hosts"]) >= 3,
        "all_estimates_within_one_percent": summary["emitted_within_one_percent"] == summary["estimate_count"],
        "all_unconditional_above_one_percent_withheld": summary["unconditional_above_one_percent_withheld"] == summary["unconditional_above_one_percent"],
        "repeat_design_complete": repeated_summary["five_allocations_per_target"],
    }
    summary["accepted"] = all(summary["protocol_gates"].values())
    result = {
        "schema": SCORE_SCHEMA,
        "prediction_freeze_sha256": freeze_sha256,
        "accounting_sha256": v4.sha256(accounting_path),
        "statistical_unit": "physical-node-allocation",
        "summary": summary,
        "primary_allocations": primary_rows,
        "repeat_allocations": repeat_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("freeze")
    command.add_argument("--v4-freeze", type=Path, required=True)
    command.add_argument("--v4-freeze-sha256", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("accounting")
    command.add_argument("--freeze", type=Path, required=True)
    command.add_argument("--primary-qacct", type=Path, required=True)
    command.add_argument("--repeat-qacct", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("score")
    command.add_argument("--freeze", type=Path, required=True)
    command.add_argument("--target-root", action="append", type=Path, required=True)
    command.add_argument("--repeat-root", action="append", type=Path, required=True)
    command.add_argument("--accounting", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.v4_freeze, args.v4_freeze_sha256, args.output)
    elif args.command == "accounting":
        accounting(args.freeze, args.primary_qacct, args.repeat_qacct, args.output)
    else:
        score(args.freeze, args.target_root, args.repeat_root, args.accounting, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
