import json
from pathlib import Path

import pytest

from scaletether.research.fixed_resource_megatron_search_v8 import SCORE_SCHEMA
from scaletether.research.fixed_resource_megatron_search_v8_paper import (
    IntegrationError,
    build,
    write_package,
)


def _allocation(task: int, ordinal: int = 1) -> dict:
    estimate = task <= 40
    return {
        "task_id": task,
        "allocation_ordinal": ordinal,
        "cohort": "paper-grid-remainder" if task <= 3 else "seeded-off-grid-interior",
        "hidden_size": (768, 1024, 1280)[(task - 1) % 3],
        "sequence_length": 544 + 32 * ((task - 1) % 40),
        "hostname": f"r{(task + ordinal) % 6 + 1}n1",
        "action": "ESTIMATE" if estimate else "MEASURE",
        "selected_tp": 2 if estimate else None,
        "requested_tps": [] if estimate else [1, 2, 4],
        "physically_fastest_tp": 2,
        "medians_us": {"1": 110.0, "2": 100.0, "4": 105.0},
        "within_candidate_cv_percent": {"1": 1.0, "2": 1.5, "4": 2.0},
        "blind_log_selected_tp": 2,
        "blind_log_regret_percent": 0.0,
        "emitted_regret_percent": 0.0 if estimate else None,
        "predicted_fastest_margin_percent": float(task) / 10,
        "method_selected_tps": {
            "log-coordinate bilinear": 2,
            "raw-coordinate bilinear": 2 if estimate else 1,
            "nearest calibration corner": 2 if estimate else 4,
        },
    }


def _score(path: Path) -> None:
    primary = [_allocation(task) for task in range(1, 81)]
    repeats = [
        _allocation(task, ordinal) for ordinal in range(2, 6) for task in range(1, 11)
    ]
    policy_names = [
        "estimate-all-log-model",
        "log-raw-selection-agreement",
        "log-nearest-selection-agreement",
        "three-model-unanimity",
    ] + [f"policy-{index}" for index in range(16)]
    risk = [
        {
            "policy": name,
            "allocation_units": 80,
            "estimated_allocation_units": 40,
            "coverage_percent": 50.0,
            "above_one_percent_regret": 0,
            "selective_risk_percent": 0.0,
            "selective_risk_wilson_95_fraction": [0.0, 0.1],
            "exact_best": 40,
            "mean_regret_percent": 0.0,
            "maximum_regret_percent": 0.0,
        }
        for name in policy_names
    ]
    path.write_text(
        json.dumps(
            {
                "schema": SCORE_SCHEMA,
                "prediction_freeze_sha256": "a" * 64,
                "accounting_sha256": "b" * 64,
                "statistical_unit": "physical-node-allocation",
                "summary": {
                    "accepted": True,
                    "protocol_gates": {"all": True},
                    "risk_coverage_primary_allocation_units": risk,
                    "repeated_low_margin": {
                        "risk_coverage_by_allocation_unit": [
                            {**row, "allocation_units": 50} for row in risk
                        ]
                    },
                    "gpu_second_accounting": {
                        "direct_primary_grid_gpu_seconds": 40000.0,
                        "selective_requested_measurement_gpu_seconds": 20000.0,
                        "calibration_gpu_seconds": 5355.296,
                        "selective_cold_start_gpu_seconds": 25355.296,
                        "repeat_validation_gpu_seconds_not_charged_to_policy": 20000.0,
                        "total_experimental_gpu_seconds": 60000.0,
                    },
                },
                "primary_allocations": primary,
                "repeat_allocations": repeats,
            }
        )
        + "\n"
    )


def test_v8_paper_package_derives_cohorts_repeats_and_cost(tmp_path: Path) -> None:
    score = tmp_path / "score.json"
    _score(score)
    package = build(score)
    assert [row["queries"] for row in package["cohort_summary"]] == [80, 3, 77]
    assert package["cohort_summary"][0]["estimate_count"] == 40
    assert len(package["near_tie_stability"]) == 10
    assert all(row["allocation_units"] == 5 for row in package["near_tie_stability"])
    assert package["cost"][-1]["gpu_seconds"] == pytest.approx(14644.704)


def test_v8_paper_package_writes_complete_csv_set(tmp_path: Path) -> None:
    score = tmp_path / "score.json"
    _score(score)
    output = tmp_path / "package"
    write_package(score, output)
    assert {path.name for path in output.iterdir()} == {
        "paper-metrics.json",
        "cohort-summary.csv",
        "risk-coverage-primary.csv",
        "risk-coverage-near-tie.csv",
        "near-tie-stability.csv",
        "cost.csv",
        "primary-cases.csv",
    }
    with pytest.raises(IntegrationError, match="absent or empty"):
        write_package(score, output)


def test_v8_paper_package_writes_latex_from_score(tmp_path: Path) -> None:
    score = tmp_path / "score.json"
    _score(score)
    output = tmp_path / "package"
    latex = tmp_path / "latex"
    write_package(score, output, latex)
    table = (latex / "fixed-resource-megatron-search-v8-table.tex").read_text()
    summary = (latex / "fixed-resource-megatron-search-v8-summary.tex").read_text()
    assert "All held-out queries & 80 & 40/40" in table
    assert "Three-model unanimity & 40/80" in table
    assert "Across 80 held-out H100 queries" in summary


def test_v8_latex_exposes_failed_gate_and_handles_zero_estimates(
    tmp_path: Path,
) -> None:
    score = tmp_path / "score.json"
    _score(score)
    document = json.loads(score.read_text())
    document["summary"]["accepted"] = False
    document["summary"]["protocol_gates"] = {"all": False}
    for row in document["primary_allocations"]:
        row["action"] = "MEASURE"
        row["selected_tp"] = None
        row["emitted_regret_percent"] = None
    score.write_text(json.dumps(document) + "\n")
    write_package(score, tmp_path / "package", tmp_path / "latex")
    summary = (
        tmp_path / "latex/fixed-resource-megatron-search-v8-summary.tex"
    ).read_text()
    assert "At least one predeclared protocol gate failed" in summary
    assert "no query received an ESTIMATE decision" in summary


def test_v8_paper_package_rejects_incomplete_primary_rows(tmp_path: Path) -> None:
    score = tmp_path / "score.json"
    _score(score)
    document = json.loads(score.read_text())
    document["primary_allocations"].pop()
    score.write_text(json.dumps(document))
    with pytest.raises(IntegrationError, match="invalid V8 score shape"):
        build(score)
