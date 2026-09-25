"""Derive reviewer-facing V8 tables from the immutable score artifact.

This module does not alter the stored predictor or score.  It converts the
allocation-level rows already present in ``score.json`` into compact,
auditable tables for manuscript integration and the reviewer artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from scaletether.research.fixed_resource_megatron_search_v8 import SCORE_SCHEMA


PACKAGE_SCHEMA = "scaletether-fixed-resource-megatron-selective-paper-package-v8"


class IntegrationError(RuntimeError):
    """Raised when a score cannot support the declared paper tables."""


def _cohort_summary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimates = [row for row in rows if row["action"] == "ESTIMATE"]
    harmful = [row for row in rows if float(row["blind_log_regret_percent"]) > 1.0]
    emitted = [float(row["emitted_regret_percent"]) for row in estimates]
    return {
        "cohort": name,
        "queries": len(rows),
        "estimate_count": len(estimates),
        "measure_count": len(rows) - len(estimates),
        "estimate_coverage_percent": 100.0 * len(estimates) / len(rows),
        "estimate_exact_best": sum(
            row["selected_tp"] == row["physically_fastest_tp"] for row in estimates
        ),
        "estimate_within_one_percent": sum(value <= 1.0 for value in emitted),
        "median_estimate_regret_percent": median(emitted) if emitted else None,
        "maximum_estimate_regret_percent": max(emitted) if emitted else None,
        "unconditional_above_one_percent": len(harmful),
        "unconditional_above_one_percent_withheld": sum(
            row["action"] == "MEASURE" for row in harmful
        ),
        "conservative_measure_count": sum(
            row["action"] == "MEASURE" and float(row["blind_log_regret_percent"]) <= 1.0
            for row in rows
        ),
        "physical_optimum_counts": {
            str(tp): sum(int(row["physically_fastest_tp"]) == tp for row in rows)
            for tp in (1, 2, 4)
        },
        "maximum_candidate_cv_percent": max(
            float(value)
            for row in rows
            for value in row["within_candidate_cv_percent"].values()
        ),
        "distinct_hosts": sorted({str(row["hostname"]) for row in rows}),
    }


def _near_tie_rows(
    primary: list[dict[str, Any]], repeats: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    primary_by_task = {int(row["task_id"]): row for row in primary}
    repeat_task_ids = sorted({int(row["task_id"]) for row in repeats})
    output: list[dict[str, Any]] = []
    for task_id in repeat_task_ids:
        units = [primary_by_task[task_id]] + [
            row for row in repeats if int(row["task_id"]) == task_id
        ]
        units.sort(key=lambda row: int(row["allocation_ordinal"]))
        if len(units) != 5 or [int(row["allocation_ordinal"]) for row in units] != [
            1,
            2,
            3,
            4,
            5,
        ]:
            raise IntegrationError(f"task {task_id} does not have five allocations")
        frozen = primary_by_task[task_id]
        optima = [int(row["physically_fastest_tp"]) for row in units]
        regrets = [float(row["blind_log_regret_percent"]) for row in units]
        output.append(
            {
                "task_id": task_id,
                "hidden_size": int(frozen["hidden_size"]),
                "sequence_length": int(frozen["sequence_length"]),
                "action": frozen["action"],
                "blind_log_selected_tp": int(frozen["blind_log_selected_tp"]),
                "predicted_fastest_margin_percent": float(
                    frozen["predicted_fastest_margin_percent"]
                ),
                "allocation_units": 5,
                "distinct_hosts": sorted({str(row["hostname"]) for row in units}),
                "physical_optimum_votes": {
                    str(tp): sum(value == tp for value in optima) for tp in (1, 2, 4)
                },
                "physical_optimum_unanimous": len(set(optima)) == 1,
                "blind_exact_best_allocations": sum(
                    int(frozen["blind_log_selected_tp"]) == optimum
                    for optimum in optima
                ),
                "blind_within_one_percent_allocations": sum(
                    value <= 1.0 for value in regrets
                ),
                "mean_blind_regret_percent": mean(regrets),
                "maximum_blind_regret_percent": max(regrets),
                "allocation_regrets_percent": regrets,
            }
        )
    if len(output) != 10:
        raise IntegrationError("near-tie table requires ten frozen targets")
    output.sort(
        key=lambda row: (
            float(row["predicted_fastest_margin_percent"]),
            int(row["hidden_size"]),
            int(row["sequence_length"]),
        )
    )
    return output


def _cost_rows(accounting: dict[str, Any]) -> list[dict[str, Any]]:
    direct = float(accounting["direct_primary_grid_gpu_seconds"])
    requested = float(accounting["selective_requested_measurement_gpu_seconds"])
    calibration = float(accounting["calibration_gpu_seconds"])
    cold = float(accounting["selective_cold_start_gpu_seconds"])
    repeat = float(accounting["repeat_validation_gpu_seconds_not_charged_to_policy"])
    total = float(accounting["total_experimental_gpu_seconds"])
    return [
        {"quantity": "direct-primary-acquisition", "gpu_seconds": direct},
        {"quantity": "selective-requested-acquisition", "gpu_seconds": requested},
        {"quantity": "shared-calibration", "gpu_seconds": calibration},
        {"quantity": "selective-cold-start", "gpu_seconds": cold},
        {"quantity": "repeat-validation-not-policy", "gpu_seconds": repeat},
        {"quantity": "total-v8-experiment", "gpu_seconds": total},
        {
            "quantity": "warm-policy-savings",
            "gpu_seconds": direct - requested,
            "percent_of_direct": 100.0 * (direct - requested) / direct,
        },
        {
            "quantity": "cold-start-net-savings",
            "gpu_seconds": direct - cold,
            "percent_of_direct": 100.0 * (direct - cold) / direct,
        },
    ]


def _near_tie_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    allocation_units = sum(int(row["allocation_units"]) for row in rows)
    within_one = sum(int(row["blind_within_one_percent_allocations"]) for row in rows)
    estimates = [row for row in rows if row["action"] == "ESTIMATE"]
    estimate_units = sum(int(row["allocation_units"]) for row in estimates)
    hosts = {host for row in rows for host in row["distinct_hosts"]}
    return {
        "targets": len(rows),
        "allocation_units": allocation_units,
        "unanimous_physical_optimum_targets": sum(
            bool(row["physical_optimum_unanimous"]) for row in rows
        ),
        "blind_within_one_percent_allocations": within_one,
        "estimate_targets": len(estimates),
        "estimate_allocation_units": estimate_units,
        "estimate_exact_best_allocations": sum(
            int(row["blind_exact_best_allocations"]) for row in estimates
        ),
        "estimate_within_one_percent_allocations": sum(
            int(row["blind_within_one_percent_allocations"]) for row in estimates
        ),
        "maximum_estimate_allocation_regret_percent": max(
            (float(row["maximum_blind_regret_percent"]) for row in estimates),
            default=None,
        ),
        "maximum_allocation_regret_percent": max(
            float(row["maximum_blind_regret_percent"]) for row in rows
        ),
        "distinct_hosts": sorted(hosts),
    }


def build(score_path: Path) -> dict[str, Any]:
    score = json.loads(score_path.read_text(encoding="utf-8"))
    primary = score.get("primary_allocations", [])
    repeats = score.get("repeat_allocations", [])
    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("statistical_unit") != "physical-node-allocation"
        or len(primary) != 80
        or len(repeats) != 40
        or [int(row.get("task_id", 0)) for row in primary] != list(range(1, 81))
    ):
        raise IntegrationError("invalid V8 score shape")
    cohorts = {
        "all": primary,
        "paper-grid-remainder": [
            row for row in primary if row["cohort"] == "paper-grid-remainder"
        ],
        "seeded-off-grid-interior": [
            row for row in primary if row["cohort"] == "seeded-off-grid-interior"
        ],
    }
    if (
        len(cohorts["paper-grid-remainder"]) != 3
        or len(cohorts["seeded-off-grid-interior"]) != 77
    ):
        raise IntegrationError("V8 cohorts do not match the frozen design")
    risk = score["summary"]["risk_coverage_primary_allocation_units"]
    repeated_risk = score["summary"]["repeated_low_margin"][
        "risk_coverage_by_allocation_unit"
    ]
    if len(risk) != 20 or len(repeated_risk) != 20:
        raise IntegrationError("risk-coverage policy matrix is incomplete")
    near_ties = _near_tie_rows(primary, repeats)
    return {
        "schema": PACKAGE_SCHEMA,
        "source_score": str(score_path),
        "prediction_freeze_sha256": score["prediction_freeze_sha256"],
        "accounting_sha256": score["accounting_sha256"],
        "statistical_unit": score["statistical_unit"],
        "protocol_gates": score["summary"]["protocol_gates"],
        "accepted": score["summary"]["accepted"],
        "cohort_summary": [
            _cohort_summary(name, rows) for name, rows in cohorts.items()
        ],
        "risk_coverage_primary": risk,
        "risk_coverage_near_tie_allocations": repeated_risk,
        "near_tie_stability": near_ties,
        "near_tie_summary": _near_tie_summary(near_ties),
        "cost": _cost_rows(score["summary"]["gpu_second_accounting"]),
        "primary_cases": primary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise IntegrationError(f"cannot write empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}\\%"


def _integer_tex(value: float) -> str:
    return f"{value:,.0f}".replace(",", "{,}")


def _write_latex(package: dict[str, Any], output_directory: Path) -> None:
    """Write the compact manuscript table from the immutable paper package."""

    output_directory.mkdir(parents=True, exist_ok=True)
    cohorts = {row["cohort"]: row for row in package["cohort_summary"]}
    all_rows = cohorts["all"]
    costs = {row["quantity"]: row for row in package["cost"]}
    near_ties = package["near_tie_summary"]
    near_tie_estimate_max = (
        "--"
        if near_ties["maximum_estimate_allocation_regret_percent"] is None
        else f"{float(near_ties['maximum_estimate_allocation_regret_percent']):.2f}"
    )
    policies = {row["policy"]: row for row in package["risk_coverage_primary"]}
    policy_labels = (
        ("estimate-all-log-model", "Log model (no gate)"),
        ("log-raw-selection-agreement", "Log/raw agreement"),
        ("log-nearest-selection-agreement", "Log/nearest agreement"),
        ("three-model-unanimity", "Three-model unanimity"),
    )

    gate_clause = (
        "All four predeclared protocol gates passed"
        if package["accepted"]
        else "At least one predeclared protocol gate failed"
    )
    if all_rows["estimate_count"]:
        maximum_regret = f"{float(all_rows['maximum_estimate_regret_percent']):.2f}"
        estimate_clause = (
            f"{all_rows['estimate_within_one_percent']}/{all_rows['estimate_count']} "
            "ESTIMATE decisions were within 1\\% of the measured optimum, with "
            f"{float(all_rows['maximum_estimate_regret_percent']):.2f}\\% maximum regret"
        )
        median_description = f"had {float(all_rows['median_estimate_regret_percent']):.2f}\\% median regret"
    else:
        maximum_regret = "--"
        estimate_clause = "no query received an ESTIMATE decision"
        median_description = "had no defined regret because no query received ESTIMATE"
    headline = (
        "\\newcommand{\\fixedresourceviiiheadline}{"
        f"{gate_clause}. Across {all_rows['queries']} held-out "
        f"H100 queries, the selective policy estimated {all_rows['estimate_count']} and "
        f"requested measurement for {all_rows['measure_count']}; {estimate_clause}."
        "}\n"
        f"\\newcommand{{\\fixedresourceviiidirect}}{{{_integer_tex(costs['direct-primary-acquisition']['gpu_seconds'])}}}\n"
        f"\\newcommand{{\\fixedresourceviiirequested}}{{{_integer_tex(costs['selective-requested-acquisition']['gpu_seconds'])}}}\n"
        f"\\newcommand{{\\fixedresourceviiicalibration}}{{{_integer_tex(costs['shared-calibration']['gpu_seconds'])}}}\n"
        f"\\newcommand{{\\fixedresourceviiicold}}{{{_integer_tex(costs['selective-cold-start']['gpu_seconds'])}}}\n"
        f"\\newcommand{{\\fixedresourceviiirepeat}}{{{_integer_tex(costs['repeat-validation-not-policy']['gpu_seconds'])}}}\n"
        f"\\newcommand{{\\fixedresourceviiiwarmpercent}}{{{costs['warm-policy-savings']['percent_of_direct']:.1f}}}\n"
        f"\\newcommand{{\\fixedresourceviiicoldratio}}{{{100.0 * costs['selective-cold-start']['gpu_seconds'] / costs['direct-primary-acquisition']['gpu_seconds']:.1f}}}\n"
        f"\\newcommand{{\\fixedresourceviiiestimates}}{{{all_rows['estimate_count']}}}\n"
        f"\\newcommand{{\\fixedresourceviiimeasures}}{{{all_rows['measure_count']}}}\n"
        f"\\newcommand{{\\fixedresourceviiiwithinestimates}}{{{all_rows['estimate_within_one_percent']}}}\n"
        f"\\newcommand{{\\fixedresourceviiimaxregret}}{{{maximum_regret}}}\n"
        f"\\newcommand{{\\fixedresourceviiimediandescription}}{{{median_description}}}\n"
        f"\\newcommand{{\\fixedresourceviiiwithheld}}{{{all_rows['unconditional_above_one_percent_withheld']}}}\n"
        f"\\newcommand{{\\fixedresourceviiiharmful}}{{{all_rows['unconditional_above_one_percent']}}}\n"
        f"\\newcommand{{\\fixedresourceviiiconservative}}{{{all_rows['conservative_measure_count']}}}\n"
        f"\\newcommand{{\\fixedresourceviiiunanimousties}}{{{near_ties['unanimous_physical_optimum_targets']}}}\n"
        f"\\newcommand{{\\fixedresourceviiiwithinties}}{{{near_ties['blind_within_one_percent_allocations']}}}\n"
        f"\\newcommand{{\\fixedresourceviiimaxtieregret}}{{{near_ties['maximum_allocation_regret_percent']:.2f}}}\n"
        f"\\newcommand{{\\fixedresourceviiitieestimates}}{{{near_ties['estimate_targets']}}}\n"
        f"\\newcommand{{\\fixedresourceviiitieestimateunits}}{{{near_ties['estimate_allocation_units']}}}\n"
        f"\\newcommand{{\\fixedresourceviiitieestimateexact}}{{{near_ties['estimate_exact_best_allocations']}}}\n"
        f"\\newcommand{{\\fixedresourceviiitiemaxestimateregret}}{{{near_tie_estimate_max}}}\n"
    )
    (output_directory / "fixed-resource-megatron-search-v8-summary.tex").write_text(
        headline, encoding="utf-8"
    )

    cohort_lines: list[str] = []
    cohort_labels = {
        "all": "All held-out queries",
        "paper-grid-remainder": "Previously unmeasured grid",
        "seeded-off-grid-interior": "Seeded off-grid sample",
    }
    for key in ("all", "paper-grid-remainder", "seeded-off-grid-interior"):
        row = cohorts[key]
        cohort_lines.append(
            "{} & {} & {}/{} & {}/{} & {} & {} \\\\".format(
                cohort_labels[key],
                row["queries"],
                row["estimate_count"],
                row["measure_count"],
                row["estimate_within_one_percent"],
                row["estimate_count"],
                _percent(row["maximum_estimate_regret_percent"]),
                f"{row['unconditional_above_one_percent_withheld']}/"
                f"{row['unconditional_above_one_percent']}",
            )
        )

    policy_lines: list[str] = []
    for key, label in policy_labels:
        row = policies[key]
        policy_lines.append(
            "{} & {}/{} & {} & {} & {}/{} & {} \\\\".format(
                label,
                row["estimated_allocation_units"],
                row["allocation_units"],
                _percent(row["coverage_percent"], 1),
                f"{row['above_one_percent_regret']}/{row['estimated_allocation_units']}",
                row["exact_best"],
                row["estimated_allocation_units"],
                _percent(row["maximum_regret_percent"]),
            )
        )

    table = (
        """\\begin{table*}[!t]
\\caption{Held-out selective-policy evaluation on H100. Each primary query uses one physical-node allocation to measure all three configurations and supply a label unavailable to the stored action. Panel (a) reports the deployed three-model-unanimity policy; ``withheld'' counts ungated primary log-model choices above 1\\% regret. Panel (b) compares four predeclared policies on the same 80 queries. Exact-choice and above-1\\%-regret fractions use the number of \\textsc{Estimate} decisions as denominator.}
\\label{tab:fixed-resource-v8}
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{4.0pt}
\\textbf{(a) Deployed policy by held-out set}\\par\\vspace{2pt}
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
Query set & Queries & Estimate/Measure & Est. within 1\\% & Max. est. regret & $>1\\%$ withheld \\\\
\\midrule
"""
        + "\n".join(cohort_lines)
        + """
\\bottomrule
\\end{tabular}

\\vspace{5pt}
\\textbf{(b) Risk--coverage policy comparison}\\par\\vspace{2pt}
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
Decision policy & Estimates & Coverage & Est. $>1\\%$ & Exact/est. & Max. regret \\\\
\\midrule
"""
        + "\n".join(policy_lines)
        + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    )
    (output_directory / "fixed-resource-megatron-search-v8-table.tex").write_text(
        table, encoding="utf-8"
    )


def write_package(
    score_path: Path, output_directory: Path, latex_directory: Path | None = None
) -> dict[str, Any]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise IntegrationError(
            "paper integration output directory must be absent or empty"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    package = build(score_path)
    (output_directory / "paper-metrics.json").write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_directory / "cohort-summary.csv", package["cohort_summary"])
    _write_csv(
        output_directory / "risk-coverage-primary.csv", package["risk_coverage_primary"]
    )
    _write_csv(
        output_directory / "risk-coverage-near-tie.csv",
        package["risk_coverage_near_tie_allocations"],
    )
    _write_csv(
        output_directory / "near-tie-stability.csv", package["near_tie_stability"]
    )
    _write_csv(output_directory / "cost.csv", package["cost"])
    _write_csv(output_directory / "primary-cases.csv", package["primary_cases"])
    if latex_directory is not None:
        _write_latex(package, latex_directory)
    return package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--latex-directory", type=Path)
    args = parser.parse_args(argv)
    write_package(args.score, args.output_directory, args.latex_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
