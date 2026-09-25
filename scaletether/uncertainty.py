from __future__ import annotations

from collections import Counter
import csv
import math
from pathlib import Path
from typing import Any

from .config import architecture, assess_hardware_identity


_RESIDUAL_FIELDS = {
    "workload_id",
    "target",
    "device_name",
    "compute_capability",
    "sm_count",
    "topology",
    "gpus",
    "model",
    "predicted_step_us",
    "observed_step_us",
    "held_out",
}


def _domain(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": str(prediction["target"]["name"]),
        "topology": str(prediction["topology"]["name"]),
        "gpus": int(prediction["requested_gpus"]),
        "model": str(prediction["model"]),
    }


def _evidence_envelope(prediction: dict[str, Any]) -> dict[str, Any]:
    """Return a transparent sensitivity envelope without a coverage claim.

    The multipliers are deliberately coarse policy assumptions. They prevent a
    point estimate backed by analytical or out-of-domain components from being
    displayed as precise, but they are not learned quantiles and must never be
    described as a confidence or prediction interval.
    """

    point = float(prediction["summary"]["step_time_us"])
    sources = Counter(
        str(event["duration_source"]) for event in prediction.get("timeline", ())
    )
    if not sources:
        source_counts = prediction.get("duration_source_counts", {})
        if isinstance(source_counts, dict):
            sources.update(
                {
                    str(source): int(count)
                    for source, count in source_counts.items()
                    if isinstance(count, int) and count > 0
                }
            )
    reasons: list[str] = []
    radius = 0.10
    grade = "medium"

    if any(source.startswith("interpolated-") for source in sources):
        radius = max(radius, 0.20)
        grade = "low"
        reasons.append("one or more component timings were interpolated")
    if any(source.startswith("analytical-") for source in sources):
        radius = max(radius, 0.50)
        grade = "low"
        reasons.append(
            "one or more collective timings use an uncalibrated analytical model"
        )
    if any(source.startswith("observed-") for source in sources):
        radius = max(radius, 1.00)
        grade = "very-low"
        reasons.append(
            "one or more collectives lack a supported counterfactual timing model"
        )
    if prediction.get("admission_assessments"):
        radius = max(radius, 1.00)
        grade = "very-low"
        reasons.append("compute/NCCL overlap is outside the exact interference domain")
    unmatched_network_contention = any(
        region.get("calibration_status") != "matched"
        for region in prediction.get("network_contention", [])
    )
    if unmatched_network_contention:
        radius = max(radius, 0.50)
        grade = "low" if grade != "very-low" else grade
        reasons.append(
            "network contention uses the uncalibrated whole-event fluid fair-share approximation"
        )
    if (
        prediction.get("policy", {}).get("nccl_cga_timing_calibration_status")
        == "unmatched"
    ):
        radius = max(radius, 1.00)
        grade = "very-low"
        reasons.append(
            "the requested NCCL CGA cluster size has no exact mode-matched timing calibration"
        )
    if (
        prediction.get("capture_completeness", {}).get("timing_recommendation_eligible")
        is False
    ):
        radius = max(radius, 1.00)
        grade = "very-low"
        reasons.append(
            "captured execution semantics are incomplete or unsupported by replay"
        )
    if any(
        "different target" in warning or "no exact signature transfer" in warning
        for warning in prediction.get("warnings", [])
    ):
        radius = max(radius, 1.00)
        grade = "very-low"
        reasons.append(
            "compute timing was transferred across architectures without a transfer model"
        )
    if any(
        "unstable repeated timing" in warning
        for warning in prediction.get("warnings", [])
    ):
        radius = max(radius, 1.00)
        grade = "very-low"
        reasons.append(
            "one or more repeated qualified kernel signatures had unstable timing"
        )
    if not reasons:
        reasons.append(
            "no held-out end-to-end residual set was supplied; the displayed range is a sensitivity envelope"
        )

    return {
        "schema_version": "0.1",
        "method": "evidence-sensitivity-envelope-v0",
        "calibrated": False,
        "coverage_claim": False,
        "requested_coverage": None,
        "domain": _domain(prediction),
        "sample_count": 0,
        "point_us": point,
        "lower_us": max(0.0, point * (1.0 - radius)),
        "upper_us": point * (1.0 + radius),
        "evidence_grade": grade,
        "recommendation_status": "abstain",
        "abstention_reasons": reasons,
        "duration_source_counts": dict(sorted(sources.items())),
        "interpretation": (
            "This is a policy sensitivity envelope, not a statistically calibrated "
            "prediction interval. Supply held-out domain-matched residuals to make a coverage claim."
        ),
    }


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid held_out value {value!r}")


def _conformal_interval(
    prediction: dict[str, Any],
    path: Path,
    coverage: float,
    residual_group: str | None = None,
) -> dict[str, Any]:
    if not 0.5 <= coverage < 1.0:
        raise ValueError("prediction coverage must be in [0.5, 1.0)")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = _RESIDUAL_FIELDS - fields
        if missing:
            raise ValueError(
                "validation residual CSV is missing columns: "
                + ", ".join(sorted(missing))
            )
        if residual_group is not None and "residual_group" not in fields:
            raise ValueError(
                "validation residual CSV requires a residual_group column when "
                "--validation-residual-group is supplied"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("validation residual CSV is empty")

    domain = _domain(prediction)
    workload_ids: set[str] = set()
    scores: list[float] = []
    observed_groups: set[str] = set()
    target_architecture = architecture(domain["target"])
    hardware_domains: set[tuple[str, str, int]] = set()
    for row_number, row in enumerate(rows, start=2):
        workload_id = row["workload_id"].strip()
        if not workload_id or workload_id in workload_ids:
            raise ValueError(
                f"validation residual row {row_number} has an empty or duplicate workload_id"
            )
        workload_ids.add(workload_id)
        if not _parse_bool(row["held_out"]):
            raise ValueError(
                f"validation residual row {row_number} is not marked held_out"
            )
        row_domain = {
            "target": row["target"].strip(),
            "topology": row["topology"].strip(),
            "gpus": int(row["gpus"]),
            "model": row["model"].strip(),
        }
        if row_domain != domain:
            raise ValueError(
                f"validation residual row {row_number} domain {row_domain} "
                f"does not match prediction domain {domain}"
            )
        hardware_identity = assess_hardware_identity(
            {
                "target": row["target"],
                "device_name": row["device_name"],
                "compute_capability": row["compute_capability"],
                "sm_count": row["sm_count"],
            },
            target_architecture,
        )
        if hardware_identity["status"] != "exact-product-match":
            raise ValueError(
                f"validation residual row {row_number} physical device is "
                "outside the exact target domain"
            )
        observed_hardware = hardware_identity["observed"]
        hardware_domains.add(
            (
                str(observed_hardware["device_name"]),
                str(observed_hardware["compute_capability"]),
                int(observed_hardware["sm_count"]),
            )
        )
        predicted = float(row["predicted_step_us"])
        observed = float(row["observed_step_us"])
        if not math.isfinite(predicted) or not math.isfinite(observed):
            raise ValueError(f"validation residual row {row_number} is non-finite")
        if predicted <= 0.0 or observed <= 0.0:
            raise ValueError(
                f"validation residual row {row_number} requires positive step times"
            )
        group = row.get("residual_group", "").strip()
        if residual_group is not None:
            if not group:
                raise ValueError(
                    f"validation residual row {row_number} has an empty residual_group"
                )
            observed_groups.add(group)
        if residual_group is None or group == residual_group:
            scores.append(abs(math.log(observed / predicted)))

    if len(hardware_domains) != 1:
        raise ValueError("validation residual CSV cannot mix physical device domains")
    if residual_group is not None and not scores:
        available = ", ".join(sorted(observed_groups)) or "none"
        raise ValueError(
            f"validation residual group {residual_group!r} has no rows; "
            f"available groups: {available}"
        )
    device_name, compute_capability, sm_count = next(iter(hardware_domains))

    # Split-conformal finite-sample correction. If the requested rank is n+1,
    # the finite calibration set cannot support the requested coverage.
    rank = math.ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        minimum = math.ceil(coverage / (1.0 - coverage))
        raise ValueError(
            f"{coverage:.3f} coverage requires at least {minimum} held-out rows; "
            f"got {len(scores)}"
        )
    score = sorted(scores)[rank - 1]
    point = float(prediction["summary"]["step_time_us"])
    unresolved = bool(prediction.get("admission_assessments"))
    cross_target = any(
        "different target" in warning or "no exact signature transfer" in warning
        for warning in prediction.get("warnings", [])
    )
    unstable_capture = any(
        "unstable repeated timing" in warning
        for warning in prediction.get("warnings", [])
    )
    unmatched_cga = (
        prediction.get("policy", {}).get("nccl_cga_timing_calibration_status")
        == "unmatched"
    )
    incomplete_capture = (
        prediction.get("capture_completeness", {}).get("timing_recommendation_eligible")
        is False
    )
    abstention_reasons = []
    if unresolved:
        abstention_reasons.append(
            "current trace has unresolved compute/NCCL admission outside the mechanistic domain"
        )
    if cross_target:
        abstention_reasons.append(
            "current trace transfers compute timing across architectures without a transfer model"
        )
    if unstable_capture:
        abstention_reasons.append(
            "current capture has unstable timing for a repeated qualified kernel signature"
        )
    if unmatched_cga:
        abstention_reasons.append(
            "the requested NCCL CGA cluster size has no exact mode-matched timing calibration"
        )
    if incomplete_capture:
        abstention_reasons.append(
            "captured execution semantics are incomplete or unsupported by replay"
        )
    coverage_claim = not unmatched_cga and not incomplete_capture
    if incomplete_capture:
        coverage_scope = (
            "no coverage claim: the current capture contains unsupported or "
            "incomplete execution semantics"
        )
    elif unmatched_cga:
        coverage_scope = (
            "no coverage claim: residual rows do not lock the requested CGA "
            "policy dimension"
        )
    else:
        coverage_scope = (
            "finite-sample marginal coverage for a new exchangeable workload in "
            "the exact locked domain"
        )
        if residual_group is not None:
            coverage_scope += f" and residual group {residual_group!r}"
    method = (
        "split-conformal-log-ratio-v0"
        if residual_group is None
        else "mondrian-split-conformal-log-ratio-v1"
    )
    assumptions = [
        "calibration and future workload errors are exchangeable",
        "the residual file was not used to fit or select the point-prediction model",
    ]
    if residual_group is None:
        assumptions.append(
            "coverage is marginal, not conditional on workload or kernel features"
        )
    else:
        assumptions.append(
            "the residual group was fixed before observing future evaluation outcomes"
        )
        assumptions.append(
            "coverage is marginal within the selected residual group, not "
            "conditional on other features"
        )
    return {
        "schema_version": "0.1",
        "method": method,
        "calibrated": True,
        "coverage_claim": coverage_claim,
        "requested_coverage": coverage,
        "coverage_scope": coverage_scope,
        "domain": domain,
        "sample_count": len(scores),
        "residual_total_count": len(rows),
        "residual_group": residual_group,
        "quantile_rank": rank,
        "log_ratio_quantile": score,
        "point_us": point,
        "lower_us": point * math.exp(-score),
        "upper_us": point * math.exp(score),
        "evidence_grade": "high" if not abstention_reasons else "low",
        "recommendation_status": "usable" if not abstention_reasons else "abstain",
        "abstention_reasons": abstention_reasons,
        "residual_source": str(path.resolve()),
        "residual_hardware_identity": {
            "status": "exact-product-match",
            "target": target_architecture.name,
            "device_name": device_name,
            "compute_capability": compute_capability,
            "sm_count": sm_count,
        },
        "assumptions": assumptions,
    }


def _prediction_claim(
    prediction: dict[str, Any], uncertainty: dict[str, Any]
) -> dict[str, Any]:
    """Summarize what the current prediction evidence is allowed to claim."""

    capture_eligible = (
        prediction.get("capture_completeness", {}).get(
            "timing_recommendation_eligible"
        )
        is True
    )
    coverage_claim = uncertainty.get("coverage_claim") is True
    recommendation_status = str(
        uncertainty.get("recommendation_status", "abstain")
    )
    recommendation_usable = capture_eligible and recommendation_status == "usable"
    if not capture_eligible:
        status = "diagnostic-only-incomplete-capture"
        interpretation = (
            "The timeline and point estimate are diagnostic only because executed "
            "semantics were not captured completely enough for a timing recommendation."
        )
    elif coverage_claim and recommendation_usable:
        status = "calibrated-domain-interval"
        interpretation = (
            "The interval carries the recorded finite-sample coverage claim inside "
            "its exact locked domain and assumptions. This is not a point-accuracy "
            "or out-of-domain claim."
        )
    elif coverage_claim:
        status = "calibrated-interval-recommendation-abstains"
        interpretation = (
            "A statistical interval claim is retained under its recorded assumptions, "
            "but unresolved evidence makes the timing recommendation abstain."
        )
    else:
        status = "estimate-only-no-accuracy-claim"
        interpretation = (
            "The point estimate and sensitivity envelope have no statistical coverage "
            "or point-accuracy claim. Domain-matched held-out validation is required."
        )
    return {
        "schema": "scaletether-prediction-claim-v1",
        "status": status,
        "point_estimate_available": True,
        "timing_recommendation_eligible": capture_eligible,
        "recommendation_usable": recommendation_usable,
        "statistical_interval_claim": coverage_claim,
        "point_accuracy_claim": False,
        "coverage_scope": uncertainty.get("coverage_scope"),
        "interpretation": interpretation,
    }


def attach_uncertainty(
    prediction: dict[str, Any],
    validation_residuals: Path | None = None,
    coverage: float = 0.90,
    residual_group: str | None = None,
) -> dict[str, Any]:
    """Return a prediction document with an auditable uncertainty section."""

    # Only top-level uncertainty fields and the summary are modified below.
    # Deep-copying the complete timeline doubled peak memory for large semantic
    # candidate searches without providing additional isolation.
    result = dict(prediction)
    result["summary"] = dict(prediction["summary"])
    uncertainty = (
        _evidence_envelope(result)
        if validation_residuals is None
        else _conformal_interval(
            result,
            validation_residuals,
            coverage,
            residual_group,
        )
    )
    result["uncertainty"] = uncertainty
    result["prediction_claim"] = _prediction_claim(result, uncertainty)
    result["summary"]["step_time_interval_us"] = {
        "lower": uncertainty["lower_us"],
        "point": uncertainty["point_us"],
        "upper": uncertainty["upper_us"],
        "coverage": uncertainty["requested_coverage"],
        "coverage_claim": uncertainty["coverage_claim"],
    }
    return result
