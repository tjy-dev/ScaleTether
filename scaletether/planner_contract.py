"""Deterministic planner-facing admission contract.

The contract separates immutable domain violations (``UNSUPPORTED``) from
missing or unstable evidence that can be repaired by another run
(``MEASURE``).  A timing estimate is returned only after both classes of
checks pass.
"""

from __future__ import annotations

from dataclasses import dataclass


ESTIMATE = "ESTIMATE"
MEASURE = "MEASURE"
UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class CalibrationDomain:
    framework_identity: str
    runtime_identity: str
    accelerator_identity: str
    sequence_length: int
    minimum_hidden_size: int
    maximum_hidden_size: int
    tp_transitions: frozenset[tuple[int, int]]
    pp_schedules: frozenset[str]


@dataclass(frozen=True)
class PlannerRequest:
    framework_identity: str
    runtime_identity: str
    accelerator_identity: str
    sequence_length: int
    hidden_size: int
    transformation: str
    source_tp: int | None = None
    target_tp: int | None = None
    pp_schedule: str | None = None
    source_measurements_complete: bool = True
    source_measurements_stable: bool = True


@dataclass(frozen=True)
class PlannerDecision:
    action: str
    reason_code: str


def decide(domain: CalibrationDomain, request: PlannerRequest) -> PlannerDecision:
    """Return a typed action using a fixed, fail-closed precedence order."""

    if request.framework_identity != domain.framework_identity:
        return PlannerDecision(UNSUPPORTED, "framework-identity-mismatch")
    if request.runtime_identity != domain.runtime_identity:
        return PlannerDecision(UNSUPPORTED, "runtime-identity-mismatch")
    if request.accelerator_identity != domain.accelerator_identity:
        return PlannerDecision(UNSUPPORTED, "accelerator-identity-mismatch")
    if request.sequence_length != domain.sequence_length:
        return PlannerDecision(UNSUPPORTED, "sequence-length-out-of-domain")
    if not domain.minimum_hidden_size <= request.hidden_size <= domain.maximum_hidden_size:
        return PlannerDecision(UNSUPPORTED, "hidden-size-extrapolation")

    if request.transformation == "tp":
        transition = (request.source_tp, request.target_tp)
        if transition not in domain.tp_transitions:
            return PlannerDecision(UNSUPPORTED, "tp-transition-unsupported")
    elif request.transformation == "pp":
        if request.pp_schedule not in domain.pp_schedules:
            return PlannerDecision(UNSUPPORTED, "pp-schedule-unsupported")
    else:
        return PlannerDecision(UNSUPPORTED, "transformation-unsupported")

    if not request.source_measurements_complete:
        return PlannerDecision(MEASURE, "source-measurements-incomplete")
    if not request.source_measurements_stable:
        return PlannerDecision(MEASURE, "source-measurements-unstable")
    return PlannerDecision(ESTIMATE, "calibration-domain-accepted")
