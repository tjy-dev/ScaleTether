from dataclasses import replace

from scaletether.planner_contract import (
    ESTIMATE,
    MEASURE,
    UNSUPPORTED,
    CalibrationDomain,
    PlannerRequest,
    decide,
)


DOMAIN = CalibrationDomain(
    framework_identity="megatron-core:pinned",
    runtime_identity="cuda-nccl:pinned",
    accelerator_identity="h100:pinned",
    sequence_length=2048,
    minimum_hidden_size=3072,
    maximum_hidden_size=6144,
    tp_transitions=frozenset({(1, 2), (2, 4)}),
    pp_schedules=frozenset({"1f1b-noninterleaved"}),
)
SUPPORTED = PlannerRequest(
    framework_identity=DOMAIN.framework_identity,
    runtime_identity=DOMAIN.runtime_identity,
    accelerator_identity=DOMAIN.accelerator_identity,
    sequence_length=DOMAIN.sequence_length,
    hidden_size=4096,
    transformation="tp",
    source_tp=1,
    target_tp=2,
)


def test_supported_request_is_estimated() -> None:
    decision = decide(DOMAIN, SUPPORTED)
    assert (decision.action, decision.reason_code) == (
        ESTIMATE,
        "calibration-domain-accepted",
    )


def test_repairable_evidence_requests_measurement() -> None:
    incomplete = decide(
        DOMAIN, replace(SUPPORTED, source_measurements_complete=False)
    )
    unstable = decide(DOMAIN, replace(SUPPORTED, source_measurements_stable=False))
    assert (incomplete.action, incomplete.reason_code) == (
        MEASURE,
        "source-measurements-incomplete",
    )
    assert (unstable.action, unstable.reason_code) == (
        MEASURE,
        "source-measurements-unstable",
    )


def test_out_of_domain_requests_are_unsupported_with_typed_reasons() -> None:
    cases = (
        (replace(SUPPORTED, hidden_size=8192), "hidden-size-extrapolation"),
        (replace(SUPPORTED, source_tp=4, target_tp=8), "tp-transition-unsupported"),
        (
            replace(SUPPORTED, framework_identity="megatron-core:other"),
            "framework-identity-mismatch",
        ),
        (
            replace(
                SUPPORTED,
                transformation="pp",
                source_tp=None,
                target_tp=None,
                pp_schedule="interleaved-custom",
            ),
            "pp-schedule-unsupported",
        ),
        (
            replace(SUPPORTED, accelerator_identity="blackwell:unmeasured"),
            "accelerator-identity-mismatch",
        ),
        (
            replace(SUPPORTED, runtime_identity="cuda-nccl:other"),
            "runtime-identity-mismatch",
        ),
    )
    for request, reason in cases:
        decision = decide(DOMAIN, request)
        assert (decision.action, decision.reason_code) == (UNSUPPORTED, reason)
