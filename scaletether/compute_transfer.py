from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import statistics
from typing import Any

from .config import architecture, assess_hardware_identity


@dataclass(frozen=True)
class ComputeTransferPoint:
    kernel_signature: str
    source_target: str
    source_device_name: str
    source_compute_capability: str
    source_sm_count: int
    target: str
    target_device_name: str
    target_compute_capability: str
    target_sm_count: int
    source_duration_us: float
    target_duration_us: float
    duration_ratio: float
    samples: int
    target_kernel_signature: str | None = None
    source_kernel_launch_signature: str | None = None
    target_kernel_launch_signature: str | None = None
    pairing_method: str | None = None
    source_workload_sha256: str | None = None
    target_workload_sha256: str | None = None
    evidence_class: str | None = None
    source_capture_kind: str | None = None
    target_capture_kind: str | None = None
    source_provenance_sha256: str | None = None
    target_provenance_sha256: str | None = None


@dataclass(frozen=True)
class ComputeTransferLookup:
    duration_us: float
    duration_ratio: float
    samples: int
    source: str


@dataclass(frozen=True)
class ComputeTransferCalibration:
    source_path: str
    points: tuple[ComputeTransferPoint, ...]

    @classmethod
    def load_csv(cls, path: Path) -> "ComputeTransferCalibration":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"compute transfer CSV is empty: {path}")
        required = {
            "kernel_signature",
            "source_target",
            "source_device_name",
            "source_compute_capability",
            "source_sm_count",
            "target",
            "target_device_name",
            "target_compute_capability",
            "target_sm_count",
            "source_duration_us",
            "target_duration_us",
        }
        missing = required - set(rows[0])
        if missing:
            raise ValueError(
                "compute transfer CSV is missing columns: " + ", ".join(sorted(missing))
            )
        audit_columns = {
            "target_kernel_signature",
            "source_kernel_launch_signature",
            "target_kernel_launch_signature",
            "source_event_id",
            "target_event_id",
            "pairing_method",
            "source_workload_sha256",
            "target_workload_sha256",
        }
        provenance_columns = {
            "evidence_class",
            "source_capture_kind",
            "target_capture_kind",
            "source_provenance_sha256",
            "target_provenance_sha256",
        }
        present_audit_columns = audit_columns & set(rows[0])
        if present_audit_columns and present_audit_columns != audit_columns:
            raise ValueError(
                "compute transfer CSV has an incomplete paired-workload audit: "
                + ", ".join(sorted(audit_columns - present_audit_columns))
            )
        audited = present_audit_columns == audit_columns
        present_provenance_columns = provenance_columns & set(rows[0])
        if (
            present_provenance_columns
            and present_provenance_columns != provenance_columns
        ):
            raise ValueError(
                "compute transfer CSV has an incomplete capture-provenance audit: "
                + ", ".join(sorted(provenance_columns - present_provenance_columns))
            )
        provenance_audited = present_provenance_columns == provenance_columns
        if provenance_audited and not audited:
            raise ValueError(
                "capture-provenance audit requires the complete paired-workload audit"
            )
        grouped: dict[
            tuple[Any, ...],
            list[tuple[float, float]],
        ] = {}
        observed_event_pairs: set[tuple[str, str, str, str]] = set()
        for row_number, row in enumerate(rows, start=2):
            signature = row["kernel_signature"].strip()
            source_target = row["source_target"].strip().lower()
            target = row["target"].strip().lower()
            if not signature or not source_target or not target:
                raise ValueError(
                    f"compute transfer row {row_number} has an empty domain key"
                )
            if source_target == target:
                raise ValueError(
                    f"compute transfer row {row_number} must cross target architectures"
                )
            source_architecture = architecture(source_target)
            target_architecture = architecture(target)
            source_identity = assess_hardware_identity(
                {
                    "target": source_target,
                    "device_name": row["source_device_name"],
                    "compute_capability": row["source_compute_capability"],
                    "sm_count": row["source_sm_count"],
                },
                source_architecture,
            )
            target_identity = assess_hardware_identity(
                {
                    "target": target,
                    "device_name": row["target_device_name"],
                    "compute_capability": row["target_compute_capability"],
                    "sm_count": row["target_sm_count"],
                },
                target_architecture,
            )
            if source_identity["status"] != "exact-product-match":
                raise ValueError(
                    f"compute transfer row {row_number} source physical device "
                    "is outside its exact target domain"
                )
            if target_identity["status"] != "exact-product-match":
                raise ValueError(
                    f"compute transfer row {row_number} destination physical "
                    "device is outside its exact target domain"
                )
            source_duration = float(row["source_duration_us"])
            target_duration = float(row["target_duration_us"])
            if source_duration <= 0.0 or target_duration <= 0.0:
                raise ValueError(
                    f"compute transfer row {row_number} requires positive durations"
                )
            audit_key: tuple[str | None, ...] = (None,) * 6
            provenance_key: tuple[str | None, ...] = (None,) * 5
            if audited:
                target_signature = row["target_kernel_signature"].strip()
                source_launch = row["source_kernel_launch_signature"].strip()
                target_launch = row["target_kernel_launch_signature"].strip()
                source_event_id = row["source_event_id"].strip()
                target_event_id = row["target_event_id"].strip()
                pairing_method = row["pairing_method"].strip()
                source_workload_sha256 = row["source_workload_sha256"].strip().lower()
                target_workload_sha256 = row["target_workload_sha256"].strip().lower()
                required_values = {
                    "target_kernel_signature": target_signature,
                    "source_kernel_launch_signature": source_launch,
                    "target_kernel_launch_signature": target_launch,
                    "source_event_id": source_event_id,
                    "target_event_id": target_event_id,
                    "pairing_method": pairing_method,
                }
                empty = sorted(
                    key for key, value in required_values.items() if not value
                )
                if empty:
                    raise ValueError(
                        f"compute transfer row {row_number} has empty paired-workload "
                        "fields: " + ", ".join(empty)
                    )
                if pairing_method not in {
                    "auto-launch-signature-v1",
                    "explicit-event-id-v1",
                }:
                    raise ValueError(
                        f"compute transfer row {row_number} has unsupported pairing method"
                    )
                for field, value in (
                    ("source_workload_sha256", source_workload_sha256),
                    ("target_workload_sha256", target_workload_sha256),
                ):
                    if len(value) != 64 or any(
                        character not in "0123456789abcdef" for character in value
                    ):
                        raise ValueError(
                            f"compute transfer row {row_number} has invalid {field}"
                        )
                pair_key = (
                    source_workload_sha256,
                    target_workload_sha256,
                    source_event_id,
                    target_event_id,
                )
                if pair_key in observed_event_pairs:
                    raise ValueError(
                        f"compute transfer row {row_number} duplicates an event pair"
                    )
                observed_event_pairs.add(pair_key)
                audit_key = (
                    target_signature,
                    source_launch,
                    target_launch,
                    pairing_method,
                    source_workload_sha256,
                    target_workload_sha256,
                )
            if provenance_audited:
                evidence_class = row["evidence_class"].strip()
                source_capture_kind = row["source_capture_kind"].strip()
                target_capture_kind = row["target_capture_kind"].strip()
                source_provenance_sha256 = (
                    row["source_provenance_sha256"].strip().lower()
                )
                target_provenance_sha256 = (
                    row["target_provenance_sha256"].strip().lower()
                )
                if evidence_class not in {
                    "physical-paired-capture-v1",
                    "synthetic-contract-v1",
                }:
                    raise ValueError(
                        f"compute transfer row {row_number} has unsupported "
                        "evidence_class"
                    )
                if not source_capture_kind or not target_capture_kind:
                    raise ValueError(
                        f"compute transfer row {row_number} has empty capture kind"
                    )
                for field, value in (
                    ("source_provenance_sha256", source_provenance_sha256),
                    ("target_provenance_sha256", target_provenance_sha256),
                ):
                    if len(value) != 64 or any(
                        character not in "0123456789abcdef" for character in value
                    ):
                        raise ValueError(
                            f"compute transfer row {row_number} has invalid {field}"
                        )
                if evidence_class == "physical-paired-capture-v1" and (
                    source_capture_kind != "torch-profiler"
                    or target_capture_kind != "torch-profiler"
                ):
                    raise ValueError(
                        f"compute transfer row {row_number} labels non-profiler "
                        "inputs as physical evidence"
                    )
                provenance_key = (
                    evidence_class,
                    source_capture_kind,
                    target_capture_kind,
                    source_provenance_sha256,
                    target_provenance_sha256,
                )
            source_observed = source_identity["observed"]
            target_observed = target_identity["observed"]
            grouped.setdefault(
                (
                    signature,
                    source_target,
                    str(source_observed["device_name"]),
                    str(source_observed["compute_capability"]),
                    int(source_observed["sm_count"]),
                    target,
                    str(target_observed["device_name"]),
                    str(target_observed["compute_capability"]),
                    int(target_observed["sm_count"]),
                    *audit_key,
                    *provenance_key,
                ),
                [],
            ).append((source_duration, target_duration))
        points = []
        for domain, samples in grouped.items():
            (
                signature,
                source_target,
                source_device_name,
                source_compute_capability,
                source_sm_count,
                target,
                target_device_name,
                target_compute_capability,
                target_sm_count,
                target_kernel_signature,
                source_kernel_launch_signature,
                target_kernel_launch_signature,
                pairing_method,
                source_workload_sha256,
                target_workload_sha256,
                evidence_class,
                source_capture_kind,
                target_capture_kind,
                source_provenance_sha256,
                target_provenance_sha256,
            ) = domain
            ratios = [target_us / source_us for source_us, target_us in samples]
            points.append(
                ComputeTransferPoint(
                    kernel_signature=signature,
                    source_target=source_target,
                    source_device_name=source_device_name,
                    source_compute_capability=source_compute_capability,
                    source_sm_count=source_sm_count,
                    target=target,
                    target_device_name=target_device_name,
                    target_compute_capability=target_compute_capability,
                    target_sm_count=target_sm_count,
                    source_duration_us=statistics.median(
                        source_us for source_us, _ in samples
                    ),
                    target_duration_us=statistics.median(
                        target_us for _, target_us in samples
                    ),
                    duration_ratio=statistics.median(ratios),
                    samples=len(samples),
                    target_kernel_signature=target_kernel_signature,
                    source_kernel_launch_signature=source_kernel_launch_signature,
                    target_kernel_launch_signature=target_kernel_launch_signature,
                    pairing_method=pairing_method,
                    source_workload_sha256=source_workload_sha256,
                    target_workload_sha256=target_workload_sha256,
                    evidence_class=evidence_class,
                    source_capture_kind=source_capture_kind,
                    target_capture_kind=target_capture_kind,
                    source_provenance_sha256=source_provenance_sha256,
                    target_provenance_sha256=target_provenance_sha256,
                )
            )
        return cls(
            source_path=str(path.resolve()),
            points=tuple(
                sorted(
                    points,
                    key=lambda point: (
                        point.source_target,
                        point.target,
                        point.kernel_signature,
                    ),
                )
            ),
        )

    def lookup(
        self,
        kernel_signature: str | None,
        source_target: str,
        target: str,
        observed_duration_us: float,
        trace_source: dict[str, Any],
    ) -> ComputeTransferLookup | None:
        if kernel_signature is None:
            return None
        try:
            source_architecture = architecture(source_target)
        except ValueError:
            return None
        if (
            assess_hardware_identity(trace_source, source_architecture)["status"]
            != "exact-product-match"
        ):
            return None
        matches = [
            point
            for point in self.points
            if point.kernel_signature == kernel_signature
            and point.source_target == source_target.lower()
            and point.target == target.lower()
            and point.evidence_class != "synthetic-contract-v1"
        ]
        if len(matches) != 1:
            return None
        point = matches[0]
        return ComputeTransferLookup(
            duration_us=observed_duration_us * point.duration_ratio,
            duration_ratio=point.duration_ratio,
            samples=point.samples,
            source="measured-exact-signature-compute-transfer",
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "kind": "exact-kernel-signature-compute-transfer",
            "source_path": self.source_path,
            "points": [asdict(point) for point in self.points],
        }
