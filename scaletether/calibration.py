from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
import math
import statistics
from typing import Any

from .config import architecture, device_name_matches_architecture
from .nccl_policy import (
    cta_policy_name,
    parse_cga_cluster_size,
    parse_cta_policy_mode,
)
from .schema import COLLECTIVES


SOFTWARE_VERSION_KEYS = (
    "torch_version",
    "cuda_version",
    "cuda_driver_version",
    "nccl_version",
)


def _affine_message_interpolation(
    message_bytes: int,
    lower_bytes: int,
    upper_bytes: int,
    lower_value: float,
    upper_value: float,
) -> float:
    """Interpolate fixed-algorithm latency with the standard alpha+beta*n model."""

    if not lower_bytes < message_bytes < upper_bytes:
        raise ValueError("affine message interpolation requires an interior point")
    fraction = (message_bytes - lower_bytes) / (upper_bytes - lower_bytes)
    return lower_value + fraction * (upper_value - lower_value)


CALIBRATABLE_COLLECTIVES = COLLECTIVES - {"unknown"}
P2P_COLLECTIVES = {"send", "recv"}
P2P_NETWORK_TIERS = {"intra_node", "inter_node"}


def normalize_calibration_collective(value: str) -> str:
    collective = value.strip().lower()
    if collective not in CALIBRATABLE_COLLECTIVES:
        raise ValueError(
            "calibration collective must be one of "
            + ", ".join(sorted(CALIBRATABLE_COLLECTIVES))
        )
    return collective


@dataclass(frozen=True)
class CalibrationPoint:
    message_bytes: int
    max_ctas: int | None
    compute_alone_us: float
    collective_alone_us: float
    overlap_compute_slowdown: float
    overlap_collective_slowdown: float
    overlap_device_us: float
    overlap_regime: str
    samples: int
    event_intervals_overlap: bool | None = None
    event_interval_overlap_fraction: float | None = None


@dataclass(frozen=True)
class CalibrationLookup:
    duration_us: float
    source: str
    lower_bytes: int
    upper_bytes: int


@dataclass(frozen=True)
class InterferenceLookup:
    compute_slowdown: float
    collective_slowdown: float
    regime: str
    source: str
    lower_bytes: int
    upper_bytes: int
    compute_alone_us: float
    collective_alone_us: float
    overlap_device_us: float
    compute_event_count: int | None
    validation_evidence_status: str | None
    validation_evidence_sha256: str | None
    validation_timeline_outcome: str | None


@dataclass(frozen=True)
class OverlapCalibration:
    target: str
    gpu_count: int
    collective: str
    compute_signature: str
    source_path: str
    points: tuple[CalibrationPoint, ...]
    device_name: str = ""
    compute_capability: str = ""
    hardware_identity_status: str = "programmatic"
    network_tier: str | None = None
    comm_stream_priority: str = "normal"
    sm_count: int | None = None
    compute_blocks_per_sm: float | None = None
    compute_blocks: int | None = None
    compute_event_count: int | None = None
    compute_sm_coverage_upper_bound: float | None = None
    nccl_cta_policy: int = 0
    nccl_nvls_ctas: int | None = None
    nccl_cga_cluster_size: int = 0
    # API acceptance is not proof that NCCL actually selected NVLS.
    nvls_runtime_status: str = "not-requested"
    software_versions: dict[str, str] = field(default_factory=dict)
    validation_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_collective = normalize_calibration_collective(self.collective)
        if self.collective != normalized_collective:
            raise ValueError("calibration collective must use its canonical name")
        if self.collective in P2P_COLLECTIVES:
            if self.network_tier not in P2P_NETWORK_TIERS:
                raise ValueError(
                    "send/recv calibration requires network_tier intra_node or inter_node"
                )
        elif self.network_tier is not None:
            raise ValueError("network_tier is only valid for send/recv calibration")
        parse_cga_cluster_size(self.nccl_cga_cluster_size)
        allowed = {"active", "inactive", "unknown", "not-requested"}
        if self.nvls_runtime_status not in allowed:
            raise ValueError("invalid NVLS runtime status")
        if self.nccl_nvls_ctas is None and self.nvls_runtime_status != "not-requested":
            raise ValueError(
                "automatic NVLS configuration must use not-requested status"
            )
        if (
            self.nccl_nvls_ctas is not None
            and self.nvls_runtime_status == "not-requested"
        ):
            raise ValueError(
                "an NVLS CTA override requires active/inactive/unknown status"
            )

    @classmethod
    def load_csv(
        cls,
        path: Path,
        target: str,
        compute_signature: str | None = None,
    ) -> "OverlapCalibration":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"calibration CSV is empty: {path}")
        required = {
            "target",
            "device_name",
            "compute_capability",
            "nccl_max_ctas",
            "message_bytes",
            "gpus",
            "sm_count",
            "compute_alone_ms",
            "nccl_alone_ms",
            "overlap_compute_ms",
            "overlap_nccl_ms",
            "overlap_device_ms",
            "compute_slowdown",
            "nccl_slowdown",
        }
        missing = required - set(rows[0])
        if missing:
            if "overlap_device_ms" in missing:
                raise ValueError(
                    "calibration CSV predates device-gated timing and is unsafe for overlap modeling"
                )
            raise ValueError(
                f"calibration CSV is missing columns: {', '.join(sorted(missing))}"
            )

        target_architecture = architecture(target)
        embedded_targets = {row["target"].strip().lower() for row in rows}
        if embedded_targets != {target_architecture.name}:
            raise ValueError(
                "calibration CSV target does not exactly match the requested target"
            )
        device_names = {row["device_name"].strip() for row in rows}
        if len(device_names) != 1 or not next(iter(device_names)):
            raise ValueError("one calibration CSV must contain one device product name")
        device_name = next(iter(device_names))
        if not device_name_matches_architecture(device_name, target_architecture):
            raise ValueError(
                "calibration target is inconsistent with the physical device "
                f"product name: target={target!r}, device={device_name!r}"
            )
        compute_capabilities = {row["compute_capability"].strip() for row in rows}
        if compute_capabilities != {target_architecture.compute_capability}:
            raise ValueError(
                "calibration compute capability does not match the target architecture"
            )

        gpu_counts = {int(row["gpus"]) for row in rows}
        if len(gpu_counts) != 1:
            raise ValueError("one calibration CSV cannot mix GPU counts")
        collectives = {
            normalize_calibration_collective(
                row.get("collective", "").strip() or "all_reduce"
            )
            for row in rows
        }
        if len(collectives) != 1:
            raise ValueError("one calibration CSV cannot mix collective operations")
        collective = next(iter(collectives))
        network_tiers = {
            row.get("network_tier", "").strip().lower() or None for row in rows
        }
        if len(network_tiers) != 1:
            raise ValueError("one calibration CSV cannot mix network tiers")
        network_tier = network_tiers.pop()
        if collective in P2P_COLLECTIVES and network_tier not in P2P_NETWORK_TIERS:
            raise ValueError(
                "send/recv calibration requires network_tier intra_node or inter_node"
            )
        if collective not in P2P_COLLECTIVES and network_tier is not None:
            raise ValueError("network_tier is only valid for send/recv calibration")
        sm_counts = {int(row["sm_count"]) for row in rows}
        if len(sm_counts) != 1 or next(iter(sm_counts)) <= 0:
            raise ValueError("one calibration CSV must contain one positive SM count")
        sm_count = next(iter(sm_counts))
        if sm_count != target_architecture.sm_count:
            raise ValueError(
                "calibration SM count does not match the target architecture"
            )
        blocks_per_sm_values = {
            float(row["compute_blocks_per_sm"])
            for row in rows
            if row.get("compute_blocks_per_sm", "").strip()
        }
        if len(blocks_per_sm_values) > 1:
            raise ValueError("one calibration CSV cannot mix compute blocks per SM")
        compute_blocks_values = {
            int(row["compute_blocks"])
            for row in rows
            if row.get("compute_blocks", "").strip()
        }
        if len(compute_blocks_values) > 1:
            raise ValueError(
                "one calibration CSV cannot mix absolute compute block counts"
            )
        compute_event_count_values = {
            int(row["compute_event_count"])
            for row in rows
            if row.get("compute_event_count", "").strip()
        }
        if len(compute_event_count_values) > 1:
            raise ValueError("one calibration CSV cannot mix compute event counts")
        blocks_per_sm = (
            None if not blocks_per_sm_values else next(iter(blocks_per_sm_values))
        )
        compute_blocks = (
            None if not compute_blocks_values else next(iter(compute_blocks_values))
        )
        compute_event_count = (
            None
            if not compute_event_count_values
            else next(iter(compute_event_count_values))
        )
        if blocks_per_sm is not None and blocks_per_sm <= 0:
            raise ValueError("calibration compute blocks per SM must be positive")
        if compute_blocks is not None and compute_blocks <= 0:
            raise ValueError(
                "calibration absolute compute block count must be positive"
            )
        if compute_event_count is not None and compute_event_count <= 0:
            raise ValueError("calibration compute event count must be positive")
        embedded_signatures = {
            row.get("compute_signature", "").strip()
            for row in rows
            if row.get("compute_signature", "").strip()
        }
        if len(embedded_signatures) > 1:
            raise ValueError("one calibration CSV cannot mix compute signatures")
        if (
            compute_signature is not None
            and embedded_signatures
            and embedded_signatures != {compute_signature}
        ):
            raise ValueError(
                "requested compute signature disagrees with calibration CSV"
            )
        if compute_signature is None and embedded_signatures:
            compute_signature = embedded_signatures.pop()
        if compute_signature is None:
            if compute_blocks is not None:
                compute_signature = f"dependent_fma_blocks{compute_blocks}_v1"
            elif blocks_per_sm is not None:
                normalized = (
                    str(int(blocks_per_sm))
                    if blocks_per_sm.is_integer()
                    else format(blocks_per_sm, "g").replace(".", "p")
                )
                compute_signature = f"dependent_fma_bpsm{normalized}_v1"
            else:
                # The original harness used four blocks per SM but predated the
                # explicit domain column. Preserve that legacy profile name.
                compute_signature = "dependent_fma_bpsm4_v1"
        coverage = (
            min(1.0, compute_blocks / sm_count)
            if compute_blocks is not None
            else (None if blocks_per_sm is None else min(1.0, blocks_per_sm))
        )
        priorities = {
            row.get("comm_stream_priority", "normal").strip().lower() for row in rows
        }
        if len(priorities) != 1 or not priorities <= {"normal", "high"}:
            raise ValueError(
                "one calibration CSV must contain one valid stream priority"
            )
        cta_policies = {
            parse_cta_policy_mode(row.get("nccl_cta_policy", "default")) for row in rows
        }
        if len(cta_policies) != 1:
            raise ValueError("one calibration CSV cannot mix NCCL CTA policies")
        cga_cluster_sizes = {
            parse_cga_cluster_size(
                0
                if row.get("nccl_cga_cluster_size", "").strip().lower()
                in {"", "auto", "default"}
                else int(row["nccl_cga_cluster_size"])
            )
            for row in rows
        }
        if len(cga_cluster_sizes) != 1:
            raise ValueError("one calibration CSV cannot mix NCCL CGA cluster sizes")
        nvls_values: set[int | None] = set()
        for row in rows:
            raw_nvls = row.get("nccl_nvls_ctas", "").strip().lower()
            nvls = None if raw_nvls in {"", "auto", "default"} else int(raw_nvls)
            if nvls is not None and nvls <= 0:
                raise ValueError("calibration NVLS CTAs must be positive or auto")
            nvls_values.add(nvls)
        if len(nvls_values) != 1:
            raise ValueError("one calibration CSV cannot mix NVLS CTA settings")
        nvls_runtime_statuses = {
            row.get("nvls_runtime_status", "").strip().lower() or "unknown"
            for row in rows
        }
        if len(nvls_runtime_statuses) != 1 or not nvls_runtime_statuses <= {
            "active",
            "inactive",
            "unknown",
            "not-requested",
        }:
            raise ValueError(
                "one calibration CSV must contain one valid NVLS runtime status"
            )
        nvls_value = next(iter(nvls_values))
        nvls_runtime_status = nvls_runtime_statuses.pop()
        if nvls_value is None:
            nvls_runtime_status = "not-requested"
        elif nvls_runtime_status == "not-requested":
            raise ValueError(
                "an NVLS CTA override cannot have runtime status not-requested"
            )
        software_versions: dict[str, str] = {}
        for key in SOFTWARE_VERSION_KEYS:
            values = {
                row.get(key, "").strip() for row in rows if row.get(key, "").strip()
            }
            if len(values) > 1:
                raise ValueError(
                    f"one calibration CSV cannot mix {key.replace('_', ' ')} values"
                )
            if values:
                software_versions[key] = values.pop()
        validation_columns = {
            "path": "validation_evidence_path",
            "sha256": "validation_evidence_sha256",
            "schema": "validation_evidence_schema",
            "status": "validation_evidence_status",
            "timeline_outcomes": "validation_timeline_outcomes",
            "occupancy_profile_path": "validation_occupancy_profile_path",
            "occupancy_profile_sha256": "validation_occupancy_profile_sha256",
        }
        validation_evidence: dict[str, Any] = {}
        populated_validation_columns = {
            key: {
                row.get(column, "").strip()
                for row in rows
                if row.get(column, "").strip()
            }
            for key, column in validation_columns.items()
        }
        populated = {
            key: values
            for key, values in populated_validation_columns.items()
            if values
        }
        if populated:
            if set(populated) != set(validation_columns) or any(
                len(values) != 1 for values in populated.values()
            ):
                raise ValueError(
                    "calibration validation evidence must be complete and consistent"
                )
            validation_evidence = {
                key: next(iter(values)) for key, values in populated.items()
            }
            if (
                len(str(validation_evidence["sha256"])) != 64
                or len(str(validation_evidence["occupancy_profile_sha256"])) != 64
            ):
                raise ValueError("calibration validation evidence has an invalid hash")
            try:
                outcomes = json.loads(str(validation_evidence["timeline_outcomes"]))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "calibration validation timeline outcomes are invalid"
                ) from error
            if not isinstance(outcomes, dict) or set(outcomes) != {"8", "16"}:
                raise ValueError(
                    "calibration validation timeline outcomes must cover CTA8 and CTA16"
                )
            validation_evidence["timeline_outcomes"] = outcomes
        grouped: dict[tuple[int | None, int], list[dict[str, str]]] = {}
        for row in rows:
            raw_ctas = row["nccl_max_ctas"].strip().lower()
            ctas = None if raw_ctas == "default" else int(raw_ctas)
            if ctas is not None and ctas <= 0:
                raise ValueError("calibration max CTAs must be positive")
            message_bytes = int(row["message_bytes"])
            if message_bytes <= 0:
                raise ValueError("calibration message size must be positive")
            grouped.setdefault((ctas, message_bytes), []).append(row)

        points = []
        for (ctas, message_bytes), samples in grouped.items():

            def median(name: str) -> float:
                return statistics.median(float(row[name]) for row in samples)

            compute_alone_us = median("compute_alone_ms") * 1000.0
            collective_alone_us = median("nccl_alone_ms") * 1000.0
            compute_slowdown = median("compute_slowdown")
            collective_slowdown = median("nccl_slowdown")
            overlap_device_us = median("overlap_device_ms") * 1000.0
            interval_overlap: bool | None = None
            interval_overlap_fraction: float | None = None
            if "event_intervals_overlap" in samples[0]:
                parsed_interval_overlap = []
                for row in samples:
                    raw_overlap = row.get("event_intervals_overlap", "").strip().lower()
                    if raw_overlap not in {"true", "false"}:
                        raise ValueError(
                            "event_intervals_overlap must contain only true or false"
                        )
                    parsed_interval_overlap.append(raw_overlap == "true")
                interval_overlap_fraction = sum(parsed_interval_overlap) / len(
                    parsed_interval_overlap
                )
                if interval_overlap_fraction == 0.5:
                    raise ValueError(
                        "one calibration point has an ambiguous 50% event-interval "
                        "overlap split"
                    )
                if len(set(parsed_interval_overlap)) == 1:
                    interval_overlap = parsed_interval_overlap[0]
            serialized_denominator = (
                compute_alone_us * max(1.0, compute_slowdown) + collective_alone_us
            )
            overlap_regime = (
                "concurrent_slowdown"
                if interval_overlap_fraction is not None
                and interval_overlap_fraction > 0.5
                else (
                    "serialized_compute_first"
                    if interval_overlap_fraction is not None
                    and interval_overlap_fraction < 0.5
                    or (
                        serialized_denominator > 0
                        and overlap_device_us / serialized_denominator >= 0.90
                    )
                    else "concurrent_slowdown"
                )
            )
            points.append(
                CalibrationPoint(
                    message_bytes=message_bytes,
                    max_ctas=ctas,
                    compute_alone_us=compute_alone_us,
                    collective_alone_us=collective_alone_us,
                    overlap_compute_slowdown=compute_slowdown,
                    overlap_collective_slowdown=collective_slowdown,
                    overlap_device_us=overlap_device_us,
                    overlap_regime=overlap_regime,
                    samples=len(samples),
                    event_intervals_overlap=interval_overlap,
                    event_interval_overlap_fraction=interval_overlap_fraction,
                )
            )
        return cls(
            target=target,
            device_name=device_name,
            compute_capability=target_architecture.compute_capability,
            hardware_identity_status="exact-product-name-match",
            gpu_count=gpu_counts.pop(),
            collective=collective,
            compute_signature=compute_signature,
            source_path=str(path.resolve()),
            points=tuple(
                sorted(
                    points,
                    key=lambda point: (
                        -1 if point.max_ctas is None else point.max_ctas,
                        point.message_bytes,
                    ),
                )
            ),
            network_tier=network_tier,
            comm_stream_priority=priorities.pop(),
            sm_count=sm_count,
            compute_blocks_per_sm=blocks_per_sm,
            compute_blocks=compute_blocks,
            compute_event_count=compute_event_count,
            compute_sm_coverage_upper_bound=coverage,
            nccl_cta_policy=cta_policies.pop(),
            nccl_nvls_ctas=nvls_value,
            nccl_cga_cluster_size=next(iter(cga_cluster_sizes)),
            nvls_runtime_status=nvls_runtime_status,
            software_versions=software_versions,
            validation_evidence=validation_evidence,
        )

    def _curve(self, max_ctas: int | None) -> list[CalibrationPoint]:
        return sorted(
            (point for point in self.points if point.max_ctas == max_ctas),
            key=lambda point: point.message_bytes,
        )

    def collective_duration(
        self,
        message_bytes: int,
        group_size: int,
        target: str,
        max_ctas: int | None,
        comm_stream_priority: str = "normal",
        nccl_cta_policy: int = 0,
        nccl_nvls_ctas: int | None = None,
        nccl_cga_cluster_size: int = 0,
        collective: str = "all_reduce",
        network_tier: str | None = None,
    ) -> CalibrationLookup | None:
        if (
            target != self.target
            or group_size != self.gpu_count
            or collective != self.collective
            or network_tier != self.network_tier
            or comm_stream_priority != self.comm_stream_priority
            or nccl_cta_policy != self.nccl_cta_policy
            or nccl_nvls_ctas != self.nccl_nvls_ctas
            or nccl_cga_cluster_size != self.nccl_cga_cluster_size
        ):
            return None
        curve = self._curve(max_ctas)
        if not curve:
            return None
        for point in curve:
            if point.message_bytes == message_bytes:
                return CalibrationLookup(
                    point.collective_alone_us,
                    "measured-nccl-calibration",
                    message_bytes,
                    message_bytes,
                )
        if (
            message_bytes < curve[0].message_bytes
            or message_bytes > curve[-1].message_bytes
        ):
            return None
        for lower, upper in zip(curve, curve[1:]):
            if lower.message_bytes < message_bytes < upper.message_bytes:
                return CalibrationLookup(
                    _affine_message_interpolation(
                        message_bytes,
                        lower.message_bytes,
                        upper.message_bytes,
                        lower.collective_alone_us,
                        upper.collective_alone_us,
                    ),
                    "interpolated-nccl-calibration",
                    lower.message_bytes,
                    upper.message_bytes,
                )
        return None

    def interference(
        self,
        message_bytes: int,
        group_size: int,
        target: str,
        max_ctas: int | None,
        compute_signature: str | None,
        comm_stream_priority: str = "normal",
        nccl_cta_policy: int = 0,
        nccl_nvls_ctas: int | None = None,
        nccl_cga_cluster_size: int = 0,
        collective: str = "all_reduce",
        network_tier: str | None = None,
    ) -> InterferenceLookup | None:
        if (
            target != self.target
            or group_size != self.gpu_count
            or collective != self.collective
            or network_tier != self.network_tier
            or compute_signature != self.compute_signature
            or comm_stream_priority != self.comm_stream_priority
            or nccl_cta_policy != self.nccl_cta_policy
            or nccl_nvls_ctas != self.nccl_nvls_ctas
            or nccl_cga_cluster_size != self.nccl_cga_cluster_size
        ):
            return None
        curve = self._curve(max_ctas)
        if not curve:
            return None
        for point in curve:
            if point.message_bytes == message_bytes:
                return InterferenceLookup(
                    compute_slowdown=max(1.0, point.overlap_compute_slowdown),
                    collective_slowdown=max(1.0, point.overlap_collective_slowdown),
                    regime=point.overlap_regime,
                    source="measured-overlap-calibration",
                    lower_bytes=message_bytes,
                    upper_bytes=message_bytes,
                    compute_alone_us=point.compute_alone_us,
                    collective_alone_us=point.collective_alone_us,
                    overlap_device_us=point.overlap_device_us,
                    compute_event_count=self.compute_event_count,
                    validation_evidence_status=self.validation_evidence.get("status"),
                    validation_evidence_sha256=self.validation_evidence.get("sha256"),
                    validation_timeline_outcome=self.validation_evidence.get(
                        "timeline_outcomes", {}
                    ).get(str(max_ctas)),
                )
        if (
            message_bytes < curve[0].message_bytes
            or message_bytes > curve[-1].message_bytes
        ):
            return None
        for lower, upper in zip(curve, curve[1:]):
            if lower.message_bytes < message_bytes < upper.message_bytes:
                if lower.overlap_regime != upper.overlap_regime:
                    return None
                fraction = (math.log(message_bytes) - math.log(lower.message_bytes)) / (
                    math.log(upper.message_bytes) - math.log(lower.message_bytes)
                )
                compute_slowdown = lower.overlap_compute_slowdown + fraction * (
                    upper.overlap_compute_slowdown - lower.overlap_compute_slowdown
                )
                collective_slowdown = lower.overlap_collective_slowdown + fraction * (
                    upper.overlap_collective_slowdown
                    - lower.overlap_collective_slowdown
                )
                return InterferenceLookup(
                    compute_slowdown=max(1.0, compute_slowdown),
                    collective_slowdown=max(1.0, collective_slowdown),
                    regime=lower.overlap_regime,
                    source="interpolated-overlap-calibration",
                    lower_bytes=lower.message_bytes,
                    upper_bytes=upper.message_bytes,
                    compute_alone_us=_affine_message_interpolation(
                        message_bytes,
                        lower.message_bytes,
                        upper.message_bytes,
                        lower.compute_alone_us,
                        upper.compute_alone_us,
                    ),
                    collective_alone_us=_affine_message_interpolation(
                        message_bytes,
                        lower.message_bytes,
                        upper.message_bytes,
                        lower.collective_alone_us,
                        upper.collective_alone_us,
                    ),
                    overlap_device_us=_affine_message_interpolation(
                        message_bytes,
                        lower.message_bytes,
                        upper.message_bytes,
                        lower.overlap_device_us,
                        upper.overlap_device_us,
                    ),
                    compute_event_count=self.compute_event_count,
                    validation_evidence_status=self.validation_evidence.get("status"),
                    validation_evidence_sha256=self.validation_evidence.get("sha256"),
                    validation_timeline_outcome=None,
                )
        return None

    def to_summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "hardware_identity_status": self.hardware_identity_status,
            "gpu_count": self.gpu_count,
            "collective": self.collective,
            "network_tier": self.network_tier,
            "compute_signature": self.compute_signature,
            "comm_stream_priority": self.comm_stream_priority,
            "sm_count": self.sm_count,
            "compute_blocks_per_sm": self.compute_blocks_per_sm,
            "compute_blocks": self.compute_blocks,
            "compute_event_count": self.compute_event_count,
            "compute_sm_coverage_upper_bound": self.compute_sm_coverage_upper_bound,
            "nccl_cta_policy": cta_policy_name(self.nccl_cta_policy),
            "nccl_cta_policy_flag": self.nccl_cta_policy,
            "nccl_nvls_ctas": self.nccl_nvls_ctas,
            "nccl_cga_cluster_size": self.nccl_cga_cluster_size,
            "nvls_runtime_status": self.nvls_runtime_status,
            "software_versions": dict(sorted(self.software_versions.items())),
            "validation_evidence": self.validation_evidence,
            "interpolation": {
                "duration_model": "affine-in-message-bytes-alpha-beta",
                "slowdown_model": "linear-in-log-message-bytes",
                "extrapolation": "disabled",
            },
            "source_path": self.source_path,
            "points": [asdict(point) for point in self.points],
        }


@dataclass(frozen=True)
class OverlapCalibrationLibrary:
    """A conservative collection of independently versioned overlap profiles."""

    profiles: tuple[OverlapCalibration, ...]

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError("an overlap calibration library cannot be empty")
        domains = [
            (
                profile.target,
                profile.device_name,
                profile.compute_capability,
                profile.hardware_identity_status,
                profile.gpu_count,
                profile.collective,
                profile.network_tier,
                profile.compute_signature,
                profile.compute_event_count,
                profile.comm_stream_priority,
                profile.nccl_cta_policy,
                profile.nccl_nvls_ctas,
                profile.nccl_cga_cluster_size,
                profile.nvls_runtime_status,
                tuple(sorted(profile.software_versions.items())),
            )
            for profile in self.profiles
        ]
        if len(domains) != len(set(domains)):
            raise ValueError("overlap calibration library contains a duplicate domain")

    @property
    def target(self) -> str:
        targets = {profile.target for profile in self.profiles}
        return next(iter(targets)) if len(targets) == 1 else "mixed"

    def collective_duration(
        self,
        message_bytes: int,
        group_size: int,
        target: str,
        max_ctas: int | None,
        comm_stream_priority: str = "normal",
        nccl_cta_policy: int = 0,
        nccl_nvls_ctas: int | None = None,
        nccl_cga_cluster_size: int = 0,
        collective: str = "all_reduce",
        network_tier: str | None = None,
    ) -> CalibrationLookup | None:
        matches = [
            (lookup, profile.nvls_runtime_status)
            for profile in self.profiles
            if (
                lookup := profile.collective_duration(
                    message_bytes,
                    group_size,
                    target,
                    max_ctas,
                    comm_stream_priority,
                    nccl_cta_policy,
                    nccl_nvls_ctas,
                    nccl_cga_cluster_size,
                    collective,
                    network_tier,
                )
            )
            is not None
        ]
        if not matches:
            return None
        if nccl_nvls_ctas is not None and len({status for _, status in matches}) != 1:
            # Active NVLS and fallback measurements are distinct execution
            # domains even when their requested communicator config is equal.
            return None
        if len(matches) == 1:
            return matches[0][0]
        # Repeated isolated-NCCL observations from different compute profiles
        # are replicates. Aggregate rather than selecting the fastest run.
        return CalibrationLookup(
            duration_us=statistics.median(item.duration_us for item, _ in matches),
            source="ensemble-nccl-calibration",
            lower_bytes=min(item.lower_bytes for item, _ in matches),
            upper_bytes=max(item.upper_bytes for item, _ in matches),
        )

    def interference(
        self,
        message_bytes: int,
        group_size: int,
        target: str,
        max_ctas: int | None,
        compute_signature: str | None,
        comm_stream_priority: str = "normal",
        nccl_cta_policy: int = 0,
        nccl_nvls_ctas: int | None = None,
        nccl_cga_cluster_size: int = 0,
        collective: str = "all_reduce",
        network_tier: str | None = None,
    ) -> InterferenceLookup | None:
        matches = [
            lookup
            for profile in self.profiles
            if (
                lookup := profile.interference(
                    message_bytes,
                    group_size,
                    target,
                    max_ctas,
                    compute_signature,
                    comm_stream_priority,
                    nccl_cta_policy,
                    nccl_nvls_ctas,
                    nccl_cga_cluster_size,
                    collective,
                    network_tier,
                )
            )
            is not None
        ]
        # Duplicate exact domains are rejected at construction, so ambiguity
        # here can only arise from malformed future profile implementations.
        return matches[0] if len(matches) == 1 else None

    def to_summary(self) -> dict[str, Any]:
        return {
            "kind": "overlap-calibration-library",
            "profile_count": len(self.profiles),
            "profiles": [profile.to_summary() for profile in self.profiles],
        }


OverlapCalibrationModel = OverlapCalibration | OverlapCalibrationLibrary


def nvls_runtime_evidence(
    calibration: OverlapCalibrationModel | None,
    target: str,
    nccl_cta_policy: int,
    nccl_nvls_ctas: int | None,
    nccl_cga_cluster_size: int = 0,
) -> dict[str, Any]:
    """Summarize algorithm-selection evidence separately from API support."""

    if nccl_nvls_ctas is None:
        return {
            "status": "not-requested",
            "applicability": "not-requested",
            "target": target,
            "source_paths": [],
        }
    if calibration is None:
        return {
            "status": "unknown",
            "applicability": "no-calibration",
            "target": target,
            "source_paths": [],
        }
    profiles = (
        calibration.profiles
        if isinstance(calibration, OverlapCalibrationLibrary)
        else (calibration,)
    )
    policy_matching = [
        profile
        for profile in profiles
        if profile.nccl_cta_policy == nccl_cta_policy
        and profile.nccl_nvls_ctas == nccl_nvls_ctas
        and profile.nccl_cga_cluster_size == nccl_cga_cluster_size
    ]
    matching = [profile for profile in policy_matching if profile.target == target]
    if not matching:
        return {
            "status": "unknown",
            "applicability": (
                "target-mismatch" if policy_matching else "policy-unmatched"
            ),
            "target": target,
            "source_paths": [],
            "rejected_source_paths": sorted(
                {profile.source_path for profile in policy_matching}
            ),
        }
    statuses = {profile.nvls_runtime_status for profile in matching}
    status = statuses.pop() if len(statuses) == 1 else "conflicting"
    return {
        "status": status,
        "applicability": "exact-target-policy-match",
        "target": target,
        "source_paths": sorted({profile.source_path for profile in matching}),
    }


def select_calibration_environment(
    calibration: OverlapCalibrationModel,
    trace_source: dict[str, Any],
) -> tuple[OverlapCalibrationModel | None, dict[str, Any]]:
    """Exclude exact version mismatches and report weaker provenance states."""

    profiles = (
        calibration.profiles
        if isinstance(calibration, OverlapCalibrationLibrary)
        else (calibration,)
    )
    source_versions = {
        key: str(trace_source[key])
        for key in SOFTWARE_VERSION_KEYS
        if trace_source.get(key) not in (None, "")
    }
    assessments: list[dict[str, Any]] = []
    candidates: list[tuple[OverlapCalibration, dict[str, Any]]] = []
    for profile in profiles:
        expected = dict(sorted(profile.software_versions.items()))
        mismatches = {
            key: {"expected": value, "observed": source_versions[key]}
            for key, value in expected.items()
            if key in source_versions and source_versions[key] != value
        }
        missing = sorted(set(expected) - set(source_versions))
        if mismatches:
            status = "mismatch"
        elif not expected:
            status = "unspecified-calibration-versions"
        elif missing:
            status = "unverifiable-trace-versions"
        else:
            status = "exact-match"
        assessment = {
            "source_path": profile.source_path,
            "compute_signature": profile.compute_signature,
            "gpu_count": profile.gpu_count,
            "expected": expected,
            "observed": source_versions,
            "missing_from_trace": missing,
            "mismatches": mismatches,
            "status": status,
        }
        assessments.append(assessment)
        if status != "mismatch":
            candidates.append((profile, assessment))

    exact = [profile for profile, item in candidates if item["status"] == "exact-match"]
    if exact:
        # Never aggregate a versionless/wildcard profile with a profile whose
        # complete version domain was verified for this trace.
        usable = exact
        for _, item in candidates:
            if item["status"] != "exact-match":
                item["status"] = "superseded-by-exact-match"
    else:
        usable = [profile for profile, _ in candidates]
    if not usable:
        selected: OverlapCalibrationModel | None = None
    elif len(usable) == 1:
        selected = usable[0]
    else:
        selected = OverlapCalibrationLibrary(tuple(usable))
    return selected, {
        "source_versions": source_versions,
        "profile_count": len(profiles),
        "usable_profile_count": len(usable),
        "excluded_profile_count": len(profiles) - len(usable),
        "profiles": assessments,
        "status": (
            "all-excluded"
            if not usable
            else ("exact" if exact else "weak-or-mixed-provenance")
        ),
    }
