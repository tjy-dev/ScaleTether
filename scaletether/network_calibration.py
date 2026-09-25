from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import architecture, assess_hardware_identity


SOFTWARE_VERSION_KEYS = (
    "torch_version",
    "cuda_version",
    "cuda_driver_version",
    "nccl_version",
)


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError(f"network contention {field} must be nonnegative")
    return result


@dataclass(frozen=True)
class NetworkContentionOperation:
    collective: str
    message_bytes: int
    group_size: int
    group_role: str | None
    network_resources: tuple[str, ...]
    network_tier: str | None
    max_ctas: int | None
    comm_stream_priority: str
    nccl_cta_policy: int
    nccl_nvls_ctas: int | None
    nccl_cga_cluster_size: int
    progress_rate: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkContentionOperation":
        resources = tuple(str(item) for item in data["network_resources"])
        if not resources or any(not item for item in resources):
            raise ValueError("network contention operation needs routed resources")
        operation = cls(
            collective=str(data["collective"]),
            message_bytes=int(data["message_bytes"]),
            group_size=int(data["group_size"]),
            group_role=(
                None if data.get("group_role") is None else str(data["group_role"])
            ),
            network_resources=resources,
            network_tier=(
                None if data.get("network_tier") is None else str(data["network_tier"])
            ),
            max_ctas=_optional_int(data.get("max_ctas"), "max_ctas"),
            comm_stream_priority=str(data["comm_stream_priority"]),
            nccl_cta_policy=int(data["nccl_cta_policy"]),
            nccl_nvls_ctas=_optional_int(data.get("nccl_nvls_ctas"), "nccl_nvls_ctas"),
            nccl_cga_cluster_size=int(data["nccl_cga_cluster_size"]),
            progress_rate=float(data["progress_rate"]),
        )
        if not operation.collective or operation.message_bytes < 0:
            raise ValueError("network contention operation domain is invalid")
        if operation.group_size <= 0:
            raise ValueError("network contention group_size must be positive")
        if operation.comm_stream_priority not in {"normal", "high"}:
            raise ValueError(
                "network contention stream priority must be normal or high"
            )
        if operation.nccl_cta_policy < 0 or operation.nccl_cga_cluster_size < 0:
            raise ValueError(
                "network contention NCCL policy values must be nonnegative"
            )
        if not 0.0 < operation.progress_rate <= 1.0:
            raise ValueError("network contention progress_rate must be in (0, 1]")
        return operation

    def domain_key(self) -> tuple[Any, ...]:
        return (
            self.collective,
            self.message_bytes,
            self.group_size,
            self.group_role,
            self.network_resources,
            self.network_tier,
            self.max_ctas,
            self.comm_stream_priority,
            self.nccl_cta_policy,
            self.nccl_nvls_ctas,
            self.nccl_cga_cluster_size,
        )

    def domain_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("progress_rate")
        result["network_resources"] = list(self.network_resources)
        return result


@dataclass(frozen=True)
class NetworkContentionPoint:
    resource: str
    operations: tuple[NetworkContentionOperation, ...]
    samples: int

    def key(self) -> tuple[Any, ...]:
        return self.resource, tuple(
            sorted(item.domain_key() for item in self.operations)
        )


@dataclass(frozen=True)
class NetworkContentionLookup:
    progress_rates: tuple[float, ...]
    samples: int
    source: str = "measured-exact-network-contention-calibration"


@dataclass(frozen=True)
class NetworkContentionCalibration:
    source_path: str
    source_sha256: str
    target: str
    device_name: str
    compute_capability: str
    sm_count: int
    topology: str
    software_versions: dict[str, str]
    points: tuple[NetworkContentionPoint, ...]

    @classmethod
    def load_json(cls, path: Path) -> "NetworkContentionCalibration":
        raw_bytes = path.read_bytes()
        data = json.loads(raw_bytes)
        if data.get("schema") != "scaletether-network-contention-calibration-v1":
            raise ValueError("unsupported network contention calibration schema")
        target = str(data.get("target", "")).strip().lower()
        target_architecture = architecture(target)
        device = data.get("device")
        if not isinstance(device, dict):
            raise ValueError(
                "network contention calibration requires exact physical device identity"
            )
        identity = assess_hardware_identity(
            {
                "target": target,
                "device_name": device.get("name"),
                "compute_capability": device.get("compute_capability"),
                "sm_count": device.get("sm_count"),
            },
            target_architecture,
        )
        if identity["status"] != "exact-product-match":
            raise ValueError(
                "network contention calibration physical device is outside the "
                "exact target domain"
            )
        observed_device = identity["observed"]
        versions = data.get("software_versions")
        if not isinstance(versions, dict) or any(
            not str(versions.get(key, "")).strip() for key in SOFTWARE_VERSION_KEYS
        ):
            raise ValueError(
                "network contention calibration requires exact torch/CUDA/driver/NCCL versions"
            )
        points = []
        seen = set()
        for index, raw_point in enumerate(data.get("points", [])):
            operations = tuple(
                NetworkContentionOperation.from_dict(item)
                for item in raw_point.get("operations", [])
            )
            samples = int(raw_point.get("samples", 0))
            point = NetworkContentionPoint(
                resource=str(raw_point.get("resource", "")),
                operations=operations,
                samples=samples,
            )
            if not point.resource or len(point.operations) < 2:
                raise ValueError(
                    f"network contention point {index} needs a resource and two operations"
                )
            if samples < 5:
                raise ValueError(
                    f"network contention point {index} requires at least five repetitions"
                )
            duplicate_rates: dict[tuple[Any, ...], float] = {}
            for operation in operations:
                previous = duplicate_rates.setdefault(
                    operation.domain_key(), operation.progress_rate
                )
                if previous != operation.progress_rate:
                    raise ValueError(
                        "identical network contention operation domains require equal rates"
                    )
            if point.key() in seen:
                raise ValueError("duplicate network contention calibration domain")
            seen.add(point.key())
            points.append(point)
        if not points:
            raise ValueError("network contention calibration contains no points")
        return cls(
            source_path=str(path.resolve()),
            source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            target=target,
            device_name=str(observed_device["device_name"]),
            compute_capability=str(observed_device["compute_capability"]),
            sm_count=int(observed_device["sm_count"]),
            topology=str(data["topology"]),
            software_versions={
                key: str(versions[key]) for key in SOFTWARE_VERSION_KEYS
            },
            points=tuple(sorted(points, key=lambda point: str(point.key()))),
        )

    def lookup(
        self,
        resource: str,
        operations: list[dict[str, Any]],
        target: str,
        topology: str,
        trace_source: dict[str, Any],
    ) -> NetworkContentionLookup | None:
        if target.lower() != self.target or topology != self.topology:
            return None
        if (
            assess_hardware_identity(trace_source, architecture(self.target))["status"]
            != "exact-product-match"
        ):
            return None
        if any(
            str(trace_source.get(key, "")) != self.software_versions[key]
            for key in SOFTWARE_VERSION_KEYS
        ):
            return None
        try:
            runtime = tuple(
                NetworkContentionOperation.from_dict({**item, "progress_rate": 1.0})
                for item in operations
            )
        except (KeyError, TypeError, ValueError):
            return None
        key = resource, tuple(sorted(item.domain_key() for item in runtime))
        matches = [point for point in self.points if point.key() == key]
        if len(matches) != 1:
            return None
        rates = {
            item.domain_key(): item.progress_rate for item in matches[0].operations
        }
        return NetworkContentionLookup(
            progress_rates=tuple(rates[item.domain_key()] for item in runtime),
            samples=matches[0].samples,
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "kind": "exact-operation-set-network-contention",
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "target": self.target,
            "device": {
                "name": self.device_name,
                "compute_capability": self.compute_capability,
                "sm_count": self.sm_count,
            },
            "topology": self.topology,
            "software_versions": dict(sorted(self.software_versions.items())),
            "point_count": len(self.points),
        }
