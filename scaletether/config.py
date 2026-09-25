from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import re


@dataclass(frozen=True)
class Parallelism:
    tp: int
    pp: int
    dp: int
    ep: int = 1

    @classmethod
    def parse(cls, text: str, gpus: int) -> "Parallelism":
        values: dict[str, int] = {}
        for assignment in text.split(","):
            if "=" not in assignment:
                raise ValueError(f"invalid parallelism assignment {assignment!r}")
            name, raw_value = assignment.split("=", 1)
            name = name.strip().lower()
            if name not in {"tp", "pp", "dp", "ep"} or name in values:
                raise ValueError(f"invalid or duplicate parallelism dimension {name!r}")
            value = int(raw_value)
            if value <= 0:
                raise ValueError(f"parallelism dimension {name!r} must be positive")
            values[name] = value
        missing = {"tp", "pp", "dp"} - values.keys()
        if missing:
            raise ValueError(f"parallelism is missing {', '.join(sorted(missing))}")
        result = cls(
            tp=values["tp"], pp=values["pp"], dp=values["dp"], ep=values.get("ep", 1)
        )
        # EP normally partitions an existing DP or TP dimension and therefore is
        # not multiplied into the global world size.
        if result.tp * result.pp * result.dp != gpus:
            raise ValueError(
                f"tp*pp*dp={result.tp * result.pp * result.dp} does not equal --gpus={gpus}"
            )
        if result.ep > max(result.dp, result.tp):
            raise ValueError("ep cannot exceed both dp and tp")
        return result

    def group_size(self, role: str) -> int:
        sizes = {"tp": self.tp, "pp": self.pp, "dp": self.dp, "ep": self.ep}
        if role not in sizes:
            raise ValueError(f"unknown process-group role {role!r}")
        return sizes[role]

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class NetworkTier:
    bandwidth_gbps: float
    latency_us: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str) -> "NetworkTier":
        tier = cls(
            bandwidth_gbps=float(data["bandwidth_GBps"]),
            latency_us=float(data["latency_us"]),
        )
        if tier.bandwidth_gbps <= 0 or tier.latency_us < 0:
            raise ValueError(f"topology tier {name!r} has invalid bandwidth or latency")
        return tier


@dataclass(frozen=True)
class Topology:
    name: str
    gpus_per_node: int
    nodes: int | None
    intra_node: NetworkTier
    inter_node: NetworkTier
    metadata: dict[str, Any]
    routes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    target: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Topology":
        if str(data.get("schema_version", "0.1")) != "0.1":
            raise ValueError("unsupported topology schema version")
        tiers = data.get("tiers", {})
        raw_routes = data.get("routing", {})
        if not isinstance(raw_routes, dict):
            raise ValueError("topology routing must be a mapping")
        routes: dict[str, tuple[str, ...]] = {}
        for tier_name in ("intra_node", "inter_node"):
            raw_route = raw_routes.get(tier_name)
            if raw_route is None:
                continue
            if not isinstance(raw_route, list) or not raw_route:
                raise ValueError(
                    f"topology route {tier_name!r} must be a non-empty list"
                )
            route = tuple(str(resource).strip() for resource in raw_route)
            if any(not resource for resource in route) or len(set(route)) != len(route):
                raise ValueError(
                    f"topology route {tier_name!r} has empty or duplicate resources"
                )
            routes[tier_name] = route
        topology = cls(
            name=str(data.get("name", "unnamed")),
            gpus_per_node=int(data["gpus_per_node"]),
            nodes=None if data.get("nodes") is None else int(data["nodes"]),
            intra_node=NetworkTier.from_dict(tiers["intra_node"], "intra_node"),
            inter_node=NetworkTier.from_dict(tiers["inter_node"], "inter_node"),
            metadata=dict(data.get("metadata", {})),
            routes=routes,
            target=(
                None
                if data.get("target") in (None, "")
                else str(data["target"]).strip().lower()
            ),
        )
        if topology.gpus_per_node <= 0 or (
            topology.nodes is not None and topology.nodes <= 0
        ):
            raise ValueError("topology gpus_per_node and nodes must be positive")
        return topology

    def validate_capacity(self, gpus: int) -> None:
        if self.nodes is not None and self.nodes * self.gpus_per_node < gpus:
            raise ValueError(
                f"topology capacity is {self.nodes * self.gpus_per_node} GPUs, below requested {gpus}"
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": "0.1",
            "name": self.name,
            "gpus_per_node": self.gpus_per_node,
            "nodes": self.nodes,
            "tiers": {
                "intra_node": {
                    "bandwidth_GBps": self.intra_node.bandwidth_gbps,
                    "latency_us": self.intra_node.latency_us,
                },
                "inter_node": {
                    "bandwidth_GBps": self.inter_node.bandwidth_gbps,
                    "latency_us": self.inter_node.latency_us,
                },
            },
            "metadata": self.metadata,
        }
        if self.target is not None:
            result["target"] = self.target
        if self.routes:
            result["routing"] = {
                name: list(resources) for name, resources in sorted(self.routes.items())
            }
        return result

    def route_resources(self, tier: str) -> tuple[str, ...]:
        if tier not in {"intra_node", "inter_node"}:
            raise ValueError(f"unknown topology tier {tier!r}")
        if tier in self.routes:
            return self.routes[tier]
        return (tier.replace("_", "-"),)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "YAML topology files require PyYAML; install scaletether with its dependencies"
            ) from error
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def load_topology(path: Path, gpus: int) -> Topology:
    data = _load_mapping(path)
    if "infrastructure" in data or (
        "devices" in data and "instances" in data and "edges" in data
    ):
        from .infragraph import topology_from_infragraph

        topology = topology_from_infragraph(data).topology
    else:
        topology = Topology.from_dict(data)
    topology.validate_capacity(gpus)
    return topology


@dataclass(frozen=True)
class Architecture:
    name: str
    family: str
    compute_capability: str
    sm_count: int


ARCHITECTURES = {
    "rtx4090": Architecture("rtx4090", "ada", "8.9", 128),
    "h100": Architecture("h100", "hopper", "9.0", 132),
    "h200": Architecture("h200", "hopper", "9.0", 132),
    "b200": Architecture("b200", "blackwell", "10.0", 148),
    "gb200": Architecture("gb200", "blackwell", "10.0", 148),
}


_DEVICE_PRODUCT_PATTERNS = {
    "rtx4090": r"(?:^|\W)RTX\s*4090(?:\W|$)",
    "h100": r"(?:^|\W)H100(?:\W|$)",
    "h200": r"(?:^|\W)H200(?:\W|$)",
    "b200": r"(?:^|\W)B200(?:\W|$)",
    "gb200": r"(?:^|\W)GB200(?:\W|$)",
}


def device_name_matches_architecture(name: str, target: Architecture) -> bool:
    """Match the physical product name, not only CC and SM count."""

    pattern = _DEVICE_PRODUCT_PATTERNS.get(target.name)
    return bool(pattern and re.search(pattern, name.strip(), flags=re.IGNORECASE))


def assess_hardware_identity(
    source: dict[str, Any], target: Architecture
) -> dict[str, Any]:
    """Assess whether provenance names one exact physical target product.

    Compute capability and SM count are not sufficient: H100/H200 and
    B200/GB200 intentionally remain separate evidence domains.
    """

    observed_target = str(source.get("target", "")).strip().lower()
    device_name = str(source.get("device_name", "")).strip()
    compute_capability = str(source.get("compute_capability", "")).strip()
    raw_sm_count = source.get("sm_count")
    try:
        sm_count = None if isinstance(raw_sm_count, bool) else int(raw_sm_count)
    except (TypeError, ValueError):
        sm_count = None
    if (
        sm_count is not None
        and isinstance(raw_sm_count, float)
        and raw_sm_count != sm_count
    ):
        sm_count = None

    missing = []
    if not observed_target:
        missing.append("target")
    if not device_name:
        missing.append("device_name")
    if not compute_capability:
        missing.append("compute_capability")
    if sm_count is None or sm_count <= 0:
        missing.append("sm_count")

    mismatches: dict[str, dict[str, Any]] = {}
    if observed_target and observed_target != target.name:
        mismatches["target"] = {
            "expected": target.name,
            "observed": observed_target,
        }
    if device_name and not device_name_matches_architecture(device_name, target):
        mismatches["device_name"] = {
            "expected_product": target.name,
            "observed": device_name,
        }
    if compute_capability and compute_capability != target.compute_capability:
        mismatches["compute_capability"] = {
            "expected": target.compute_capability,
            "observed": compute_capability,
        }
    if sm_count is not None and sm_count > 0 and sm_count != target.sm_count:
        mismatches["sm_count"] = {
            "expected": target.sm_count,
            "observed": sm_count,
        }

    if mismatches:
        status = "mismatch"
    elif missing:
        status = "unverified-missing-fields"
    else:
        status = "exact-product-match"
    return {
        "status": status,
        "expected": {
            "target": target.name,
            "physical_product": target.name,
            "compute_capability": target.compute_capability,
            "sm_count": target.sm_count,
        },
        "observed": {
            "target": observed_target or None,
            "device_name": device_name or None,
            "compute_capability": compute_capability or None,
            "sm_count": sm_count if sm_count is not None and sm_count > 0 else None,
        },
        "missing_fields": missing,
        "mismatches": mismatches,
    }


def architecture(name: str) -> Architecture:
    try:
        return ARCHITECTURES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown target {name!r}; choose one of {', '.join(sorted(ARCHITECTURES))}"
        ) from error
