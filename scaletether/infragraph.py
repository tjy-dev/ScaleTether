from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import NetworkTier, Topology


INFRAGRAPH_SCHEMA_VERSION = "2.4.0"
INFRAGRAPH_COMMIT = "2701fa51334b22767744e404b613503783ae9ecb"
ASTRA_SERVICE_VERSION = "1.4.0"
ASTRA_SERVICE_COMMIT = "00bca4549a6cf5d44177fb0863c71a91a095ef13"
ASTRA_SERVICE_API_VERSION = "1.4.0"
ASTRA_SERVICE_INFRAGRAPH_VERSION = "2.0.0"
ASTRA_SERVICE_INFRAGRAPH_COMMIT = "8bf8483a6d2e22de7089369b29b5b5ee50728b62"
ASTRA_SERVICE_CORE_COMMIT = "518bd513ae110428cd62eb60efc0f3993fd53c70"
PROFILE = "scaletether-two-tier-v1"
_DESCRIPTION_PREFIX = f"{PROFILE}:"


@dataclass(frozen=True)
class InfraGraphProfile:
    topology: "Topology"
    effective_nodes: int


def _physical(tier: "NetworkTier") -> dict[str, Any]:
    return {
        "bandwidth": {
            "choice": "gigabytes_per_second",
            "gigabytes_per_second": tier.bandwidth_gbps,
        },
        "latency": {"choice": "us", "us": tier.latency_us},
    }


def topology_to_infragraph(
    topology: "Topology", required_gpus: int | None = None
) -> dict[str, Any]:
    """Translate the deliberately small scaletether topology into InfraGraph.

    InfraGraph is more expressive than scaletether's current two-tier model.  The
    profile marker makes the generated subset reversible and prevents a future
    importer from silently flattening arbitrary InfraGraph graphs.
    """

    if required_gpus is not None and required_gpus <= 0:
        raise ValueError("required_gpus must be positive")
    minimum_nodes = (
        1
        if required_gpus is None
        else math.ceil(required_gpus / topology.gpus_per_node)
    )
    effective_nodes = topology.nodes or minimum_nodes
    if effective_nodes < minimum_nodes:
        raise ValueError("topology has insufficient nodes for the requested GPUs")
    profile = {
        "profile": PROFILE,
        "infragraph_schema_version": INFRAGRAPH_SCHEMA_VERSION,
        "infragraph_commit": INFRAGRAPH_COMMIT,
        "effective_nodes": effective_nodes,
        "source_topology": topology.to_dict(),
    }
    description = _DESCRIPTION_PREFIX + json.dumps(
        profile, sort_keys=True, separators=(",", ":")
    )
    intra_physical = _physical(topology.intra_node)
    inter_physical = _physical(topology.inter_node)
    return {
        "name": "scaletether-cluster",
        "description": description,
        "devices": [
            {
                "name": "scaletether-compute-node",
                "description": "Compute node emitted by scaletether",
                "components": [
                    {
                        "name": "gpu",
                        "description": "GPU/XPU accelerator",
                        "count": topology.gpus_per_node,
                        "choice": "xpu",
                    },
                    {
                        "name": "scaleup-switch",
                        "description": "Abstract intra-node scale-up fabric",
                        "count": 1,
                        "choice": "switch",
                    },
                    {
                        "name": "nic",
                        "description": "Abstract scale-out NIC",
                        "count": 1,
                        "choice": "nic",
                    },
                ],
                "links": [
                    {
                        "name": "scaletether-intra-node",
                        "description": "Effective intra-node tier",
                        "physical": intra_physical,
                    }
                ],
                "edges": [
                    {
                        "ep1": {"component": f"gpu[0:{topology.gpus_per_node}]"},
                        "ep2": {"component": "scaleup-switch[0]"},
                        "scheme": "many2many",
                        "link": "scaletether-intra-node",
                    },
                    {
                        "ep1": {"component": "scaleup-switch[0]"},
                        "ep2": {"component": "nic[0]"},
                        "scheme": "one2one",
                        "link": "scaletether-intra-node",
                    },
                ],
            },
            {
                "name": "scaletether-fabric-switch",
                "description": "Abstract scale-out fabric switch emitted by scaletether",
                "components": [
                    {"name": "fabric-asic", "count": 1, "choice": "switch"},
                    {
                        "name": "fabric-port",
                        "count": effective_nodes,
                        "choice": "port",
                    },
                ],
                "links": [
                    {
                        "name": "scaletether-fabric-backplane",
                        "description": "Effective scale-out switch tier",
                        "physical": inter_physical,
                    }
                ],
                "edges": [
                    {
                        "ep1": {"component": "fabric-asic[0]"},
                        "ep2": {"component": f"fabric-port[0:{effective_nodes}]"},
                        "scheme": "many2many",
                        "link": "scaletether-fabric-backplane",
                    }
                ],
            },
        ],
        "links": [
            {
                "name": "scaletether-inter-node",
                "description": "Effective inter-node tier",
                "physical": inter_physical,
            }
        ],
        "instances": [
            {
                "name": "compute-node",
                "device": "scaletether-compute-node",
                "count": effective_nodes,
            },
            {
                "name": "fabric-switch",
                "device": "scaletether-fabric-switch",
                "count": 1,
            },
        ],
        "edges": [
            {
                "ep1": {
                    "instance": f"compute-node[0:{effective_nodes}]",
                    "component": "nic[0]",
                },
                "ep2": {
                    "instance": "fabric-switch[0]",
                    "component": f"fabric-port[0:{effective_nodes}]",
                },
                "scheme": "one2one",
                "link": "scaletether-inter-node",
            }
        ],
    }


def _named(items: Any, name: str, kind: str) -> dict[str, Any]:
    if not isinstance(items, list):
        raise ValueError(f"InfraGraph {kind} inventory must be a list")
    matches = [
        item for item in items if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"InfraGraph profile requires exactly one {kind} named {name!r}"
        )
    return matches[0]


def _check_tier(link: dict[str, Any], tier: "NetworkTier", name: str) -> None:
    try:
        physical = link["physical"]
        bandwidth = physical["bandwidth"]
        latency = physical["latency"]
        if bandwidth["choice"] != "gigabytes_per_second":
            raise ValueError
        if latency["choice"] != "us":
            raise ValueError
        actual_bandwidth = float(bandwidth["gigabytes_per_second"])
        actual_latency = float(latency["us"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"InfraGraph {name} must use GB/s bandwidth and microsecond latency"
        ) from error
    if not math.isclose(actual_bandwidth, tier.bandwidth_gbps) or not math.isclose(
        actual_latency, tier.latency_us
    ):
        raise ValueError(f"InfraGraph {name} disagrees with its reversible profile")


def topology_from_infragraph(data: dict[str, Any]) -> InfraGraphProfile:
    """Import only the exact reversible scaletether InfraGraph profile."""

    infrastructure = data.get("infrastructure", data)
    if not isinstance(infrastructure, dict):
        raise ValueError("InfraGraph input must contain an infrastructure mapping")
    description = infrastructure.get("description")
    if not isinstance(description, str) or not description.startswith(
        _DESCRIPTION_PREFIX
    ):
        raise ValueError(
            "arbitrary InfraGraph import is not supported: expected the reversible "
            f"{PROFILE} profile emitted by scaletether"
        )
    try:
        marker = json.loads(description[len(_DESCRIPTION_PREFIX) :])
        if marker["profile"] != PROFILE:
            raise ValueError
        effective_nodes = int(marker["effective_nodes"])
        from .config import Topology

        topology = Topology.from_dict(marker["source_topology"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid scaletether reversible profile marker") from error
    if effective_nodes <= 0:
        raise ValueError("InfraGraph effective node count must be positive")

    compute = _named(infrastructure.get("devices"), "scaletether-compute-node", "device")
    fabric = _named(infrastructure.get("devices"), "scaletether-fabric-switch", "device")
    gpu = _named(compute.get("components"), "gpu", "component")
    nic = _named(compute.get("components"), "nic", "component")
    port = _named(fabric.get("components"), "fabric-port", "component")
    if gpu.get("choice") != "xpu" or int(gpu.get("count", 0)) != topology.gpus_per_node:
        raise ValueError(
            "InfraGraph GPU inventory disagrees with its reversible profile"
        )
    if nic.get("choice") != "nic" or int(nic.get("count", 0)) != 1:
        raise ValueError(
            "InfraGraph NIC inventory disagrees with its reversible profile"
        )
    if port.get("choice") != "port" or int(port.get("count", 0)) != effective_nodes:
        raise ValueError("InfraGraph fabric ports disagree with its reversible profile")
    compute_instances = _named(
        infrastructure.get("instances"), "compute-node", "instance"
    )
    if (
        compute_instances.get("device") != "scaletether-compute-node"
        or int(compute_instances.get("count", 0)) != effective_nodes
    ):
        raise ValueError(
            "InfraGraph compute instances disagree with its reversible profile"
        )
    intra = _named(compute.get("links"), "scaletether-intra-node", "link")
    inter = _named(infrastructure.get("links"), "scaletether-inter-node", "link")
    _check_tier(intra, topology.intra_node, "intra-node link")
    _check_tier(inter, topology.inter_node, "inter-node link")
    if topology.nodes is not None and topology.nodes != effective_nodes:
        raise ValueError("InfraGraph node count disagrees with its reversible profile")
    return InfraGraphProfile(topology=topology, effective_nodes=effective_nodes)


def _require_exact_keys(
    value: Any, required: set[str], optional: set[str], context: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"ASTRA-sim Service {context} must be a mapping")
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise ValueError(
            f"ASTRA-sim Service {context} is missing fields: "
            + ", ".join(sorted(missing))
        )
    if extra:
        raise ValueError(
            f"ASTRA-sim Service {context} has unsupported fields: "
            + ", ".join(sorted(extra))
        )
    return value


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"ASTRA-sim Service {context} must be a positive integer")
    if value > 2_147_483_647:
        raise ValueError(f"ASTRA-sim Service {context} exceeds int32")
    return value


def _validate_service_link(link: Any, context: str) -> None:
    link = _require_exact_keys(link, {"name", "physical"}, {"description"}, context)
    physical = _require_exact_keys(
        link["physical"], {"bandwidth", "latency"}, set(), f"{context}.physical"
    )
    bandwidth = _require_exact_keys(
        physical["bandwidth"],
        {"choice", "gigabytes_per_second"},
        set(),
        f"{context}.physical.bandwidth",
    )
    latency = _require_exact_keys(
        physical["latency"],
        {"choice", "us"},
        set(),
        f"{context}.physical.latency",
    )
    if bandwidth["choice"] != "gigabytes_per_second":
        raise ValueError(
            f"ASTRA-sim Service {context} must use gigabytes_per_second"
        )
    if latency["choice"] != "us":
        raise ValueError(f"ASTRA-sim Service {context} must use microseconds")
    for value, name in (
        (bandwidth["gigabytes_per_second"], "bandwidth"),
        (latency["us"], "latency"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"ASTRA-sim Service {context} {name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(
                f"ASTRA-sim Service {context} {name} must be positive and finite"
            )


def _validate_service_edge(edge: Any, context: str, infrastructure: bool) -> None:
    edge = _require_exact_keys(
        edge, {"ep1", "ep2", "scheme", "link"}, set(), context
    )
    if edge["scheme"] not in {"one2one", "many2many", "ring"}:
        raise ValueError(f"ASTRA-sim Service {context} has an unsupported scheme")
    endpoint_required = {"instance", "component"} if infrastructure else {"component"}
    endpoint_optional = set() if infrastructure else {"device"}
    for endpoint_name in ("ep1", "ep2"):
        _require_exact_keys(
            edge[endpoint_name],
            endpoint_required,
            endpoint_optional,
            f"{context}.{endpoint_name}",
        )


def _validate_astra_service_1_4_subset(infrastructure: Any) -> None:
    """Validate the exact InfraGraph 2.0 subset accepted by our adapter.

    This is intentionally narrower than the complete upstream schema. It
    mirrors only fields emitted by ``topology_to_infragraph`` and rejects
    unknown fields rather than silently losing them in a version conversion.
    """

    infrastructure = _require_exact_keys(
        infrastructure,
        {"name", "description", "devices", "links", "instances", "edges"},
        set(),
        "infrastructure",
    )
    for inventory in ("devices", "links", "instances", "edges"):
        if not isinstance(infrastructure[inventory], list):
            raise ValueError(
                f"ASTRA-sim Service infrastructure.{inventory} must be a list"
            )
    for device_index, raw_device in enumerate(infrastructure["devices"]):
        context = f"infrastructure.devices[{device_index}]"
        device = _require_exact_keys(
            raw_device,
            {"name", "components", "links", "edges"},
            {"description"},
            context,
        )
        for inventory in ("components", "links", "edges"):
            if not isinstance(device[inventory], list):
                raise ValueError(f"ASTRA-sim Service {context}.{inventory} must be a list")
        for component_index, component in enumerate(device["components"]):
            component_context = f"{context}.components[{component_index}]"
            component = _require_exact_keys(
                component,
                {"name", "count", "choice"},
                {"description"},
                component_context,
            )
            _require_positive_int(component["count"], f"{component_context}.count")
            if component["choice"] not in {
                "custom",
                "device",
                "cpu",
                "xpu",
                "nic",
                "memory",
                "port",
                "switch",
            }:
                raise ValueError(
                    f"ASTRA-sim Service {component_context}.choice is unsupported"
                )
        for link_index, link in enumerate(device["links"]):
            _validate_service_link(link, f"{context}.links[{link_index}]")
        for edge_index, edge in enumerate(device["edges"]):
            _validate_service_edge(
                edge, f"{context}.edges[{edge_index}]", infrastructure=False
            )
    for link_index, link in enumerate(infrastructure["links"]):
        _validate_service_link(link, f"infrastructure.links[{link_index}]")
    for instance_index, instance in enumerate(infrastructure["instances"]):
        context = f"infrastructure.instances[{instance_index}]"
        instance = _require_exact_keys(
            instance, {"name", "device", "count"}, {"description"}, context
        )
        _require_positive_int(instance["count"], f"{context}.count")
    for edge_index, edge in enumerate(infrastructure["edges"]):
        _validate_service_edge(
            edge, f"infrastructure.edges[{edge_index}]", infrastructure=True
        )


def adapt_infragraph_for_astra_service(data: dict[str, Any]) -> dict[str, Any]:
    """Return a lossless ASTRA-sim Service v1.4.0 infrastructure document.

    The supported scaletether profile is simultaneously valid under the project's
    InfraGraph 2.4 representation and the service's InfraGraph 2.0 subset, so
    no graph rewrite is needed. The function nevertheless performs an exact
    regeneration check and rejects any unrecognized or mutated field. This
    ensures that a future schema difference cannot become a silent downgrade.
    """

    infrastructure = data.get("infrastructure", data)
    if not isinstance(infrastructure, dict):
        raise ValueError("InfraGraph input must contain an infrastructure mapping")
    profile = topology_from_infragraph(infrastructure)
    expected = topology_to_infragraph(
        profile.topology,
        required_gpus=profile.effective_nodes * profile.topology.gpus_per_node,
    )
    if infrastructure != expected:
        raise ValueError(
            "InfraGraph document is not the exact canonical scaletether two-tier "
            "profile; refusing a potentially lossy ASTRA-sim Service conversion"
        )
    _validate_astra_service_1_4_subset(infrastructure)
    return copy.deepcopy(infrastructure)


def topology_to_astra_service_infragraph(
    topology: "Topology", required_gpus: int | None = None
) -> dict[str, Any]:
    """Generate the loss-checked Service v1.4.0 representation directly."""

    return adapt_infragraph_for_astra_service(
        topology_to_infragraph(topology, required_gpus)
    )


def dump_infragraph(
    topology: "Topology", path: Path, required_gpus: int | None = None
) -> None:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("InfraGraph YAML export requires PyYAML") from error
    path.write_text(
        yaml.safe_dump(
            topology_to_infragraph(topology, required_gpus),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def dump_astra_service_infragraph(
    topology: "Topology", path: Path, required_gpus: int | None = None
) -> None:
    """Write the loss-checked ASTRA-sim Service v1.4.0 infrastructure YAML."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("InfraGraph YAML export requires PyYAML") from error
    path.write_text(
        yaml.safe_dump(
            topology_to_astra_service_infragraph(topology, required_gpus),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
