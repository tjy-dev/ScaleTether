from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

from .chakra import CHAKRA_COMMIT
from .config import Topology, load_topology
from .infragraph import (
    ASTRA_SERVICE_API_VERSION,
    ASTRA_SERVICE_COMMIT,
    ASTRA_SERVICE_CORE_COMMIT,
    ASTRA_SERVICE_INFRAGRAPH_COMMIT,
    ASTRA_SERVICE_INFRAGRAPH_VERSION,
    ASTRA_SERVICE_VERSION,
    topology_from_infragraph,
    topology_to_astra_service_infragraph,
)


BUNDLE_SCHEMA = "scaletether-astra-service-request-bundle-v1"
CONFIG_SCHEMA = "astra-sim-service-openapi-config-1.4.0"
_WORKLOAD_DIRECTORY = "workload"
_WORKLOAD_BASENAME = "workload"
_WORKLOAD_PREFIX = f"{_WORKLOAD_DIRECTORY}/{_WORKLOAD_BASENAME}"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Any, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Chakra export {field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_rank_inputs(
    record: Mapping[str, Any], root: Path, expected_ranks: int
) -> tuple[Path, tuple[Path, ...], Path]:
    if record.get("chakra_commit") != CHAKRA_COMMIT:
        raise ValueError("Chakra export does not match the project schema pin")
    semantic = record.get("semantic_validation")
    if not isinstance(semantic, dict):
        raise ValueError("Chakra export lacks semantic validation")
    if semantic.get("status") != "exact-protobuf-roundtrip":
        raise ValueError("Chakra export did not pass exact protobuf round-trip")
    if semantic.get("rank_files_validated") != expected_ranks:
        raise ValueError("Chakra semantic validation rank count disagrees with target")

    prefix = _resolve_path(record.get("prefix"), root, "prefix")
    raw_rank_files = record.get("rank_files")
    if not isinstance(raw_rank_files, list) or len(raw_rank_files) != expected_ranks:
        raise ValueError("Chakra export must contain exactly one file per target rank")
    rank_files = tuple(
        _resolve_path(value, root, f"rank_files[{rank}]")
        for rank, value in enumerate(raw_rank_files)
    )
    if len(set(rank_files)) != len(rank_files):
        raise ValueError("Chakra export rank files must be unique")
    for rank, rank_file in enumerate(rank_files):
        expected = prefix.with_name(f"{prefix.name}.{rank}.et").resolve()
        if rank_file != expected:
            raise ValueError(
                f"Chakra rank {rank} path does not match its declared prefix"
            )
        if not rank_file.is_file():
            raise FileNotFoundError(f"Chakra rank file does not exist: {rank_file}")

    communicator = _resolve_path(
        record.get("communicator_file"), root, "communicator_file"
    )
    expected_communicator = prefix.with_name(
        prefix.name + ".comm-groups.json"
    ).resolve()
    if communicator != expected_communicator:
        raise ValueError("Chakra communicator path does not match its prefix")
    if not communicator.is_file():
        raise FileNotFoundError(
            f"Chakra communicator file does not exist: {communicator}"
        )
    return prefix, rank_files, communicator


def _communicator_config(path: Path, ranks: int) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Chakra communicator file is not valid JSON") from error
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Chakra communicator file must contain at least one group")
    result = []
    seen_identifiers: set[str] = set()
    for identifier, members in sorted(raw.items(), key=lambda item: item[0]):
        if not isinstance(identifier, str) or not identifier.isdecimal():
            raise ValueError("Chakra communicator identifiers must be decimal strings")
        if identifier in seen_identifiers:
            raise ValueError("Chakra communicator identifiers must be unique")
        seen_identifiers.add(identifier)
        if not isinstance(members, list) or not members:
            raise ValueError(f"Chakra communicator {identifier} must be non-empty")
        if any(
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= ranks
            for rank in members
        ):
            raise ValueError(
                f"Chakra communicator {identifier} contains an invalid rank"
            )
        if len(members) != len(set(members)):
            raise ValueError(
                f"Chakra communicator {identifier} contains duplicate ranks"
            )
        result.append({"identifier": identifier, "npu_list": members})
    covered = {rank for group in result for rank in group["npu_list"]}
    if covered != set(range(ranks)):
        raise ValueError("Chakra communicator groups do not cover every target rank")
    return result


def _system_configuration(dimensions: int) -> dict[str, Any]:
    algorithms = ["ring"] * dimensions
    return {
        "scheduling_policy": "LIFO",
        "all_reduce_implementation": algorithms,
        "reduce_scatter_implementation": algorithms,
        "all_gather_implementation": algorithms,
        "all_to_all_implementation": algorithms,
        "collective_optimization": "baseline",
        "local_reduction_delay": 0,
        "active_chunks_per_dimension": 1,
        # The public analytical backend schedules endpoint work as a future
        # event and asserts strict time progress. Its canonical examples use
        # 10 ns; zero reaches the event queue but aborts at equal timestamps.
        "endpoint_delay": 10,
        "preferred_dataset_splits": 1,
        "peak_perf": 0,
        "local_mem_bw": 0,
        "roofline_enabled": 0,
        "trace_enabled": 0,
        "replay_only": 0,
    }


def _service_annotations() -> dict[str, Any]:
    """Emit the device annotations required by the released Service runtime.

    These annotations describe the two canonical instance classes; they do
    not alter the canonical InfraGraph document. Service v1.4.0 uses them to
    distinguish hosts from switches before translating InfraGraph into its
    analytical topology.
    """

    return {
        "nodes": [
            {
                "name": "compute-node",
                "attributes": [
                    {
                        "attribute": "device_name",
                        "value": "scaletether-compute-node",
                    },
                    {"attribute": "device_type", "value": "host"},
                ],
            },
            {
                "name": "fabric-switch",
                "attributes": [
                    {
                        "attribute": "device_name",
                        "value": "scaletether-fabric-switch",
                    },
                    {"attribute": "device_type", "value": "switch"},
                ],
            },
        ]
    }


def _explicit_analytical_network(
    topology: Topology, effective_nodes: int
) -> list[dict[str, Any]]:
    """Lower the canonical two-tier topology into Service analytical dimensions.

    Service v1.4.0 documents bandwidth in GB/s and latency in ns for an
    explicit analytical network.  Use that interface directly instead of the
    released InfraGraph translator: the translator drops the inter-node
    dimension for a multi-GPU host replicated across nodes and also forwards
    its internally normalized Gbit/s and millisecond values without converting
    them to the explicit schema's units.

    The canonical InfraGraph remains in the request as provenance.  The
    explicit dimensions are a loss-checked lowering of the same bounded
    scaletether two-tier profile, not an independently supplied topology.
    """

    dimensions: list[dict[str, Any]] = []
    if topology.gpus_per_node > 1:
        dimensions.append(
            {
                "topology": "switch",
                "npus_count": topology.gpus_per_node,
                "bandwidth": topology.intra_node.bandwidth_gbps,
                "latency": topology.intra_node.latency_us * 1000.0,
            }
        )
    if effective_nodes > 1:
        dimensions.append(
            {
                "topology": "switch",
                "npus_count": effective_nodes,
                "bandwidth": topology.inter_node.bandwidth_gbps,
                "latency": topology.inter_node.latency_us * 1000.0,
            }
        )
    return dimensions


def _service_configuration(
    topology: Topology,
    required_gpus: int,
    communicator_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    infrastructure = topology_to_astra_service_infragraph(topology, required_gpus)
    profile = topology_from_infragraph(infrastructure)
    capacity = profile.effective_nodes * profile.topology.gpus_per_node
    if capacity != required_gpus:
        raise ValueError(
            "ASTRA-sim Service v1 only supports full-node scaletether placements; "
            "a partial final node requires an explicit placement-aware graph"
        )
    if required_gpus < 2:
        raise ValueError("ASTRA-sim Service analytical bundles require at least 2 GPUs")
    network = _explicit_analytical_network(
        profile.topology, profile.effective_nodes
    )
    dimensions = len(network)
    if dimensions < 1 or dimensions > 3:
        raise ValueError("ASTRA-sim Service analytical topology must have 1-3 dimensions")
    lowered_capacity = 1
    for dimension in network:
        lowered_capacity *= dimension["npus_count"]
    if lowered_capacity != required_gpus:
        raise ValueError(
            "ASTRA-sim Service analytical lowering rank product disagrees with target"
        )
    # The released congestion-aware analytical backend rejects any network
    # with more than one dimension.  Preserve the two-tier representation by
    # selecting the released multidimensional congestion-unaware backend
    # instead of flattening the topology or silently dropping a tier.
    backend = (
        "analytical_congestion_aware"
        if dimensions == 1
        else "analytical_congestion_unaware"
    )
    return {
        "common_config": {
            "workload": _WORKLOAD_PREFIX,
            "system": _system_configuration(dimensions),
            "communicator_group": communicator_groups,
            "remote_memory": {
                "memory_type": "NO_MEMORY_EXPANSION",
                "remote_mem_latency": 0,
                "remote_mem_bw": 0,
            },
            "cmd_parameters": {
                "num_queues_per_dim": 1,
                "comm_scale": 1.0,
                "injection_scale": 1.0,
                "rendezvous_protocol": False,
            },
        },
        "network_backend": {
            "choice": backend,
            backend: {
                "topology": {
                    "choice": "network",
                    "network": network,
                }
            },
        },
        "infragraph": {
            "infrastructure": infrastructure,
            "annotations": _service_annotations(),
        },
    }


def _write_workload_archive(path: Path, rank_files: tuple[Path, ...]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for rank, rank_file in enumerate(rank_files):
            info = zipfile.ZipInfo(
                f"{_WORKLOAD_PREFIX}.{rank}.et",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, rank_file.read_bytes())


def build_astra_service_bundle(
    chakra_export: Mapping[str, Any],
    topology: Topology,
    required_gpus: int,
    output_dir: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic, fail-closed Service v1.4.0 request bundle."""

    if isinstance(required_gpus, bool) or not isinstance(required_gpus, int):
        raise ValueError("required_gpus must be an integer")
    if required_gpus <= 0:
        raise ValueError("required_gpus must be positive")
    root = Path.cwd() if source_root is None else source_root.resolve()
    _, rank_files, communicator_path = _load_rank_inputs(
        chakra_export, root, required_gpus
    )
    communicator_groups = _communicator_config(communicator_path, required_gpus)
    configuration = _service_configuration(
        topology, required_gpus, communicator_groups
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"ASTRA-sim Service bundle directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    archive_path = output_dir / "workload.zip"
    manifest_path = output_dir / "manifest.json"
    config_path.write_bytes(_canonical_json_bytes(configuration))
    _write_workload_archive(archive_path, rank_files)

    rank_inputs = [
        {
            "rank": rank,
            "source": str(rank_file),
            "sha256": _sha256_file(rank_file),
            "size_bytes": rank_file.stat().st_size,
            "archive_path": f"{_WORKLOAD_PREFIX}.{rank}.et",
        }
        for rank, rank_file in enumerate(rank_files)
    ]
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "status": "complete",
        "requested_gpus": required_gpus,
        "configuration_schema": CONFIG_SCHEMA,
        "software": {
            "astra_service_version": ASTRA_SERVICE_VERSION,
            "astra_service_commit": ASTRA_SERVICE_COMMIT,
            "astra_service_api_version": ASTRA_SERVICE_API_VERSION,
            "astra_service_infragraph_version": ASTRA_SERVICE_INFRAGRAPH_VERSION,
            "astra_service_infragraph_commit": ASTRA_SERVICE_INFRAGRAPH_COMMIT,
            "astra_service_core_commit": ASTRA_SERVICE_CORE_COMMIT,
            "chakra_commit": CHAKRA_COMMIT,
        },
        "inputs": {
            "rank_files": rank_inputs,
            "communicator_file": {
                "source": str(communicator_path),
                "sha256": _sha256_file(communicator_path),
                "size_bytes": communicator_path.stat().st_size,
                "group_count": len(communicator_groups),
            },
            "chakra_semantic_validation": chakra_export["semantic_validation"],
        },
        "outputs": {
            "config": {
                "path": config_path.name,
                "sha256": _sha256_file(config_path),
                "size_bytes": config_path.stat().st_size,
            },
            "workload_archive": {
                "path": archive_path.name,
                "sha256": _sha256_file(archive_path),
                "size_bytes": archive_path.stat().st_size,
            },
        },
        "claims": {
            "chakra_semantic_roundtrip": True,
            "service_request_schema_compatible": True,
            "analytical_topology_loss_checked": True,
            "service_infragraph_translation_used": False,
            "timing_claim_authorized": False,
            "service_execution_observed": False,
            "astra_sim_3_equivalence": False,
            "large_scale_timing_calibrated": False,
        },
    }
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    validate_astra_service_bundle(output_dir)
    return manifest


def validate_astra_service_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ASTRA-sim Service bundle manifest is invalid") from error
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("ASTRA-sim Service bundle manifest is not complete v1")
    software = manifest.get("software")
    expected_software = {
        "astra_service_version": ASTRA_SERVICE_VERSION,
        "astra_service_commit": ASTRA_SERVICE_COMMIT,
        "astra_service_api_version": ASTRA_SERVICE_API_VERSION,
        "astra_service_infragraph_version": ASTRA_SERVICE_INFRAGRAPH_VERSION,
        "astra_service_infragraph_commit": ASTRA_SERVICE_INFRAGRAPH_COMMIT,
        "astra_service_core_commit": ASTRA_SERVICE_CORE_COMMIT,
        "chakra_commit": CHAKRA_COMMIT,
    }
    if software != expected_software:
        raise ValueError("ASTRA-sim Service bundle software identity changed")
    requested_gpus = manifest.get("requested_gpus")
    if (
        isinstance(requested_gpus, bool)
        or not isinstance(requested_gpus, int)
        or requested_gpus <= 0
    ):
        raise ValueError("ASTRA-sim Service bundle requested_gpus is invalid")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("ASTRA-sim Service bundle output inventory is missing")
    expected_output_paths = {
        "config": "config.json",
        "workload_archive": "workload.zip",
    }
    for name in ("config", "workload_archive"):
        record = outputs.get(name)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"ASTRA-sim Service bundle {name} record is invalid")
        if record["path"] != expected_output_paths[name]:
            raise ValueError(f"ASTRA-sim Service bundle {name} path changed")
        path = bundle_dir / record["path"]
        if not path.is_file():
            raise ValueError(f"ASTRA-sim Service bundle {name} is missing")
        if path.stat().st_size != record["size_bytes"] or _sha256_file(path) != record[
            "sha256"
        ]:
            raise ValueError(f"ASTRA-sim Service bundle {name} changed")

    config_path = bundle_dir / outputs["config"]["path"]
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ASTRA-sim Service bundle config is invalid") from error
    if config.get("common_config", {}).get("workload") != _WORKLOAD_PREFIX:
        raise ValueError("ASTRA-sim Service workload prefix changed")
    infrastructure = config.get("infragraph", {}).get("infrastructure")
    if not isinstance(infrastructure, dict):
        raise ValueError("ASTRA-sim Service bundle lacks infrastructure")
    profile = topology_from_infragraph(infrastructure)

    config_groups = config.get("common_config", {}).get("communicator_group")
    if not isinstance(config_groups, list) or not config_groups:
        raise ValueError("ASTRA-sim Service bundle communicator groups are missing")
    group_identifiers: set[str] = set()
    covered_ranks: set[int] = set()
    for group in config_groups:
        if not isinstance(group, dict) or set(group) != {"identifier", "npu_list"}:
            raise ValueError("ASTRA-sim Service bundle communicator group is invalid")
        identifier = group["identifier"]
        members = group["npu_list"]
        if (
            not isinstance(identifier, str)
            or not identifier.isdecimal()
            or identifier in group_identifiers
        ):
            raise ValueError(
                "ASTRA-sim Service bundle communicator identifier is invalid"
            )
        group_identifiers.add(identifier)
        if (
            not isinstance(members, list)
            or not members
            or len(members) != len(set(members))
            or any(
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 0
                or rank >= requested_gpus
                for rank in members
            )
        ):
            raise ValueError("ASTRA-sim Service bundle communicator ranks are invalid")
        covered_ranks.update(members)
    if covered_ranks != set(range(requested_gpus)):
        raise ValueError("ASTRA-sim Service bundle communicators lack rank coverage")
    expected_config = _service_configuration(
        profile.topology, requested_gpus, config_groups
    )
    configured_backend = config.get("network_backend", {}).get("choice")
    topology_choice = (
        config.get("network_backend", {})
        .get(configured_backend, {})
        .get("topology", {})
        .get("choice")
        if isinstance(configured_backend, str)
        else None
    )
    if topology_choice == "infragraph":
        # Retained v1 evidence predates the explicit, unit-correct lowering.
        # Validate its exact former representation without upgrading its
        # claims or silently treating it as the corrected path.
        expected_backend = expected_config["network_backend"]["choice"]
        if configured_backend != expected_backend:
            # Retained one-dimensional evidence used congestion-aware.  There
            # is no retained multidimensional legacy success to grandfather.
            raise ValueError("ASTRA-sim Service legacy backend choice changed")
        expected_config["network_backend"][expected_backend]["topology"] = {
            "choice": "infragraph"
        }
    if config != expected_config:
        raise ValueError("ASTRA-sim Service bundle config is not canonical")

    rank_records = manifest.get("inputs", {}).get("rank_files")
    if not isinstance(rank_records, list) or len(rank_records) != requested_gpus:
        raise ValueError("ASTRA-sim Service bundle rank inventory is incomplete")
    expected_rank_record_keys = {
        "rank",
        "source",
        "sha256",
        "size_bytes",
        "archive_path",
    }
    for rank, record in enumerate(rank_records):
        if not isinstance(record, dict) or set(record) != expected_rank_record_keys:
            raise ValueError("ASTRA-sim Service bundle rank record is invalid")
        if record["rank"] != rank or record["archive_path"] != (
            f"{_WORKLOAD_PREFIX}.{rank}.et"
        ):
            raise ValueError("ASTRA-sim Service bundle rank identity changed")
        if (
            not isinstance(record["size_bytes"], int)
            or record["size_bytes"] <= 0
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
        ):
            raise ValueError("ASTRA-sim Service bundle rank digest is invalid")
    archive_path = bundle_dir / outputs["workload_archive"]["path"]
    expected_names = [record["archive_path"] for record in rank_records]
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.namelist() != expected_names:
            raise ValueError("ASTRA-sim Service workload archive rank names changed")
        for record in rank_records:
            payload = archive.read(record["archive_path"])
            if len(payload) != record["size_bytes"] or _sha256_bytes(payload) != record[
                "sha256"
            ]:
                raise ValueError(
                    f"ASTRA-sim Service archived rank {record['rank']} changed"
                )
    claims = manifest.get("claims")
    expected_claims = {
        "chakra_semantic_roundtrip": True,
        "service_request_schema_compatible": True,
        "analytical_topology_loss_checked": True,
        "service_infragraph_translation_used": False,
        "timing_claim_authorized": False,
        "service_execution_observed": False,
        "astra_sim_3_equivalence": False,
        "large_scale_timing_calibrated": False,
    }
    if topology_choice == "infragraph":
        expected_claims.pop("analytical_topology_loss_checked")
        expected_claims.pop("service_infragraph_translation_used")
    if claims != expected_claims:
        raise ValueError("ASTRA-sim Service bundle must forbid a timing claim")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a pinned ASTRA-sim Service v1.4.0 request bundle"
    )
    parser.add_argument("--chakra-export", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--gpus", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    export_path = args.chakra_export.resolve()
    record = json.loads(export_path.read_text(encoding="utf-8"))
    topology = load_topology(args.topology, args.gpus)
    manifest = build_astra_service_bundle(
        record,
        topology,
        args.gpus,
        args.output,
        source_root=export_path.parent,
    )
    print(f"bundle={args.output.resolve()}")
    print(f"config_sha256={manifest['outputs']['config']['sha256']}")
    print(
        "workload_archive_sha256="
        + manifest["outputs"]["workload_archive"]["sha256"]
    )
    print("timing_claim_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
