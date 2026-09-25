"""Score a frozen row/column layout candidate against a physical H100 trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import WorkloadTrace


class MegatronLayoutH100ValidationError(ValueError):
    """The frozen candidate or physical target is not admissible evidence."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collectives(trace: WorkloadTrace, rank: int) -> list[dict[str, Any]]:
    events = [
        event
        for event in trace.events
        if event.rank == rank and event.kind == "collective"
    ]
    return [
        {
            "collective": event.collective,
            "message_bytes": event.message_bytes,
            "group_role": event.group_role,
            "group_size": event.group_size,
        }
        for event in events
    ]


def validate(
    candidate_path: Path,
    freeze_path: Path,
    target_path: Path,
    correctness_path: Path,
) -> dict[str, Any]:
    candidate = WorkloadTrace.load(candidate_path)
    target = WorkloadTrace.load(target_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    candidate_sha = _sha(candidate_path)
    if freeze != {
        "schema": "scaletether-megatron-layout-prospective-freeze-v1",
        "candidate_sha256": candidate_sha,
        "target_layout": "row-gelu-column",
        "target_started": False,
    }:
        raise MegatronLayoutH100ValidationError("candidate freeze is not exact")
    if correctness.get("status") != "passed" or float(
        correctness.get("max_relative_to_peak", 1.0)
    ) > float(correctness.get("tolerance", 0.0)):
        raise MegatronLayoutH100ValidationError("dense-equivalence check failed")

    source = target.source
    if (
        source.get("target") != "h100"
        or source.get("compute_capability") != "9.0"
        or source.get("sm_count") != 132
        or "H100" not in str(source.get("device_name", ""))
    ):
        raise MegatronLayoutH100ValidationError("target is not exact H100 evidence")
    limitations = target.metadata.get("capture_limitations", [])
    if limitations:
        raise MegatronLayoutH100ValidationError(
            f"target capture has limitations: {limitations}"
        )
    measurement = target.metadata.get("framework_measurement")
    if not isinstance(measurement, dict) or measurement.get("layer") != {
        "kind": "megatron-core-row-gelu-column-mlp-v1",
        "row_input_is_parallel": True,
        "column_gather_output": True,
        "sequence_parallel": False,
    }:
        raise MegatronLayoutH100ValidationError("target layout declaration differs")
    report = candidate.metadata.get("megatron_mlp_layout_rewrite", {})
    tp = int(report.get("tensor_parallel_size", 0))
    if tp not in {2, 4} or measurement.get("tensor_parallel_size") != tp:
        raise MegatronLayoutH100ValidationError("TP width differs from frozen target")
    if correctness.get("tensor_parallel_size") != tp:
        raise MegatronLayoutH100ValidationError("correctness TP differs from candidate")
    target_model = measurement.get("model")
    candidate_model = report.get("model")
    if not isinstance(target_model, dict) or not isinstance(candidate_model, dict):
        raise MegatronLayoutH100ValidationError("model dimensions are missing")
    for target_key, candidate_key in (
        ("num_layers", "num_layers"),
        ("hidden_size", "hidden_size"),
        ("ffn_hidden_size", "ffn_hidden_size"),
        ("sequence_length", "sequence_length"),
        ("micro_batch_size", "micro_batch_size"),
        ("activation_element_bytes", "activation_element_bytes"),
    ):
        if target_model.get(target_key) != candidate_model.get(candidate_key):
            raise MegatronLayoutH100ValidationError(
                f"target {target_key} differs from frozen candidate"
            )

    comparisons: list[dict[str, Any]] = []
    for rank in range(tp):
        expected = _collectives(candidate, rank)
        merged_observed = _collectives(target, rank)
        rank_path = target_path.with_name(f"workload.rank{rank}.json")
        if not rank_path.is_file():
            raise MegatronLayoutH100ValidationError(
                f"rank-local target trace is missing: {rank_path}"
            )
        rank_target = WorkloadTrace.load(rank_path)
        rank_observed = _collectives(rank_target, rank)
        if len(merged_observed) != len(rank_observed) or any(
            merged["collective"] != local["collective"]
            or merged["message_bytes"] != local["message_bytes"]
            for merged, local in zip(merged_observed, rank_observed)
        ):
            raise MegatronLayoutH100ValidationError(
                f"rank {rank} merged and rank-local collective identities differ"
            )
        observed = [
            {
                "collective": local["collective"],
                "message_bytes": local["message_bytes"],
                "group_role": merged["group_role"],
                "group_size": local["group_size"],
            }
            for merged, local in zip(merged_observed, rank_observed)
        ]
        if observed != expected:
            raise MegatronLayoutH100ValidationError(
                f"rank {rank} collective structure differs: "
                f"expected={expected}, observed={observed}"
            )
        comparisons.append(
            {"rank": rank, "status": "exact", "collectives": observed}
        )
    return {
        "schema": "scaletether-megatron-layout-h100-validation-v1",
        "status": "passed",
        "claim": "exact collective structure and dense numerical equivalence",
        "timing_claim": "not-evaluated",
        "candidate_sha256": candidate_sha,
        "target_sha256": _sha(target_path),
        "correctness_sha256": _sha(correctness_path),
        "rank_comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(args.candidate, args.freeze, args.target, args.correctness)
    if args.output.exists():
        parser.error("output already exists")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
