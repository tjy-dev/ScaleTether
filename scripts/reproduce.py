"""Verify the release data and reproduce results without GPUs or network access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_archive(name: str, destination: Path) -> None:
    archive_path = ROOT / "data" / name
    checksums = json.loads((ROOT / "data/checksums.json").read_text())
    if digest(archive_path.read_bytes()) != checksums[name]:
        raise ValueError(f"Archive checksum mismatch: {name}")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        expected = {row["path"]: row["sha256"] for row in manifest["members"]}
        if set(archive.namelist()) != set(expected) | {"MANIFEST.json"}:
            raise ValueError(f"Archive member list mismatch: {name}")
        for member, sha in expected.items():
            path = destination / member
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Invalid archive path: {member}")
            data = archive.read(member)
            if digest(data) != sha:
                raise ValueError(f"Member checksum mismatch: {member}")
        archive.extractall(destination)
    print(f"Verified {name}: {len(expected)} files", flush=True)


def reproduce(directory: Path) -> None:
    evaluation = directory / "evaluation"
    verified_archive("evaluation.zip", evaluation)
    verified_archive("workloads.zip", directory / "workloads")
    commands = [
        ["validate_scaletether_artifact.py"],
        ["generate_scaletether_tables.py", "--check"],
        ["score_agentic_interface_benchmark.py", "--check"],
        ["score_agentic_interface_replication_v2.py", "--check"],
        ["score_agentic_interface_prose_v3.py", "--check"],
        ["agentic_interface_matched_v4.py", "score", "--check"],
        ["-m", "unittest", "test_agentic_interface_matched_v4"],
    ]
    for command in commands:
        print(f"Checking {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, *command], cwd=evaluation, check=True)
    tables = {
        path: path.read_bytes()
        for path in (evaluation / "generated").glob("camera-ready-*-table.tex")
    }
    subprocess.run([sys.executable, "make_camera_ready_results.py"], cwd=evaluation, check=True)
    if any(path.read_bytes() != content for path, content in tables.items()):
        raise ValueError("Regenerated camera-ready tables differ from the archived tables")
    print("All reproduction checks passed.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Keep extracted data in a new directory")
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        reproduce(args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="scaletether-") as temporary:
            reproduce(Path(temporary))


if __name__ == "__main__":
    main()
