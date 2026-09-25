"""Recompute one frozen four-GPU timing decision before reading its outcome."""

import argparse
import json
from pathlib import Path
import zipfile

from scaletether.research.fixed_resource_megatron_search_v5 import _corner_logs
from scaletether.research.fixed_resource_megatron_search_v8 import _candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=1, choices=range(1, 81))
    args = parser.parse_args()
    archive_path = Path(__file__).resolve().parents[1] / "data/evaluation.zip"
    prefix = "evidence/scaletether/"
    with zipfile.ZipFile(archive_path) as archive:
        calibration = json.loads(archive.read(prefix + "fixed-resource-megatron-search-v4-freeze.json"))
        frozen = json.loads(archive.read(prefix + "fixed-resource-megatron-search-v8-freeze.json"))
        target = next(row for row in frozen["targets"] if row["task_id"] == args.case)
        decision = _candidate(_corner_logs(calibration), target["hidden_size"], target["sequence_length"])
        if any(decision[key] != target[key] for key in decision):
            raise ValueError("Recomputed decision differs from the pre-acquisition freeze")
        # Open the held-out outcome only after the decision has been computed.
        measured = json.loads(archive.read(prefix + "fixed-resource-megatron-search-v8.json"))
    observation = next(row for row in measured["primary_allocations"] if row["task_id"] == args.case)
    times = observation["medians_us"]
    selected = decision["selected_tp"]
    if decision["action"] == "MEASURE":
        selected = int(min(times, key=times.get))
    print(json.dumps({"case": args.case, **decision}, indent=2))
    print("Verified against the decision frozen before target acquisition.")
    print(json.dumps({
        "mode": "offline replay of recorded H100 measurements",
        "final_selected_tp": selected,
        "measurements_requested": decision["requested_tps"],
        "observed_regret_percent": 100 * (times[str(selected)] / min(times.values()) - 1),
    }, indent=2))


if __name__ == "__main__":
    main()
