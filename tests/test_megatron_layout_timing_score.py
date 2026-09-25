from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scaletether.megatron_layout_timing_score import score, validate_bracket


def write_bracket(root: Path, tp: int, outer: int, error: float = 2.0) -> Path:
    case = root / f"tp{tp}-{outer}"
    case.mkdir()
    freeze = {
        "target_observed": False,
        "created_unix_ns": 1,
        "anchor_hidden_sizes": [1024, 4096],
        "target_hidden_size": 2048,
    }
    encoded = (json.dumps(freeze) + "\n").encode()
    (case / "prediction-freeze.json").write_bytes(encoded)
    record = {"samples_us": [10.0] * 20, "median_us": 10.0}
    names = ("source_h1024", "target_h1024", "source_h4096", "target_h4096",
             "heldout_source_left", "heldout_target", "heldout_source_right")
    result = {
        "schema": "scaletether-megatron-layout-timing-bracket-v1",
        "tensor_parallel_size": tp,
        "outer_repetition": outer,
        "measured_repetitions": 20,
        "prediction_freeze_sha256": hashlib.sha256(encoded).hexdigest(),
        "target_started_unix_ns": 2,
        "prediction_absolute_percentage_error": error,
        "source_control_drift_percent": 1.0,
        "records": {name: dict(record) for name in names},
    }
    path = case / "result.json"
    path.write_text(json.dumps(result))
    return path


def test_scores_two_complete_groups(tmp_path: Path) -> None:
    paths = [write_bracket(tmp_path, tp, outer) for tp in (2, 4) for outer in range(1, 6)]
    report = score(paths)
    assert report["overall_gate"] == "passed"
    assert report["groups"]["tp4"]["median_absolute_percentage_error"] == 2.0


def test_rejects_target_before_freeze(tmp_path: Path) -> None:
    path = write_bracket(tmp_path, 2, 1)
    value = json.loads(path.read_text())
    value["target_started_unix_ns"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="did not start after"):
        validate_bracket(path)


def test_fails_if_one_bracket_exceeds_gate(tmp_path: Path) -> None:
    paths = [write_bracket(tmp_path, tp, outer, 6.0 if (tp, outer) == (4, 5) else 2.0)
             for tp in (2, 4) for outer in range(1, 6)]
    report = score(paths)
    assert report["overall_gate"] == "failed"
    assert report["groups"]["tp4"]["timing_gate"] == "failed"


def test_accepts_a_different_frozen_anchor_bracket(tmp_path: Path) -> None:
    path = write_bracket(tmp_path, 2, 1)
    case = path.parent
    freeze_path = case / "prediction-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["anchor_hidden_sizes"] = [2048, 4096]
    freeze["target_hidden_size"] = 3584
    encoded = (json.dumps(freeze) + "\n").encode()
    freeze_path.write_bytes(encoded)
    value = json.loads(path.read_text())
    value["prediction_freeze_sha256"] = hashlib.sha256(encoded).hexdigest()
    value["records"]["source_h2048"] = value["records"].pop("source_h1024")
    value["records"]["target_h2048"] = value["records"].pop("target_h1024")
    path.write_text(json.dumps(value))
    assert validate_bracket(path)["outer_repetition"] == 1
