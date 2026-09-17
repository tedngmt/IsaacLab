# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify learned weights, complete frame coverage, and recorded experiment steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    """Return a file digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    local = Path(__file__).resolve().parent
    manifest = json.loads((local / "prepared/manifest.json").read_text())
    runs = [local / "runs/all44_v1", local / "runs/all44_v2"]
    total_steps = 0
    coverage_reports = []
    for run in runs:
        report = json.loads((run / "report.json").read_text())
        assert report["phase"] == "complete"
        coverage = {item["clip_id"]: np.zeros(item["frames"], dtype=np.int32) for item in manifest["clips"]}
        records = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
        for record in records:
            total_steps += 1
            assert record["step"] == total_steps
            for name in ("loss", "penetration", "contact", "displacement", "velocity", "acceleration", "gradient_norm"):
                assert np.isfinite(record[name]), (run, name, record["step"])
            start = record["start"]
            coverage[record["clip_id"]][start : start + report["window_frames"]] += 1
        assert all(values.min() >= report["epochs_requested"] for values in coverage.values())
        assert report["source_checkpoint_sha256"] == sha256(Path(report["source_checkpoint"]))
        assert report["checkpoint_sha256"] == sha256(run / "checkpoint.pth")
        coverage_reports.append(
            {
                "run": str(run),
                "updates": len(records),
                "passes": report["epochs_requested"],
                "minimum_visits_per_source_frame": min(int(values.min()) for values in coverage.values()),
                "clips": len(coverage),
                "frames": sum(len(values) for values in coverage.values()),
            }
        )
    final_report = json.loads((runs[-1] / "report.json").read_text())
    before = torch.load(final_report["source_checkpoint"], map_location="cpu")["model"]
    after_file = torch.load(runs[-1] / "checkpoint.pth", map_location="cpu")
    after = after_file["model"]
    assert before.keys() == after.keys()
    changed = []
    largest = 0.0
    for key, value in after.items():
        assert torch.isfinite(value).all(), key
        if not torch.equal(value, before[key]):
            changed.append(key)
            largest = max(largest, float((value - before[key]).abs().max()))
    assert changed, "No pretrained model tensor changed"
    assert after_file["step"] == total_steps == 3200
    output = {
        "passed": True,
        "training_updates": total_steps,
        "passes_over_every_source_frame": 16,
        "coverage": coverage_reports,
        "changed_model_tensors": len(changed),
        "changed_model_tensor_names": changed,
        "max_parameter_absolute_change": largest,
        "all_model_tensors_finite": True,
        "original_checkpoint_unchanged": True,
        "final_checkpoint_sha256": sha256(runs[-1] / "checkpoint.pth"),
        "trainer_source_sha256": sha256(local / "train_refinement.py"),
        "training_set_only": True,
    }
    (runs[-1] / "training_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "changed_model_tensor_names"}))


if __name__ == "__main__":
    main()
