# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify completed GraspXL mug refinement training and native-frame provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, description: str) -> None:
    """Reject an incomplete or inconsistent verification condition."""
    if not condition:
        raise ValueError(description)


def resolve(base: Path, value: str) -> Path:
    """Resolve a manifest path against its containing directory."""
    path = Path(value)
    return path if path.is_absolute() else base / path


def expected_windows(clips: list[dict], window: int) -> set[tuple[str, int]]:
    """Reconstruct the trainer's complete windows including overlapping tails."""
    result = set()
    for clip in clips:
        count = clip["frames"]
        for start in range(0, count, window):
            if min(count, start + window) - start >= 8:
                result.add((clip["clip_id"], start))
        result.add((clip["clip_id"], max(0, count - window)))
    return result


def verify_coverage(report: dict, records: list[dict], clips: list[dict]) -> dict:
    """Verify finite steps and coverage of every native frame in every requested pass."""
    passes = int(report["epochs_requested"])
    window = int(report["window_frames"])
    previous_steps = int(report.get("previous_training_steps", 0))
    expected = expected_windows(clips, window)
    require(passes > 0 and records, "No completed training passes were recorded")
    require(report["windows_per_epoch"] == len(expected), "Recorded window count differs from native-frame windows")
    require(len(records) == passes * len(expected), "Training log lacks some requested updates")
    require(report["training_steps"] == previous_steps + len(records), "Final report step count differs from log")
    counts = {clip["clip_id"]: np.zeros((passes, clip["frames"]), dtype=np.int32) for clip in clips}
    epoch_windows = [Counter() for _ in range(passes)]
    fields = ("loss", "penetration", "contact", "displacement", "velocity", "acceleration", "gradient_norm")
    for offset, record in enumerate(records, 1):
        require(record["step"] == previous_steps + offset, f"Nonconsecutive training step at log row {offset}")
        require(all(np.isfinite(record[name]) for name in fields),
                f"Non-finite training value at step {record['step']}")
        epoch = int(record["epoch"])
        require(epoch == (offset - 1) // len(expected) + 1, "Training epoch numbering differs from step order")
        clip_id, start = record["clip_id"], int(record["start"])
        require((clip_id, start) in expected, f"Unexpected training window: {clip_id}, {start}")
        epoch_windows[epoch - 1][(clip_id, start)] += 1
        counts[clip_id][epoch - 1, start : start + window] += 1
    expected_counter = Counter({item: 1 for item in expected})
    require(all(value == expected_counter for value in epoch_windows), "A training pass duplicated or omitted a window")
    require(all(np.all(value >= 1) for value in counts.values()),
            "Some source frames were omitted from a training pass")
    return {
        "updates_this_run": len(records),
        "previous_training_steps": previous_steps,
        "total_checkpoint_steps": report["training_steps"],
        "passes_over_every_source_frame": passes,
        "windows_per_pass": len(expected),
        "all_loss_components_and_gradient_norms_finite": True,
        "every_native_frame_seen_in_every_pass": True,
        "minimum_visits_per_source_frame": min(int(value.sum(axis=0).min()) for value in counts.values()),
        "clips": len(clips), "frames": sum(clip["frames"] for clip in clips),
        "per_clip": [
            {"clip_id": clip_id, "frames": value.shape[1], "passes": value.shape[0],
             "minimum_visits_each_pass": int(value.min()),
             "minimum_total_visits_per_frame": int(value.sum(axis=0).min()),
             "maximum_total_visits_per_frame": int(value.sum(axis=0).max())}
            for clip_id, value in counts.items()
        ],
    }


def verify_geometry(run: Path, prepared: Path, manifest: dict) -> dict:
    """Verify right-only geometry, original object poses [m], and all native frames."""
    exported = json.loads((run / "geometry_manifest.json").read_text())
    exported_clips = {entry["clip_id"]: entry for entry in exported["clips"]}
    expected_ids = {entry["clip_id"] for entry in manifest["clips"]}
    require(set(exported_clips) == expected_ids, "Exported geometry does not contain all prepared clips")
    require(len(exported["clips"]) == len(expected_ids), "Duplicate exported clip IDs")
    require(exported["source_fps"] is None, "Unknown physical source rate was replaced with an assumed rate")
    require(exported["playback_fps"] == 30, "Unexpected comparison playback rate")
    evaluation = json.loads((run / "evaluation_final.json").read_text())
    evaluated = {entry["clip_id"]: entry for entry in evaluation["per_clip"]}
    require(set(evaluated) == expected_ids, "Final evaluation omits or adds clips")
    objects_path = resolve(prepared, manifest["object_path"])
    require(sha256(objects_path) == manifest["object_sha256"], "Prepared native object file changed")
    with np.load(objects_path, allow_pickle=False) as archive:
        native_vertices = archive["object_vertices_canonical"]
        native_faces = archive["object_faces"]
    baseline_error = 0.0
    checked = []
    forbidden_keys = {"vertices_l_before", "vertices_l_after", "faces_l", "x_lhand_before", "x_lhand_after"}
    for clip in manifest["clips"]:
        clip_id, frames = clip["clip_id"], clip["frames"]
        require(frames == clip["original_frames"], f"Native frame count changed: {clip_id}")
        require(clip["source_stride"] == 1, f"Native frames were subsampled: {clip_id}")
        require(not clip["is_lhand"] and clip["is_rhand"], f"Unexpected hand presence: {clip_id}")
        require(clip["source_fps"] is None and not clip["source_fps_known"], f"Source timing changed: {clip_id}")
        require(sha256(Path(clip["source_path"])) == clip["source_sha256"], f"Original dataset file changed: {clip_id}")
        prepared_path = resolve(prepared, clip["clip_path"])
        require(sha256(prepared_path) == clip["clip_sha256"], f"Prepared training data changed: {clip_id}")
        geometry_path = resolve(run, exported_clips[clip_id]["geometry_path"])
        metadata_path = resolve(run, exported_clips[clip_id]["metadata_path"])
        metadata = json.loads(metadata_path.read_text())
        require(sha256(geometry_path) == metadata["geometry_sha256"], f"Geometry hash differs from metadata: {clip_id}")
        with np.load(prepared_path, allow_pickle=False) as source, np.load(
            geometry_path, allow_pickle=False
        ) as geometry:
            require(not forbidden_keys.intersection(geometry.files), f"Absent left hand was exported: {clip_id}")
            require(not source["valid_mask_lhand"].any(), f"Absent left hand validity mask is active: {clip_id}")
            require(not source["lhand_contact_mask"].any(), f"Absent left hand contact mask is active: {clip_id}")
            require(not source["lhand_vertices"].any(), f"Absent left hand source vertices are nonzero: {clip_id}")
            require(source["valid_mask_rhand"].all() and source["valid_mask_obj"].all(),
                    f"Native right-hand or object frames are masked out: {clip_id}")
            np.testing.assert_array_equal(source["source_frame_indices"], np.arange(frames))
            np.testing.assert_array_equal(geometry["source_frame_indices"], np.arange(frames))
            np.testing.assert_array_equal(geometry["object_vertices_canonical"], native_vertices)
            np.testing.assert_array_equal(geometry["object_faces"], native_faces)
            np.testing.assert_array_equal(geometry["obj_rot"], source["object_rotation"])
            np.testing.assert_array_equal(geometry["obj_trans"], source["object_translation"])
            np.testing.assert_array_equal(geometry["x_rhand_before"], source["x_rhand"])
            for stage in ("before", "after"):
                vertices = geometry[f"vertices_r_{stage}"]
                require(vertices.shape == (frames, 778, 3) and np.isfinite(vertices).all(),
                        f"Invalid right-hand output: {clip_id}, {stage}")
                require(evaluated[clip_id][stage]["vertex_samples"] == frames * 778,
                        f"Evaluation includes wrong vertex/frame count: {clip_id}, {stage}")
            difference = float(np.abs(geometry["vertices_r_before"] - source["rhand_vertices"]).max())
            require(difference < 1e-6, f"Before-training reference geometry changed: {clip_id}")
            baseline_error = max(baseline_error, difference)
        checked.append({"clip_id": clip_id, "frames": frames, "geometry_sha256": metadata["geometry_sha256"],
                        "original_source_sha256_unchanged": clip["source_sha256"]})
    return {
        "right_hand_only": True,
        "absent_left_masked_and_not_exported": True,
        "evaluation_vertices_per_frame": 778,
        "all_native_frames_retained": True,
        "source_frame_stride": 1,
        "physical_source_fps": None,
        "playback_fps": 30,
        "playback_timing_is_assumed_not_resampled": True,
        "native_object_geometry_and_poses_unchanged": True,
        "original_source_files_unchanged": True,
        "before_mano_reconstruction_max_abs_error_m": baseline_error,
        "matched_sequences": sum(not clip["supplemental"] for clip in manifest["clips"]),
        "supplemental_tabletop_sequences": sum(clip["supplemental"] for clip in manifest["clips"]),
        "clips": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    local = Path(__file__).resolve().parent
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=local / "prepared")
    args = parser.parse_args()
    run, prepared = args.run_dir.resolve(), args.prepared.resolve()
    torch.set_num_threads(2)
    report_path = run / "report.json"
    report = json.loads(report_path.read_text())
    require(report.get("phase") == "complete", "Training/export is not complete; verification must wait")
    require(report.get("refiner_trained") is True and report.get("generator_trained") is False,
            "This verifier expects the recorded refiner-only experiment")
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["dataset"] == "GraspXL", "Prepared manifest is not GraspXL")
    require(len(manifest["clips"]) == manifest["sequences"], "Manifest sequence total is inconsistent")
    require(sum(clip["frames"] for clip in manifest["clips"]) == manifest["frames"],
            "Manifest frame total is inconsistent")
    require(set(report["clips"]) == {clip["clip_id"] for clip in manifest["clips"]}, "Training omitted prepared clips")
    require(report["prepared_manifest_sha256"] == sha256(manifest_path),
            "Prepared manifest changed after training began")
    require(report["sdf_sha256"] == sha256(prepared / "sdf.npz"), "Training signed-distance grid changed")
    source_path = Path(report["source_checkpoint"])
    source_hash = sha256(source_path)
    require(report["source_checkpoint_sha256"] == source_hash, "Released source checkpoint changed")
    checkpoint_path = run / "checkpoint.pth"
    checkpoint_hash = sha256(checkpoint_path)
    require(report["checkpoint_sha256"] == checkpoint_hash, "Trained checkpoint differs from completed report")
    log_path = run / "training.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    coverage = verify_coverage(report, records, manifest["clips"])
    before = torch.load(source_path, map_location="cpu")["model"]
    trained = torch.load(checkpoint_path, map_location="cpu")
    after = trained["model"]
    require(before.keys() == after.keys(), "Trained model state keys differ from the released architecture")
    require(trained["step"] == report["training_steps"], "Saved checkpoint step differs from completed training log")
    require(trained["upstream_checkpoint_sha256"] == source_hash, "Saved checkpoint source provenance differs")
    changed = []
    largest = 0.0
    for key, value in after.items():
        require(torch.isfinite(value).all().item(), f"Non-finite trained model tensor: {key}")
        require(value.shape == before[key].shape and value.dtype == before[key].dtype,
                f"Model tensor shape or dtype changed: {key}")
        if not torch.equal(value, before[key]):
            changed.append(key)
            largest = max(largest, float((value - before[key]).abs().max()))
    require(changed, "No pretrained model tensor changed")
    require(any(key.startswith("seqTransEncoder.") for key in changed), "No transformer encoder tensor changed")
    require(trained.get("optimizer", {}).get("state"), "Checkpoint lacks evidence of optimizer updates")
    for state in trained["optimizer"]["state"].values():
        for name, value in state.items():
            require(bool(torch.isfinite(value).all()) if torch.is_tensor(value) else bool(np.isfinite(value)),
                    f"Non-finite optimizer state: {name}")
    geometry = verify_geometry(run, prepared, manifest)
    require(sha256(source_path) == source_hash, "Source checkpoint changed while verifying")
    output = {
        "passed": True, "run_dir": str(run), "dataset": "GraspXL", "training_set_only": True,
        "training_updates": coverage["updates_this_run"],
        "passes_over_every_source_frame": coverage["passes_over_every_source_frame"],
        "coverage": coverage, "provenance": geometry,
        "changed_model_tensors": len(changed), "changed_model_tensor_names": changed,
        "max_parameter_absolute_change": largest,
        "all_model_tensors_finite": True, "all_optimizer_state_finite": True,
        "original_checkpoint_unchanged": True, "source_checkpoint_sha256": source_hash,
        "final_checkpoint_sha256": checkpoint_hash, "training_log_sha256": sha256(log_path),
        "trainer_source_sha256_at_verification": sha256(local / "train_refinement.py"),
        "prepared_manifest_sha256": sha256(manifest_path), "completed_report_sha256": sha256(report_path),
        "scope": "Refiner training and in-sample source-motion cleanup; not new text generation or physics validation",
    }
    output_path = run / "training_verification.json"
    temporary = output_path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_path)
    print(json.dumps({"passed": True, "output": str(output_path), "updates": coverage["updates_this_run"],
                      "passes": coverage["passes_over_every_source_frame"], "clips": coverage["clips"],
                      "frames": coverage["frames"], "changed_model_tensors": len(changed),
                      "right_hand_only": True}), flush=True)


if __name__ == "__main__":
    main()
