# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add frame-level mug penetration measurements to comparison geometry.

Existing geometry arrays and evaluation reports are preserved. The signed
distance convention is negative in mug material and positive in empty space.
Contact proximity is geometric and does not establish grasp forces or stability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

ANNOTATION_KEYS = {
    "penetration_count_before",
    "penetration_count_after",
    "penetration_max_before_mm",
    "penetration_max_after_mm",
}


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write JSON atomically without allowing non-finite numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".annotating")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class MugDistance:
    """Interpolate a fixed mug signed distance field [m] at canonical points [m]."""

    def __init__(self, path: Path, device: str):
        self.device = torch.device(device)
        with np.load(path, allow_pickle=False) as data:
            self.grid = torch.as_tensor(data["sdf_grid"], device=self.device)[None, None]
            self.lower = torch.as_tensor(data["grid_min"], device=self.device)
            self.upper = torch.as_tensor(data["grid_max"], device=self.device)
            self.vertices = data["canonical_vertices"]
            self.faces = data["canonical_faces"]

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        shape = points.shape[:-1]
        normalized = 2 * (points - self.lower) / (self.upper - self.lower) - 1
        values = functional.grid_sample(
            self.grid, normalized.reshape(1, 1, 1, -1, 3),
            mode="bilinear", padding_mode="border", align_corners=True,
        ).reshape(shape)
        outside = torch.maximum(self.lower - points, points - self.upper).clamp_min(0).norm(dim=-1)
        return torch.where(outside > 0, values.clamp_min(0) + outside, values)


def evaluation_summary(evaluation: dict) -> dict:
    """Summarize per-clip changes from the existing, unchanged evaluation."""
    measures = {}
    for metric in ("vertices_deeper_than_2_mm", "vertices_deeper_than_5_mm", "maximum_penetration_mm"):
        categories = {"improved": [], "unchanged": [], "worsened": []}
        for clip in evaluation["per_clip"]:
            difference = clip["after"][metric] - clip["before"][metric]
            category = "improved" if difference < 0 else ("worsened" if difference > 0 else "unchanged")
            categories[category].append(clip["clip_id"])
        measures[metric] = {
            category: {"count": len(clips), "clip_ids": clips} for category, clips in categories.items()
        }
    return {"clips": len(evaluation["per_clip"]), "metrics": measures, "aggregate": evaluation["aggregate"]}


@torch.no_grad()
def annotate_clip(
    clip: dict,
    manifest_dir: Path,
    sdf: MugDistance,
    sdf_path: Path,
    sdf_sha256: str,
    chunk_frames: int,
    evaluation: dict,
) -> dict:
    """Append penetration counts and maximum depths [mm] without changing geometry."""
    path = Path(clip["geometry_path"])
    if not path.is_absolute():
        path = manifest_dir / path
    metadata_path = Path(clip.get("metadata_path", path.with_suffix(".json")))
    if not metadata_path.is_absolute():
        metadata_path = manifest_dir / metadata_path
    initial_hash = digest(path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("geometry_sha256") != initial_hash:
        raise ValueError(f"Geometry hash does not match its metadata: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    np.testing.assert_array_equal(arrays["object_vertices_canonical"], sdf.vertices)
    np.testing.assert_array_equal(arrays["object_faces"], sdf.faces)
    frames = len(arrays["obj_trans"])
    metrics = {}
    distances_per_side = {}
    for stage in ("before", "after"):
        distance_chunks = []
        for start in range(0, frames, chunk_frames):
            end = min(frames, start + chunk_frames)
            vertices = np.concatenate([
                arrays[f"vertices_l_{stage}"][start:end], arrays[f"vertices_r_{stage}"][start:end]
            ], axis=1)
            world = torch.as_tensor(vertices, dtype=torch.float32, device=sdf.device)
            translation = torch.as_tensor(arrays["obj_trans"][start:end], dtype=torch.float32, device=sdf.device)
            rotation = torch.as_tensor(arrays["obj_rot"][start:end], dtype=torch.float32, device=sdf.device)
            canonical = torch.einsum("tvi,tij->tvj", world - translation[:, None], rotation)
            distance_chunks.append(sdf(canonical).cpu().numpy())
        distances = np.concatenate(distance_chunks)
        if not np.isfinite(distances).all():
            raise ValueError(f"Non-finite signed distance: {clip['clip_id']}")
        distances_per_side[stage] = distances
        metrics[f"penetration_count_{stage}"] = np.sum(distances < -0.002, axis=1).astype(np.int32)
        metrics[f"penetration_max_{stage}_mm"] = (np.maximum(-distances, 0).max(axis=1) * 1000).astype(np.float32)

    source_contacts = np.abs(distances_per_side["before"]) < 0.003
    source_contact_samples = int(source_contacts.sum())
    summary = {
        "clip_id": clip["clip_id"], "frames": frames,
        "vertex_samples": int(source_contacts.size),
        "source_contact_definition": "Original hand vertex within 3 mm absolute signed distance of mug surface",
        "source_contact_vertex_samples": source_contact_samples,
        "before": {}, "after": {},
    }
    for stage, distances in distances_per_side.items():
        stage_summary = {
            "vertices_deeper_than_2_mm": int(metrics[f"penetration_count_{stage}"].sum()),
            "maximum_penetration_mm": float(metrics[f"penetration_max_{stage}_mm"].max()),
            "frames_with_penetration_over_2_mm": int((metrics[f"penetration_count_{stage}"] > 0).sum()),
            "evaluation_count_difference": int(metrics[f"penetration_count_{stage}"].sum())
            - evaluation[stage]["vertices_deeper_than_2_mm"],
        }
        for threshold_mm in (3, 5):
            nearby = np.abs(distances) < threshold_mm / 1000
            retained = int((nearby & source_contacts).sum())
            stage_summary[f"vertices_within_{threshold_mm}_mm"] = int(nearby.sum())
            stage_summary[f"vertices_within_{threshold_mm}_mm_fraction"] = float(nearby.mean())
            stage_summary[f"source_contacts_within_{threshold_mm}_mm"] = retained
            stage_summary[f"source_contacts_within_{threshold_mm}_mm_fraction"] = (
                retained / source_contact_samples if source_contact_samples else None
            )
        summary[stage] = stage_summary

    rewrite = any(name not in arrays or not np.array_equal(arrays[name], value) for name, value in metrics.items())
    if rewrite:
        temporary = path.with_suffix(".annotating.npz")
        np.savez_compressed(temporary, **{**arrays, **metrics})
        with np.load(temporary, allow_pickle=False) as written:
            for name, array in arrays.items():
                if name not in ANNOTATION_KEYS:
                    if array.dtype != written[name].dtype:
                        raise ValueError(f"Array dtype changed while annotating {name}")
                    np.testing.assert_array_equal(array, written[name])
            for name, array in metrics.items():
                np.testing.assert_array_equal(array, written[name])
        if digest(path) != initial_hash:
            raise ValueError("Source geometry changed concurrently during annotation")
        temporary.replace(path)
    metadata.setdefault("geometry_sha256_before_annotation", initial_hash)
    metadata["geometry_sha256"] = digest(path)
    metadata["frame_metric_annotation"] = {
        "description": "Both hands: number of vertices deeper than 2 mm in mug material and maximum depth per frame",
        "method": "Native watertight mug signed-distance grid with trilinear interpolation",
        "sdf_path": str(sdf_path), "sdf_sha256": sdf_sha256,
        "distance_sign": "Negative in material, positive in empty cup cavity and handle opening",
        "world_to_canonical": "(world_vertices - obj_trans) @ obj_rot",
        "arrays": sorted(ANNOTATION_KEYS), "existing_geometry_arrays_preserved_exactly": True,
        "precision_note": "Grid spacing 0.75 mm; counts near the threshold can differ slightly by CPU/GPU precision",
        "contact_summary": summary,
    }
    write_json(metadata_path, metadata)
    summary["geometry_sha256_before_annotation"] = metadata["geometry_sha256_before_annotation"]
    summary["geometry_sha256"] = metadata["geometry_sha256"]
    summary["npz_rewritten"] = rewrite
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    local = Path(__file__).resolve().parent
    parser.add_argument("--geometry_manifest", type=Path, required=True)
    parser.add_argument("--sdf", type=Path, default=local / "prepared/sdf.npz")
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument("--clip_id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk_frames", type=int, default=128)
    args = parser.parse_args()
    if args.chunk_frames < 1:
        parser.error("chunk_frames must be positive")
    started = time.perf_counter()
    torch.set_num_threads(2)
    manifest_path = args.geometry_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    evaluation_path = manifest_path.parent / "evaluation_final.json"
    evaluation_hash = digest(evaluation_path)
    evaluation = json.loads(evaluation_path.read_text())
    by_clip = {clip["clip_id"]: clip for clip in evaluation["per_clip"]}
    selected = [clip for clip in manifest["clips"] if args.clip_id is None or clip["clip_id"] == args.clip_id]
    if not selected:
        raise ValueError("No clips matched the requested clip_id")
    sdf_path = args.sdf.resolve()
    sdf_sha256 = digest(sdf_path)
    sdf = MugDistance(sdf_path, args.device)
    results = []
    for clip in selected:
        result = annotate_clip(
            clip, manifest_path.parent, sdf, sdf_path, sdf_sha256, args.chunk_frames, by_clip[clip["clip_id"]]
        )
        results.append(result)
        print(json.dumps({"annotated": clip["clip_id"], "frames": result["frames"]}), flush=True)
    contact_samples = sum(clip["source_contact_vertex_samples"] for clip in results)
    vertex_samples = sum(clip["vertex_samples"] for clip in results)
    aggregate = {}
    for stage in ("before", "after"):
        values = {
            "vertices_deeper_than_2_mm": sum(clip[stage]["vertices_deeper_than_2_mm"] for clip in results),
            "maximum_penetration_mm": max(clip[stage]["maximum_penetration_mm"] for clip in results),
            "frames_with_penetration_over_2_mm": sum(
                clip[stage]["frames_with_penetration_over_2_mm"] for clip in results
            ),
        }
        for threshold_mm in (3, 5):
            key = f"vertices_within_{threshold_mm}_mm"
            values[key] = sum(clip[stage][key] for clip in results)
            values[f"{key}_fraction"] = values[key] / vertex_samples
            key = f"source_contacts_within_{threshold_mm}_mm"
            values[key] = sum(clip[stage][key] for clip in results)
            values[f"{key}_fraction"] = values[key] / contact_samples if contact_samples else None
        aggregate[stage] = values
    if digest(evaluation_path) != evaluation_hash:
        raise ValueError("Existing evaluation changed during annotation")
    report = {
        "completed": True,
        "geometry_manifest": str(manifest_path),
        "annotated_clips": len(results), "manifest_clips": len(manifest["clips"]),
        "partial_annotation": len(results) != len(manifest["clips"]),
        "annotated_frames": sum(clip["frames"] for clip in results),
        "vertex_samples": vertex_samples, "source_contact_vertex_samples": contact_samples,
        "sdf_sha256": sdf_sha256, "device": args.device,
        "evaluation_final_sha256_unchanged": evaluation_hash,
        "evaluation_summary_all_clips": evaluation_summary(evaluation),
        "contact_metric_scope": "Only annotated clips; absolute SDF surface proximity, no contact forces or stability",
        "aggregate_annotated_clips": aggregate,
        "per_clip": results, "elapsed_seconds": time.perf_counter() - started,
    }
    output = (args.summary_path or manifest_path.parent / "summary.json").resolve()
    write_json(output, report)
    print(json.dumps({"completed": True, "summary": str(output), "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
