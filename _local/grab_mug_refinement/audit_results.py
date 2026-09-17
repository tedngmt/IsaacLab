# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit refined GRAB mug geometry against native triangles, independently of training.

Run in the existing soma-x environment for Trimesh's R-tree dependency. Sampling
uses fixed clip fractions; it is not selected according to refinement results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from trimesh.ray.ray_util import contains_points

RAY_DIRECTIONS = (np.array([0.439, 0.617, 0.653]), np.array([0.38, 0.72, 0.57]))
_WORKER_MESH = None
_WORKER_GRID = None


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write a report atomically so an interrupted audit remains identifiable."""
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def surface_distances(mesh: trimesh.Trimesh, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact triangle distances [m], two-ray signs, and disagreement flags.

    Only points within a bounding box expanded by 0.01 m need triangle queries:
    all other points are necessarily outside and farther than the contact
    threshold. Their unsigned distances are infinity, not approximations.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    near = np.logical_and(vertices >= mesh.bounds[0] - 0.01, vertices <= mesh.bounds[1] + 0.01).all(axis=-1)
    distance = np.full(len(vertices), np.inf, dtype=np.float64)
    inside = np.zeros(len(vertices), dtype=bool)
    disagreement = np.zeros(len(vertices), dtype=bool)
    selected = np.flatnonzero(near)
    if len(selected):
        points = vertices[selected]
        _, distances, _ = trimesh.proximity.closest_point(mesh, points)
        parity = [contains_points(mesh.ray, points, check_direction=direction) for direction in RAY_DIRECTIONS]
        distance[selected] = distances
        inside[selected] = parity[0] & parity[1]
        disagreement[selected] = parity[0] != parity[1]
    return distance, inside, disagreement


def penetration_metrics(distance: np.ndarray, inside: np.ndarray, disagreement: np.ndarray) -> dict:
    """Summarize confirmed material penetration depths [mm] at hand vertices."""
    depth = np.where(inside, distance, 0.0) * 1000.0
    return {
        "vertex_samples": len(distance),
        "inside_vertices_confirmed_two_rays": int(inside.sum()),
        "ray_direction_disagreements_excluded": int(disagreement.sum()),
        "vertices_deeper_than_2_mm": int((depth > 2.0).sum()),
        "vertices_deeper_than_5_mm": int((depth > 5.0).sum()),
        "fraction_deeper_than_2_mm": float((depth > 2.0).mean()),
        "fraction_deeper_than_5_mm": float((depth > 5.0).mean()),
        "maximum_penetration_depth_mm": float(depth.max(initial=0.0)),
        "mean_penetration_all_vertices_mm": float(depth.mean()),
        "mean_penetration_inside_vertices_mm": float(depth[inside].mean()) if inside.any() else 0.0,
    }


def grid_comparison(
    grid: dict, points: np.ndarray, distance: np.ndarray, inside: np.ndarray, disagreement: np.ndarray
) -> dict:
    """Compare interpolated signed distances [m] with independent exact queries."""
    selected = np.isfinite(distance) & ~disagreement
    selected &= np.logical_and(points >= grid["grid_min"], points <= grid["grid_max"]).all(axis=-1)
    exact = np.where(inside[selected], -distance[selected], distance[selected])
    coordinates = ((points[selected] - grid["grid_min"]) / float(grid["spacing_m"])).T
    sampled = map_coordinates(grid["sdf_grid"], coordinates[[2, 1, 0]], order=1, mode="nearest", prefilter=False)
    error = np.abs(sampled - exact) * 1000.0
    away = np.abs(exact) > 2 * float(grid["spacing_m"])
    return {
        "exact_query_points": len(exact),
        "max_abs_error_mm": float(error.max(initial=0.0)),
        "mean_abs_error_mm": float(error.mean()) if len(error) else 0.0,
        "p95_abs_error_mm": float(np.percentile(error, 95)) if len(error) else 0.0,
        "sign_mismatches": int(((sampled < 0) != (exact < 0)).sum()),
        "points_farther_than_two_grid_voxels": int(away.sum()),
        "sign_mismatches_farther_than_two_grid_voxels": int(((sampled[away] < 0) != (exact[away] < 0)).sum()),
        "depth_2_mm_classification_mismatches": int(((sampled < -0.002) != (exact < -0.002)).sum()),
        "depth_5_mm_classification_mismatches": int(((sampled < -0.005) != (exact < -0.005)).sum()),
    }


def frame_metrics(mesh: trimesh.Trimesh, before: np.ndarray, after: np.ndarray, grid: dict | None = None) -> dict:
    """Measure before/after hand vertices in canonical object coordinates [m]."""
    result = {}
    queries = {}
    for stage, points in (("before", before), ("after", after)):
        queries[stage] = surface_distances(mesh, points)
        distance, inside, disagreement = queries[stage]
        result[stage] = penetration_metrics(distance, inside, disagreement)
        if grid is not None:
            result[stage]["independent_sdf_check"] = grid_comparison(grid, points, distance, inside, disagreement)
    before_distance, before_inside, before_disagree = queries["before"]
    after_distance, after_inside, after_disagree = queries["after"]
    contact = (before_distance < 0.003) & ~before_disagree
    retained = contact & (after_distance < 0.003) & ~after_disagree
    retained_nonpenetrating = retained & ~(after_inside & (after_distance > 0.002))
    movement = np.linalg.norm(after - before, axis=-1) * 1000.0
    result["contact"] = {
        "baseline_vertices_within_3_mm_of_surface": int(contact.sum()),
        "same_vertices_still_within_3_mm": int(retained.sum()),
        "retained_fraction": float(retained.sum() / contact.sum()) if contact.any() else None,
        "same_vertices_within_3_mm_without_penetration_over_2_mm": int(retained_nonpenetrating.sum()),
        "retained_without_deep_penetration_fraction": (
            float(retained_nonpenetrating.sum() / contact.sum()) if contact.any() else None
        ),
        "baseline_contact_vertices_already_deeper_than_2_mm": int(
            (contact & before_inside & (before_distance > 0.002)).sum()
        ),
        "after_contact_ray_disagreements": int((contact & after_disagree).sum()),
    }
    result["vertex_displacement"] = {
        "mean_mm": float(movement.mean()),
        "p95_mm": float(np.percentile(movement, 95)),
        "max_mm": float(movement.max(initial=0.0)),
    }
    return result


def aggregate_frames(frames: list[dict]) -> dict:
    """Aggregate parallel before/after sampled-frame contact measurements."""
    result = {"sampled_frames": len(frames)}
    for stage in ("before", "after"):
        count = sum(frame[stage]["vertex_samples"] for frame in frames)
        result[stage] = {
            "vertex_samples": count,
            "vertices_deeper_than_2_mm": sum(frame[stage]["vertices_deeper_than_2_mm"] for frame in frames),
            "vertices_deeper_than_5_mm": sum(frame[stage]["vertices_deeper_than_5_mm"] for frame in frames),
            "sampled_frames_with_depth_over_2_mm": sum(
                frame[stage]["vertices_deeper_than_2_mm"] > 0 for frame in frames
            ),
            "sampled_frames_with_depth_over_5_mm": sum(
                frame[stage]["vertices_deeper_than_5_mm"] > 0 for frame in frames
            ),
            "maximum_penetration_depth_mm": max(frame[stage]["maximum_penetration_depth_mm"] for frame in frames),
            "mean_penetration_all_vertices_mm": sum(
                frame[stage]["mean_penetration_all_vertices_mm"] * frame[stage]["vertex_samples"] for frame in frames
            )
            / count,
            "ray_direction_disagreements_excluded": sum(
                frame[stage]["ray_direction_disagreements_excluded"] for frame in frames
            ),
        }
        for threshold in (2, 5):
            result[stage][f"fraction_deeper_than_{threshold}_mm"] = (
                result[stage][f"vertices_deeper_than_{threshold}_mm"] / count
            )
        checks = [frame[stage]["independent_sdf_check"] for frame in frames if "independent_sdf_check" in frame[stage]]
        if checks:
            total = sum(check["exact_query_points"] for check in checks)
            result[stage]["independent_sdf_check"] = {
                "exact_query_points": total,
                "max_abs_error_mm": max(check["max_abs_error_mm"] for check in checks),
                "mean_abs_error_mm": sum(check["mean_abs_error_mm"] * check["exact_query_points"] for check in checks)
                / max(total, 1),
                "sign_mismatches": sum(check["sign_mismatches"] for check in checks),
                "sign_mismatches_farther_than_two_grid_voxels": sum(
                    check["sign_mismatches_farther_than_two_grid_voxels"] for check in checks
                ),
                "depth_2_mm_classification_mismatches": sum(
                    check["depth_2_mm_classification_mismatches"] for check in checks
                ),
                "depth_5_mm_classification_mismatches": sum(
                    check["depth_5_mm_classification_mismatches"] for check in checks
                ),
            }
    contact_total = sum(frame["contact"]["baseline_vertices_within_3_mm_of_surface"] for frame in frames)
    retained_total = sum(frame["contact"]["same_vertices_still_within_3_mm"] for frame in frames)
    result["contact"] = {
        "baseline_vertices_within_3_mm_of_surface": contact_total,
        "same_vertices_still_within_3_mm": retained_total,
        "retained_fraction": retained_total / contact_total if contact_total else None,
    }
    return result


def validate_geometry(arrays: dict, original: dict, subject: dict, obj: dict) -> dict:
    """Validate all-frame identity and geometry before measuring training effects."""
    for key, expected in (
        ("source_frame_indices", original["source_frame_indices"]),
        ("obj_rot", original["object_rotation"]),
        ("obj_trans", original["object_translation"]),
        ("object_vertices_canonical", obj["object_vertices_canonical"]),
        ("object_faces", obj["object_faces"]),
        ("faces_l", subject["lhand_faces"]),
        ("faces_r", subject["rhand_faces"]),
    ):
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f"Source geometry changed: {key}")
    count = len(original["source_frame_indices"])
    baseline_error = 0.0
    for side, native in (("l", "lhand"), ("r", "rhand")):
        for stage in ("before", "after"):
            values = arrays[f"vertices_{side}_{stage}"]
            if values.shape != (count, 778, 3) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {side}/{stage} hand geometry")
        error = float(np.abs(arrays[f"vertices_{side}_before"] - original[f"{native}_vertices"]).max())
        baseline_error = max(baseline_error, error)
    if baseline_error > 1e-6:
        raise ValueError(f"Before geometry differs from prepared personalized MANO: {baseline_error} m")
    return {
        "all_object_rotations_unchanged": True,
        "all_object_translations_unchanged": True,
        "all_source_frame_indices_unchanged": True,
        "native_object_mesh_unchanged": True,
        "hand_topology_unchanged": True,
        "before_matches_personalized_mano_max_abs_error_m": baseline_error,
    }


def _initialize_worker(prepared: str) -> None:
    """Load a private mesh query index and SDF grid once per worker."""
    global _WORKER_MESH, _WORKER_GRID
    root = Path(prepared)
    obj = dict(np.load(root / "object.npz", allow_pickle=False))
    _WORKER_MESH = trimesh.Trimesh(obj["object_vertices_canonical"], obj["object_faces"], process=False)
    _WORKER_GRID = dict(np.load(root / "sdf.npz", allow_pickle=False))


def _audit_frame(task: tuple) -> tuple[int, dict]:
    """Measure one fixed sample in a reusable process-local triangle index."""
    clip_index, frame, source_frame, before, after = task
    result = frame_metrics(_WORKER_MESH, before, after, _WORKER_GRID)
    result.update(export_frame=frame, source_frame=source_frame)
    return clip_index, result


def main() -> None:
    """Write exact-triangle sampled metrics and all-frame identity checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path(__file__).with_name("prepared"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometry_manifest = args.geometry_manifest.resolve()
    manifest = json.loads(geometry_manifest.read_text())
    prepared = args.prepared.resolve()
    source_manifest = json.loads((prepared / "manifest.json").read_text())
    source_entries = {entry["clip_id"]: entry for entry in source_manifest["clips"]}
    entries = manifest["clips"]
    if len(entries) != 44 or {entry["clip_id"] for entry in entries} != set(source_entries):
        raise ValueError("Final audit requires all 44 GRAB mug sequences exactly once")
    obj = dict(np.load(prepared / "object.npz", allow_pickle=False))
    mesh = trimesh.Trimesh(obj["object_vertices_canonical"], obj["object_faces"], process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Mug must be an oriented, closed solid")
    special = np.array([[-0.015, 0.0, 0.0], [-0.015, 0.0, 0.040], [0.040, 0.0, 0.0], [0.050, 0.0, 0.0]])
    expected = np.array([False, False, False, True])
    for direction in RAY_DIRECTIONS:
        if not np.array_equal(contains_points(mesh.ray, special, check_direction=direction), expected):
            raise ValueError("Empty cup cavity or handle opening was incorrectly classified as solid")
    report = {
        "completed": False,
        "scope": "Training-set refinement on all 44 GRAB mug clips; not held-out generalization or physics validation",
        "geometry_manifest": str(geometry_manifest),
        "geometry_manifest_sha256": digest(geometry_manifest),
        "preparation_manifest_sha256": digest(prepared / "manifest.json"),
        "sdf_sha256": digest(prepared / "sdf.npz"),
        "method": "Exact nearest native triangles; containment confirmed by two deterministic ray directions",
        "sampling": "Round((T-1)*[0.25,0.50,0.75]) fixed independently of before/after results",
        "contact_definition": (
            "Baseline vertex within 3 mm of mug surface; same vertex must remain within 3 mm afterward"
        ),
        "ray_direction_disagreement_policy": (
            "Excluded from confirmed penetration and contact classification; counted explicitly"
        ),
        "mesh": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": True,
            "solid_volume_m3": mesh.volume,
        },
        "cavity_and_handle_probes_passed": True,
        "limitations": [
            "132 sampled frames do not prove that unexamined frames or triangle intersections are collision-free.",
            "Vertex penetration is not intersecting mesh volume, contact force, or dynamic grasp stability.",
            "Retention uses vertex identity and proximity; sliding contact can change which vertices touch.",
            "Personalized MANO differs from the previous cropped SMPL-X videos; before/after here share MANO shape.",
            "Noisy original motions are training inputs, not collision-free ground-truth targets.",
        ],
        "clips": [],
    }
    csv_rows = []
    all_frame_movement_sum = 0.0
    all_frame_movement_count = 0
    all_frame_movement_max = 0.0
    sample_tasks = []
    records = []
    for clip_index, entry in enumerate(entries):
        clip_id = entry["clip_id"]
        path = Path(entry["geometry_path"])
        if not path.is_absolute():
            path = geometry_manifest.parent / path
        arrays = dict(np.load(path, allow_pickle=False))
        original = dict(np.load(prepared / source_entries[clip_id]["clip_path"], allow_pickle=False))
        subject = dict(np.load(prepared / source_entries[clip_id]["subject_path"], allow_pickle=False))
        validation = validate_geometry(arrays, original, subject, obj)
        before = np.concatenate((arrays["vertices_l_before"], arrays["vertices_r_before"]), axis=1)
        after = np.concatenate((arrays["vertices_l_after"], arrays["vertices_r_after"]), axis=1)
        movement = np.linalg.norm(after - before, axis=-1) * 1000.0
        all_frame_movement_sum += float(movement.astype(np.float64).sum())
        all_frame_movement_count += movement.size
        all_frame_movement_max = max(all_frame_movement_max, float(movement.max()))
        record = {
            "clip_id": clip_id,
            "geometry_sha256": digest(path),
            "validation": validation,
            "all_frame_vertex_displacement": {
                "frames": len(before),
                "vertex_samples": movement.size,
                "mean_mm": float(movement.mean()),
                "p95_mm": float(np.percentile(movement, 95)),
                "max_mm": float(movement.max()),
            },
            "frames": [],
        }
        selected_frames = np.rint((len(before) - 1) * np.array([0.25, 0.50, 0.75])).astype(int)
        for frame in selected_frames:
            rotation = arrays["obj_rot"][frame].astype(np.float64)
            translation = arrays["obj_trans"][frame].astype(np.float64)
            before_local = (before[frame].astype(np.float64) - translation) @ rotation
            after_local = (after[frame].astype(np.float64) - translation) @ rotation
            sample_tasks.append(
                (clip_index, int(frame), int(arrays["source_frame_indices"][frame]), before_local, after_local)
            )
        records.append(record)
        print(json.dumps({"validated_all_frames": clip_id, "frames": len(before)}), flush=True)
    if args.workers < 1:
        parser.error("workers must be positive")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=_initialize_worker, initargs=(str(prepared),)
    ) as executor:
        for clip_index, metrics in executor.map(_audit_frame, sample_tasks):
            record = records[clip_index]
            record["frames"].append(metrics)
            row = {
                "clip_id": record["clip_id"],
                "export_frame": metrics["export_frame"],
                "source_frame": metrics["source_frame"],
            }
            for stage in ("before", "after"):
                row.update(
                    {f"{stage}_{key}": value for key, value in metrics[stage].items() if not isinstance(value, dict)}
                )
            row.update({f"contact_{key}": value for key, value in metrics["contact"].items()})
            row.update({f"displacement_{key}": value for key, value in metrics["vertex_displacement"].items()})
            csv_rows.append(row)
            if len(record["frames"]) == 3:
                record["summary"] = aggregate_frames(record["frames"])
                report["clips"].append(record)
                write_json(output / "exact_contact_audit.json", report)
                print(json.dumps({"audited": record["clip_id"], "summary": record["summary"]}), flush=True)
    frames = [frame for clip in report["clips"] for frame in clip["frames"]]
    summary = aggregate_frames(frames)
    summary["clips"] = len(entries)
    summary["all_frame_vertex_displacement"] = {
        "vertex_samples": all_frame_movement_count,
        "mean_mm": all_frame_movement_sum / all_frame_movement_count,
        "max_mm": all_frame_movement_max,
    }
    summary["clips_with_lower_sampled_over_2_mm_vertex_count"] = sum(
        clip["summary"]["after"]["vertices_deeper_than_2_mm"] < clip["summary"]["before"]["vertices_deeper_than_2_mm"]
        for clip in report["clips"]
    )
    summary["clips_with_higher_sampled_over_2_mm_vertex_count"] = sum(
        clip["summary"]["after"]["vertices_deeper_than_2_mm"] > clip["summary"]["before"]["vertices_deeper_than_2_mm"]
        for clip in report["clips"]
    )
    report["summary"] = summary
    report["elapsed_seconds"] = time.monotonic() - started
    report["completed"] = True
    write_json(output / "exact_contact_audit.json", report)
    with (output / "exact_contact_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    clip_rows = []
    for clip in report["clips"]:
        row = {"clip_id": clip["clip_id"]}
        for stage in ("before", "after"):
            row.update(
                {
                    f"{stage}_{key}": value
                    for key, value in clip["summary"][stage].items()
                    if not isinstance(value, dict)
                }
            )
        row.update({f"contact_{key}": value for key, value in clip["summary"]["contact"].items()})
        row.update(
            {f"all_frame_displacement_{key}": value for key, value in clip["all_frame_vertex_displacement"].items()}
        )
        clip_rows.append(row)
    with (output / "exact_contact_clips.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(clip_rows[0]))
        writer.writeheader()
        writer.writerows(clip_rows)
    print(json.dumps({"completed": True, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
