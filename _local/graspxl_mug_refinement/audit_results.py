# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit refined GraspXL right-hand geometry independently of the training loss.

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
        "ray_direction_disagreements_farther_than_2_mm": int((disagreement & (distance > 0.002)).sum()),
        "ray_direction_disagreement_max_surface_distance_mm": float(distance[disagreement].max(initial=0.0) * 1000.0),
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
    contact = before_distance < 0.003
    retained = contact & (after_distance < 0.003)
    retained_nonpenetrating = retained & ~after_disagree & ~(after_inside & (after_distance > 0.002))
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
        "before_contact_ray_disagreements": int((contact & before_disagree).sum()),
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
            "ray_direction_disagreements_farther_than_2_mm": sum(
                frame[stage]["ray_direction_disagreements_farther_than_2_mm"] for frame in frames
            ),
            "ray_direction_disagreement_max_surface_distance_mm": max(
                frame[stage]["ray_direction_disagreement_max_surface_distance_mm"] for frame in frames
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
    for key in (
        "same_vertices_within_3_mm_without_penetration_over_2_mm",
        "baseline_contact_vertices_already_deeper_than_2_mm",
        "after_contact_ray_disagreements",
        "before_contact_ray_disagreements",
    ):
        result["contact"][key] = sum(frame["contact"][key] for frame in frames)
    result["contact"]["retained_without_deep_penetration_fraction"] = (
        result["contact"]["same_vertices_within_3_mm_without_penetration_over_2_mm"] / contact_total
        if contact_total
        else None
    )
    return result


def resolve(base: Path, name: str) -> Path:
    """Resolve an absolute or manifest-relative asset path."""
    path = Path(name)
    return path if path.is_absolute() else base / path


def load_arrays(path: Path) -> dict:
    """Load numeric geometry arrays without object deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_pose(values: np.ndarray, count: int, name: str) -> dict:
    """Check finite pose translations [m] and nondegenerate rotation-6D pairs."""
    if values.shape != (count, 99) or not np.isfinite(values).all():
        raise ValueError(f"Invalid finite pose array: {name}")
    rotations = values[:, 3:].reshape(count, 16, 3, 2)
    first, second = rotations[..., 0], rotations[..., 1]
    first_norm = np.linalg.norm(first, axis=-1)
    second_norm = np.linalg.norm(second, axis=-1)
    cross_norm = np.linalg.norm(np.cross(first, second), axis=-1)
    if (first_norm < 1e-8).any() or (second_norm < 1e-8).any() or (cross_norm < 1e-8).any():
        raise ValueError(f"Degenerate rotation-6D pair: {name}")
    return {
        "all_finite": True,
        "all_rotation_6d_pairs_nondegenerate": True,
        "minimum_pair_cross_norm": float(cross_norm.min()),
        "anatomical_joint_limits_checked": False,
    }


def validate_geometry(arrays: dict, original: dict, reference: dict, subject: dict, obj: dict) -> dict:
    """Check unchanged source meshes and poses over every right-hand frame [m]."""
    for key, expected in (
        ("source_frame_indices", original["source_frame_indices"]),
        ("obj_rot", original["object_rotation"]),
        ("obj_trans", original["object_translation"]),
        ("object_vertices_canonical", obj["object_vertices_canonical"]),
        ("object_faces", obj["object_faces"]),
        ("faces_r", subject["rhand_faces"]),
        ("x_rhand_before", original["x_rhand"]),
    ):
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f"Prepared source geometry or before pose changed: {key}")
    count = len(original["source_frame_indices"])
    if count not in (100, 155) or not np.array_equal(arrays["source_frame_indices"], np.arange(count)):
        raise ValueError("Every original frame must be present at stride one")
    if original["is_lhand"].item() or original["valid_mask_lhand"].any():
        raise ValueError("GraspXL left hand must be absent and excluded from the audit")
    if not original["is_rhand"].item() or not original["valid_mask_rhand"].all():
        raise ValueError("Every right-hand frame must be valid")
    for stage in ("before", "after"):
        vertices = arrays[f"vertices_r_{stage}"]
        if vertices.shape != (count, 778, 3) or not np.isfinite(vertices).all():
            raise ValueError(f"Invalid {stage} right-hand geometry")
    baseline_error = float(np.abs(arrays["vertices_r_before"] - original["rhand_vertices"]).max())
    reference_error = float(np.abs(arrays["vertices_r_before"] - reference["original_vertices"]).max())
    if max(baseline_error, reference_error) > 1e-6:
        raise ValueError("Before MANO geometry differs from the original source cache by more than 1 micrometer")
    reference_errors = {}
    for key, cached_key in (
        ("obj_rot", "object_rotation"),
        ("obj_trans", "object_translation"),
        ("object_vertices_canonical", "object_vertices_canonical"),
    ):
        values, cached = arrays[key], reference[cached_key]
        if not np.array_equal(values, cached.astype(values.dtype)):
            raise ValueError(f"Object differs from source cache beyond storage precision: {key}")
        reference_errors[key] = float(np.abs(values.astype(np.float64) - cached.astype(np.float64)).max())
    for key, cached_key in (
        ("object_faces", "object_faces"),
        ("faces_r", "original_faces"),
        ("source_frame_indices", "source_frame_indices"),
    ):
        if not np.array_equal(arrays[key], reference[cached_key]):
            raise ValueError(f"Source cache topology or timing changed: {key}")
    rotation = arrays["obj_rot"]
    if not np.isfinite(rotation).all() or not np.isfinite(arrays["obj_trans"]).all():
        raise ValueError("Object poses are not finite")
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=2e-4):
        raise ValueError("Object rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise ValueError("Object rotations are not proper active rotations")
    poses = {stage: validate_pose(arrays[f"x_rhand_{stage}"], count, stage) for stage in ("before", "after")}
    return {
        "all_object_rotations_unchanged_from_prepared": True,
        "all_object_translations_unchanged_from_prepared": True,
        "all_source_frame_indices_unchanged": True,
        "native_object_mesh_unchanged": True,
        "hand_topology_unchanged": True,
        "left_hand_absent_and_excluded": True,
        "before_matches_prepared_mano_max_abs_error_m": baseline_error,
        "before_matches_original_source_cache_max_abs_error_m": reference_error,
        "source_cache_object_rounding_max_abs_errors": reference_errors,
        "finite_poses": poses,
    }


def load_collision_mesh(prepared: Path) -> tuple[trimesh.Trimesh, dict]:
    """Verify an unchanged native triangle surface and its query topology [m]."""
    native_path = prepared / "object.npz"
    native = load_arrays(native_path)
    native_mesh = trimesh.Trimesh(native["object_vertices_canonical"], native["object_faces"], process=False)
    proxy_path = prepared / "collision_proxy.npz"
    if proxy_path.exists():
        proxy = load_arrays(proxy_path)
        if not np.array_equal(
            native["object_vertices_canonical"][native["object_faces"]],
            proxy["object_vertices_canonical"][proxy["object_faces"]],
        ):
            raise ValueError("Collision proxy changes native ordered triangle coordinates")
        if not np.array_equal(proxy["face_source_indices"], np.arange(len(native["object_faces"]))):
            raise ValueError("Collision proxy reorders native faces")
        if not np.array_equal(
            proxy["object_vertices_canonical"], native["object_vertices_canonical"][proxy["vertex_source_indices"]]
        ):
            raise ValueError("Collision proxy includes changed source vertices")
        mesh = trimesh.Trimesh(proxy["object_vertices_canonical"], proxy["object_faces"], process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError("Reindexed query topology is not a positively oriented watertight mesh")
        mode = "Reindexed watertight proxy with bitwise-identical ordered native triangles"
        query_path = proxy_path
    else:
        mesh, query_path = native_mesh, native_path
        mode = "Native object triangles; nonmanifold topology can make parity classification ambiguous"
    triangles = np.ascontiguousarray(native["object_vertices_canonical"][native["object_faces"]])
    probe_points = np.array(
        [[-0.018, 0.0, 0.0], [-0.018, 0.04, 0.0], [0.043, 0.0, 0.0], [0.057, 0.0, 0.0], [-0.018, -0.05, 0.0]]
    )
    expected_inside = np.array([False, False, False, True, True])
    for direction in RAY_DIRECTIONS:
        if not np.array_equal(contains_points(mesh.ray, probe_points, check_direction=direction), expected_inside):
            raise ValueError("Known cavity, rim, handle opening, or material probes failed")
    return mesh, {
        "query_surface": mode,
        "query_mesh_path": str(query_path),
        "query_mesh_sha256": digest(query_path),
        "native_object_sha256": digest(native_path),
        "ordered_triangle_coordinates_sha256": hashlib.sha256(triangles.tobytes()).hexdigest(),
        "ordered_triangle_coordinates_bitwise_identical": True,
        "native_vertices": len(native_mesh.vertices),
        "query_vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "native_watertight": bool(native_mesh.is_watertight),
        "query_watertight": bool(mesh.is_watertight),
        "query_winding_consistent": bool(mesh.is_winding_consistent),
        "oriented_volume_m3": float(mesh.volume),
        "surface_positions_changed": False,
        "self_intersections_removed": False,
        "cavity_handle_and_material_probes_passed": True,
        "probe_points_m": probe_points.tolist(),
        "probe_expected_inside": expected_inside.tolist(),
    }


def _initialize_worker(prepared: str) -> None:
    """Create process-local triangle and optional SDF query structures."""
    global _WORKER_MESH, _WORKER_GRID
    root = Path(prepared)
    _WORKER_MESH, _ = load_collision_mesh(root)
    grid_path = root / "sdf.npz"
    _WORKER_GRID = load_arrays(grid_path) if grid_path.exists() else None


def _audit_frame(task: tuple) -> tuple[int, dict]:
    """Measure one predetermined frame in canonical object coordinates [m]."""
    clip_index, frame, source_frame, before, after = task
    result = frame_metrics(_WORKER_MESH, before, after, _WORKER_GRID)
    result.update(export_frame=frame, source_frame=source_frame)
    return clip_index, result


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a flat audit table atomically."""
    temporary = path.with_suffix(".writing.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    """Audit all 18 training clips without modifying geometry or source data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path(__file__).with_name("prepared"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    started = time.monotonic()
    run_dir, prepared = args.run_dir.resolve(), args.prepared.resolve()
    geometry_manifest = run_dir / "geometry_manifest.json"
    manifest = json.loads(geometry_manifest.read_text())
    source_manifest_path = prepared / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_entries = {entry["clip_id"]: entry for entry in source_manifest["clips"]}
    entries = manifest["clips"]
    if (
        len(entries) != 18
        or len(source_entries) != 18
        or {entry["clip_id"] for entry in entries} != set(source_entries)
    ):
        raise ValueError("Final audit requires all 18 selected GraspXL mug sequences exactly once")
    if sum(bool(entry["supplemental"]) for entry in source_entries.values()) != 3:
        raise ValueError("Expected 15 matched diverse-grasp sequences and 3 supplemental tabletop sequences")
    obj = load_arrays(prepared / "object.npz")
    _, mesh_info = load_collision_mesh(prepared)
    report = {
        "completed": False,
        "scope": "Training-set right-hand refinement on all 18 selected GraspXL mug clips; not held-out evaluation",
        "clip_count": 18,
        "matched_clip_count": 15,
        "supplemental_clip_count": 3,
        "source_fps": None,
        "playback_fps_assumed": 30,
        "geometry_manifest": str(geometry_manifest),
        "geometry_manifest_sha256": digest(geometry_manifest),
        "preparation_manifest_sha256": digest(source_manifest_path),
        "audit_script_sha256": digest(Path(__file__)),
        "sdf_sha256": digest(prepared / "sdf.npz") if (prepared / "sdf.npz").exists() else None,
        "method": "Exact nearest native triangles; inside classification agreed by two deterministic ray directions",
        "ray_directions": [direction.tolist() for direction in RAY_DIRECTIONS],
        "sampling": "Round((T-1)*[0.25,0.50,0.75]); fixed independently of refinement results",
        "contact_definition": (
            "Unsigned distance within 3 mm of surface; same vertex must remain within 3 mm afterward. "
            "Unsigned retention includes ray-ambiguous samples; signed-depth claims exclude them."
        ),
        "ray_direction_disagreement_policy": "Exclude disagreements from signed depths; count them explicitly",
        "mesh": mesh_info,
        "limitations": [
            "54 sampled frames do not establish collision freedom at unexamined times or between hand triangles.",
            "Vertex penetration is not intersecting volume, contact force, or dynamic grasp stability.",
            "Ray agreement is an operational inside estimate; self-intersections and thin folded surfaces can remain.",
            "Collision reindexing does not move triangles, repair geometry, or remove source mesh self-intersections.",
            "Contact retention uses vertex identity and proximity; sliding can change which vertices touch.",
            "No source contact labels exist; contact proximity is measured, not ground-truth supervision.",
            "Finite poses and nondegenerate rotation pairs do not establish anatomical plausibility.",
            "Source FPS is unknown; assumed 30 FPS playback is not the physical motion duration.",
            "All 18 clips were used in training, so improvement here does not establish generalization.",
        ],
        "clips": [],
    }
    records, sample_tasks = [], []
    movement_sum, movement_count, movement_max = 0.0, 0, 0.0
    source_hashes = {}
    for clip_index, entry in enumerate(entries):
        clip_id = entry["clip_id"]
        source_entry = source_entries[clip_id]
        geometry_path = resolve(geometry_manifest.parent, entry["geometry_path"])
        metadata_path = resolve(geometry_manifest.parent, entry["metadata_path"])
        source_path = resolve(prepared, source_entry["clip_path"])
        cache_path = Path(source_entry["reference_cache"])
        for path, expected in (
            (source_path, source_entry["clip_sha256"]),
            (cache_path, source_entry["reference_cache_sha256"]),
        ):
            current = digest(path)
            if current != expected:
                raise ValueError(f"Prepared source provenance hash changed: {path}")
            source_hashes[str(path)] = current
        arrays = load_arrays(geometry_path)
        original = load_arrays(source_path)
        reference = load_arrays(cache_path)
        subject = load_arrays(resolve(prepared, source_entry["subject_path"]))
        metadata = json.loads(metadata_path.read_text())
        if metadata["clip_id"] != clip_id or metadata.get("geometry_sha256") != digest(geometry_path):
            raise ValueError(f"Geometry metadata hash or identity differs: {clip_id}")
        if metadata["source_fps"] is not None or metadata["playback_fps"] != 30:
            raise ValueError("Source rate must remain unknown and assumed playback must remain 30 FPS")
        validation = validate_geometry(arrays, original, reference, subject, obj)
        before, after = arrays["vertices_r_before"], arrays["vertices_r_after"]
        expected_count = 100 if source_entry["supplemental"] else 155
        if len(before) != expected_count or metadata["source_frames"] != expected_count:
            raise ValueError(f"Source frames were omitted or added: {clip_id}")
        movement = np.linalg.norm(after.astype(np.float64) - before, axis=-1) * 1000.0
        movement_sum += float(movement.sum())
        movement_count += movement.size
        movement_max = max(movement_max, float(movement.max()))
        record = {
            "clip_id": clip_id,
            "supplemental": bool(source_entry["supplemental"]),
            "geometry_path": str(geometry_path),
            "geometry_sha256": digest(geometry_path),
            "metadata_sha256": digest(metadata_path),
            "prepared_clip_sha256": source_hashes[str(source_path)],
            "original_reference_cache_sha256": source_hashes[str(cache_path)],
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
    output_path = run_dir / "independent_audit.json"
    write_json(output_path, report)
    frame_rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=_initialize_worker, initargs=(str(prepared),)
    ) as executor:
        for clip_index, metrics in executor.map(_audit_frame, sample_tasks):
            record = records[clip_index]
            record["frames"].append(metrics)
            row = {
                "clip_id": record["clip_id"],
                "supplemental": record["supplemental"],
                "export_frame": metrics["export_frame"],
                "source_frame": metrics["source_frame"],
            }
            for stage in ("before", "after"):
                row.update(
                    {f"{stage}_{key}": value for key, value in metrics[stage].items() if not isinstance(value, dict)}
                )
            row.update({f"contact_{key}": value for key, value in metrics["contact"].items()})
            row.update({f"displacement_{key}": value for key, value in metrics["vertex_displacement"].items()})
            frame_rows.append(row)
            if len(record["frames"]) == 3:
                record["summary"] = aggregate_frames(record["frames"])
                report["clips"].append(record)
                write_json(output_path, report)
                print(json.dumps({"audited": record["clip_id"], "summary": record["summary"]}), flush=True)
    frames = [frame for clip in report["clips"] for frame in clip["frames"]]
    summary = aggregate_frames(frames)
    summary["clips"] = len(entries)
    summary["all_frame_vertex_displacement"] = {
        "vertex_samples": movement_count,
        "mean_mm": movement_sum / movement_count,
        "max_mm": movement_max,
    }
    for relation, compare in (("lower", np.less), ("higher", np.greater), ("equal", np.equal)):
        summary[f"clips_with_{relation}_sampled_over_2_mm_vertex_count"] = sum(
            bool(
                compare(
                    clip["summary"]["after"]["vertices_deeper_than_2_mm"],
                    clip["summary"]["before"]["vertices_deeper_than_2_mm"],
                )
            )
            for clip in report["clips"]
        )
    report["summary"] = summary
    report["subsets"] = {
        name: aggregate_frames(
            [frame for clip in report["clips"] if clip["supplemental"] == supplemental for frame in clip["frames"]]
        )
        for name, supplemental in (("matched_diverse_grasp", False), ("supplemental_tabletop", True))
    }
    for record in records:
        if digest(Path(record["geometry_path"])) != record["geometry_sha256"]:
            raise ValueError(f"Geometry changed during the audit: {record['clip_id']}")
    if digest(geometry_manifest) != report["geometry_manifest_sha256"]:
        raise ValueError("Geometry manifest changed during the audit")
    report["elapsed_seconds"] = time.monotonic() - started
    report["completed"] = True
    write_json(output_path, report)
    write_csv(run_dir / "independent_audit_frames.csv", frame_rows)
    clip_rows = []
    for clip in report["clips"]:
        row = {"clip_id": clip["clip_id"], "supplemental": clip["supplemental"]}
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
    write_csv(run_dir / "independent_audit_clips.csv", clip_rows)
    print(json.dumps({"completed": True, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
