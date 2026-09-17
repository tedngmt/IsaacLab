# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare all 18 selected GraspXL mug clips for right-hand Text2HOI refinement.

Run in the existing text2hoi Conda environment. This script preserves original
MANO geometry and object motion; it performs no training, fitting, or conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh

UID = "12652c6e0aaf44dda32aab816f433baf"
TABLETOP_BIAS = np.asarray([0.09566994, 0.00638343, 0.0061863], dtype=np.float32)


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    """Save finite numeric arrays; unknown source FPS is explicitly NaN."""
    for key, value in arrays.items():
        if key != "source_fps" and value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {path.name}:{key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def farthest_point_indices(vertices: np.ndarray, count: int) -> np.ndarray:
    """Select deterministic unique vertex indices for a point cloud [m]."""
    if len(np.unique(vertices, axis=0)) < count:
        raise ValueError("Not enough distinct mesh vertices")
    selected = np.empty(count, dtype=np.int64)
    distance = np.full(len(vertices), np.inf)
    candidate = int(np.square(vertices - vertices.mean(axis=0)).sum(axis=1).argmax())
    for index in range(count):
        selected[index] = candidate
        distance = np.minimum(distance, np.square(vertices - vertices[candidate]).sum(axis=1))
        distance[selected[: index + 1]] = -1
        candidate = int(distance.argmax())
    return selected


@torch.no_grad()
def main() -> None:
    """Encode all source frames and validate hands/object reconstruction [m]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).with_name("prepared"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    projects = args.projects_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = projects / "Text2HOI"
    local = projects / "IsaacLab/_local"
    source_root = projects / "GraspXL_MANO_51Objects_GRABMatched"
    cache_root = local / "mug_comparisons/cache/GraspXL"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from lib.models.mano import build_mano_aa
    from lib.utils.proc_output import get_hand_joints_w_tip, get_hand_verts, get_transformed_obj_pc
    from lib.utils.rot import axis_angle_to_rot6d, rotmat_to_rot6d

    torch.set_num_threads(4)
    pairs = json.loads((source_root / "metadata/pairs.json").read_text())
    matched = {
        f"{pair['source_archive']}__{Path(pair['mano_path']).stem}": pair
        for pair in pairs
        if pair["grab_object"] == "mug"
    }
    cache_paths = sorted(cache_root.glob("*.npz"))
    expected = set(matched) | {f"tabletop_large__mano_{index}" for index in range(3)}
    if len(matched) != 15 or {path.stem for path in cache_paths} != expected:
        raise ValueError("Expected exactly 15 matched and 3 supplemental tabletop mug clips")
    models = {side: build_mano_aa(is_rhand=side == "rhand", flat_hand=True) for side in ("lhand", "rhand")}
    nonflat = smplx.MANO(
        str(projects / "SOMA-X/assets/MANO/MANO_RIGHT.pkl"),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
        create_transl=False,
    )
    assets = {}
    for side, model in models.items():
        assets[f"{side}_v_template"] = model.v_template.numpy()
        assets[f"{side}_model_betas"] = np.zeros(10, dtype=np.float32)
        assets[f"{side}_source_betas"] = np.zeros(10, dtype=np.float32)
        assets[f"{side}_faces"] = model.faces.astype(np.int32)
    save_arrays(output / "subjects/mano_zero.npz", **assets)

    cache_first = np.load(cache_paths[0], allow_pickle=False)
    object_vertices = cache_first["object_vertices_canonical"].astype(np.float32)
    object_faces = cache_first["object_faces"].astype(np.int32)
    mesh = trimesh.Trimesh(object_vertices, object_faces, process=False)
    points_idx = farthest_point_indices(object_vertices, 1024)
    points = object_vertices[points_idx]
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)[points_idx]
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    centroid = points.mean(axis=0)
    scale = np.linalg.norm(points - centroid, axis=-1).max()
    save_arrays(
        output / "object.npz",
        object_vertices_canonical=object_vertices,
        object_faces=object_faces,
        object_points=points,
        object_normals=normals,
        object_points_normalized=(points - centroid) / scale,
        object_centroid=centroid,
        object_scale=np.asarray(scale, dtype=np.float32),
        object_point_indices=points_idx,
    )
    points_t = torch.from_numpy(points)[None]
    vertices_t = torch.from_numpy(object_vertices)[None]
    records = []
    maximum_hand_error = maximum_object_error = maximum_pose_roundtrip = 0.0

    for cache_path in cache_paths:
        clip_id = cache_path.stem
        cache = np.load(cache_path, allow_pickle=False)
        indices = cache["source_frame_indices"]
        supplemental = clip_id.startswith("tabletop_")
        if supplemental:
            provenance = json.loads(cache_path.with_suffix(".json").read_text())
            source_path = Path(provenance["source_path"])
            expected_frames = 100
            subset = "tabletop_large"
        else:
            pair = matched[clip_id]
            source_path = source_root / pair["mano_path"]
            expected_frames = 155
            subset = pair["source_archive"]
        source = np.load(source_path, allow_pickle=True).item()
        hand, obj = source["right_hand"], source[UID]
        if set(hand) != {"trans", "rot", "pose"}:
            raise ValueError(f"Unrecognized hand identity/pose fields: {clip_id}")
        if len(indices) != expected_frames or not np.array_equal(indices, np.arange(expected_frames)):
            raise ValueError(f"Full native timeline not preserved: {clip_id}")
        if "angle" in obj and np.abs(obj["angle"]).max() != 0:
            raise ValueError(f"Unexpected object articulation: {clip_id}")
        np.testing.assert_array_equal(cache["object_vertices_canonical"].astype(np.float32), object_vertices)
        np.testing.assert_array_equal(cache["object_faces"], object_faces)
        np.testing.assert_array_equal(cache["original_faces"], models["rhand"].faces)
        rotation = torch.from_numpy(hand["rot"])
        source_pose = torch.from_numpy(hand["pose"])
        if supplemental:
            full_pose = source_pose
            translation = torch.from_numpy(hand["trans"] - TABLETOP_BIAS)
        else:
            full_pose = source_pose + nonflat.hand_mean[None]
            translation = torch.from_numpy(hand["trans"])
        pose = torch.cat((rotation, full_pose), dim=-1)
        x_hand = torch.cat((translation, axis_angle_to_rot6d(pose.reshape(-1, 3)).reshape(len(indices), 96)), dim=-1)
        active_rotation = cache["object_rotation"].astype(np.float32)
        object_translation = cache["object_translation"].astype(np.float32)
        x_obj = torch.cat(
            (torch.from_numpy(object_translation), rotmat_to_rot6d(torch.from_numpy(active_rotation).transpose(1, 2))),
            dim=-1,
        )
        dummy_left = torch.cat(
            (
                torch.zeros(len(indices), 3),
                axis_angle_to_rot6d(torch.zeros(len(indices) * 16, 3)).reshape(len(indices), 96),
            ),
            dim=-1,
        )
        vertex_parts, joint_parts, contact_indices_parts, contact_distances_parts = [], [], [], []
        hand_error = object_error = pose_error = 0.0
        for start in range(0, len(indices), args.batch_size):
            interval = slice(start, start + args.batch_size)
            vertices = get_hand_verts(x_hand[None, interval], models["rhand"])[0]
            joints = get_hand_joints_w_tip(x_hand[None, interval], models["rhand"])[0]
            direct = (
                models["rhand"](
                    betas=torch.zeros(len(vertices), 10),
                    global_orient=rotation[interval],
                    hand_pose=full_pose[interval],
                ).vertices
                + translation[interval, None]
            )
            pose_error = max(pose_error, float((vertices - direct).abs().max()))
            hand_error = max(hand_error, float(np.abs(vertices.numpy() - cache["original_vertices"][interval]).max()))
            object_world = get_transformed_obj_pc(x_obj[None, interval], vertices_t, "grab")[0].numpy()
            expected_object = np.einsum("tij,kj->tki", active_rotation[interval], object_vertices)
            expected_object += object_translation[interval, None]
            object_error = max(object_error, float(np.abs(object_world - expected_object).max()))
            object_points = get_transformed_obj_pc(x_obj[None, interval], points_t, "grab")[0]
            distances = torch.cdist(joints, object_points, compute_mode="donot_use_mm_for_euclid_dist")
            near_distance, near_indices = distances.min(dim=-1)
            vertex_parts.append(vertices.numpy())
            joint_parts.append(joints.numpy())
            contact_indices_parts.append(near_indices.numpy())
            contact_distances_parts.append(near_distance.numpy())
        if max(hand_error, object_error, pose_error) > 1e-6:
            raise ValueError(f"Source reconstruction mismatch: {clip_id}: {hand_error}, {object_error}, {pose_error}")
        maximum_hand_error = max(maximum_hand_error, hand_error)
        maximum_object_error = max(maximum_object_error, object_error)
        maximum_pose_roundtrip = max(maximum_pose_roundtrip, pose_error)
        rhand_distances = np.concatenate(contact_distances_parts)
        arrays = {
            "x_lhand": dummy_left.numpy(),
            "x_rhand": x_hand.numpy(),
            "x_obj": x_obj.numpy(),
            "lhand_vertices": np.zeros((len(indices), 778, 3), dtype=np.float32),
            "rhand_vertices": np.concatenate(vertex_parts),
            "lhand_joints": np.zeros((len(indices), 21, 3), dtype=np.float32),
            "rhand_joints": np.concatenate(joint_parts),
            "lhand_contact_point_indices": np.zeros((len(indices), 21), dtype=np.int64),
            "rhand_contact_point_indices": np.concatenate(contact_indices_parts),
            "lhand_contact_distances": np.zeros((len(indices), 21), dtype=np.float32),
            "rhand_contact_distances": rhand_distances,
            "lhand_contact_mask": np.zeros((len(indices), 21), dtype=bool),
            "rhand_contact_mask": rhand_distances < 0.02,
            "valid_mask_lhand": np.zeros(len(indices), dtype=bool),
            "valid_mask_rhand": np.ones(len(indices), dtype=bool),
            "valid_mask_obj": np.ones(len(indices), dtype=bool),
            "is_lhand": np.asarray(False),
            "is_rhand": np.asarray(True),
            "object_rotation": active_rotation,
            "object_translation": object_translation,
            "source_frame_indices": indices,
            "source_fps": np.asarray(np.nan, dtype=np.float32),
            "playback_fps": np.asarray(30.0, dtype=np.float32),
        }
        path = output / "clips" / f"{clip_id}.npz"
        save_arrays(path, **arrays)
        record = {
            "clip_id": clip_id,
            "dataset": "GraspXL",
            "subject": "mano_zero",
            "gender": "neutral",
            "action": "grasp",
            "source_subset": subset,
            "supplemental": supplemental,
            "frames": len(indices),
            "original_frames": len(indices),
            "source_fps": None,
            "source_fps_known": False,
            "playback_fps": 30,
            "source_stride": 1,
            "is_lhand": False,
            "is_rhand": True,
            "clip_path": str(path.relative_to(output)),
            "subject_path": "subjects/mano_zero.npz",
            "source_path": str(source_path),
            "source_sha256": digest(source_path),
            "reference_cache": str(cache_path),
            "reference_cache_sha256": digest(cache_path),
            "clip_sha256": digest(path),
            "native_mano_cache_max_abs_error_m": hand_error,
            "object_roundtrip_max_abs_error_m": object_error,
            "pose_roundtrip_max_abs_error_m": pose_error,
            "rhand_frames_with_joint_proximity_under_20_mm": int((rhand_distances < 0.02).any(axis=-1).sum()),
            "pose_convention": (
                "Tabletop source is flat-hand mean; subtract fixed wrist bias from translation"
                if supplemental
                else "Matched source uses nonflat mean; add MANO hand_mean to finger pose before encoding"
            ),
        }
        records.append(record)
        print(json.dumps({"prepared": clip_id, "frames": len(indices), "hand_error_m": hand_error}), flush=True)

    manifest = {
        "schema_version": 1,
        "dataset": "GraspXL",
        "purpose": "All 18 selected-mug source motions; training-set refinement, not held-out generalization",
        "sequences": len(records),
        "frames": sum(record["frames"] for record in records),
        "matched_sequences": 15,
        "supplemental_tabletop_sequences": 3,
        "model_representation": (
            "Original right MANO, standard zero-beta template, flat_hand_mean=True after pose adaptation"
        ),
        "absent_left_policy": (
            "Left hand is absent: all validity/contact masks false; stored vertices/joints zero. "
            "Dummy x_lhand uses zero translation and valid identity6D rotations for safe upstream decoding. "
            "Exclude left hand from every loss, metric, render, and contact computation."
        ),
        "source_contact_annotations": "Not available; no source_contact arrays are fabricated",
        "source_fps": None,
        "source_fps_known": False,
        "playback_fps": 30,
        "temporal_policy": "All native frames retained at stride1; 30 FPS is an assumed playback rate, not resampling",
        "object_rotation_policy": "Source active R retained; x_obj encodes R.T for Text2HOI's GRAB row-vector decoder",
        "model_dataset_decoder": "grab",
        "object_uid": UID,
        "object_path": "object.npz",
        "object_sha256": digest(output / "object.npz"),
        "object_point_cloud": (
            "1024 deterministic farthest-point native vertices; normalized only for PointNet features"
        ),
        "object_vertices": len(object_vertices),
        "object_faces": len(object_faces),
        "object_watertight": bool(mesh.is_watertight),
        "object_winding_consistent": bool(mesh.is_winding_consistent),
        "object_volume_m3": float(mesh.volume),
        "subject_paths": {"mano_zero": "subjects/mano_zero.npz"},
        "subject_sha256": {"mano_zero": digest(output / "subjects/mano_zero.npz")},
        "max_native_mano_cache_abs_error_m": maximum_hand_error,
        "max_object_roundtrip_abs_error_m": maximum_object_error,
        "max_pose_roundtrip_abs_error_m": maximum_pose_roundtrip,
        "source_datasets_modified": False,
        "soma_fitting_performed": False,
        "clips": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Prepared {manifest['sequences']} clips / {manifest['frames']} frames at {output}")


if __name__ == "__main__":
    main()
