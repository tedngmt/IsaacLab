# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare all GRAB mug clips for personalized MANO motion refinement.

Run with the text2hoi environment. This preparation preserves full clip timing
at 30 FPS and does not optimize motions, train a model, or change source files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def zip_array(archive: zipfile.ZipFile, suffix: str) -> np.ndarray:
    """Read a unique numeric array from an archive."""
    names = [name for name in archive.namelist() if Path(name).name == suffix]
    if len(names) != 1:
        raise ValueError(f"Expected one {suffix}: {names}")
    return np.load(io.BytesIO(archive.read(names[0])), allow_pickle=False)


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    """Save finite numeric arrays without pickle payloads."""
    for key, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {path.name}:{key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


@torch.no_grad()
def main() -> None:
    """Reconstruct original personalized MANO hands and report differences [m]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).with_name("prepared"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    projects = args.projects_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = projects / "Text2HOI"
    grab = projects / "Grab Dataset"
    cache_root = projects / "IsaacLab/_local/mug_comparisons/cache/GRAB"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from lib.models.mano import build_mano_aa
    from lib.utils.proc_output import get_hand_joints_w_tip, get_hand_verts, get_transformed_obj_pc
    from lib.utils.rot import axis_angle_to_rot6d, rotmat_to_rot6d

    torch.set_num_threads(4)
    cache_paths = sorted(cache_root.glob("*.npz"))
    if len(cache_paths) != 44:
        raise ValueError(f"Expected all 44 GRAB mug clips, found {len(cache_paths)}")
    with zipfile.ZipFile(grab / "tools__smplx_correspondence.zip") as archive:
        hand_ids = {side: zip_array(archive, f"{side}_smplx_ids.npy") for side in ("lhand", "rhand")}
    all_hand_ids = np.unique(np.concatenate(list(hand_ids.values())))
    cache_hand_ids = {side: np.searchsorted(all_hand_ids, ids) for side, ids in hand_ids.items()}

    with (repo / "data/grab/obj.pkl").open("rb") as stream:
        object_data = pickle.load(stream)
    first = np.load(cache_paths[0], allow_pickle=False)
    object_vertices = first["object_vertices_canonical"].astype(np.float32)
    object_faces = first["object_faces"].astype(np.int32)
    points = object_data["obj_pcs"]["mug"].astype(np.float32)
    normals = object_data["obj_pc_normals"]["mug"].astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    point_error, native_nearest = cKDTree(object_vertices).query(points)
    if points.shape != (1024, 3) or point_error.max() > 0.002:
        raise ValueError("Pretrained object points do not match native GRAB mug coordinates")
    centroid = points.mean(axis=0)
    scale = np.linalg.norm(points - centroid, axis=1).max()
    save_arrays(
        output / "object.npz",
        object_vertices_canonical=object_vertices,
        object_faces=object_faces,
        object_points=points,
        object_normals=normals,
        object_points_normalized=(points - centroid) / scale,
        object_centroid=centroid,
        object_scale=np.asarray(scale, dtype=np.float32),
        pretrained_decimated_point_indices=object_data["point_sets"]["mug"].astype(np.int64),
        nearest_native_vertex_indices=native_nearest.astype(np.int64),
    )
    points_t = torch.from_numpy(points)[None]
    sampled_vertices_t = torch.from_numpy(object_vertices[::41])[None]
    models = {side: build_mano_aa(is_rhand=side == "rhand", flat_hand=True) for side in ("lhand", "rhand")}
    standard_templates = {side: model.v_template.clone() for side, model in models.items()}
    subjects = {}
    records = []
    maximum_roundtrip = 0.0
    maximum_object_error = 0.0

    for cache_path in cache_paths:
        clip_id = cache_path.stem
        subject, motion = clip_id.split("__", 1)
        with zipfile.ZipFile(grab / f"grab__{subject}.zip") as archive:
            members = [name for name in archive.namelist() if Path(name).name == f"{motion}.npz"]
            if len(members) != 1:
                raise ValueError(f"Nonunique original source: {clip_id}")
            source_bytes = archive.read(members[0])
            source = dict(np.load(io.BytesIO(source_bytes), allow_pickle=True))
        gender = str(source["gender"].item())
        if source["obj_name"].item() != "mug" or float(source["framerate"]) != 120.0:
            raise ValueError(f"Unexpected original source identity: {clip_id}")
        if subject not in subjects:
            assets = {}
            with zipfile.ZipFile(grab / f"tools__subject_meshes__{gender}.zip") as archive:
                for side in ("lhand", "rhand"):
                    name = f"{gender}/{subject}_{side}.ply"
                    mesh = trimesh.load(io.BytesIO(archive.read(name)), file_type="ply", process=False)
                    assets[f"{side}_v_template"] = np.asarray(mesh.vertices, dtype=np.float32)
                    assets[f"{side}_source_betas"] = zip_array(archive, f"{subject}_{side}_betas.npy").astype(
                        np.float32
                    )
                    assets[f"{side}_model_betas"] = np.zeros(10, dtype=np.float32)
                    assets[f"{side}_faces"] = models[side].faces.astype(np.int32)
                    assets[f"{side}_smplx_vertex_ids"] = hand_ids[side].astype(np.int64)
            save_arrays(output / "subjects" / f"{subject}.npz", **assets)
            subjects[subject] = assets

        cache = np.load(cache_path, allow_pickle=False)
        indices = cache["source_frame_indices"]
        if not np.array_equal(indices, np.arange(0, int(source["n_frames"]), 4)):
            raise ValueError(f"Full source timeline not preserved: {clip_id}")
        if not np.array_equal(cache["object_faces"], object_faces):
            raise ValueError(f"Object topology changed: {clip_id}")
        arrays = {
            "source_frame_indices": indices,
            "source_fps": np.asarray(120.0, dtype=np.float32),
            "playback_fps": np.asarray(30.0, dtype=np.float32),
            "valid_mask_lhand": np.ones(len(indices), dtype=bool),
            "valid_mask_rhand": np.ones(len(indices), dtype=bool),
            "valid_mask_obj": np.ones(len(indices), dtype=bool),
            "object_rotation": cache["object_rotation"].astype(np.float32),
            "object_translation": cache["object_translation"].astype(np.float32),
        }
        # Text2HOI GRAB decodes row vectors as v @ R; cache stores active R.
        row_rotation = torch.from_numpy(arrays["object_rotation"]).transpose(1, 2)
        x_obj = torch.cat((torch.from_numpy(arrays["object_translation"]), rotmat_to_rot6d(row_rotation)), dim=-1)
        arrays["x_obj"] = x_obj.numpy()
        side_reports = {}
        for side in ("lhand", "rhand"):
            params = source[side].item()["params"]
            pose = torch.from_numpy(
                np.concatenate((params["global_orient"][indices], params["fullpose"][indices]), axis=-1)
            )
            trans = torch.from_numpy(params["transl"][indices])
            x_hand = torch.cat((trans, axis_angle_to_rot6d(pose.reshape(-1, 3)).reshape(len(indices), 96)), dim=-1)
            arrays[f"x_{side}"] = x_hand.numpy()
            model = models[side]
            model.v_template.copy_(torch.from_numpy(subjects[subject][f"{side}_v_template"]))
            vertices_parts, joints_parts, near_indices_parts, near_dist_parts = [], [], [], []
            native_error_parts = []
            native_original = cache["original_vertices"][:, cache_hand_ids[side]]
            for start in range(0, len(indices), args.batch_size):
                interval = slice(start, start + args.batch_size)
                current = x_hand[None, interval]
                vertices = get_hand_verts(current, model)[0]
                joints = get_hand_joints_w_tip(current, model)[0]
                direct = (
                    model(
                        betas=torch.zeros(len(vertices), 10),
                        global_orient=pose[interval, :3],
                        hand_pose=pose[interval, 3:],
                    ).vertices
                    + trans[interval, None]
                )
                maximum_roundtrip = max(maximum_roundtrip, float((vertices - direct).abs().max()))
                obj_points = get_transformed_obj_pc(x_obj[None, interval], points_t, "grab")[0]
                distance = torch.cdist(joints, obj_points, compute_mode="donot_use_mm_for_euclid_dist")
                near_dist, near_indices = distance.min(dim=-1)
                vertices_np = vertices.numpy()
                native_error_parts.append(np.linalg.norm(vertices_np - native_original[interval], axis=-1))
                vertices_parts.append(vertices_np)
                joints_parts.append(joints.numpy())
                near_indices_parts.append(near_indices.numpy())
                near_dist_parts.append(near_dist.numpy())
            arrays[f"{side}_vertices"] = np.concatenate(vertices_parts)
            arrays[f"{side}_joints"] = np.concatenate(joints_parts)
            arrays[f"{side}_contact_point_indices"] = np.concatenate(near_indices_parts)
            arrays[f"{side}_contact_distances"] = np.concatenate(near_dist_parts)
            arrays[f"{side}_contact_mask"] = arrays[f"{side}_contact_distances"] < 0.02
            # These are native GRAB annotations mapped through SMPL-X vertex IDs.
            arrays[f"{side}_source_contact"] = source["contact"].item()["body"][indices][:, hand_ids[side]] != 0
            native_errors = np.concatenate(native_error_parts)
            sample = np.unique(np.linspace(0, len(indices) - 1, min(5, len(indices)), dtype=int))
            model.v_template.copy_(standard_templates[side])
            standard = get_hand_verts(x_hand[None, sample], model)[0].numpy()
            standard_errors = np.linalg.norm(standard - native_original[sample], axis=-1)
            side_reports[side] = {
                "personalized_mano_vs_smplx_mean_m": float(native_errors.mean()),
                "personalized_mano_vs_smplx_p95_m": float(np.percentile(native_errors, 95)),
                "personalized_mano_vs_smplx_max_m": float(native_errors.max()),
                "standard_mano_vs_smplx_sampled_mean_m": float(standard_errors.mean()),
                "standard_mano_vs_smplx_sampled_max_m": float(standard_errors.max()),
                "joint_contact_frames": int(arrays[f"{side}_contact_mask"].any(axis=-1).sum()),
            }
        sampled_objects = get_transformed_obj_pc(x_obj[None], sampled_vertices_t, "grab")[0].numpy()
        expected = np.einsum("tij,kj->tki", arrays["object_rotation"], object_vertices[::41])
        expected += arrays["object_translation"][:, None]
        object_error = float(np.abs(sampled_objects - expected).max())
        maximum_object_error = max(maximum_object_error, object_error)
        if maximum_roundtrip > 1e-5 or maximum_object_error > 1e-6:
            raise ValueError("Text2HOI encoding does not preserve source MANO or object geometry")
        path = output / "clips" / f"{clip_id}.npz"
        save_arrays(path, **arrays)
        record = {
            "clip_id": clip_id,
            "subject": subject,
            "gender": gender,
            "action": str(source["motion_intent"].item()),
            "frames": len(indices),
            "original_frames": int(source["n_frames"]),
            "clip_path": str(path.relative_to(output)),
            "subject_path": f"subjects/{subject}.npz",
            "source_archive": str(grab / f"grab__{subject}.zip"),
            "source_member": members[0],
            "source_member_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "reference_cache": str(cache_path),
            "reference_cache_sha256": digest(cache_path),
            "clip_sha256": digest(path),
            "object_roundtrip_max_abs_error_m": object_error,
            "representation_comparison": side_reports,
        }
        records.append(record)
        print(json.dumps({"prepared": clip_id, "frames": len(indices), "comparison": side_reports}), flush=True)

    manifest = {
        "schema_version": 1,
        "purpose": (
            "All-data GRAB mug refinement; output comparisons are training-set refinement, not held-out generalization"
        ),
        "model_representation": (
            "Personalized native MANO hands, flat_hand_mean=True, use_pca=False, zero model betas; "
            "source fullpose already includes mean"
        ),
        "source_betas_policy": (
            "Saved for provenance only; already represented by subject v_template; do not apply again"
        ),
        "smplx_policy": (
            "MANO is not numerically identical to cropped personalized SMPL-X; "
            "per-clip all-frame differences explicitly recorded"
        ),
        "object_rotation_policy": "x_obj encodes source row-vector Rodrigues R; cached active object_rotation=R.T",
        "source_fps": 120,
        "playback_fps": 30,
        "temporal_policy": (
            "Every fourth native frame, including leading/trailing noncontact; "
            "complete 44 clips; no 150-frame truncation"
        ),
        "contact_target_policy": (
            "Nearest pretrained mug point for each of 21 MANO joints, contact mask distance<0.02 m; "
            "geometric proximity, not verified physical contact"
        ),
        "point_cloud_policy": (
            "Exact pretrained obj.pkl mug points in native metric coordinates; "
            "indices belong to upstream decimated mesh, not native full mesh"
        ),
        "object_points_nearest_native_vertex_mean_m": float(point_error.mean()),
        "object_points_nearest_native_vertex_max_m": float(point_error.max()),
        "object_path": "object.npz",
        "object_sha256": digest(output / "object.npz"),
        "upstream_object_pickle": str((repo / "data/grab/obj.pkl").resolve()),
        "upstream_object_pickle_sha256": digest(repo / "data/grab/obj.pkl"),
        "subject_paths": {subject: f"subjects/{subject}.npz" for subject in subjects},
        "subject_sha256": {subject: digest(output / "subjects" / f"{subject}.npz") for subject in subjects},
        "sequences": len(records),
        "frames": sum(record["frames"] for record in records),
        "max_hand_roundtrip_abs_error_m": maximum_roundtrip,
        "max_object_roundtrip_abs_error_m": maximum_object_error,
        "clips": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {manifest['sequences']} sequences / {manifest['frames']} frames at {output}")


if __name__ == "__main__":
    main()
