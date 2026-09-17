# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit source placement and sampled hand penetration against the GRAB mug solid."""

import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial import cKDTree
from trimesh.ray.ray_util import contains_points

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parents[2]
sys.path.insert(0, str(PROJECTS / "SOMA-X"))
from soma._smpl_family_loader import ensure_chumpy_compat

CLIPS = ("s1__mug_drink_1", "s1__mug_drink_3", "s1__mug_drink_4", "s2__mug_drink_2",
         "s1__mug_lift", "s1__mug_pass_1")


def archive_payload(path: Path, filename: str) -> bytes:
    """Read exactly one source archive member matching its filename."""
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if Path(name).name == filename]
        if len(candidates) != 1:
            raise ValueError(candidates)
        return archive.read(candidates[0])


def contact_metrics(mesh: trimesh.Trimesh, vertices: np.ndarray) -> dict:
    """Measure only vertices classified inside the closed mug material [m]."""
    inside = mesh.contains(vertices)
    points = vertices[inside]
    confirmed = contains_points(mesh.ray, points, check_direction=np.array([0.38, 0.72, 0.57]))
    inconsistent = int((~confirmed).sum())
    points = points[confirmed]
    if len(points):
        _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    else:
        distances = np.zeros(0)
    return {
        "total_hand_vertices": len(vertices),
        "inside_solid_vertices_confirmed_two_ray_directions": len(points),
        "ray_direction_disagreements_excluded": inconsistent,
        "vertices_deeper_than_2_mm": int((distances > 0.002).sum()),
        "vertices_deeper_than_5_mm": int((distances > 0.005).sum()),
        "fraction_deeper_than_2_mm": float((distances > 0.002).sum() / len(vertices)),
        "fraction_deeper_than_5_mm": float((distances > 0.005).sum() / len(vertices)),
        "maximum_penetration_depth_mm": float(distances.max(initial=0) * 1000),
        "mean_depth_of_inside_vertices_mm": float(distances.mean() * 1000) if len(distances) else 0,
    }


def main() -> None:
    ensure_chumpy_compat()
    torch.set_num_threads(2)
    started = time.monotonic()
    source_mesh_bytes = archive_payload(PROJECTS / "Grab Dataset/tools__object_meshes__contact_meshes.zip", "mug.ply")
    packaged_mesh_bytes = (PROJECTS / "GRAB_SOMA_51Objects/objects/mug/mesh.ply").read_bytes()
    if source_mesh_bytes != packaged_mesh_bytes:
        raise ValueError("Packaged mug differs from original GRAB contact mesh")
    mesh = trimesh.load(io.BytesIO(source_mesh_bytes), file_type="ply", process=False, force="mesh")
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Reliable solid penetration classification requires a closed oriented mesh")
    with zipfile.ZipFile(PROJECTS / "Grab Dataset/tools__smplx_correspondence.zip") as archive:
        original_ids = np.unique(np.concatenate([
            np.load(io.BytesIO(archive.read(next(name for name in archive.namelist()
                                                if Path(name).name == f"{side}_smplx_ids.npy"))), allow_pickle=False)
            for side in ("lhand", "rhand")
        ]))
    probes = np.asarray([[-0.015, 0, z] for z in (-0.060, -0.050, -0.040, 0.0, 0.040, 0.060)])
    probe_inside = mesh.contains(probes)
    if probe_inside[3] or probe_inside[4]:
        raise ValueError("Expected hollow cup cavity was classified as solid")
    result = {
        "mesh": {"watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
                 "connected_bodies": int(mesh.body_count), "vertices": len(mesh.vertices), "faces": len(mesh.faces),
                 "solid_volume_m3": float(mesh.volume), "bounds_m": mesh.bounds.tolist(),
                 "source_archive_mesh_bytes_identical": True, "source_mesh_sha256": hashlib.sha256(source_mesh_bytes).hexdigest()},
        "hollow_cavity_check": {"canonical_probe_positions_m": probes.tolist(),
                                "inside_solid": probe_inside.tolist(),
                                "interpretation": "Cup empty cavity at z=0 and z=0.04 is outside solid; lower mug material is inside"},
        "method": "Two-direction ray-parity containment in watertight mug; exact nearest-triangle distance for confirmed interior hand vertices",
        "source_reconstruction_reference": "https://github.com/otaheri/GRAB/blob/master/examples/visualize_grab.py",
        "object_transform_reference": "https://github.com/otaheri/GRAB/blob/master/tools/objectmodel.py",
        "camera_independent": True,
        "physics_or_collision_response_applied": False,
        "samples_per_clip": "25%, 50%, 75% of exported clip frames, mapped back to original 120 FPS indices",
        "limitations": [
            "Sampled vertices and frames do not provide complete mesh-mesh collision volume or prove absence of other intersections.",
            "Hand topologies have different vertex densities; raw counts are not directly comparable model-performance scores.",
            "Distances measure penetration of the mug material only, not fingers in the empty cup cavity or handle opening.",
            "Original source fitting can already contain residual contact errors; SOMA fitting can add or reduce local errors.",
            "Source annotated-object distance uses nearest full-body vertex, an approximation to surface distance, and includes mouth contact during drinking.",
        ],
        "clips": [],
    }
    for clip_id in CLIPS:
        subject, stem = clip_id.split("__")
        with np.load(io.BytesIO(archive_payload(PROJECTS / "Grab Dataset" / f"grab__{subject}.zip", stem + ".npz")),
                     allow_pickle=True) as archive:
            source = dict(archive)
        gender = str(source["gender"].item())
        source_template = archive_payload(PROJECTS / "Grab Dataset" / f"tools__subject_meshes__{gender}.zip", subject + ".ply")
        package_template = (PROJECTS / "GRAB_SOMA_51Objects/subjects" / subject / "smplx_template.ply").read_bytes()
        if source_template != package_template:
            raise ValueError(f"Subject template differs: {subject}")
        template = trimesh.load(io.BytesIO(source_template), file_type="ply", process=False, maintain_order=True)
        with np.load(ROOT / "cache/GRAB" / f"{clip_id}.npz", allow_pickle=False) as archive:
            cache = dict(archive)
        with np.load(ROOT / "revisions/grab_object_rotation_v1/cache/GRAB" / f"{clip_id}.npz", allow_pickle=False) as archive:
            old_rotation = archive["object_rotation"]
        np.testing.assert_array_equal(cache["object_vertices_canonical"], mesh.vertices)
        np.testing.assert_array_equal(cache["object_faces"], mesh.faces)
        frames = np.rint((len(cache["source_frame_indices"]) - 1) * np.asarray([0.25, 0.50, 0.75])).astype(int)
        source_frames = cache["source_frame_indices"][frames]
        # Reconstruct directly from original source template and every source parameter,
        # following the official GRAB example independently of our viewer.
        model = smplx.create(str(PROJECTS / "SOMA-X/assets/SMPLX" / f"SMPLX_{gender.upper()}.npz"),
                             model_type="smplx", gender=gender, num_pca_comps=int(source["n_comps"]),
                             v_template=np.asarray(template.vertices), batch_size=len(frames))
        params = source["body"].item()["params"]
        with torch.no_grad():
            body = model(**{key: torch.as_tensor(value[source_frames], dtype=torch.float32)
                            for key, value in params.items()}).vertices.numpy()
        source_error = float(np.abs(body[:, original_ids] - cache["original_vertices"][frames]).max())
        if source_error > 2e-6:
            raise ValueError(f"Original source reconstruction mismatch: {clip_id}: {source_error}")
        item = {"clip_id": clip_id, "source_subject_template_bytes_identical": True,
                "source_pca_components": int(source["n_comps"]), "source_all_body_parameters_used": list(params),
                "official_original_reconstruction_max_abs_error_m": source_error, "frames": []}
        for sample, frame in enumerate(frames):
            rotation, translation = cache["object_rotation"][frame], cache["object_translation"][frame]
            metric = {"export_frame": int(frame), "source_frame": int(source_frames[sample])}
            for side in ("original", "soma"):
                local = (cache[f"{side}_vertices"][frame] - translation) @ rotation
                metric[side] = contact_metrics(mesh, local)
            labeled = source["contact"].item()["object"][source_frames[sample]] > 0
            body_tree = cKDTree(body[sample])
            comparison = {"annotated_object_vertices": int(labeled.sum())}
            for name, matrix in (("corrected", rotation), ("previous_wrong_rotation", old_rotation[frame])):
                points = np.asarray(mesh.vertices)[labeled] @ matrix.T + translation
                distances = body_tree.query(points)[0]
                comparison[name] = {"median_nearest_body_vertex_distance_mm": float(np.median(distances) * 1000) if len(distances) else None,
                                    "fraction_within_5_mm": float((distances < 0.005).mean()) if len(distances) else None}
            metric["annotated_contact_alignment"] = comparison
            item["frames"].append(metric)
            print(json.dumps({"clip": clip_id, "source_frame": metric["source_frame"],
                              "original": metric["original"], "soma": metric["soma"]}), flush=True)
        result["clips"].append(item)
        (ROOT / "qa/contact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    flat = [frame for clip in result["clips"] for frame in clip["frames"]]
    result["summary"] = {"clips": len(result["clips"]), "sampled_frames": len(flat),
                         "elapsed_seconds": time.monotonic() - started}
    for side in ("original", "soma"):
        result["summary"][side] = {
            "sampled_frames_with_vertices_deeper_than_2_mm": sum(frame[side]["vertices_deeper_than_2_mm"] > 0 for frame in flat),
            "sampled_frames_with_vertices_deeper_than_5_mm": sum(frame[side]["vertices_deeper_than_5_mm"] > 0 for frame in flat),
            "maximum_penetration_depth_mm": max(frame[side]["maximum_penetration_depth_mm"] for frame in flat),
        }
    result["completed"] = True
    (ROOT / "qa/contact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
