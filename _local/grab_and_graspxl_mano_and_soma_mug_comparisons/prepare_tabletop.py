# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare three supplemental tabletop mug clips without modifying source data.

The official tabletop convention uses flat MANO means and a fixed wrist bias.
An explicit, numerically verified adapter allows reuse of the existing SOMA
fitter. These are new local SOMA fits, separate from the released matched pairs.
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
from scipy.spatial.transform import Rotation

DIRECTORY = Path(__file__).resolve().parent
PROJECTS = DIRECTORY.parents[2]
SOMA_REPO = PROJECTS / "SOMA-X"
sys.path.insert(0, str(SOMA_REPO))
from soma._smpl_family_loader import ensure_chumpy_compat
from tools.hand.convert_graspxl_to_soma import Converter, validate_sequence

UID = "12652c6e0aaf44dda32aab816f433baf"
SOURCE_ROOT = PROJECTS / "GraspXL_SOMA_51Objects_GRABMatched/extra_assets/source_tabletop/large" / UID
MESH_PATH = PROJECTS / "GraspXL_MANO_51Objects_GRABMatched/objects" / f"29_mug_{UID}/mesh.obj"
ASSETS = SOMA_REPO / "assets"
BIAS = np.asarray([0.09566994, 0.00638343, 0.0061863], dtype=np.float32)
OFFICIAL_URL = "https://github.com/zdchan/GraspXL_visualization/blob/main/scripts/visualizer_sharpa_table_top.py"
SUPPLEMENTAL = DIRECTORY / "supplemental"


def digest(path: Path) -> str:
    """Hash a local provenance file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data: dict) -> None:
    """Write metadata atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def save_npz(path: Path, arrays: dict) -> None:
    """Write numeric or string arrays atomically without pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


@torch.no_grad()
def prepare_cpu() -> None:
    """Validate every source frame against the official tabletop formula [m]."""
    flat = smplx.MANO(str(ASSETS / "MANO/MANO_RIGHT.pkl"), is_rhand=True, use_pca=False,
                     flat_hand_mean=True, create_transl=False)
    nonflat = smplx.MANO(str(ASSETS / "MANO/MANO_RIGHT.pkl"), is_rhand=True, use_pca=False,
                        flat_hand_mean=False, create_transl=False)
    for index in range(3):
        clip_id = f"tabletop_large__mano_{index}"
        source_path = SOURCE_ROOT / f"mano_{index}.npy"
        source = np.load(source_path, allow_pickle=True).item()
        uid, count = validate_sequence(source)
        if uid != UID or count != 100:
            raise ValueError(f"Unexpected supplemental source: {source_path}")
        hand = source["right_hand"]
        pose = torch.from_numpy(hand["pose"])
        rot = torch.from_numpy(hand["rot"])
        trans = torch.from_numpy(hand["trans"])
        betas = torch.zeros(count, 10)
        official = flat(global_orient=rot, hand_pose=pose, betas=betas).vertices
        official = official - torch.from_numpy(BIAS)[None, None] + trans[:, None]
        adapted_pose = pose - nonflat.hand_mean[None]
        adapted_trans = trans - torch.from_numpy(BIAS)[None]
        adapted = nonflat(global_orient=rot, hand_pose=adapted_pose, betas=betas).vertices
        adapted = adapted + adapted_trans[:, None]
        frame_errors = (official - adapted).abs().amax(dim=(1, 2))
        if float(frame_errors.max()) > 1e-6:
            raise ValueError(f"Tabletop convention adapter differs: {clip_id}")
        reference_path = SUPPLEMENTAL / "reference" / f"{clip_id}.npz"
        save_npz(reference_path, {
            "original_vertices": official.numpy(),
            "original_faces": np.asarray(flat.faces, dtype=np.int32),
            "adapted_pose": adapted_pose.numpy(),
            "adapted_trans": adapted_trans.numpy(),
            "source_rot": hand["rot"],
            "adapter_frame_max_abs_error_m": frame_errors.numpy(),
        })
        report = {
            "clip_id": clip_id,
            "source_path": str(source_path),
            "source_sha256": digest(source_path),
            "source_frames": count,
            "source_fps": None,
            "source_subset": "tabletop_large",
            "supplemental": True,
            "soma_conversion": "new_local_fit",
            "source_mano_flat_hand_mean": True,
            "source_mano_betas": "zero",
            "source_wrist_bias_m": BIAS.astype(float).tolist(),
            "source_reconstruction": "SMPLX MANO(flat_hand_mean=True).vertices - wrist_bias + source_translation",
            "adapter": (
                "pose_nonflat = source_pose - MANO.hand_mean; "
                "translation_nonflat = source_translation - wrist_bias"
            ),
            "adapter_validated_frames": count,
            "adapter_max_abs_error_m": float(frame_errors.max()),
            "official_reconstruction_url": OFFICIAL_URL,
            "official_reconstruction_function": "construct_mano_meshes_per_object",
            "source_mesh_path": str(MESH_PATH),
            "source_mesh_sha256": digest(MESH_PATH),
            "mesh_rescaling": False,
            "reference_path": str(reference_path),
            "reference_sha256": digest(reference_path),
            "preparation_script_sha256": digest(Path(__file__)),
        }
        save_json(SUPPLEMENTAL / "reference" / f"{clip_id}.json", report)
        print(json.dumps({"stage": "cpu", "clip_id": clip_id, "frames": count,
                          "adapter_max_abs_error_m": report["adapter_max_abs_error_m"]}), flush=True)


def fit_and_export(device: str, batch_size: int) -> None:
    """Fit local SOMA poses and validate all geometry against the source [m]."""
    converter = Converter(argparse.Namespace(data_root=ASSETS, device=device, batch_size=batch_size,
                                             bcd_iters=1, lie_iters=3, storage="compact", fps=None))
    mesh = trimesh.load(MESH_PATH, process=False, force="mesh")
    reports = []
    for index in range(3):
        clip_id = f"tabletop_large__mano_{index}"
        source_path = SOURCE_ROOT / f"mano_{index}.npy"
        provenance = json.loads((SUPPLEMENTAL / "reference" / f"{clip_id}.json").read_text())
        if digest(source_path) != provenance["source_sha256"] or digest(MESH_PATH) != provenance["source_mesh_sha256"]:
            raise ValueError("Source or mesh changed since CPU verification")
        reference_path = Path(provenance["reference_path"])
        if digest(reference_path) != provenance["reference_sha256"]:
            raise ValueError("CPU reference changed")
        with np.load(reference_path, allow_pickle=False) as archive:
            reference = dict(archive)
        source = np.load(source_path, allow_pickle=True).item()
        adapted = {key: dict(value) for key, value in source.items()}
        adapted["right_hand"]["pose"] = reference["adapted_pose"]
        adapted["right_hand"]["trans"] = reference["adapted_trans"]
        destination = SUPPLEMENTAL / "soma" / f"{clip_id}.npz"
        conversion = converter.convert(adapted, destination, f"mano_tabletop/large/{UID}/mano_{index}.npy",
                                       Path("large") / UID / f"mano_{index}.npy")
        with np.load(destination, allow_pickle=False) as archive:
            motion = dict(archive)
        for field, source_field in (("object_trans", "trans"), ("object_rot", "rot")):
            if not np.array_equal(motion[field], source[UID][source_field]):
                raise ValueError(f"Object transform changed: {clip_id}/{field}")
        motion.update(source_subset=np.asarray("tabletop_large"),
                      source_mano_flat_hand_mean=np.asarray(True),
                      source_mano_wrist_bias_m=BIAS,
                      conversion_provenance=np.asarray("New local SOMA fit for supplemental comparison video"),
                      source_sha256=np.asarray(provenance["source_sha256"]))
        save_npz(destination, motion)
        original_parts, soma_parts, errors = [], [], []
        with torch.no_grad():
            for begin in range(0, 100, batch_size):
                selected = slice(begin, min(begin + batch_size, 100))
                def tensor(array):
                    return torch.as_tensor(array[selected], dtype=torch.float32, device=device)
                mano = converter.mano(global_orient=tensor(reference["source_rot"]),
                                      hand_pose=tensor(reference["adapted_pose"]),
                                      betas=torch.zeros(len(reference["source_rot"][selected]), 10, device=device))
                translation = tensor(reference["adapted_trans"])
                original = mano.vertices + translation[:, None]
                fitted = converter.hand.pose(tensor(motion["poses"]), pose2rot=True,
                                             absolute_pose=bool(motion["absolute_pose"]),
                                             global_translation=tensor(motion["transl"]))["vertices"]
                wrist = mano.joints[:, :1]
                target = converter.hand.identity_model._to_soma_interp(mano.vertices - wrist)
                target = target + wrist + translation[:, None]
                original_parts.append(original.cpu().numpy())
                soma_parts.append(fitted.cpu().numpy())
                errors.append((fitted - target).norm(dim=-1).cpu().numpy())
        gpu_original = np.concatenate(original_parts)
        soma = np.concatenate(soma_parts)
        vertex_errors = np.concatenate(errors)
        original_error = float(np.abs(gpu_original - reference["original_vertices"]).max())
        replay_error = float(np.abs(vertex_errors.mean(axis=1) - motion["fit_mean_vertex_error_m"]).max())
        if original_error > 1e-6 or replay_error > 5e-5:
            raise ValueError(f"Tabletop validation failed: original={original_error}, SOMA fit replay={replay_error}")
        arrays = {
            "original_vertices": reference["original_vertices"],
            "soma_vertices": soma,
            "object_vertices_canonical": np.asarray(mesh.vertices),
            "object_rotation": Rotation.from_rotvec(source[UID]["rot"]).as_matrix(),
            "object_translation": source[UID]["trans"],
            "original_faces": reference["original_faces"],
            "soma_faces": converter.hand.faces.cpu().numpy().astype(np.int32),
            "object_faces": np.asarray(mesh.faces, dtype=np.int32),
            "source_frame_indices": np.arange(100, dtype=np.int64),
            "source_fps": np.asarray(np.nan),
            "playback_fps": np.asarray(30.0),
            "original_frame_count": np.asarray(100, dtype=np.int64),
        }
        if any(not np.isfinite(value).all() for key, value in arrays.items() if key != "source_fps"):
            raise ValueError(f"Nonfinite geometry: {clip_id}")
        cache_path = DIRECTORY / "cache/GraspXL" / f"{clip_id}.npz"
        save_npz(cache_path, arrays)
        report = {
            **provenance,
            "validated": True,
            "dataset": "GraspXL",
            "subset": "tabletop_large",
            "original_label": "Original MANO (tabletop flat mean and wrist bias)",
            "soma_label": "SOMA (new local tabletop fit)",
            "output_path": str(cache_path),
            "output_sha256": digest(cache_path),
            "soma_path": str(destination),
            "soma_sha256": digest(destination),
            "output_frames": 100,
            "playback_fps": 30,
            "source_fps_known": False,
            "duration_seconds": 100 / 30,
            "units": "meters",
            "original_vertex_count": arrays["original_vertices"].shape[1],
            "soma_vertex_count": soma.shape[1],
            "object_vertex_count": len(mesh.vertices),
            "all_frame_gpu_original_vs_official_cpu_max_abs_error_m": original_error,
            "all_frame_soma_replay_vs_saved_fit_error_max_abs_error_m": replay_error,
            "soma_fit_mean_vertex_error_m": float(vertex_errors.mean()),
            "soma_fit_max_frame_mean_vertex_error_m": float(vertex_errors.mean(axis=1).max()),
            "soma_fit_max_vertex_error_m": float(vertex_errors.max()),
            "object_transforms_exactly_preserved": True,
            "object_transform_convention": "vertices_world = vertices_canonical @ rotation.T + translation",
            "conversion": conversion,
            "converter_sha256": digest(SOMA_REPO / "tools/hand/convert_graspxl_to_soma.py"),
            "conversion_settings": {"batch_size": batch_size, "bcd_iters": 1, "lie_iters": 3, "storage": "compact"},
        }
        save_json(cache_path.with_suffix(".json"), report)
        save_json(SUPPLEMENTAL / "reports" / f"{clip_id}.json", report)
        reports.append(report)
        print(json.dumps({"stage": "fit_complete", "clip_id": clip_id, "source_frames": 100,
                          "original_error_m": original_error,
                          "fit_mean_vertex_error_m": report["soma_fit_mean_vertex_error_m"]}), flush=True)
    save_json(SUPPLEMENTAL / "conversion_summary.json", {
        "validated": True,
        "clips": reports,
        "total_clips": 3,
        "total_frames": 300,
        "scope": (
            "Three supplemental raw tabletop motions for the same selected large mug; "
            "newly fitted local SOMA poses"
        ),
        "original_datasets_modified": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "fit", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    ensure_chumpy_compat()
    torch.set_num_threads(4)
    if args.stage in ("cpu", "all"):
        prepare_cpu()
    if args.stage in ("fit", "all"):
        fit_and_export(args.device, args.batch_size)


if __name__ == "__main__":
    main()
