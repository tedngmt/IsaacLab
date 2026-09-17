# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cache paired original/SOMA mug geometry using the local viewer conventions.

Run with the soma-x environment. GRAB uses both cropped SMPL-X hands; GraspXL
uses its original right MANO hand. Mesh coordinates and object scale stay in
meters. GRAB is sampled at 30 FPS from its documented 120 FPS, while every
GraspXL frame is retained with an explicitly assumed playback rate of 30 FPS.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

LOCAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCAL))
import mano_soma_viewer as viewer

SCHEMA_VERSION = 1
REQUIRED_KEYS = {
    "original_vertices",
    "soma_vertices",
    "object_vertices_canonical",
    "object_rotation",
    "object_translation",
    "original_faces",
    "soma_faces",
    "object_faces",
    "source_frame_indices",
    "source_fps",
    "playback_fps",
    "original_frame_count",
}


def file_digest(path: Path) -> str:
    """Return a SHA-256 digest without reading a complete large file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def clip_entries(dataset: str) -> list[dict]:
    """Select strict mug entries, excluding the separately named coffeemug."""
    if dataset == "GRAB":
        manifest = json.loads((viewer.GRAB_ROOT / "manifest.json").read_text())
        return [
            {
                "clip_id": f"{item['subject']}__{Path(item['path']).stem}",
                "entry": item["path"],
                "source_frames": item["frames"],
                "subject": item["subject"],
            }
            for item in sorted(manifest["motions"], key=lambda item: item["path"])
            if item["object"] == "mug"
        ]
    pairs = json.loads((viewer.MANO_ROOT / "metadata/pairs.json").read_text())
    return [
        {
            "clip_id": f"{item['source_archive']}__{Path(item['mano_path']).stem}",
            "entry": item,
            "source_frames": item["frames"],
            "subject": None,
        }
        for item in sorted(pairs, key=lambda item: item["mano_path"])
        if item["grab_object"] == "mug"
    ]


def input_signature(dataset: str, item: dict) -> dict:
    """Record source identity and exporter version for safe cache resumption."""
    if dataset == "GRAB":
        paths = [
            viewer.GRAB_ROOT / item["entry"],
            viewer.ROOT / "Grab Dataset" / f"grab__{item['subject']}.zip",
            viewer.GRAB_ROOT / "manifest.json",
        ]
    else:
        paths = [
            viewer.MANO_ROOT / item["entry"]["mano_path"],
            viewer.SOMA_ROOT / item["entry"]["soma_path"],
            viewer.MANO_ROOT / item["entry"]["mesh_path"],
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "clip_id": item["clip_id"],
        "source_frames": item["source_frames"],
        "playback_fps": 30,
        "exporter_sha256": file_digest(Path(__file__)),
        "viewer_sha256": file_digest(Path(viewer.__file__)),
        "sources": [
            {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ],
    }


class SequenceCache:
    """Keep one personalized subject resident and reuse its prepared identity."""

    def __init__(self, device: str):
        self.device = device
        self.key = None
        self.sequence = None

    def load(self, dataset: str, item: dict) -> viewer.Sequence:
        """Load clip motion while reusing the current subject's mesh models."""
        key = (dataset, item["subject"])
        if self.key != key:
            self.sequence = None
            gc.collect()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            self.sequence = viewer.Sequence(dataset, item["entry"], self.device)
            self.key = key
            return self.sequence
        sequence = self.sequence
        if dataset == "GRAB":
            with np.load(viewer.GRAB_ROOT / item["entry"], allow_pickle=False) as archive:
                motion = dict(archive)
            for name in ("subject_template_path", "gender"):
                if str(sequence.motion[name].item()) != str(motion[name].item()):
                    raise ValueError(f"Personalized identity changed within subject: {item['clip_id']}")
            with zipfile.ZipFile(viewer.ROOT / "Grab Dataset" / f"grab__{item['subject']}.zip") as archive:
                members = [name for name in archive.namelist() if Path(name).name == Path(item["entry"]).name]
                if len(members) != 1:
                    raise ValueError(f"Expected exactly one source clip: {members}")
                with np.load(io.BytesIO(archive.read(members[0])), allow_pickle=True) as source:
                    sequence.source = dict(source)
            mesh_path = viewer.GRAB_ROOT / str(motion["package_mesh_path"].item())
        else:
            with np.load(viewer.SOMA_ROOT / item["entry"]["soma_path"], allow_pickle=False) as archive:
                motion = dict(archive)
            sequence.source = np.load(viewer.MANO_ROOT / item["entry"]["mano_path"], allow_pickle=True).item()
            mesh_path = viewer.MANO_ROOT / item["entry"]["mesh_path"]
        sequence.motion = motion
        sequence.count = len(motion["poses"])
        sequence.mesh = trimesh.load(mesh_path, process=False, force="mesh")
        return sequence


@torch.no_grad()
def reconstruct(sequence: viewer.Sequence, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct original and converted hand vertices in shared coordinates [m]."""
    motion = sequence.motion
    betas = torch.zeros(len(indices), 10, device=sequence.device)
    if sequence.kind == "GraspXL":
        hand = sequence.source["right_hand"]
        original = sequence.original(
            global_orient=sequence.tensor(hand["rot"][indices]),
            hand_pose=sequence.tensor(hand["pose"][indices]),
            betas=betas,
        ).vertices
        original = original + sequence.tensor(hand["trans"][indices])[:, None]
        soma = sequence.layer.pose(
            sequence.tensor(motion["poses"][indices]),
            pose2rot=motion["poses"].ndim == 3,
            absolute_pose=bool(motion["absolute_pose"]),
            global_translation=sequence.tensor(motion["transl"][indices]),
        )["vertices"]
    else:
        params = sequence.source["body"].item()["params"]
        keys = (
            "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
            "jaw_pose", "leye_pose", "reye_pose", "expression", "transl",
        )
        original = sequence.original(
            **{name: sequence.tensor(params[name][indices]) for name in keys}, betas=betas
        ).vertices
        soma = viewer.replay(sequence.layer, motion, indices)["vertices"]
        original_ids, soma_ids = sequence.hand_vertex_ids
        original = original[:, original_ids]
        soma = soma[:, soma_ids]
    return original.cpu().numpy(), soma.cpu().numpy()


def verify_cache(path: Path, report_path: Path, signature: dict) -> bool:
    """Resume only intact geometry with identical inputs and completed validation."""
    if not path.exists() or not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
        if report["input_signature"] != signature or not report["validated"]:
            return False
        if file_digest(path) != report["output_sha256"]:
            return False
        with np.load(path, allow_pickle=False) as archive:
            if not REQUIRED_KEYS.issubset(archive.files):
                return False
            for name in REQUIRED_KEYS - {"source_fps"}:
                if not np.isfinite(archive[name]).all():
                    return False
            if len(archive["original_vertices"]) != report["output_frames"]:
                return False
        return True
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def export_clip(cache: SequenceCache, dataset: str, item: dict, output_root: Path, batch_size: int) -> dict:
    """Save a full clip and validate three frames against the existing viewer."""
    output_path = output_root / dataset / f"{item['clip_id']}.npz"
    report_path = output_path.with_suffix(".json")
    signature = input_signature(dataset, item)
    if verify_cache(output_path, report_path, signature):
        print(json.dumps({"status": "cached", "dataset": dataset, "clip_id": item["clip_id"]}), flush=True)
        return json.loads(report_path.read_text())
    start = time.monotonic()
    sequence = cache.load(dataset, item)
    if sequence.count != item["source_frames"]:
        raise ValueError(f"Source frame count differs: {item['clip_id']}")
    indices = np.arange(0, sequence.count, 4 if dataset == "GRAB" else 1, dtype=np.int64)
    original_parts, soma_parts = [], []
    for begin in range(0, len(indices), batch_size):
        original, soma = reconstruct(sequence, indices[begin : begin + batch_size])
        original_parts.append(original)
        soma_parts.append(soma)
    original = np.concatenate(original_parts)
    soma = np.concatenate(soma_parts)
    faces = sequence.hand_faces if dataset == "GRAB" else (sequence.original_faces, sequence.faces)
    arrays = {
        "original_vertices": original,
        "soma_vertices": soma,
        "object_vertices_canonical": np.asarray(sequence.mesh.vertices),
        "object_rotation": Rotation.from_rotvec(sequence.motion["object_rot"][indices].reshape(-1, 3)).as_matrix(),
        "object_translation": sequence.motion["object_trans"][indices].reshape(-1, 3),
        "original_faces": np.asarray(faces[0], dtype=np.int32),
        "soma_faces": np.asarray(faces[1], dtype=np.int32),
        "object_faces": np.asarray(sequence.mesh.faces, dtype=np.int32),
        "source_frame_indices": indices,
        "source_fps": np.asarray(120.0 if dataset == "GRAB" else np.nan),
        "playback_fps": np.asarray(30.0),
        "original_frame_count": np.asarray(sequence.count, dtype=np.int64),
    }
    for name, array in arrays.items():
        if name != "source_fps" and not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {name}: {item['clip_id']}")
    for mesh_name in ("original", "soma", "object"):
        vertices = arrays[f"{mesh_name}_vertices" if mesh_name != "object" else "object_vertices_canonical"]
        triangle = arrays[f"{mesh_name}_faces"]
        vertex_count = vertices.shape[-2]
        if triangle.min() < 0 or triangle.max() >= vertex_count:
            raise ValueError(f"Invalid {mesh_name} faces")
    errors = []
    for output_index in sorted({0, len(indices) // 2, len(indices) - 1}):
        source_index = int(indices[output_index])
        reference, _ = sequence.display_geometry(sequence.frame(source_index), hands_only=True)
        object_vertices = arrays["object_vertices_canonical"] @ arrays["object_rotation"][output_index].T
        object_vertices += arrays["object_translation"][output_index]
        error = {
            "source_frame": source_index,
            "original_max_abs_error_m": float(np.abs(original[output_index] - reference[0]).max()),
            "soma_max_abs_error_m": float(np.abs(soma[output_index] - reference[1]).max()),
            "object_max_abs_error_m": float(np.abs(object_vertices - reference[2]).max()),
        }
        if max(error[key] for key in error if key.endswith("_m")) > 5e-5:
            raise ValueError(f"Batched reconstruction differs from viewer: {error}")
        errors.append(error)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, output_path)
    report = {
        "validated": True,
        "input_signature": signature,
        "dataset": dataset,
        "clip_id": item["clip_id"],
        "output_path": str(output_path),
        "output_sha256": file_digest(output_path),
        "original_label": "Original GRAB (SMPL-X hands)" if dataset == "GRAB" else "Original MANO (right hand)",
        "source_frames": sequence.count,
        "source_fps": 120 if dataset == "GRAB" else None,
        "source_fps_known": dataset == "GRAB",
        "playback_fps": 30,
        "output_frames": len(indices),
        "duration_seconds": len(indices) / 30,
        "original_vertex_count": original.shape[1],
        "soma_vertex_count": soma.shape[1],
        "object_vertex_count": len(arrays["object_vertices_canonical"]),
        "object_transform_convention": "vertices_world = vertices_canonical @ rotation.T + translation",
        "units": "meters",
        "viewer_validation": errors,
        "elapsed_seconds": time.monotonic() - start,
    }
    temporary_report = report_path.with_suffix(".tmp.json")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary_report, report_path)
    print(json.dumps({"status": "complete", **{key: report[key] for key in (
        "dataset", "clip_id", "output_frames", "elapsed_seconds", "output_path"
    )}}), flush=True)
    return report


def main() -> None:
    """Export selected strict-mug clips to resumable geometry archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("GRAB", "GraspXL", "all"), default="all")
    parser.add_argument("--clip_id", action="append", help="Repeat to select multiple clip IDs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", type=Path, default=Path(__file__).with_name("cache"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    torch.set_num_threads(4)
    viewer.ensure_chumpy_compat()
    datasets = ("GRAB", "GraspXL") if args.dataset == "all" else (args.dataset,)
    selected = [(dataset, item) for dataset in datasets for item in clip_entries(dataset)]
    if args.clip_id:
        requested = set(args.clip_id)
        selected = [(dataset, item) for dataset, item in selected if item["clip_id"] in requested]
        missing = requested - {item["clip_id"] for _, item in selected}
        if missing:
            parser.error(f"Unknown selected clip IDs: {sorted(missing)}")
    cache = SequenceCache(args.device)
    for dataset, item in selected:
        export_clip(cache, dataset, item, args.output_root.resolve(), args.batch_size)
    print(json.dumps({"status": "finished", "clips": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
