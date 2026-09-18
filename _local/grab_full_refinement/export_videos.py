# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Apply a trained refiner to every released GRAB recording and export videos.

This is recorded-motion cleanup, not the official text-generation benchmark.
Both columns use the same standard MANO shape used by upstream training.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--clip_limit", type=int, default=0)
    parser.add_argument("--frame_limit", type=int, default=0)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    local = Path(__file__).resolve().parent
    repo = Path("/home/nmt/Projects/Text2HOI")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from lib.datasets.datasets import get_dataset
    from lib.models.mano import build_mano_aa
    from lib.utils.model_utils import build_refiner
    from lib.utils.proc import proc_refiner_input
    from lib.utils.proc_output import get_hand_verts
    from lib.utils.rot import rot6d_to_rotmat

    torch.set_num_threads(4)
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=grab"])
    config = EasyDict(OmegaConf.to_container(cfg, resolve=True))
    dataset = get_dataset("Motiongrab", config.dataset)
    left = build_mano_aa(is_rhand=False, flat_hand=True).cuda()
    right = build_mano_aa(is_rhand=True, flat_hand=True).cuda()
    refiner = build_refiner(config)
    state = torch.load(checkpoint, map_location="cpu")
    refiner.load_state_dict(state["model"])
    refiner.eval()
    checkpoint_hash = digest(checkpoint)
    reports = []
    mesh_archive = zipfile.ZipFile("/home/nmt/Projects/Grab Dataset/tools__object_meshes__contact_meshes.zip")
    for index in range(min(len(dataset), args.clip_limit or len(dataset))):
        name = str(dataset.obj_name[index])
        action = str(dataset.action_name[index]).replace(" ", "_")
        identifier = f"{index:04d}_{name}_{action}"
        movie = output / name / (identifier + ".mp4")
        report_path = movie.with_suffix(".json")
        if report_path.exists() and movie.exists():
            previous = json.loads(report_path.read_text())
            if previous["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Existing video uses another checkpoint: " + str(movie))
            if previous["video_sha256"] != digest(movie):
                raise ValueError("Existing video checksum mismatch: " + str(movie))
            reports.append(previous)
            continue
        count = int(dataset.nframes[index])
        if args.frame_limit:
            count = min(count, args.frame_limit)
        originals = [
            torch.as_tensor(np.asarray(values[index][:count], dtype=np.float32), device="cuda")
            for values in (dataset.x_lhand, dataset.x_rhand, dataset.x_obj)
        ]
        present = [bool(dataset.is_lhand[index]), bool(dataset.is_rhand[index])]
        _, points, normals, _ = dataset.object_model(name)
        pc = torch.as_tensor(points, dtype=torch.float32, device="cuda")[None]
        normal = torch.as_tensor(normals, dtype=torch.float32, device="cuda")[None]
        coverage = torch.zeros(1, 1024, device="cuda")
        for side in ("l", "r"):
            indices = np.asarray(getattr(dataset, side + "cov_idx")[index], dtype=np.int64)
            coverage[0, torch.as_tensor(indices, device="cuda")] = 1
        predicted = [torch.zeros_like(originals[0]), torch.zeros_like(originals[1])]
        denominator = torch.zeros(count, 1, device="cuda")
        starts = sorted(set(list(range(0, max(count - 150, 0) + 1, 75)) + [max(count - 150, 0)]))
        for start in starts:
            length = min(150, count - start)
            chunks = []
            for value in originals:
                chunk = torch.zeros(1, 150, value.shape[-1], device="cuda")
                chunk[0, :length] = value[start : start + length]
                chunks.append(chunk)
            masks = [torch.arange(150, device="cuda")[None] < length for _ in range(3)]
            for side in range(2):
                masks[side] &= present[side]
            il, ir, _ = proc_refiner_input(*chunks, left, right, pc, normal, *masks, coverage, "grab")
            results = refiner(il, ir, valid_mask_lhand=masks[0], valid_mask_rhand=masks[1])
            blend = torch.hann_window(length + 2, periodic=False, device="cuda")[1:-1, None]
            for side in range(2):
                predicted[side][start : start + length] += results[side][0, :length] * blend
            denominator[start : start + length] += blend
        predicted = [value / denominator for value in predicted]
        if index == 0:

            def infer_window():
                inputs_left, inputs_right, _ = proc_refiner_input(
                    *chunks, left, right, pc, normal, *masks, coverage, "grab"
                )
                return refiner(inputs_left, inputs_right, valid_mask_lhand=masks[0], valid_mask_rhand=masks[1])

            for _ in range(5):
                infer_window()
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            for _ in range(30):
                infer_window()
            torch.cuda.synchronize()
            seconds = (time.perf_counter() - inference_start) / 30
            (output / "inference_benchmark.json").write_text(
                json.dumps(
                    dict(
                        seconds_per_window=seconds,
                        valid_frames_per_window=length,
                        padded_frames_per_window=150,
                        valid_frames_per_second=length / seconds,
                        includes="GPU refiner input preparation (MANO joints/proximity) and refiner forward",
                        excludes="diffusion generation, disk loading, final mesh export, video rendering",
                        repetitions=30,
                        warmup=5,
                        checkpoint_sha256=checkpoint_hash,
                    ),
                    indent=2,
                )
                + "\n"
            )
        before, after, faces = [], [], []
        displacement = []
        for side, layer in enumerate((left, right)):
            if not present[side]:
                continue
            vertices = []
            for params in (originals[side], predicted[side]):
                vertices.append(
                    torch.cat(
                        [
                            get_hand_verts(params[start : start + 150][None], layer)[0].cpu()
                            for start in range(0, count, 150)
                        ]
                    ).numpy()
                )
            if not all(np.isfinite(v).all() for v in vertices):
                raise FloatingPointError("Nonfinite exported hand geometry: " + identifier)
            faces.append(np.asarray(layer.faces, dtype=np.int32) + 778 * len(before))
            before.append(vertices[0])
            after.append(vertices[1])
            displacement.append(np.linalg.norm(vertices[1] - vertices[0], axis=-1))
        if not before:
            raise ValueError("No active hands: " + identifier)
        mesh = trimesh.load(
            io.BytesIO(mesh_archive.read("contact_meshes/" + name + ".ply")), file_type="ply", process=False
        )
        geometry = output / "render_input.npz"
        np.savez(
            geometry,
            original_vertices=np.concatenate(before, axis=1),
            soma_vertices=np.concatenate(after, axis=1),
            original_faces=np.concatenate(faces),
            soma_faces=np.concatenate(faces),
            object_vertices_canonical=np.asarray(mesh.vertices, dtype=np.float32),
            object_faces=np.asarray(mesh.faces, dtype=np.int32),
            object_rotation=rot6d_to_rotmat(originals[2][:, 3:9]).transpose(1, 2).cpu().numpy(),
            object_translation=originals[2][:, :3].cpu().numpy(),
        )
        movie.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "/home/nmt/miniconda3/envs/graspxl/bin/python",
                str(local / "render_video.py"),
                "--geometry",
                str(geometry),
                "--output",
                str(movie),
                "--label",
                identifier,
            ],
            check=True,
        )
        report = dict(
            clip_index=index,
            clip_id=identifier,
            object=name,
            frames=count,
            fps=30,
            checkpoint_sha256=checkpoint_hash,
            checkpoint_epoch=state.get("epoch"),
            video_sha256=digest(movie),
            video=str(movie.relative_to(output)),
            mean_displacement_mm=float(np.concatenate(displacement, axis=1).mean() * 1000),
            mode="recorded-motion cleanup; standard zero-beta MANO; unchanged object trajectory",
            temporal_scope="entire retained contact interval from upstream preprocessing; raw lead-in/tail excluded",
            evaluation="in-sample visualization, not text-generation or held-out accuracy",
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        (output / "manifest.json").write_text(json.dumps(dict(completed=False, clips=reports), indent=2) + "\n")
        print(json.dumps(report), flush=True)
    (output / "manifest.json").write_text(json.dumps(dict(completed=True, clips=reports), indent=2) + "\n")
    (output / "render_input.npz").unlink(missing_ok=True)
    mesh_archive.close()


if __name__ == "__main__":
    main()
