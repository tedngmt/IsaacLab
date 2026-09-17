# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt the released Text2HOI refiner to all 18 selected GraspXL mug motions.

This is an explicitly in-sample motion-cleanup experiment, not a reproduction of
the text-conditioned generation benchmark. Original trajectories are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir


def write_json(path: Path, value: dict) -> None:
    """Write an experiment report atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class MugDistance:
    """Sample a native mug's signed distance field [m], negative inside material."""

    def __init__(self, path: Path):
        with np.load(path) as archive:
            self.grid = torch.as_tensor(archive["sdf_grid"], device="cuda")[None, None]
            self.lower = torch.as_tensor(archive["grid_min"], device="cuda")
            self.upper = torch.as_tensor(archive["grid_max"], device="cuda")

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        shape = points.shape[:-1]
        normalized = 2 * (points - self.lower) / (self.upper - self.lower) - 1
        sampled = F.grid_sample(
            self.grid,
            normalized.reshape(1, 1, 1, -1, 3),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(shape)
        # The padded box contains the entire solid; far-away points must never
        # inherit an interior border value, nor count as retained contacts.
        outside = torch.maximum(self.lower - points, points - self.upper).clamp_min(0).norm(dim=-1)
        return torch.where(outside > 0, sampled.clamp_min(0) + outside, sampled)


def main() -> None:  # noqa: C901 - standalone experiment lifecycle with shared model state
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=local / "prepared")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--clip_limit", type=int, default=0)
    args = parser.parse_args()
    if args.window < 8 or args.window > 150 or args.epochs < 1:
        parser.error("window must be 8..150 and epochs positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = args.prepared.resolve()
    repo = local.parents[2] / "Text2HOI"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.setdefault("WANDB_MODE", "disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("Run with Text2HOI's Conda environment and GPU access.")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    from lib.models.mano import build_mano_aa
    from lib.utils.model_utils import build_refiner
    from lib.utils.proc import proc_refiner_input
    from lib.utils.proc_output import get_hand_verts

    manifest = json.loads((prepared / "manifest.json").read_text())
    entries = manifest["clips"]
    if args.clip_limit:
        entries = entries[: args.clip_limit]
    objects = dict(np.load(prepared / "object.npz"))

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device="cuda")

    points = tensor(objects["object_points"])[None]
    normals = tensor(objects["object_normals"])[None]
    sdf = MugDistance(prepared / "sdf.npz")
    layers = {}
    for entry in entries:
        subject = entry["subject"]
        if subject in layers:
            continue
        assets = dict(np.load(prepared / "subjects" / f"{subject}.npz"))
        pair = []
        for side in ("lhand", "rhand"):
            layer = build_mano_aa(is_rhand=side == "rhand", flat_hand=True).cuda()
            layer.v_template.copy_(tensor(assets[f"{side}_v_template"]))
            layer.requires_grad_(False)
            pair.append(layer)
        layers[subject] = pair

    with initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        config = compose(config_name="config", overrides=["dataset=grab"])
    model = build_refiner(config, test=True)
    model.requires_grad_(True)
    source_weights = repo / config.refiner.weight_path
    source_hash = digest(source_weights)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(saved["model"], strict=True)
        previous_steps = int(saved.get("step", 0))
        del saved
    else:
        previous_steps = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    # Conservative adaptation: at most 15 mm root offset and bounded rotation
    # representation changes. Same wrapper is used before and after training.
    limits = tensor([0.015] * 3 + [0.15] * 96)

    def predict(item: dict, start: int, end: int, raw: bool = False):
        left, right = item["inputs"][0][None, start:end], item["inputs"][1][None, start:end]
        mask = torch.ones((1, end - start), dtype=torch.bool, device="cuda")
        result = model(left, right, valid_mask_lhand=torch.zeros_like(mask), valid_mask_rhand=mask)
        if raw:
            return result
        return tuple(
            x[..., :99] + limits * torch.tanh(0.1 * (y - x[..., :99]) / limits) for x, y in zip((left, right), result)
        )

    def local_vertices(item: dict, hand_params: tuple, start: int, end: int):
        # GraspXL contains only the right hand. The dummy left input is required
        # by the upstream interface and never enters any loss or measurement.
        verts = get_hand_verts(hand_params[1], layers[item["subject"]][1])[0]
        local_verts = torch.einsum(
            "tvi,tij->tvj",
            verts - item["translation"][start:end, None],
            item["rotation"][start:end],
        )
        return verts, local_verts

    started = time.monotonic()
    report = {
        "experiment": "graspxl_mug_text2hoi_refiner_source_cleanup_v1",
        "scope": "All clips used in training; in-sample reconstruction, no held-out generalization claim.",
        "source_checkpoint": str(source_weights),
        "source_checkpoint_sha256": source_hash,
        "prepared_manifest_sha256": digest(prepared / "manifest.json"),
        "sdf_sha256": digest(prepared / "sdf.npz"),
        "checkpoint_residual_wrapper": "0.1 scale; tanh cap 0.015 m translation / 0.15 rotation-6D components",
        "loss": (
            "Native mug SDF penetration, source surface-contact preservation, vertex and temporal correction penalties"
        ),
        "unchanged": ["native object geometry and poses", "all source frames", "standard zero-beta MANO template"],
        "source_fps": None,
        "playback_fps": 30,
        "temporal_policy": "All native frames retained; 30 FPS chosen for playback, physical source rate unknown",
        "hand_policy": "Right hand only; absent left masked and excluded from losses, metrics, and export",
        "object_topology": "Collision proxy splits vertex fans only; all native ordered triangle coordinates unchanged",
        "generator_trained": False,
        "refiner_trained": True,
        "physics_simulation": False,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "window_frames": args.window,
        "learning_rate": args.learning_rate,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "phase": "preparing_features",
        "clips": [],
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
        "resume_checkpoint_sha256": digest(args.resume) if args.resume else None,
        "previous_training_steps": previous_steps,
        "optimizer_resume": False,
    }
    write_json(output / "report.json", report)
    data = []
    for entry in entries:
        clip_id = entry["clip_id"]
        arrays = dict(np.load(prepared / "clips" / f"{clip_id}.npz"))
        hands = [tensor(arrays[f"x_{side}"])[None] for side in ("lhand", "rhand")]
        obj = tensor(arrays["x_obj"])[None]
        nframes = obj.shape[1]
        item = {
            "clip_id": clip_id,
            "subject": entry["subject"],
            "entry": entry,
            "arrays": arrays,
            "nframes": nframes,
            "translation": tensor(arrays["object_translation"]),
            "rotation": tensor(arrays["object_rotation"]),
        }
        # Coverage is conditioning from this observed source sequence. This is
        # source refinement, and must not be described as unconditioned generation.
        coverage_array = np.zeros(1024, dtype=np.float32)
        for side in ("rhand",):
            active_points = arrays[f"{side}_contact_point_indices"][arrays[f"{side}_contact_mask"]]
            coverage_array[np.unique(active_points)] = 1
        coverage = tensor(coverage_array)[None]
        feature_parts = [[], []]
        baseline_parts = []
        with torch.no_grad():
            for start in range(0, nframes, 128):
                end = min(nframes, start + 128)
                mask = torch.ones((1, end - start), dtype=torch.bool, device="cuda")
                features = proc_refiner_input(
                    hands[0][:, start:end],
                    hands[1][:, start:end],
                    obj[:, start:end],
                    *layers[item["subject"]],
                    points,
                    normals,
                    torch.zeros_like(mask),
                    mask,
                    mask,
                    coverage,
                    "grab",
                )
                for side in (0, 1):
                    feature_parts[side].append(features[side][0])
                verts, _ = local_vertices(item, (hands[0][:, start:end], hands[1][:, start:end]), start, end)
                baseline_parts.append(verts)
            item["inputs"] = [torch.cat(parts) for parts in feature_parts]
            item["before"] = torch.cat(baseline_parts)
            baseline_error = float((item["before"] - tensor(arrays["rhand_vertices"])).abs().max())
            if baseline_error > 1e-6:
                raise ValueError(f"Source MANO geometry changed for {clip_id}: {baseline_error} m")
            local_before = torch.einsum("tvi,tij->tvj", item["before"] - item["translation"][:, None], item["rotation"])
            item["before_sdf"] = sdf(local_before)
            item["near"] = item["before_sdf"].abs() < 0.012
            item["contact"] = item["before_sdf"].abs() < 0.003
        data.append(item)
        print(json.dumps({"features": clip_id, "frames": nframes, "baseline_max_error_m": baseline_error}), flush=True)
    report["clip_count"] = len(data)
    report["frame_count"] = sum(item["nframes"] for item in data)
    report["clips"] = [item["clip_id"] for item in data]

    def measure(item: dict, before: torch.Tensor, after: torch.Tensor, distances: torch.Tensor) -> dict:
        depth = (-distances).clamp_min(0) * 1000
        movement = (after - before).norm(dim=-1) * 1000
        selected = item["contact"]
        correction = (after - before) * 1000
        return {
            "vertices_deeper_than_2_mm": int((depth > 2).sum()),
            "vertices_deeper_than_5_mm": int((depth > 5).sum()),
            "frames_with_penetration_over_2_mm": int((depth > 2).any(dim=1).sum()),
            "maximum_penetration_mm": float(depth.max()),
            "mean_penetration_all_vertices_mm": float(depth.mean()),
            "mean_vertex_displacement_mm": float(movement.mean()),
            "max_vertex_displacement_mm": float(movement.max()),
            "source_contacts_within_3_mm_fraction": float((distances[selected].abs() < 0.003).float().mean())
            if selected.any()
            else None,
            "correction_velocity_rms_mm_per_frame": float(correction.diff(dim=0).square().mean().sqrt()),
            "vertex_samples": depth.numel(),
        }

    def infer(item: dict, raw: bool = False):
        length = item["nframes"]
        accum = [torch.zeros((length, 99), device="cuda") for _ in range(2)]
        denominator = torch.zeros((length, 1), device="cuda")
        stride = args.window // 2
        starts = list(range(0, max(length - args.window + 1, 1), stride))
        starts = sorted(set(starts + [max(0, length - args.window)]))
        for start in starts:
            end = min(length, start + args.window)
            predictions = predict(item, start, end, raw=raw)
            weights = torch.hann_window(end - start, periodic=False, device="cuda").clamp_min(0.05)[:, None]
            for side in range(2):
                accum[side][start:end] += predictions[side][0] * weights
            denominator[start:end] += weights
        params = tuple((value / denominator)[None] for value in accum)
        chunks = []
        distances = []
        for start in range(0, length, 128):
            end = min(length, start + 128)
            verts, local_verts = local_vertices(item, tuple(value[:, start:end] for value in params), start, end)
            chunks.append(verts)
            distances.append(sdf(local_verts))
        return params, torch.cat(chunks), torch.cat(distances)

    def evaluate(stage: str, export: bool = False) -> dict:
        model.eval()
        results = []
        export_entries = []
        if export:
            (output / "geometry").mkdir(exist_ok=True)
        with torch.no_grad():
            for item in data:
                params, verts, distances = infer(item)
                result = {
                    "clip_id": item["clip_id"],
                    "before": measure(item, item["before"], item["before"], item["before_sdf"]),
                    "after": measure(item, item["before"], verts, distances),
                }
                results.append(result)
                if export:
                    arrays = item["arrays"]
                    filename = output / "geometry" / (item["clip_id"] + ".npz")
                    before = item["before"].cpu().numpy()
                    after = verts.cpu().numpy()
                    np.savez_compressed(
                        filename,
                        vertices_r_before=before,
                        vertices_r_after=after,
                        faces_r=layers[item["subject"]][1].faces,
                        object_vertices_canonical=objects["object_vertices_canonical"],
                        object_faces=objects["object_faces"],
                        obj_rot=arrays["object_rotation"],
                        obj_trans=arrays["object_translation"],
                        source_frame_indices=arrays["source_frame_indices"],
                        x_rhand_before=arrays["x_rhand"],
                        x_rhand_after=params[1][0].cpu().numpy(),
                        penetration_count_before=(item["before_sdf"] < -0.002).sum(dim=1).cpu().numpy(),
                        penetration_count_after=(distances < -0.002).sum(dim=1).cpu().numpy(),
                        penetration_max_before_mm=(
                            (-item["before_sdf"]).clamp_min(0).max(dim=1).values.cpu().numpy() * 1000
                        ),
                        penetration_max_after_mm=(-distances).clamp_min(0).max(dim=1).values.cpu().numpy() * 1000,
                    )
                    metadata = {
                        **item["entry"],
                        "clip_id": item["clip_id"],
                        "source_fps": None,
                        "playback_fps": 30,
                        "split": "training",
                        "source_frames": item["entry"]["original_frames"],
                        "checkpoint": str(output / "checkpoint.pth"),
                        "comparison": (
                            "Original GraspXL right MANO vs adapted Text2HOI refiner; all clips used in training"
                        ),
                        "metrics": result,
                        "geometry_sha256": digest(filename),
                    }
                    metadata_path = filename.with_suffix(".json")
                    write_json(metadata_path, metadata)
                    export_entries.append(
                        {
                            **item["entry"],
                            "clip_id": item["clip_id"],
                            "geometry_path": str(filename),
                            "metadata_path": str(metadata_path),
                        }
                    )
                print(json.dumps({"evaluation": stage, **result}), flush=True)
        aggregate = {}
        for side in ("before", "after"):
            aggregate[side] = {
                "vertices_deeper_than_2_mm": sum(value[side]["vertices_deeper_than_2_mm"] for value in results),
                "vertices_deeper_than_5_mm": sum(value[side]["vertices_deeper_than_5_mm"] for value in results),
                "maximum_penetration_mm": max(value[side]["maximum_penetration_mm"] for value in results),
                "mean_vertex_displacement_mm": sum(
                    value[side]["mean_vertex_displacement_mm"] * value[side]["vertex_samples"] for value in results
                )
                / sum(value[side]["vertex_samples"] for value in results),
            }
        if export:
            write_json(
                output / "geometry_manifest.json",
                {"clips": export_entries, "source_fps": None, "playback_fps": 30, "scope": report["scope"]},
            )
        result = {
            "per_clip": results,
            "aggregate": aggregate,
            "metric_method": "0.75 mm native mug signed-distance grid, trilinear interpolation; vertex samples only",
        }
        write_json(output / f"evaluation_{stage}.json", result)
        return result

    if not args.export_only:
        report["phase"] = "evaluating_before_training"
        write_json(output / "report.json", report)
        report["evaluation_initial"] = evaluate("initial")["aggregate"]
        report["phase"] = "training"
        write_json(output / "report.json", report)
        windows = [
            (i, start, min(start + args.window, item["nframes"]))
            for i, item in enumerate(data)
            for start in range(0, item["nframes"], args.window)
            if min(start + args.window, item["nframes"]) - start >= 8
        ]
        # Include tails so every source frame enters at least one training window.
        for i, item in enumerate(data):
            tail = (i, max(0, item["nframes"] - args.window), item["nframes"])
            if tail not in windows:
                windows.append(tail)
        report["windows_per_epoch"] = len(windows)
        step = previous_steps
        with (output / "training.jsonl").open("w") as stream:
            for epoch in range(args.epochs):
                model.eval()  # Disable dropout; gradients remain enabled for deterministic cleanup.
                values = []
                for index in rng.permutation(len(windows)):
                    item_index, start, end = windows[index]
                    item = data[item_index]
                    optimizer.zero_grad(set_to_none=True)
                    params = predict(item, start, end)
                    vertices, local_verts = local_vertices(item, params, start, end)
                    distances = sdf(local_verts)
                    source = item["before"][start:end]
                    original_distances = item["before_sdf"][start:end]
                    near = item["near"][start:end]
                    contacts = item["contact"][start:end]
                    correction = (vertices - source) * 1000
                    penetration = F.relu(0.0002 - distances) * 1000
                    # Fixed source denominator also penalizes new penetration at vertices
                    # that were initially outside the near-contact set.
                    pen_loss = penetration.square().sum() / near.sum().clamp_min(1)
                    target_distance = original_distances.clamp_min(0.0002)
                    contact_loss = (
                        ((distances - target_distance) * 1000).square() * contacts
                    ).sum() / contacts.sum().clamp_min(1)
                    displacement = correction.square().mean()
                    velocity = correction.diff(dim=0).square().mean()
                    acceleration = correction.diff(dim=0, n=2).square().mean()
                    total = pen_loss + 0.3 * contact_loss + 0.03 * displacement + 0.1 * velocity + 0.05 * acceleration
                    if not torch.isfinite(total):
                        raise RuntimeError(f"Non-finite loss: {item['clip_id']}:{start}")
                    total.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError("Non-finite gradient")
                    optimizer.step()
                    step += 1
                    record = {
                        "step": step,
                        "epoch": epoch + 1,
                        "clip_id": item["clip_id"],
                        "start": start,
                        "loss": float(total),
                        "penetration": float(pen_loss),
                        "contact": float(contact_loss),
                        "displacement": float(displacement),
                        "velocity": float(velocity),
                        "acceleration": float(acceleration),
                        "gradient_norm": float(grad_norm),
                    }
                    stream.write(json.dumps(record) + "\n")
                    values.append(record)
                stream.flush()
                summary = {
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": float(np.mean([v["loss"] for v in values])),
                    "penetration": float(np.mean([v["penetration"] for v in values])),
                    "elapsed_seconds": time.monotonic() - started,
                }
                print("EPOCH " + json.dumps(summary), flush=True)
                report["training_progress"] = summary
                write_json(output / "report.json", report)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": report,
                "upstream_checkpoint_sha256": source_hash,
            },
            output / "checkpoint.pth",
        )
        report["training_steps"] = step
    report["phase"] = "exporting"
    write_json(output / "report.json", report)
    report["evaluation_final"] = evaluate("final", export=True)["aggregate"]
    report["phase"] = "complete"
    report["elapsed_seconds"] = time.monotonic() - started
    report["original_checkpoint_unchanged"] = digest(source_weights) == source_hash
    if (output / "checkpoint.pth").exists():
        report["checkpoint_sha256"] = digest(output / "checkpoint.pth")
    write_json(output / "report.json", report)
    print("COMPLETE " + json.dumps(report["evaluation_final"]), flush=True)


if __name__ == "__main__":
    main()
