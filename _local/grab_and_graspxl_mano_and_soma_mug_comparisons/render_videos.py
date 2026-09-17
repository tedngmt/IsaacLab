# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render synchronized original/SOMA mug comparisons with an overview and close-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from camera import build_camera_specs
from pytorch3d.renderer import (
    BlendParams,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

WIDTH, HEIGHT = 1920, 1080
PANEL_WIDTH, PANEL_HEIGHT = 960, 464
HEADER, FOOTER = 88, 64
FPS = 30
STYLE_VERSION = "mug_comparison_overview_detail_v1"
GRAB_STYLE_VERSION = "mug_comparison_world_follow_handle_corrected_rotation_v3"
COLORS = {"original": (0.22, 0.52, 0.88), "soma": (0.96, 0.53, 0.16), "object": (0.52, 0.63, 0.60)}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def caption(frame, value, position, size=0.7, color=(238, 241, 245), thickness=1):
    cv2.putText(frame, value, position, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)


def compose_frame(panels, kind, clip, index, count, source_index, source_count):
    canvas = np.full((HEIGHT, WIDTH, 3), (31, 25, 19), dtype=np.uint8)
    for row in range(2):
        for column in range(2):
            y, x = HEADER + row * PANEL_HEIGHT, column * PANEL_WIDTH
            canvas[y : y + PANEL_HEIGHT, x : x + PANEL_WIDTH] = panels[row * 2 + column][..., ::-1]
            detail_title = (
                "HANDLE CLOSE-UP  |  camera follows; mug motion preserved"
                if kind == "GRAB"
                else "HANDLE CLOSE-UP  |  object stabilized; hands may leave this view"
            )
            title = "FULL HAND MOTION" if row == 0 else detail_title
            cv2.rectangle(canvas, (x + 12, y + 10), (x + (251 if row == 0 else 904), y + 45), (247, 249, 251), -1)
            caption(canvas, title, (x + 24, y + 34), 0.62, (74, 64, 52), 1)
    cv2.line(canvas, (WIDTH // 2, HEADER), (WIDTH // 2, HEIGHT - FOOTER), (224, 226, 230), 2)
    cv2.line(canvas, (0, HEADER + PANEL_HEIGHT), (WIDTH, HEADER + PANEL_HEIGHT), (224, 226, 230), 2)
    name = clip["clip_id"].replace("__", " / ")
    caption(canvas, f"{kind}  /  MUG COMPARISON", (24, 30), 0.76, (250, 249, 247), 1)
    caption(canvas, name, (620, 30), 0.73, (213, 221, 229), 1)
    original_label = "Original GRAB - SMPL-X hands" if kind == "GRAB" else "Original MANO - right hand"
    caption(canvas, original_label, (24, 68), 0.78, (247, 171, 92), 1)
    soma_label = "SOMA reconstruction (new tabletop fit)" if clip.get("supplemental") else "SOMA reconstruction"
    caption(canvas, soma_label, (984, 68), 0.78, (94, 171, 250), 1)
    timing = (
        "Original timing | 120 fps source -> 30 fps export"
        if kind == "GRAB"
        else "30 fps playback assumed | source frame rate unknown"
    )
    caption(canvas, timing, (24, HEIGHT - 35), 0.64, (223, 227, 233), 1)
    caption(
        canvas,
        "Matched camera + scale in each pair; gray-green mug is unchanged",
        (24, HEIGHT - 10),
        0.54,
        (170, 181, 191),
        1,
    )
    caption(
        canvas,
        f"Frame {index + 1}/{count}   Source {source_index + 1}/{source_count}   {index / FPS:.2f} s",
        (1130, HEIGHT - 31),
        0.66,
        (233, 235, 241),
        1,
    )
    cv2.rectangle(canvas, (WIDTH - 345, HEIGHT - 15), (WIDTH - 24, HEIGHT - 10), (86, 75, 61), -1)
    cv2.rectangle(
        canvas,
        (WIDTH - 345, HEIGHT - 15),
        (WIDTH - 345 + int(321 * (index + 1) / count), HEIGHT - 10),
        (176, 190, 205),
        -1,
    )
    return canvas


class Renderer:
    def __init__(self, device: str):
        self.device = torch.device(device)
        settings = RasterizationSettings(
            image_size=(PANEL_HEIGHT, PANEL_WIDTH),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=64,
            max_faces_per_bin=100000,
            perspective_correct=False,
            cull_backfaces=False,
        )
        self.renderer = MeshRenderer(
            MeshRasterizer(raster_settings=settings),
            HardPhongShader(
                device=self.device,
                blend_params=BlendParams(background_color=(0.973, 0.980, 0.988)),
                materials=Materials(device=self.device, shininess=24, specular_color=((0.1, 0.1, 0.1),)),
            ),
        )

    @torch.no_grad()
    def render_clip(self, data, specs, start, end):
        vertices, triangles, colors, camera_r, camera_t, eyes, focal = [], [], [], [], [], [], []
        canonical = data["object_vertices_canonical"]
        upright = torch.as_tensor(specs["canonical_to_z_up"], dtype=torch.float32, device=self.device)
        fixed_mug = canonical @ upright.T
        cameras = {}
        for name in ("overview", "detail"):
            spec = specs[name]
            rotation, translation = look_at_view_transform(
                eye=(spec["eye"],),
                at=(spec["target"],),
                up=(spec["up"],),
                device=self.device,
            )
            cameras[name] = (rotation[0], translation[0], spec["eye"], 2.0 / spec["height"])
        for frame in range(start, end):
            rotation = data["object_rotation"][frame]
            translation = data["object_translation"][frame]
            world_mug = canonical @ rotation.T + translation
            for view in ("overview", "detail"):
                for side in ("original", "soma"):
                    hand = data[f"{side}_vertices"][frame]
                    mug = world_mug
                    if view == "detail" and specs["detail"]["coordinate_frame"] != "world_follow_handle":
                        hand = ((hand - translation) @ rotation) @ upright.T
                        mug = fixed_mug
                    vertices.append(torch.cat([mug, hand], dim=0))
                    triangles.append(data[f"{side}_scene_faces"])
                    colors.append(data[f"{side}_scene_colors"])
                    frame_camera = cameras[view]
                    if view == "detail" and "detail_frames" in specs:
                        spec = specs["detail_frames"][frame]
                        camera_rotation, camera_translation = look_at_view_transform(
                            eye=(spec["eye"],), at=(spec["target"],), up=(spec["up"],), device=self.device
                        )
                        frame_camera = (camera_rotation[0], camera_translation[0], spec["eye"], 2.0 / spec["height"])
                    camera_r.append(frame_camera[0])
                    camera_t.append(frame_camera[1])
                    eyes.append(frame_camera[2])
                    focal.append((frame_camera[3], frame_camera[3]))
        meshes = Meshes(verts=vertices, faces=triangles, textures=TexturesVertex(verts_features=colors))
        cameras = OrthographicCameras(
            device=self.device,
            R=torch.stack(camera_r),
            T=torch.stack(camera_t),
            focal_length=focal,
        )
        lights = PointLights(
            device=self.device,
            location=eyes,
            ambient_color=((0.58, 0.58, 0.58),),
            diffuse_color=((0.42, 0.42, 0.42),),
            specular_color=((0.08, 0.08, 0.08),),
        )
        rendered = self.renderer(meshes, cameras=cameras, lights=lights)
        if not torch.isfinite(rendered).all():
            raise ValueError("Renderer produced non-finite pixels.")
        return (
            (rendered[..., :3].clamp(0, 1) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
            .reshape(end - start, 4, PANEL_HEIGHT, PANEL_WIDTH, 3)
        )


def render_one(args, renderer, kind, clip):
    style_version = GRAB_STYLE_VERSION if kind == "GRAB" else STYLE_VERSION
    cache = args.cache_root / kind / (clip["clip_id"] + ".npz")
    cache_report = json.loads(cache.with_suffix(".json").read_text())
    if not cache_report["validated"]:
        raise ValueError("Geometry cache has not passed reconstruction checks.")
    output = args.output_root / kind / (clip["clip_id"] + ".mp4")
    sidecar = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_hash = digest(cache)
    if cache_hash != cache_report["output_sha256"]:
        raise ValueError("Geometry cache checksum changed.")
    if output.exists() and sidecar.exists() and not args.preview_only:
        existing = json.loads(sidecar.read_text())
        if (
            existing.get("completed")
            and existing.get("cache_sha256") == cache_hash
            and existing.get("style_version") == style_version
            and existing.get("renderer_backend", "pytorch3d") == args.backend
        ):
            print(f"SKIP {kind}/{clip['clip_id']} (validated existing export)", flush=True)
            return existing
        raise FileExistsError(f"Existing video has incompatible provenance: {output}")
    with np.load(cache, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    count = len(arrays["original_vertices"])
    specs = build_camera_specs(arrays, kind, aspect=PANEL_WIDTH / PANEL_HEIGHT)
    data = {}
    for key in (
        "original_vertices",
        "soma_vertices",
        "object_vertices_canonical",
        "object_rotation",
        "object_translation",
    ):
        data[key] = torch.as_tensor(arrays[key], dtype=torch.float32, device=renderer.device)
    object_faces = torch.as_tensor(arrays["object_faces"], dtype=torch.int64, device=renderer.device)
    object_count = len(data["object_vertices_canonical"])
    for side in ("original", "soma"):
        hand_faces = torch.as_tensor(arrays[f"{side}_faces"], dtype=torch.int64, device=renderer.device)
        data[f"{side}_scene_faces"] = torch.cat([object_faces, hand_faces + object_count], dim=0)
        data[f"{side}_scene_colors"] = torch.cat(
            [
                torch.tensor(COLORS["object"], device=renderer.device).expand(object_count, 3),
                torch.tensor(COLORS[side], device=renderer.device).expand(data[f"{side}_vertices"].shape[1], 3),
            ],
            dim=0,
        )
    qa_dir = args.qa_root / kind / clip["clip_id"]
    qa_dir.mkdir(parents=True, exist_ok=True)
    selected = {0, count // 2, count - 1}
    started = time.perf_counter()
    if args.preview_only:
        for index in sorted(selected):
            panels = renderer.render_clip(data, specs, index, index + 1)[0]
            canvas = compose_frame(
                panels, kind, clip, index, count, int(arrays["source_frame_indices"][index]), clip["source_frames"]
            )
            assert cv2.imwrite(str(qa_dir / f"frame_{index:05d}.png"), canvas)
        print(f"PREVIEW {kind}/{clip['clip_id']}: {qa_dir}", flush=True)
        return {"dataset": kind, "clip_id": clip["clip_id"], "preview_only": True, "cameras": specs}
    partial = output.with_name(output.stem + ".partial.mp4")
    if partial.exists():
        raise FileExistsError(f"Preserve/review incomplete export before retrying: {partial}")
    command = [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-n",
        str(partial),
    ]
    encoder_log = qa_dir / "ffmpeg.log"
    report = {
        "completed": False,
        "style_version": style_version,
        "dataset": kind,
        "clip_id": clip["clip_id"],
        "video": str(output),
        "cache_sha256": cache_hash,
        "object_transform_convention": cache_report.get("object_transform_convention"),
        "source_object_rotation_convention": cache_report.get("source_object_rotation_convention"),
        "frames": count,
        "output_frames": count,
        "source_frames": clip["source_frames"],
        "source_fps": clip["source_fps"],
        "fps": FPS,
        "playback_fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "cameras": specs,
        "original_label": cache_report["original_label"],
        "renderer_backend": args.backend,
        "renderer_info": getattr(renderer, "renderer_info", "PyTorch3D CUDA"),
        "supplemental": bool(clip.get("supplemental")),
        "soma_conversion": clip.get("soma_conversion", "existing_dataset_conversion"),
        "scope": (
            "Source reconstruction comparison; top shows full hand motion; bottom follows the handle "
            "with a world-up camera while retaining the original world-space mug and hand motion."
            if kind == "GRAB"
            else "Source reconstruction comparison; top shows full hand motion, "
            "bottom shows an explicitly cropped object-stabilized grip view."
        ),
        "all_geometry_unmodified": True,
    }
    sidecar.write_text(json.dumps(report, indent=2) + "\n")
    with encoder_log.open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)

        def encode_batch(rendered, start):
            """Compose and encode in order while the main thread renders the next batch."""
            for offset, panels in enumerate(rendered):
                index = start + offset
                canvas = compose_frame(
                    panels,
                    kind,
                    clip,
                    index,
                    count,
                    int(arrays["source_frame_indices"][index]),
                    clip["source_frames"],
                )
                process.stdin.write(canvas.tobytes())
                if index in selected:
                    assert cv2.imwrite(str(qa_dir / f"frame_{index:05d}.png"), canvas)
            end = start + len(rendered)
            if end % 120 < args.batch_size or end == count:
                print(f"FRAMES {kind}/{clip['clip_id']} {end}/{count}", flush=True)

        try:
            with ThreadPoolExecutor(max_workers=1) as encoder:
                pending = None
                for start in range(0, count, args.batch_size):
                    end = min(start + args.batch_size, count)
                    rendered = renderer.render_clip(data, specs, start, end)
                    if pending is not None:
                        pending.result()
                    pending = encoder.submit(encode_batch, rendered, start)
                if pending is not None:
                    pending.result()
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"Video encoder failed; see {encoder_log}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    partial.rename(output)
    probe = subprocess.run(
        [
            "/usr/bin/ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert int(stream["nb_frames"]) == count and (stream["width"], stream["height"]) == (WIDTH, HEIGHT)
    assert stream["codec_name"] == "h264" and stream["avg_frame_rate"] == "30/1"
    report.update(
        completed=True,
        elapsed_seconds=time.perf_counter() - started,
        video_sha256=digest(output),
        duration_seconds=count / FPS,
        peak_cuda_allocated_mib=(
            torch.cuda.max_memory_allocated() / 1024**2 if torch.device(renderer.device).type == "cuda" else 0.0
        ),
        video_probe=stream,
    )
    sidecar.write_text(json.dumps(report, indent=2) + "\n")
    print(f"DONE {kind}/{clip['clip_id']} {count} frames in {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def main():
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "GRAB", "GraspXL"), default="all")
    parser.add_argument("--clip_id", action="append")
    parser.add_argument("--cache_root", type=Path, default=local / "cache")
    parser.add_argument("--output_root", type=Path, default=local / "share")
    parser.add_argument("--qa_root", type=Path, default=local / "qa")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("egl", "pytorch3d"), default="egl")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch_size must be positive")
    torch.set_num_threads(4)
    if args.backend == "pytorch3d":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU rendering requires CUDA access in the text2hoi environment.")
        torch.cuda.reset_peak_memory_stats()
        renderer = Renderer(args.device)
    else:
        from render_gl import GLRenderer

        renderer = GLRenderer()
    inventory = json.loads((local / "inventory.json").read_text())
    results = []
    for kind, dataset in inventory["datasets"].items():
        if args.dataset not in ("all", kind):
            continue
        for clip in dataset["clips"]:
            if args.clip_id and clip["clip_id"] not in args.clip_id:
                continue
            results.append(render_one(args, renderer, kind, clip))
    if not results:
        raise ValueError("No clips matched the requested dataset/clip IDs.")
    print(f"Completed {len(results)} clips.", flush=True)


if __name__ == "__main__":
    main()
