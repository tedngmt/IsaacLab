# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render and package GraspXL mug MANO reference versus Text2HOI refinement videos."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
HELPERS = BASE.parent / "mug_comparisons"
sys.path.insert(0, str(HELPERS))
from camera import build_camera_specs
from package_videos import inspect_video, make_compilation, sha256
from render_gl import GLRenderer

WIDTH, HEIGHT = 1920, 1080
PANEL_WIDTH, PANEL_HEIGHT = 960, 464
HEADER, FOOTER = 88, 64
FPS = 30
STYLE_VERSION = "graspxl_mano_text2hoi_before_after_object_stabilized_v1"
TRAINING_SCOPE = "Trained on all 18 GraspXL mug clips; training-set comparison, not held-out evaluation"
ROTATION_CONVENTION = "Active rotation: world_vertices = canonical_vertices @ obj_rot.T + obj_trans"
PENETRATION_KEYS = (
    "penetration_count_before",
    "penetration_count_after",
    "penetration_max_before_mm",
    "penetration_max_after_mm",
)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _resolve(base: Path, name: str) -> Path:
    path = Path(name)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    clips = manifest["clips"]
    if not clips:
        raise ValueError("The geometry manifest has no clips")
    seen = set()
    resolved = []
    for item in clips:
        clip = dict(item)
        identifier = clip["clip_id"]
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", identifier) or identifier in seen:
            raise ValueError(f"Unsafe or repeated clip identifier: {identifier}")
        seen.add(identifier)
        clip["geometry_path"] = _resolve(path.parent, clip["geometry_path"])
        clip["metadata_path"] = _resolve(path.parent, clip["metadata_path"])
        resolved.append(clip)
    return manifest, resolved


def _faces(value: np.ndarray, vertex_count: int, name: str) -> np.ndarray:
    if value.ndim != 2 or value.shape[1] != 3 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"Expected integer triangular faces for {name}")
    if len(value) == 0 or value.min() < 0 or value.max() >= vertex_count:
        raise ValueError(f"Face indices are outside the {name} mesh")
    return value.astype(np.int64, copy=False)


def _geometry(clip: dict) -> tuple[dict, dict, np.ndarray]:
    """Load the original and refined MANO right hand in world coordinates [m]."""
    metadata = json.loads(clip["metadata_path"].read_text())
    if metadata.get("clip_id", clip["clip_id"]) != clip["clip_id"]:
        raise ValueError("Geometry metadata identifies a different clip")
    with np.load(clip["geometry_path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    count = len(arrays["vertices_r_before"])
    if not count:
        raise ValueError("The motion is empty")
    for stage in ("before", "after"):
        key = f"vertices_r_{stage}"
        if arrays[key].shape != (count, 778, 3) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Expected finite {key} with shape ({count}, 778, 3)")
    source_fps = metadata["source_fps"]
    playback_fps = float(metadata.get("playback_fps", FPS))
    source_frames = metadata["source_frames"]
    if isinstance(source_frames, bool) or not isinstance(source_frames, int) or source_frames < 1:
        raise ValueError("source_frames must be a positive integer")
    if source_fps is not None or playback_fps != FPS:
        raise ValueError("GraspXL source FPS must be null; viewing playback is assumed to be 30 FPS")
    indices = np.asarray(arrays["source_frame_indices"])
    expected_indices = np.arange(source_frames)
    if not np.array_equal(indices, expected_indices) or len(indices) != count:
        raise ValueError("Source frame indices must retain every source frame at stride one")
    rotation = np.asarray(arrays["obj_rot"], dtype=np.float32)
    translation = np.asarray(arrays["obj_trans"], dtype=np.float32)
    if rotation.shape != (count, 3, 3) or translation.shape != (count, 3):
        raise ValueError("Object transforms must correspond to every displayed frame")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Object transforms contain non-finite values")
    orthogonality = rotation @ rotation.transpose(0, 2, 1)
    if not np.allclose(orthogonality, np.eye(3), atol=2e-4) or not np.allclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise ValueError("obj_rot must contain proper active rotation matrices")
    canonical = np.asarray(arrays["object_vertices_canonical"], dtype=np.float32)
    if canonical.ndim != 2 or canonical.shape[1] != 3 or not np.isfinite(canonical).all():
        raise ValueError("Object mesh must have finite canonical vertices [m]")
    faces = _faces(arrays["faces_r"], 778, "right hand")
    # The shared renderer's internal column names refer only to its two inputs.
    # Both inputs here are MANO hands; no SOMA surface is used or displayed.
    data = {
        "original_vertices": arrays["vertices_r_before"],
        "soma_vertices": arrays["vertices_r_after"],
        "original_faces": faces,
        "soma_faces": faces,
        "object_vertices_canonical": canonical,
        "object_faces": _faces(arrays["object_faces"], len(canonical), "object"),
        "object_rotation": rotation,
        "object_translation": translation,
        "playback_fps": np.asarray(FPS),
    }
    present = [key for key in PENETRATION_KEYS if key in arrays]
    if present and len(present) != len(PENETRATION_KEYS):
        raise ValueError("Provide all four optional penetration-caption arrays together")
    for key in present:
        values = np.asarray(arrays[key])
        if values.shape != (count,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Expected nonnegative finite per-frame values for {key}")
        if key.startswith("penetration_count") and (
            (values > 778).any() or not np.array_equal(values, np.rint(values))
        ):
            raise ValueError(f"Expected whole-vertex counts from zero to 778 for {key}")
        data[key] = values
    return data, metadata, indices.astype(np.int64)


def _caption(frame: np.ndarray, text: str, xy: tuple[int, int], size=0.7, color=(238, 241, 245)) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_DUPLEX, size, color, 1, cv2.LINE_AA)


def _compose(
    panels: np.ndarray, clip_id: str, frame: int, count: int, source_index: int, data: dict | None = None
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (31, 25, 19), dtype=np.uint8)
    for row in range(2):
        for column in range(2):
            y, x = HEADER + row * PANEL_HEIGHT, column * PANEL_WIDTH
            canvas[y : y + PANEL_HEIGHT, x : x + PANEL_WIDTH] = panels[row * 2 + column][..., ::-1]
            if row == 1:
                label = "HANDLE CLOSE-UP | object stabilized; hand may leave this view"
                cv2.rectangle(canvas, (x + 12, y + 10), (x + 854, y + 45), (247, 249, 251), -1)
                _caption(canvas, label, (x + 24, y + 34), 0.60, (74, 64, 52))
            if row == 1 and data is not None and PENETRATION_KEYS[0] in data:
                stage = "before" if column == 0 else "after"
                inside_count = int(data[f"penetration_count_{stage}"][frame])
                depth_mm = float(data[f"penetration_max_{stage}_mm"][frame])
                detail = f"Inside >2 mm: {inside_count} vertices | max {depth_mm:.1f} mm (approx.)"
                width = cv2.getTextSize(detail, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)[0][0]
                cv2.rectangle(canvas, (x + 12, y + 49), (x + width + 36, y + 77), (247, 249, 251), -1)
                _caption(canvas, detail, (x + 24, y + 69), 0.52, (74, 64, 52))
    cv2.line(canvas, (WIDTH // 2, HEADER), (WIDTH // 2, HEIGHT - FOOTER), (224, 226, 230), 2)
    cv2.line(canvas, (0, HEADER + PANEL_HEIGHT), (WIDTH, HEADER + PANEL_HEIGHT), (224, 226, 230), 2)
    _caption(canvas, "GraspXL MUG | TEXT2HOI REFINEMENT", (24, 29), 0.75)
    _caption(canvas, clip_id.replace("__", " / "), (1000, 29), 0.72, (213, 221, 229))
    _caption(canvas, "BEFORE | original MANO right hand", (24, 67), 0.75, (247, 171, 92))
    _caption(canvas, "AFTER | Text2HOI refinement, same MANO identity", (984, 67), 0.72, (94, 171, 250))
    _caption(
        canvas, "Trained on these 18 clips | training-set comparison, not held-out evaluation", (24, HEIGHT - 36), 0.59
    )
    _caption(
        canvas,
        "30 fps assumed; source FPS unknown | same object + camera | no collision response",
        (24, HEIGHT - 11),
        0.51,
    )
    _caption(
        canvas,
        f"Frame {frame + 1}/{count} | Source {source_index + 1} | Playback {frame / FPS:.2f} s",
        (1310, HEIGHT - 31),
        0.49,
    )
    return canvas


def _signature(clip: dict) -> dict:
    return {
        "style_version": STYLE_VERSION,
        "geometry_sha256": sha256(clip["geometry_path"]),
        "metadata_sha256": sha256(clip["metadata_path"]),
        "renderer_script_sha256": sha256(Path(__file__)),
        "camera_script_sha256": sha256(HELPERS / "camera.py"),
        "gl_script_sha256": sha256(HELPERS / "render_gl.py"),
    }


def _render_one(args, renderer, clip: dict) -> dict:
    output = args.output_dir / "clips" / f"{clip['clip_id']}.mp4"
    report_path = output.with_suffix(".json")
    signature = _signature(clip)
    if output.exists() and not args.preview_only:
        if report_path.exists():
            existing = json.loads(report_path.read_text())
            if existing.get("completed") and existing.get("signature") == signature:
                if sha256(output) != existing.get("video_sha256"):
                    raise ValueError(f"Existing video hash changed: {output}")
                print(f"SKIP {clip['clip_id']} (verified existing render)", flush=True)
                return existing
        raise FileExistsError(f"Existing render differs; choose a new --output_dir: {output}")
    data, metadata, indices = _geometry(clip)
    count = len(indices)
    specs = build_camera_specs(data, "GraspXL", aspect=PANEL_WIDTH / PANEL_HEIGHT)
    specs["overview"]["framing"] = "The MANO right hand before and after refinement, every frame; 10% padding"
    if specs["detail"]["coordinate_frame"] != "object_stabilized_z_up":
        raise ValueError("Expected the original GraspXL object-stabilized close-up camera")
    qa = args.qa_dir / clip["clip_id"]
    qa.mkdir(parents=True, exist_ok=True)
    selected = {0, count // 2, count - 1}
    if args.preview_only:
        for frame in sorted(selected):
            panels = renderer.render_clip(data, specs, frame, frame + 1)[0]
            image = _compose(panels, clip["clip_id"], frame, count, int(indices[frame]), data)
            if not cv2.imwrite(str(qa / f"frame_{frame:05d}.png"), image):
                raise RuntimeError(f"Could not write preview for {clip['clip_id']}")
        _write_json(qa / "preview.json", {"clip_id": clip["clip_id"], "signature": signature, "cameras": specs})
        print(f"PREVIEW {clip['clip_id']}: {qa}", flush=True)
        return {"clip_id": clip["clip_id"], "preview_only": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    if partial.exists():
        raise FileExistsError(f"Review the unfinished render before retrying: {partial}")
    report = {
        "completed": False,
        "clip_id": clip["clip_id"],
        "signature": signature,
        "frames": count,
        "source_frames": int(metadata["source_frames"]),
        "source_fps": None,
        "playback_fps": FPS,
        "source_frame_stride": 1,
        "playback_fps_assumed": True,
        "resolution": [WIDTH, HEIGHT],
        "object_transform_convention": ROTATION_CONVENTION,
        "object_motion_unchanged": True,
        "before_label": "Original GraspXL MANO right hand",
        "after_label": "Text2HOI refinement, same MANO identity",
        "hand_surface_model": "MANO right hand in both columns",
        "training_scope": TRAINING_SCOPE,
        "subject": metadata.get("subject", clip.get("subject")),
        "action": metadata.get("action", clip.get("action")),
        "supplemental": bool(metadata.get("supplemental", clip.get("supplemental", False))),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "cameras": specs,
        "renderer_info": renderer.renderer_info,
        "penetration_captions": PENETRATION_KEYS[0] in data,
        "penetration_caption_definition": (
            "Approximate signed-distance-field count of hand vertices deeper than 2 mm and maximum depth [mm]; "
            "not an intersection volume"
        ),
    }
    _write_json(report_path, report)
    started = time.perf_counter()
    command = [
        "ffmpeg",
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
        str(args.encoder_threads),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-n",
        str(partial),
    ]
    with (qa / "ffmpeg.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)

        def encode_batch(rendered: np.ndarray, start: int) -> None:
            """Encode ordered frames while the main thread renders the next batch."""
            for offset, views in enumerate(rendered):
                frame = start + offset
                image = _compose(views, clip["clip_id"], frame, count, int(indices[frame]), data)
                process.stdin.write(image.tobytes())
                if frame in selected and not cv2.imwrite(str(qa / f"frame_{frame:05d}.png"), image):
                    raise RuntimeError("Could not write QA frame")
            end = start + len(rendered)
            if end % 120 < args.batch_size or end == count:
                print(f"FRAMES {clip['clip_id']} {end}/{count}", flush=True)

        try:
            with ThreadPoolExecutor(max_workers=1) as encoder:
                pending = None
                for start in range(0, count, args.batch_size):
                    end = min(count, start + args.batch_size)
                    panels = renderer.render_clip(data, specs, start, end)
                    if pending is not None:
                        pending.result()
                    pending = encoder.submit(encode_batch, panels, start)
                if pending is not None:
                    pending.result()
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"Encoder failed; inspect {qa / 'ffmpeg.log'}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    validation = inspect_video(partial, count, full_decode=False)
    partial.replace(output)
    report.update(
        completed=True,
        video_sha256=validation["sha256"],
        duration_seconds=count / FPS,
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json(report_path, report)
    print(f"DONE {clip['clip_id']} {count} frames in {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def _validate_one(output_dir: Path, clip: dict) -> dict:
    relative = Path("clips") / f"{clip['clip_id']}.mp4"
    sidecar = output_dir / relative.with_suffix(".json")
    report = json.loads(sidecar.read_text())
    if report.get("completed") is not True or report.get("signature") != _signature(clip):
        raise ValueError(f"Incomplete or stale video: {clip['clip_id']}")
    metadata = json.loads(clip["metadata_path"].read_text())
    expected_frames = int(metadata["source_frames"])
    validation = inspect_video(output_dir / relative, expected_frames)
    if validation["sha256"] != report["video_sha256"]:
        raise ValueError(f"Video hash does not match its render report: {relative}")
    return {
        "clip_id": clip["clip_id"],
        "video_path": relative.as_posix(),
        "report_path": relative.with_suffix(".json").as_posix(),
        "report_sha256": sha256(sidecar),
        "subject": report.get("subject"),
        "action": report.get("action"),
        "supplemental": bool(report.get("supplemental", False)),
        "source_frames": int(metadata["source_frames"]),
        "source_fps": None,
        "playback_fps": FPS,
        "playback_fps_assumed": True,
        "signature": report["signature"],
        **validation,
    }


def _write_index(output_dir: Path, validation: dict) -> None:
    data = {
        "clips": [
            {
                key: item[key]
                for key in (
                    "clip_id",
                    "video_path",
                    "report_path",
                    "subject",
                    "action",
                    "duration_seconds",
                    "frames",
                    "supplemental",
                )
            }
            for item in validation["clips"]
        ],
        "compilation": validation["compilation"],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>GraspXL mug · Text2HOI refinement</title>
<style>:root{color-scheme:dark;font-family:system-ui,Segoe UI,sans-serif;background:#101722;color:#edf2fa}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:28px}
h1{margin:0 0 12px;letter-spacing:-.035em}p{line-height:1.55;color:#bfd0e4}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:20px}
video{width:100%;background:#080d14;aspect-ratio:16/9}
.player{border:1px solid #364960;border-radius:10px;overflow:hidden;background:#182434}.caption{padding:16px}
h2{font-size:18px;margin:0 0 10px}.meta{font-size:14px;color:#bfd0e4}a{color:#a7d3ff}
button,input{font:inherit;color:inherit;border:1px solid #415875;background:#1d2c3e;padding:10px;border-radius:7px}
button{cursor:pointer}button:hover{background:#2a4768}.all{margin:0 0 18px}input{width:100%}
.clips{display:flex;flex-direction:column;gap:7px;max-height:680px;overflow-y:auto}.clip{text-align:left;font-size:14px}
.clip small{display:block;color:#aec3da}.clip[aria-current="true"]{background:#264969}
.notice{padding:14px 17px;background:#1a293a;border-left:3px solid #78ade0}.links{display:flex;gap:18px;margin-top:12px}
@media(max-width:900px){main{padding:18px 12px}.layout{grid-template-columns:1fr}.clips{max-height:320px}}
</style></head><body><main><h1>GraspXL mug: before and after Text2HOI refinement</h1>
<p>All 18 available clips for the selected large GraspXL mug: 15 diverse-grasp clips and 3 supplemental tabletop
clips. Left: original MANO right hand. Right: Text2HOI refinement using the same MANO identity. The top row shows
world-space hand motion; the bottom row shows an object-stabilized handle close-up.</p>
<p class="notice"><strong>Training-set comparison:</strong> the refinement model was trained on these 18 clips.
These views show its behavior on training data and do not measure performance on unseen motions.</p>
<button class="all" id="all">Play all 18 clips</button><div class="layout"><section><div class="player">
<video id="video" controls playsinline preload="metadata"></video><div class="caption"><h2 id="title"></h2>
<div class="meta" id="meta"></div><div class="links"><a id="download" download>Save video</a>
<a id="report">Render report</a><a href="README.txt">About this comparison</a></div></div></div>
<p class="notice">Both columns use the same MANO surface identity, object trajectory, camera and source frame.
Only the right hand is displayed. No SOMA model is shown.</p>
<p>Every source frame is retained. Source FPS is unknown; 30 FPS is an assumed viewing rate, so the videos do not
establish the real motion duration. Close-ups may crop the hand away from the mug; the overview retains it.
Playback applies no collision response. Visible overlap can remain in source or refined hand surfaces.</p>
<p>When present, the close-up captions count hand vertices deeper than 2 mm inside an approximate signed-distance
field and show maximum depth in millimeters. These estimates describe sampled vertices, not intersection volume.</p>
</section><aside><input id="search" type="search" aria-label="Find a clip" placeholder="Sequence or tabletop…">
<p id="count"></p><div class="clips" id="clips"></div></aside></div>
<p><a href="validation.json">Validation, chapters and SHA-256 hashes</a>
· Offline viewing · Keep this folder together.</p>
</main><script>const data=__DATA__;
const video=document.getElementById('video'),list=document.getElementById('clips');
const search=document.getElementById('search');
let selected='';function duration(s){const n=Math.round(s);return Math.floor(n/60)+':'+String(n%60).padStart(2,'0');}
function select(item,play){selected=item.video_path;video.src=item.video_path;
document.getElementById('title').textContent=item.clip_id||'All 18 GraspXL mug clips';
document.getElementById('meta').textContent=duration(item.duration_seconds)+' · '+item.frames+' frames · 30 FPS assumed'
+(item.supplemental?' · Supplemental tabletop motion':'');
document.getElementById('download').href=item.video_path;const report=document.getElementById('report');
report.hidden=!item.report_path;if(item.report_path)report.href=item.report_path;render();
if(play)video.play().catch(()=>{});}
function render(){const q=search.value.toLowerCase().trim();const filtered=data.clips.filter(item=>
!q||[item.clip_id,item.subject,item.action].join(' ').toLowerCase().includes(q));list.replaceChildren();
document.getElementById('count').textContent=filtered.length+' of 18 clips';
for(const item of filtered){const b=document.createElement('button');b.className='clip';b.textContent=item.clip_id;
b.setAttribute('aria-current',String(selected===item.video_path));const d=document.createElement('small');
d.textContent=duration(item.duration_seconds)+(item.supplemental?' · Tabletop supplement':' · Diverse grasp');
b.append(d);
b.addEventListener('click',()=>select(item,true));list.append(b);}}
search.addEventListener('input',render);document.getElementById('all').addEventListener('click',()=>select(data.compilation,true));
select(data.clips[0],false);</script></body></html>"""
    (output_dir / "index.html").write_text(page.replace("__DATA__", payload), encoding="utf-8")


def _package(args, clips: list[dict]) -> None:
    if len(clips) != 18:
        raise ValueError("Final packaging requires a manifest containing all 18 GraspXL mug clips")
    missing = [
        str(Path("clips") / (clip["clip_id"] + suffix))
        for clip in clips
        for suffix in (".mp4", ".json")
        if not (args.output_dir / "clips" / (clip["clip_id"] + suffix)).is_file()
    ]
    if missing:
        raise FileNotFoundError("All renders must finish before packaging: " + ", ".join(missing))
    checked = {}
    with ThreadPoolExecutor(max_workers=args.decode_jobs) as pool:
        futures = {pool.submit(_validate_one, args.output_dir, clip): clip["clip_id"] for clip in clips}
        for future in as_completed(futures):
            checked[futures[future]] = future.result()
            print(f"VALIDATED {len(checked)}/18: {futures[future]}", flush=True)
    ordered = [checked[clip["clip_id"]] for clip in clips]
    supplemental_count = sum(item["supplemental"] for item in ordered)
    if supplemental_count != 3:
        raise ValueError("Expected 15 matched diverse-grasp clips and 3 supplemental tabletop clips")
    for item in ordered:
        expected_frames = 100 if item["supplemental"] else 155
        if item["frames"] != expected_frames:
            raise ValueError(f"Incomplete source motion for {item['clip_id']}: expected {expected_frames} frames")
    compilation = make_compilation("GraspXL_Text2HOI_before_after", ordered, args.output_dir)
    validation = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(args.manifest),
        "clip_count": 18,
        "matched_clip_count": len(ordered) - supplemental_count,
        "supplemental_clip_count": supplemental_count,
        "missing_clips": [],
        "training_scope": TRAINING_SCOPE,
        "comparison": "Original MANO before versus Text2HOI-refined MANO",
        "source_scope": "Selected large mug 12652c6e0aaf44dda32aab816f433baf: 15 diverse-grasp + 3 tabletop clips",
        "source_fps": None,
        "playback_fps": FPS,
        "playback_fps_assumed": True,
        "object_transform_convention": ROTATION_CONVENTION,
        "clips": ordered,
        "compilation": compilation,
    }
    _write_json(args.output_dir / "validation.json", validation)
    _write_index(args.output_dir, validation)
    readme = f"""GraspXL MUG — BEFORE AND AFTER TEXT2HOI REFINEMENT

Open index.html to browse all 18 clips, or play {compilation["video_path"]}.
The compilation has 18 native chapters for players with chapter-menu support.
This folder works offline; keep its relative file layout intact.

Scope: all 18 available motions for the selected large GraspXL mug,
object UID 12652c6e0aaf44dda32aab816f433baf: 15 diverse-grasp clips from the
matched dataset and 3 supplemental tabletop source motions. Other mug object
identities in the full GraspXL archives are outside this selected-object scope.

Left: original MANO right hand reconstructed from GraspXL.
Right: Text2HOI-refined motion using the same MANO hand identity.
Only the right hand is displayed in each column. No SOMA surface is used.

{TRAINING_SCOPE}.
Visual changes on these clips do not establish generalization or better grasps.

Top row: complete hand motion in world coordinates.
Bottom row: an object-stabilized handle close-up, using the same convention as
the original GraspXL dataset comparison videos. The hand away from the mug may
leave this close-up but remains in the overview. Object mesh, native scale,
trajectory, camera and timing
are identical before and after. The comparison isolates the hand refinement.

Geometry files provide active object rotations; world-space rendering uses
canonical_vertices @ obj_rot.T + obj_trans. Stabilized close-ups apply the same
inverse object transform to the object and hand, followed by a common view rotation.
Playback has no collision response. Source or refined hand surfaces can still overlap.
When present, per-frame captions use the approximate object signed-distance field:
the number of hand vertices deeper than 2 mm inside, and maximum depth in mm.
These are sampled-vertex estimates, not intersection volume measurements.

Every source frame is shown at an assumed 30 FPS viewing rate. Source FPS is
unknown, so these video durations do not establish the real motion durations.
MP4 / H.264, 1920 x 1080. Total compilation frames: {compilation["frames"]}.
Playback duration: {compilation["duration_seconds"]:.3f} seconds.
All 18 videos and the compilation passed frame-count, format, checksum and
full-decoding checks. validation.json contains checks, hashes and chapter offsets.

This viewing package includes only videos and reports. Licensed hand/body model
assets, source motion arrays, model checkpoints and geometry caches are excluded.
"""
    (args.output_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(f"PACKAGED 18 comparisons: {args.output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="Unique viewing directory for this training run")
    parser.add_argument("--qa_dir", type=Path)
    parser.add_argument("--stage", choices=("render", "package", "all"), default="render")
    parser.add_argument("--clip_id", action="append", default=[])
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--encoder_threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--decode_jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--worker_count", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.worker_index < args.worker_count:
        parser.error("Batch size must be positive and worker index must be within worker count")
    if (args.preview_only or args.clip_id or args.worker_count != 1) and args.stage != "render":
        parser.error("Preview, clip selection and worker partitioning require --stage render")
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    args.qa_dir = args.qa_dir.resolve() if args.qa_dir else args.output_dir.parent / f"{args.output_dir.name}_qa"
    _, clips = _manifest(args.manifest)
    if args.stage in ("render", "all"):
        requested = set(args.clip_id)
        if requested - {clip["clip_id"] for clip in clips}:
            parser.error("Unknown --clip_id")
        selected = [clip for clip in clips if not requested or clip["clip_id"] in requested]
        selected = selected[args.worker_index :: args.worker_count]
        renderer = GLRenderer(width=PANEL_WIDTH, height=PANEL_HEIGHT)
        print("RENDERER", renderer.renderer_info, flush=True)
        for clip in selected:
            _render_one(args, renderer, clip)
    if args.stage in ("package", "all"):
        _package(args, clips)


if __name__ == "__main__":
    main()
