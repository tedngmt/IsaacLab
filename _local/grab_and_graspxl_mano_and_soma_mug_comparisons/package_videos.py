# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate all mug comparison videos and build a portable viewing package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXPECTED_COUNTS = {"GRAB": 44, "GraspXL": 18}
VIDEO_FPS = 30
SIGNATURE_FIELDS = (
    "codec_name",
    "codec_tag_string",
    "profile",
    "level",
    "width",
    "height",
    "pix_fmt",
    "time_base",
    "r_frame_rate",
    "avg_frame_rate",
    "extradata_size",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a bounded multimedia command and preserve failures with their details."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{result.stderr[-10000:]}")
    return result


def sha256(path: Path) -> str:
    """Hash a file without loading the complete video into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_video(path: Path, expected_frames: int, full_decode: bool = True) -> dict:
    """Check H.264 format, exact frame count, and decoding of a comparison video."""
    probe = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )
    streams = [stream for stream in probe["streams"] if stream.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    video = streams[0]
    frame_count = video.get("nb_frames")
    if frame_count in (None, "N/A"):
        counted = json.loads(
            run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-threads",
                    "2",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "json",
                    str(path),
                ]
            ).stdout
        )
        frame_count = counted["streams"][0]["nb_read_frames"]
    actual = {
        "codec": video["codec_name"],
        "width": video["width"],
        "height": video["height"],
        "fps": float(Fraction(video["avg_frame_rate"])),
        "frames": int(frame_count),
    }
    expected = {"codec": "h264", "width": 1920, "height": 1080, "fps": VIDEO_FPS, "frames": expected_frames}
    if actual != expected:
        raise ValueError(f"Unexpected video metadata for {path}: {actual}; expected {expected}")
    expected_duration = expected_frames / VIDEO_FPS
    duration = float(video.get("duration", probe["format"]["duration"]))
    if abs(duration - expected_duration) > 1 / VIDEO_FPS + 0.001:
        raise ValueError(f"Unexpected duration for {path}: {duration}; expected {expected_duration}")
    stderr = ""
    if full_decode:
        result = run(
            [
                "ffmpeg",
                "-hide_banner",
                "-v",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-threads",
                "2",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-threads",
                "2",
                "-f",
                "null",
                "-",
            ]
        )
        stderr = result.stderr.strip()
        if stderr:
            raise ValueError(f"Decoder reported errors for {path}: {stderr[-10000:]}")
    return {
        **actual,
        "duration_seconds": duration,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "full_decode_passed": full_decode,
        "decode_error_log": stderr,
        "stream_signature": {key: video.get(key) for key in SIGNATURE_FIELDS},
    }


def validate_clip(dataset: str, clip: dict, share: Path) -> dict:
    """Validate one clip and attach source provenance using portable relative paths."""
    clip_id = clip["clip_id"]
    relative = Path(dataset) / f"{clip_id}.mp4"
    sidecar_relative = relative.with_suffix(".json")
    sidecar = share / sidecar_relative
    report = json.loads(sidecar.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"Expected a report object in {sidecar}")
    if report.get("completed") is not True or (report.get("dataset"), report.get("clip_id")) != (dataset, clip_id):
        raise ValueError(f"Render is incomplete or its identity does not match: {sidecar}")
    stride = 4 if dataset == "GRAB" else 1
    source_frames = clip["source_frames"]
    expected_frames = len(range(0, source_frames, stride))
    if report.get("source_frames") != source_frames or report.get("output_frames") != expected_frames:
        raise ValueError(f"Render report frame counts do not match the inventory: {sidecar}")
    metadata = inspect_video(share / relative, expected_frames)
    if report.get("video_sha256") != metadata["sha256"]:
        raise ValueError(f"Video hash does not match the completed render report: {sidecar}")
    soma_motion_path = Path(clip["soma_path"])
    if soma_motion_path.is_absolute():
        soma_motion_path = soma_motion_path.relative_to(BASE.parents[2])
    provenance = {
        "source_frames": source_frames,
        "source_fps": clip["source_fps"],
        "source_frame_stride": stride,
        "source_motion_path": clip.get("source_motion_path", clip.get("mano_path", clip["soma_path"])),
        "source_motion_base": clip.get("source_motion_base", "source_dataset_root"),
        "soma_motion_path": soma_motion_path.as_posix(),
        "soma_motion_base": clip.get("soma_motion_base", "soma_dataset_root"),
        "source_member": clip["source_member"],
        "source_archive": clip.get("source_archive", Path(clip.get("source_path", "")).name),
        "source_duration_seconds": clip["source_duration_seconds"],
        "playback_fps_assumed": dataset == "GraspXL",
        "conversion_note": clip.get("conversion_note", "Existing paired SOMA conversion"),
    }
    return {
        "dataset": dataset,
        "clip_id": clip_id,
        "video_path": relative.as_posix(),
        "report_path": sidecar_relative.as_posix(),
        "report_sha256": sha256(sidecar),
        "subject": clip.get("subject"),
        "action": clip.get("action"),
        "supplemental": bool(clip.get("supplemental", False)),
        "source": provenance,
        **metadata,
    }


def make_compilation(dataset: str, clips: list[dict], share: Path) -> dict:
    """Concatenate identically encoded clips without changing their image quality."""
    signatures = {json.dumps(clip["stream_signature"], sort_keys=True) for clip in clips}
    if len(signatures) != 1:
        raise ValueError(f"Video stream parameters differ within {dataset}; cannot safely concatenate")
    output = share / f"{dataset}_all_mugs.mp4"
    partial = share / f"{dataset}_all_mugs.partial.mp4"
    cursor = 0
    chapters = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffmetadata", delete=False) as stream:
        metadata_path = Path(stream.name)
        stream.write(";FFMETADATA1\n")
        for clip in clips:
            end_frame = cursor + clip["frames"]
            chapter_title = clip["clip_id"] + (" (tabletop supplement)" if clip["supplemental"] else "")
            if any(character in chapter_title for character in "\\\n\r=;#"):
                raise ValueError(f"Unexpected special character in chapter title: {chapter_title}")
            stream.write(f"[CHAPTER]\nTIMEBASE=1/{VIDEO_FPS}\nSTART={cursor}\nEND={end_frame}\n")
            stream.write(f"title={chapter_title}\n")
            chapters.append(
                {
                    "clip_id": clip["clip_id"],
                    "title": chapter_title,
                    "start_frame": cursor,
                    "start_seconds": cursor / VIDEO_FPS,
                    "end_seconds": end_frame / VIDEO_FPS,
                }
            )
            cursor = end_frame
    # Generated identifiers contain no quotes; still quote paths according to ffconcat syntax.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as stream:
        concat_path = Path(stream.name)
        stream.write("ffconcat version 1.0\n")
        for clip in clips:
            filename = str((share / clip["video_path"]).resolve()).replace("'", "'\\''")
            stream.write(f"file '{filename}'\n")
            # MP4 container durations may be rounded to milliseconds. Use the
            # exact frame duration to prevent timestamp gaps between clips.
            stream.write(f"duration {clip['frames'] / VIDEO_FPS:.12f}\n")
    try:
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-f",
                "ffmetadata",
                "-i",
                str(metadata_path),
                "-map",
                "0:v:0",
                "-map_metadata",
                "1",
                "-map_chapters",
                "1",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(partial),
            ]
        )
        metadata = inspect_video(partial, sum(clip["frames"] for clip in clips))
        native_chapters = json.loads(
            run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_chapters",
                    "-of",
                    "json",
                    str(partial),
                ]
            ).stdout
        )["chapters"]
        if len(native_chapters) != len(chapters):
            raise ValueError(f"Compilation has an unexpected number of native chapters: {dataset}")
        for native, chapter in zip(native_chapters, chapters):
            if native.get("tags", {}).get("title") != chapter["title"]:
                raise ValueError(f"Native chapter title does not match: {dataset} / {chapter['clip_id']}")
            for bound in ("start", "end"):
                if abs(float(native[f"{bound}_time"]) - chapter[f"{bound}_seconds"]) > 0.001:
                    raise ValueError(f"Native chapter timing does not match: {dataset} / {chapter['clip_id']}")
        metadata["native_chapters_verified"] = True
        partial.replace(output)
    finally:
        concat_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    return {"dataset": dataset, "video_path": output.name, "clip_count": len(clips), "chapters": chapters, **metadata}


def format_duration(seconds: float) -> str:
    """Format an approximate viewing duration."""
    minutes, remaining = divmod(round(seconds), 60)
    return f"{minutes}m {remaining:02d}s"


def write_readme(share: Path, validation: dict) -> None:
    """Document video interpretation without including licensed model or motion files."""
    compilations = {item["dataset"]: item for item in validation["compilations"]}
    count_grab = compilations["GRAB"]["clip_count"]
    count_graspxl = compilations["GraspXL"]["clip_count"]
    count_supplemental = sum(item["supplemental"] for item in validation["clips"])
    count_matched = count_graspxl - count_supplemental
    content = f"""MUG MOTIONS — ORIGINAL AND SOMA COMPARISONS
Generated {validation["created_at_utc"]}

Open index.html in a browser, or play either combined video:
  GRAB_all_mugs.mp4: {count_grab} clips, {format_duration(compilations["GRAB"]["duration_seconds"])}
  GraspXL_all_mugs.mp4: {count_graspxl} clips, {format_duration(compilations["GraspXL"]["duration_seconds"])}
Individual MP4 files are in GRAB/ and GraspXL/. Each has a JSON render report.
The index runs offline and uses only relative paths. Keep this folder together.
The combined videos have native chapters for players that support chapter menus.

WHAT THE PANELS SHOW
Left: original source model. Right: fitted SOMA conversion of the same motion.
Top: hands-only overview in world coordinates, including the moving object.
GRAB bottom: the camera follows the handle in world space while the mug keeps
its original trajectory and tilt. The mug is not rotated into a stationary
object frame; original and SOMA share the same moving camera at every frame.
GraspXL bottom: the existing object-stabilized close-up keeps the mug stationary
to inspect fingers and the handle. This GraspXL view is unchanged.
Both close-ups may crop hands away from the mug; the overview retains them.
The rows show the same source frames from different cameras. Camera tracking
changes the view, not the underlying hand motion or object trajectory.
Corresponding original/SOMA panels share camera, scale, object, and time.

GRAB
All {count_grab} strict object='mug' clips from the original ten-subject GRAB archives,
paired with their SOMA conversions. The original full-body model is SMPL-X;
its hand surfaces are cropped for display. The label 'SMPL-X hands' is accurate;
these exports do not independently reconstruct a standalone MANO hand model.
Both hands are retained. Body, face, arms, and table are hidden.
Source rate: 120 FPS. Every fourth source frame is shown at 30 FPS, preserving
real-time speed; duration rounding at the end of each clip is less than 1/30 s.
Actions: 20 drink, 6 lift, 3 offhand, 9 pass, and 6 toast clips.
Correction: GRAB object rotations now follow the official GRAB row-vector
convention. Earlier GRAB exports used the opposite rotation and are superseded.
This is a playback correction, not an edit to the dataset or a contact repair.
GraspXL playback is unchanged.

CORRECTION TO EARLIER GRAB EXPORTS
These GRAB videos replace earlier exports that applied the mug rotation in the
wrong direction. Object positions now follow the official GRAB row-vector
convention: vertices @ Rodrigues(source_rotation) + translation. The source
motion and hand reconstruction were not edited. This fixes the viewer's object
alignment; it is not a physical contact repair. GraspXL videos are unchanged.
Playback displays stored positions without collision response. Small overlaps
can remain in the original captured fits and in their SOMA conversions.

GRASPXL
All {count_matched} mug sequences in the local matched dataset: three diverse-grasp
source archives, five sequences each, 155 frames per clip. Also included are
{count_supplemental} supplemental tabletop clips for the same mug, 100 frames each;
their SOMA conversions were fitted locally for this video export. These extras
are labeled tabletop supplements and were not in the existing paired dataset.
Original right-hand MANO versus fitted SOMA; every source frame is shown.
Scope: the selected large mug, UID 12652c6e0aaf44dda32aab816f433baf.
Other mug identities in the complete raw GraspXL archives are outside this
selected-mug comparison. The fitting of the extras is a format conversion,
not Text2HOI or reinforcement-learning training.
Playback is 30 FPS (ASSUMED): the source data does not record its frame rate.
The resulting video duration is not a measured real-world motion duration.

INTERPRETATION
These are source-dataset motions and their conversions, not Text2HOI-generated
motions or a demonstration of IsaacLab physics or reinforcement learning.
The GRAB and GraspXL mugs have different mesh geometry, and their motions have
different origins. Inspect handle use qualitatively; these videos alone do not
establish which model is more accurate or which dataset trains a better policy.
SOMA is an approximate surface fit. Small differences are not contact scores.
Captured and fitted surfaces can still overlap. Playback has no collision
response; apparent intersections alone do not establish dataset quality.

FORMAT AND VERIFICATION
MP4 / H.264, 1920 x 1080, 30 FPS. All {validation["validated_clip_count"]} videos and both compilations
were checked for expected frame count, codec, size, rate, and full-file decode.
validation.json records checks, source paths relative to their dataset roots,
durations, file sizes, and SHA-256 hashes. Compilation chapter offsets are also
recorded there. No licensed raw datasets, hand/body model assets, or meshes are
included in this viewing folder.
"""
    (share / "README.txt").write_text(content, encoding="utf-8")


def write_index(share: Path, validation: dict) -> None:
    """Build a self-contained offline clip browser with native video playback."""
    data = {
        "clips": [
            {
                key: item[key]
                for key in (
                    "dataset",
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
        "compilations": [
            {key: item[key] for key in ("dataset", "video_path", "clip_count", "duration_seconds")}
            for item in validation["compilations"]
        ],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mug motions · Original and SOMA</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#101722;color:#eaf0fa}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:32px 28px 48px}
h1{font-size:clamp(24px,3vw,36px);margin:0 0 10px;letter-spacing:-.04em}p{line-height:1.55}
.intro{color:#b7c6dc;max-width:1100px;margin:0 0 20px}.actions{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 22px}
button,select,input{font:inherit;color:inherit;background:#1b293b;
border:1px solid #425775;border-radius:8px;padding:10px 13px}
button{cursor:pointer}button:hover,button:focus-visible{background:#2a4565}a{color:#a5d3ff}a:hover{color:white}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:22px}
.player{background:#172230;border:1px solid #34455a;border-radius:12px;overflow:hidden}
video{display:block;width:100%;aspect-ratio:16/9;background:#080d14}
.caption{padding:16px 18px}.caption h2{font-size:18px;margin:0 0 7px}
.meta{color:#b7c6dc;font-size:14px}.links{display:flex;gap:18px;margin-top:12px;flex-wrap:wrap}
.notice{padding:14px 17px;border-left:3px solid #6ca5df;background:#192739;color:#cdd9ec;font-size:14px}
aside{min-width:0}.filters{display:grid;gap:10px}input,select{width:100%}
.count{color:#aebed4;font-size:14px;margin:12px 0}
.clips{display:flex;flex-direction:column;gap:7px;max-height:660px;overflow-y:auto;padding-right:5px}
.clip{text-align:left;line-height:1.45;padding:10px 12px;font-size:14px}
.clip span{display:block;font-size:12px;color:#aec1d8}
.clip[aria-current="true"]{background:#244665;border-color:#78bbf6}
.footer{color:#9fb0c8;font-size:13px;margin-top:22px}
@media(max-width:900px){main{padding:22px 14px}.layout{grid-template-columns:1fr}.clips{max-height:360px}}
</style></head><body><main>
<h1>Mug motions: original and SOMA</h1>
<p class="intro">__TOTAL__ synchronized comparisons: all __GRAB_COUNT__ GRAB mug clips and
__GRASPXL_COUNT__ clips for your selected GraspXL mug (__MATCHED_COUNT__ matched clips plus
__SUPPLEMENTAL_COUNT__ clearly labeled tabletop supplements with new SOMA conversions).
Original on the left; fitted SOMA on the right. The top row shows the hands in world coordinates;
the bottom row shows the grip and handle. GRAB uses a camera that follows the handle in world space;
GraspXL keeps its existing object-stabilized close-up.</p>
<div class="actions" id="compilations"></div>
<div class="layout"><section>
<div class="player"><video id="video" controls playsinline preload="metadata"></video><div class="caption">
<h2 id="title">Select a clip</h2><div class="meta" id="meta"></div><div class="links">
<a id="download" download>Save video</a><a id="report">Render report</a><a href="README.txt">About these comparisons</a>
</div></div></div>
<p class="notice"><strong>Reading the views:</strong> GRAB shows cropped SMPL-X hands; GraspXL uses MANO.
In GRAB's close-up, the camera follows the handle while the mug retains its original trajectory and tilt.
GraspXL's close-up keeps the mug stationary. Original and SOMA share the same camera at each source frame.
Both close-ups may crop hands away from the mug; the overview retains them.
GRAB plays at real-time speed (120 FPS sampled to 30 FPS). GraspXL plays at an assumed 30 FPS because
its source rate is unknown.</p>
<p class="notice">The two datasets use different mug meshes and source motions. Handle use can be compared visually,
but these videos do not establish model accuracy. They show the original datasets and SOMA conversions,
not generated Text2HOI motion or IsaacLab physics. Captured and fitted surfaces may overlap;
playback does not apply collision response.</p>
<p class="notice">GRAB object rotations now follow the official GRAB row-vector convention.
These corrected views supersede the earlier GRAB exports. This fixes playback, without changing the
source dataset or repairing contacts. GraspXL playback is unchanged.</p>
<p class="notice">Updated GRAB videos correct a mug-rotation error in earlier exports.
Use this revised collection when assessing hand-object alignment. GraspXL videos are unchanged.</p>
</section><aside><div class="filters"><label>Dataset<select id="dataset"><option value="">Both datasets</option>
<option>GRAB</option><option>GraspXL</option></select></label><label>Find a clip<input id="search" type="search"
placeholder="Subject, action, or sequence…"></label></div><div class="count" id="count"></div>
<div class="clips" id="clips"></div></aside></div>
<p class="footer">Offline viewing · 1920 × 1080 · H.264 MP4 · Keep this folder together.
<a href="validation.json">Validation and SHA-256 hashes</a></p>
</main><script>
const data=__DATA__;
const video=document.getElementById('video'), title=document.getElementById('title');
const meta=document.getElementById('meta');
const download=document.getElementById('download'), report=document.getElementById('report');
const dataset=document.getElementById('dataset'), search=document.getElementById('search');
const list=document.getElementById('clips');
let selected='';
function duration(seconds){const total=Math.round(seconds);
return Math.floor(total/60)+':'+String(total%60).padStart(2,'0');}
function select(item,play){selected=item.video_path;video.src=item.video_path;
title.textContent=item.clip_id?item.dataset+' / '+item.clip_id:item.dataset+' — all mug clips';
const counts=item.clip_count?item.clip_count+' clips':item.frames+' frames';
const label=item.dataset==='GRAB'?'Original SMPL-X hands vs SOMA · real-time':'Original MANO vs SOMA · 30 FPS assumed';
meta.textContent=duration(item.duration_seconds)+' · '+counts+' · '+label;
download.href=item.video_path;report.hidden=!item.report_path;if(item.report_path)report.href=item.report_path;
render();if(play)video.play().catch(()=>{});}
function render(){const query=search.value.toLowerCase().trim();
const visible=data.clips.filter(item=>(!dataset.value||item.dataset===dataset.value)&&
(!query||[item.dataset,item.clip_id,item.subject,item.action].join(' ').toLowerCase().includes(query)));
document.getElementById('count').textContent=visible.length+' of '+data.clips.length+' clips';list.replaceChildren();
for(const item of visible){const button=document.createElement('button');button.className='clip';
button.setAttribute('aria-current',String(selected===item.video_path));
button.textContent=item.clip_id;const detail=document.createElement('span');
detail.textContent=item.dataset+' · '+duration(item.duration_seconds)+(item.action?' · '+item.action:'')+
(item.supplemental?' · tabletop supplement':'');
button.append(detail);button.addEventListener('click',()=>select(item,true));list.append(button);}}
for(const item of data.compilations){const button=document.createElement('button');
button.textContent='Play all '+item.dataset+' · '+item.clip_count+' clips · '+duration(item.duration_seconds);
button.addEventListener('click',()=>select(item,true));document.getElementById('compilations').append(button);}
dataset.addEventListener('change',render);search.addEventListener('input',render);select(data.clips[0],false);
</script></body></html>
"""
    supplemental_count = sum(item["supplemental"] for item in validation["clips"])
    replacements = {
        "__DATA__": payload,
        "__TOTAL__": str(validation["validated_clip_count"]),
        "__GRAB_COUNT__": str(validation["expected_clip_counts"]["GRAB"]),
        "__GRASPXL_COUNT__": str(validation["expected_clip_counts"]["GraspXL"]),
        "__SUPPLEMENTAL_COUNT__": str(supplemental_count),
        "__MATCHED_COUNT__": str(validation["expected_clip_counts"]["GraspXL"] - supplemental_count),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    (share / "index.html").write_text(template, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=BASE / "inventory.json")
    parser.add_argument("--share_dir", type=Path, default=BASE / "share")
    parser.add_argument("--decode_jobs", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            parser.error(f"{executable} must be available on PATH")
    inventory = json.loads(args.inventory.read_text())
    total_expected = sum(EXPECTED_COUNTS.values())
    share = args.share_dir.resolve()
    tasks = []
    missing = []
    for dataset, expected in EXPECTED_COUNTS.items():
        clips = inventory["datasets"][dataset]["clips"]
        if len(clips) != expected:
            raise ValueError(f"Expected {expected} {dataset} clips, found {len(clips)}")
        for clip in clips:
            tasks.append((dataset, clip))
            for suffix in (".mp4", ".json"):
                path = share / dataset / f"{clip['clip_id']}{suffix}"
                if not path.is_file():
                    missing.append(str(path.relative_to(share)))
    if missing:
        raise FileNotFoundError(
            f"All {total_expected} renders must be complete; missing {len(missing)} files:\n" + "\n".join(missing)
        )
    results = {}
    with ThreadPoolExecutor(max_workers=args.decode_jobs) as pool:
        futures = {
            pool.submit(validate_clip, dataset, clip, share): (dataset, clip["clip_id"]) for dataset, clip in tasks
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            print(f"Validated {len(results):02d}/{total_expected}: {key[0]}/{key[1]}", flush=True)
    ordered = [results[(dataset, clip["clip_id"])] for dataset, clip in tasks]
    compilations = []
    for dataset in EXPECTED_COUNTS:
        print(f"Building and validating {dataset} compilation", flush=True)
        compilations.append(make_compilation(dataset, [item for item in ordered if item["dataset"] == dataset], share))
    validation = {
        "schema_version": 1,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_clip_counts": EXPECTED_COUNTS,
        "validated_clip_count": len(ordered),
        "missing_clips": [],
        "inventory_sha256": sha256(args.inventory),
        "decode_workers": args.decode_jobs,
        "decode_threads_per_worker": 2,
        "source_roots": {
            dataset: {key: Path(inventory["datasets"][dataset][key]).name for key in ("source_root", "soma_root")}
            for dataset in EXPECTED_COUNTS
        },
        "clips": ordered,
        "compilations": compilations,
    }
    (share / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    write_readme(share, validation)
    write_index(share, validation)
    print(f"PASS: {len(ordered)} clips, 2 compilations, offline index and validation report: {share}", flush=True)


if __name__ == "__main__":
    main()
