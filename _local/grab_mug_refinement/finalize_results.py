# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package validated videos and measurements without model or raw motion assets."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    local = Path(__file__).resolve().parent
    run = local / "runs/all44_v2"
    share = run / "share"
    report = json.loads((run / "report.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    audit = json.loads((run / "audit/exact_contact_audit.json").read_text())
    validation = json.loads((share / "validation.json").read_text())
    verification = json.loads((run / "training_verification.json").read_text())
    assert report["phase"] == "complete" and summary["completed"] and audit["completed"]
    assert validation["status"] == "passed"
    assert verification["passed"] and verification["training_updates"] == 3200
    assert len(validation["clips"]) == 44 and validation["compilation"]["frames"] == 13171
    assert len(list((share / "clips").glob("*.mp4"))) == 44
    for clip in validation["clips"]:
        assert digest(share / clip["video_path"]) == clip["sha256"]
    assert digest(share / validation["compilation"]["video_path"]) == validation["compilation"]["sha256"]
    result = summary["evaluation_summary_all_clips"]
    values = result["aggregate"]
    before, after = values["before"], values["after"]
    reduction2 = 100 * (1 - after["vertices_deeper_than_2_mm"] / before["vertices_deeper_than_2_mm"])
    reduction5 = 100 * (1 - after["vertices_deeper_than_5_mm"] / before["vertices_deeper_than_5_mm"])
    better = result["metrics"]["vertices_deeper_than_2_mm"]["improved"]["count"]
    worse = result["metrics"]["vertices_deeper_than_2_mm"]["worsened"]["clip_ids"]
    sampled = audit["summary"]
    exact_before = sampled["before"]["vertices_deeper_than_2_mm"]
    exact_after = sampled["after"]["vertices_deeper_than_2_mm"]
    exact_reduction = 100 * (1 - exact_after / exact_before)
    exact_max_before = sampled["before"]["maximum_penetration_depth_mm"]
    exact_max_after = sampled["after"]["maximum_penetration_depth_mm"]
    before2 = before["vertices_deeper_than_2_mm"]
    after2 = after["vertices_deeper_than_2_mm"]
    readme = f"""# GRAB mug training results

The Text2HOI GRAB **refiner** was fine-tuned on all 44 clips: 3,200 optimizer
updates, 16 passes covering every one of the 13,171 displayed source frames.
The motion generator was not trained. This is recorded-motion refinement with
locally adapted collision/contact losses, not new text-conditioned generation or
Isaac reinforcement learning.

## Comparison videos

Open `index.html`, a video in `clips/`, or `{validation["compilation"]["video_path"]}`.
The compilation is 7 minutes 19 seconds, with 44 chapters. Both columns retain
the same subject's personalized MANO anatomy, mug motion, timing and camera.
The top row shows both hands; the bottom follows the mug handle. Video captions
show approximate per-frame penetration counts and maximum depth.

## Full-sequence measurements

| Measurement | Before | After |
|---|---:|---:|
| Vertex samples deeper than 2 mm inside mug material | {before2:,} | {after2:,} |
| Vertex samples deeper than 5 mm | {before["vertices_deeper_than_5_mm"]:,} | {after["vertices_deeper_than_5_mm"]:,} |
| Maximum estimated depth | {before["maximum_penetration_mm"]:.2f} mm | {after["maximum_penetration_mm"]:.2f} mm |
| Mean hand-vertex adjustment | 0 mm | {after["mean_vertex_displacement_mm"]:.2f} mm |

This is a **{reduction2:.1f}% reduction** in samples deeper than 2 mm and a
**{reduction5:.1f}% reduction** deeper than 5 mm. {better}/44 clips improved on the
2 mm count. Three worsened: {", ".join(worse)}. Individual frames and maximum
depth can worsen even when a clip's total count improves. Intersections remain;
the worst depth is almost unchanged.

Across all frames, 91.39% of source vertices initially within 3 mm of the surface
remain within 3 mm afterward, and 99.94% remain within 5 mm. These are proximity
measurements, not verified force-bearing contact. Removing penetration alone does
not prove stable grasping.

Full-sequence values use a 0.75 mm signed-distance grid of the native hollow mug.
Empty cup and handle spaces are excluded from its solid. CPU video annotations
count one fewer baseline sample at the exact 2 mm threshold than GPU evaluation
(43,087 versus 43,088); this is floating-point threshold sensitivity.

## Independent audit

Exact native-triangle distances and two ray directions were checked at fixed
25%, 50%, and 75% positions in every clip: 132 sampled frames. Those checks found
{exact_before:,} versus {exact_after:,} vertex samples deeper than 2 mm
({exact_reduction:.1f}% fewer). Sampled maximum depth slightly worsened:
{exact_max_before:.3f} to {exact_max_after:.3f} mm. In this independent sample,
34 clips improved on the 2 mm count, 5 were unchanged, and 5 worsened.
90.8% of the same baseline near-surface vertices remained within 3 mm.
These sampled exact results and the full-video approximate results cover
different frame sets and use different distance calculations. See
`exact_contact_audit.json` and the two CSV files for per-frame and per-clip results.
Every exported frame also passed checks that object geometry, object poses,
source timing and baseline MANO geometry were preserved.

## Limits

All 44 clips were used in training. This is an **in-sample comparison**; it does
not establish performance on unseen data. It uses personalized standalone MANO
on both sides. Earlier MANO-versus-SOMA videos used SMPL-X hand crops on their
original side, which differ by about 1.0–1.2 mm on average (up to 9.52 mm).
Do not attribute that representation difference to this model's improvement.

The videos contain no SOMA conversion and no physics/collision response. Vertex
counts are not collision volume. No forces, friction, or gravity stability were
tested. Original datasets and downloaded checkpoints remain unchanged.
"""
    (run / "RESULTS.md").write_text(readme)
    (share / "RESULTS.md").write_text(readme)
    copies = {
        "report.json": "training_report.json",
        "summary.json": "summary.json",
        "evaluation_final.json": "evaluation_final.json",
        "training_verification.json": "training_verification.json",
        "audit/exact_contact_audit.json": "exact_contact_audit.json",
        "audit/exact_contact_frames.csv": "exact_contact_frames.csv",
        "audit/exact_contact_clips.csv": "exact_contact_clips.csv",
    }
    for source, destination in copies.items():
        shutil.copyfile(run / source, share / destination)
    shutil.copyfile(local / "prepared/sdf.json", share / "sdf_validation.json")
    index_path = share / "index.html"
    index = index_path.read_text()
    marker = '<p id="training-results"'
    if marker not in index:
        index = index.replace(
            '<button class="all"',
            '<p id="training-results" class="notice"><strong>Approximate full-video result:</strong> '
            f"{reduction2:.1f}% fewer vertex samples deeper than 2 mm; {better}/44 clips improved, "
            f"3 worsened. Mean hand adjustment: {after['mean_vertex_displacement_mm']:.2f} mm. "
            f"Independent 132-frame mesh check: {exact_reduction:.1f}% fewer such samples, "
            "with a slightly worse maximum depth. "
            '<a href="RESULTS.md">Results and limitations</a> · '
            '<a href="exact_contact_clips.csv">Independent per-clip audit</a></p><button class="all"',
        )
        index_path.write_text(index)
    archive = local / "GRAB_Mug_Text2HOI_Before_After.zip"
    temporary = archive.with_suffix(".partial.zip")
    files = sorted(path for path in share.rglob("*") if path.is_file())
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in files:
            assert path.suffix.lower() not in {".npz", ".npy", ".pkl", ".pth", ".ply", ".obj"}, path
            zipped.write(path, arcname=Path("GRAB_Mug_Text2HOI_Before_After") / path.relative_to(share))
    with zipfile.ZipFile(temporary) as zipped:
        assert zipped.testzip() is None
        assert len(zipped.namelist()) == len(files)
    temporary.replace(archive)
    archive_hash = digest(archive)
    archive.with_suffix(".sha256").write_text(f"{archive_hash}  {archive.name}\n")
    completion = {
        "complete": True,
        "clips": 44,
        "frames": 13171,
        "training_updates": 3200,
        "share": str(share),
        "compilation": str(share / validation["compilation"]["video_path"]),
        "zip": str(archive),
        "zip_bytes": archive.stat().st_size,
        "zip_sha256": archive_hash,
        "files_in_zip": len(files),
        "zip_crc_passed": True,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "exact_audit_completed": True,
    }
    (run / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion))


if __name__ == "__main__":
    main()
