# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package validated GraspXL comparison videos and measured training results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    """Return a streaming file digest."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=local / "runs/all18_v1")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    share = run / "share"
    report = json.loads((run / "report.json").read_text())
    evaluation = json.loads((run / "evaluation_final.json").read_text())
    verification = json.loads((run / "training_verification.json").read_text())
    audit = json.loads((run / "independent_audit.json").read_text())
    videos = json.loads((share / "validation.json").read_text())
    assert report["phase"] == "complete" and verification["passed"] and audit["completed"]
    assert videos["status"] == "passed" and len(videos["clips"]) == 18
    assert videos["compilation"]["frames"] == 2625
    assert len(list((share / "clips").glob("*.mp4"))) == 18
    for item in [*videos["clips"], videos["compilation"]]:
        assert digest(share / item["video_path"]) == item["sha256"]
    assert report["checkpoint_sha256"] == digest(run / "checkpoint.pth")
    before, after = (evaluation["aggregate"][side] for side in ("before", "after"))
    groups = {"improved": [], "unchanged": [], "worsened": []}
    for item in evaluation["per_clip"]:
        difference = item["after"]["vertices_deeper_than_2_mm"] - item["before"]["vertices_deeper_than_2_mm"]
        key = "improved" if difference < 0 else "worsened" if difference > 0 else "unchanged"
        groups[key].append(item["clip_id"])
    sampled_before, sampled_after = (audit["summary"][side] for side in ("before", "after"))
    before2, after2 = (item["vertices_deeper_than_2_mm"] for item in (before, after))
    before5, after5 = (item["vertices_deeper_than_5_mm"] for item in (before, after))
    exact_before2, exact_after2 = (item["vertices_deeper_than_2_mm"] for item in (sampled_before, sampled_after))
    exact_before_max, exact_after_max = (
        item["maximum_penetration_depth_mm"] for item in (sampled_before, sampled_after)
    )
    contact_retention = 100 * audit["summary"]["contact"]["retained_fraction"]
    matched = audit["subsets"]["matched_diverse_grasp"]
    tabletop = audit["subsets"]["supplemental_tabletop"]
    reduction = (
        100 * (1 - after["vertices_deeper_than_2_mm"] / before["vertices_deeper_than_2_mm"])
        if before["vertices_deeper_than_2_mm"]
        else None
    )
    summary = {
        "completed": True,
        "training_set_only": True,
        "clips": 18,
        "frames": 2625,
        "aggregate": evaluation["aggregate"],
        "clips_by_change_in_2_mm_count": groups,
        "percent_reduction_in_2_mm_count": reduction,
        "independent_sample": audit["summary"],
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    text = f"""# GraspXL mug refinement results

The released Text2HOI GRAB **refiner** was fine-tuned on all 18 selected GraspXL
mug clips: {report["training_steps"]:,} optimizer updates and
{report["epochs_requested"]} passes over every native frame. There are 15 matched
clips and 3 supplemental tabletop clips, with 2,625 frames in total.
The before geometry reproduces the original GraspXL MANO viewer geometry.

This adapts the refiner to recorded motions using collision/contact and motion
preservation losses. It does not train the text-to-motion generator, a SOMA
model, or an Isaac reinforcement-learning controller. Every clip appears in
training; these results do not measure performance on unseen data.

## Videos

Open `index.html` or `{videos["compilation"]["video_path"]}`. All 18 clips have
individual MP4 files and compilation chapters. Playback is 87.5 seconds at
30 FPS. All native frames are retained; the physical source frame rate is unknown.

Both columns use the same original right-hand MANO identity, mug geometry,
object trajectory, frame order, and camera. Only the hand motion is refined.
The top row shows the whole hand; the bottom follows the mug handle.

## Full-sequence approximate measurements

| Measurement | Original | Refined |
|---|---:|---:|
| Vertex-frame samples deeper than 2 mm | {before2:,} | {after2:,} |
| Vertex-frame samples deeper than 5 mm | {before5:,} | {after5:,} |
| Maximum estimated penetration | {before["maximum_penetration_mm"]:.3f} mm | {after["maximum_penetration_mm"]:.3f} mm |
| Mean hand-vertex adjustment | 0 mm | {after["mean_vertex_displacement_mm"]:.3f} mm |

On the 2 mm count, {len(groups["improved"])}/18 clips improved,
{len(groups["unchanged"])} were unchanged, and {len(groups["worsened"])} worsened.
Worsened clips: {", ".join(groups["worsened"]) or "none"}.
Detailed per-clip values, including contact proximity retention, are in
`evaluation_final.json`. A lower total does not imply every frame improved.

The same vertex can be counted in many frames. These totals are not counts of
distinct vertices, holes, or intersection volume. There are 778 right-hand
vertices per frame and 2,042,250 vertex-frame samples overall. Submillimeter
adjustments can be difficult to see, and opaque surfaces hide interior vertices.

Full-sequence metrics use a 0.75 mm signed-distance grid. Independent validation
found a maximum interpolation error of 0.372 mm on 1,000 probes, so values near
thresholds are approximate. Empty cup and handle space are not solid material.

## Independent exact geometry check

At fixed 25%, 50%, and 75% frame positions in every clip, exact triangle
distances and two ray directions found:

| Measurement on 54 sampled frames | Original | Refined |
|---|---:|---:|
| Vertex-frame samples deeper than 2 mm | {exact_before2:,} | {exact_after2:,} |
| Maximum penetration depth | {exact_before_max:.3f} mm | {exact_after_max:.3f} mm |

In these sampled frames, **{contact_retention:.2f}%** of source vertices initially
within 3 mm of the surface remain within 3 mm. The matched subset's count changes
{matched["before"]["vertices_deeper_than_2_mm"]} to {matched["after"]["vertices_deeper_than_2_mm"]},
while the tabletop subset changes {tabletop["before"]["vertices_deeper_than_2_mm"]}
to {tabletop["after"]["vertices_deeper_than_2_mm"]}. The tabletop sample therefore
worsens slightly despite a lower full-sequence approximate count. Retaining
proximity and reducing penetration are separate objectives; neither proves
stable contact under physical forces.

See `independent_audit.json` and its per-frame/per-clip CSV files for contact
retention, ray ambiguity exclusions, and matched/tabletop subset results.
Exact sampled checks and approximate full-sequence metrics cover different
frames and use different distance methods.

## Limits and reproducibility

The source mug has nonmanifold shared edges. A separate collision copy duplicates
11 vertex indices to represent continuous surface patches: **all 3,000 ordered
triangle coordinates remain bitwise identical**. No surface is moved, added,
deleted, or filled. Remaining folded triangles and ray ambiguities are documented
in `collision_proxy_validation.json`; a topology correction does not prove an
ideal physical solid.

Contact measurements are unsigned surface proximity, not verified forces or
contact labels. Penetration reduction alone does not establish a stable grasp.
No physics simulation, friction, gravity, or lift-success test is performed.
Original datasets and the downloaded checkpoint remain unchanged.

The trained checkpoint and source geometry are retained locally outside this
viewing package. `training_verification.json` checks actual changed weights,
finite losses, and complete frame coverage. Every video passed full decoding,
frame-count, and checksum checks recorded in `validation.json`.
"""
    (run / "RESULTS.md").write_text(text)
    (share / "RESULTS.md").write_text(text)
    for source, destination in {
        "report.json": "training_report.json",
        "summary.json": "summary.json",
        "evaluation_initial.json": "evaluation_initial.json",
        "evaluation_final.json": "evaluation_final.json",
        "training_verification.json": "training_verification.json",
        "independent_audit.json": "independent_audit.json",
        "independent_audit_frames.csv": "independent_audit_frames.csv",
        "independent_audit_clips.csv": "independent_audit_clips.csv",
    }.items():
        shutil.copyfile(run / source, share / destination)
    for name in ("sdf.json", "collision_proxy.json", "collision_proxy_validation.json"):
        shutil.copyfile(local / "prepared" / name, share / name)
    page = share / "index.html"
    html = page.read_text()
    if 'id="training-results"' not in html:
        html = html.replace(
            '<button class="all"',
            '<p id="training-results" class="notice"><strong>Measured training-set result:</strong> '
            f"{before['vertices_deeper_than_2_mm']:,} → {after['vertices_deeper_than_2_mm']:,} "
            f"vertex-frame samples deeper than 2 mm; {len(groups['improved'])}/18 clips improved, "
            f"{len(groups['worsened'])} worsened. Mean hand adjustment: "
            f"{after['mean_vertex_displacement_mm']:.3f} mm. "
            '<a href="RESULTS.md">Results and limitations</a> · '
            '<a href="independent_audit_clips.csv">Independent per-clip checks</a></p><button class="all"',
        )
        page.write_text(html)
    archive = local / "GraspXL_Mug_Text2HOI_Before_After.zip"
    temporary = archive.with_suffix(".partial.zip")
    files = sorted(path for path in share.rglob("*") if path.is_file())
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in files:
            assert path.suffix.lower() not in {".npz", ".npy", ".pkl", ".pth", ".ply", ".obj"}
            zipped.write(path, arcname=Path("GraspXL_Mug_Text2HOI_Before_After") / path.relative_to(share))
    with zipfile.ZipFile(temporary) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist()) == len(files)
    temporary.replace(archive)
    archive_hash = digest(archive)
    archive.with_suffix(".sha256").write_text(f"{archive_hash}  {archive.name}\n")
    completed = {
        "complete": True,
        "clips": 18,
        "frames": 2625,
        "training_updates": report["training_steps"],
        "share": str(share),
        "compilation": str(share / videos["compilation"]["video_path"]),
        "zip": str(archive),
        "zip_bytes": archive.stat().st_size,
        "zip_sha256": archive_hash,
        "files_in_zip": len(files),
        "zip_crc_passed": True,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "exact_audit_completed": True,
    }
    (run / "completion.json").write_text(json.dumps(completed, indent=2) + "\n")
    print(json.dumps(completed))


if __name__ == "__main__":
    main()
