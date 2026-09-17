# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the corrected GRAB revision using two disjoint, balanced clip groups."""

import json
import subprocess
from pathlib import Path


def main():
    revision = Path(__file__).resolve().parent
    base = revision.parents[1]
    inventory = json.loads((base / "inventory.json").read_text())
    groups = [[], []]
    frames = [0, 0]
    for clip in sorted(inventory["datasets"]["GRAB"]["clips"], key=lambda item: -item["source_frames"]):
        group = min(range(2), key=frames.__getitem__)
        groups[group].append(clip["clip_id"])
        frames[group] += (clip["source_frames"] + 3) // 4
    assert len(set(groups[0]) | set(groups[1])) == 44
    assert not set(groups[0]) & set(groups[1])
    (revision / "render_groups.json").write_text(json.dumps({"clips": groups, "frames": frames}, indent=2) + "\n")
    jobs, logs = [], []
    try:
        for index, clips in enumerate(groups):
            command = [
                "/home/nmt/miniconda3/envs/text2hoi/bin/python",
                "-u",
                str(base / "render_videos.py"),
                "--dataset",
                "GRAB",
                "--output_root",
                str(revision / "share"),
                "--qa_root",
                str(revision / "qa"),
            ]
            for clip_id in clips:
                command += ["--clip_id", clip_id]
            log = (revision / f"render_group_{index}.log").open("w")
            logs.append(log)
            jobs.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
            print(f"Started group {index}: {len(clips)} clips, {frames[index]} frames", flush=True)
        codes = [job.wait() for job in jobs]
        if any(codes):
            raise RuntimeError(f"Rendering failed; inspect per-group logs: {codes}")
        print("All 44 corrected GRAB clips rendered.", flush=True)
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
                job.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
