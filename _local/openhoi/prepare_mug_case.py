# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the trusted local GraspXL mug sources and save an OpenHOI case inventory."""

import hashlib
import json
import pickletools
import re
from pathlib import Path

import numpy as np


def main() -> None:
    """Validate source arrays and pairing; this does not run a generative model."""
    setup_root = Path(__file__).resolve().parent
    workspace = setup_root.parents[1]
    mano_root = workspace.parent / "GraspXL_MANO_51Objects_GRABMatched"
    soma_root = workspace.parent / "GraspXL_SOMA_51Objects_GRABMatched"
    manifest = json.loads((mano_root / "manifest.json").read_text(encoding="utf-8"))
    pairs = json.loads((mano_root / "metadata/pairs.json").read_text(encoding="utf-8"))
    (mug,) = [entry for entry in manifest["objects"] if entry["grab_object"] == "mug"]
    mug_pairs = [entry for entry in pairs if entry["uid"] == mug["uid"]]
    if {entry["mano_path"] for entry in mug_pairs} != set(mug["motion_paths"]):
        raise ValueError("Mug sequence paths differ between source metadata files.")

    for pair in mug_pairs:
        mano_path = mano_root / pair["mano_path"]
        soma_path = soma_root / pair["soma_path"]
        for path, hash_key in ((mano_path, "mano_sha256"), (soma_path, "soma_sha256")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != pair[hash_key]:
                raise ValueError(f"Source hash mismatch: {path}")
        # These are the user's trusted original GraspXL dictionaries.
        source = np.load(mano_path, allow_pickle=True).item()
        for name, fields in (
            ("right_hand", {"rot": 3, "trans": 3, "pose": 45}),
            (mug["uid"], {"rot": 3, "trans": 3}),
        ):
            for field, width in fields.items():
                array = np.asarray(source[name][field])
                if array.shape != (pair["frames"], width) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid array: {mano_path}, {name}/{field}")
        with np.load(soma_path, allow_pickle=False) as converted:
            if len(converted["poses"]) != pair["frames"]:
                raise ValueError(f"Paired frame count mismatch: {soma_path}")
            for source_field, converted_field in (("rot", "object_rot"), ("trans", "object_trans")):
                if not np.array_equal(source[mug["uid"]][source_field], converted[converted_field]):
                    raise ValueError(f"Paired object trajectory mismatch: {soma_path}")

    mesh = mano_root / mug["mesh_path"]
    if hashlib.sha256(mesh.read_bytes()).hexdigest() != mug["mesh_sha256"]:
        raise ValueError("Mug mesh hash mismatch.")
    cache = setup_root / "repo/Affordance-DrivenHOIDiffusion/afford/data_grab.pkl"
    # Read pickle opcodes without deserializing the upstream object graph.
    cached_mug_strings = sorted(
        {
            value
            for _, value, _ in pickletools.genops(cache.read_bytes())
            if isinstance(value, str) and len(value) < 256 and re.search(r"\bmug\b", value.lower())
        }
    )
    benchmark_path = workspace / "_local/benchmarks/graspxl_openhoi_text2hoi_v1.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    (split_entry,) = [entry for entry in benchmark["objects"] if entry["uid"] == mug["uid"]]
    report = {
        "status": "source_inventory_verified_model_inference_pending",
        "scope": (
            "Hash, finite-array, frame-count, and paired object-trajectory checks only; "
            "no contact or model validation."
        ),
        "model": "OpenHOI",
        "object": "mug",
        "source_uid": mug["uid"],
        "source_object_key": mug["object_key"],
        "benchmark_split": split_entry["split"],
        "sequence_count": len(mug_pairs),
        "frame_count": sum(pair["frames"] for pair in mug_pairs),
        "source_fps": manifest["fps"],
        "unit": manifest["unit"],
        "mano_convention": manifest["mano_convention"],
        "proposed_prompt": "Lift mug with right hand.",
        "cached_grab_mug_text_strings": cached_mug_strings,
        "cached_affordance_warning": (
            "GRAB cached maps are tied to their sampled mesh points, not the different GraspXL mug."
        ),
        "mesh_path_relative_to_mano_package": mug["mesh_path"],
        "mesh_sha256": mug["mesh_sha256"],
        "pairs": mug_pairs,
    }
    output = setup_root / "mug_case.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key not in {"pairs", "mano_convention"}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
