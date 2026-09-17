# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression check for GRAB's passive object rotation and GraspXL's active one."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from smplx.lbs import batch_rodrigues

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mano_soma_viewer as viewer


def check(kind: str) -> None:
    """Compare actual viewer object geometry with each source convention [m]."""
    sequence = viewer.Sequence.__new__(viewer.Sequence)
    sequence.kind = kind
    sequence.device = "cpu"
    vertices = np.asarray([[0.03, 0.04, -0.02], [-0.02, 0.01, 0.05]], dtype=np.float32)
    rotvec = np.asarray([[0.3, -0.8, 1.2]], dtype=np.float32)
    translation = np.asarray([[0.1, -0.2, 0.3]], dtype=np.float32)
    zero_vertices = torch.zeros(1, 1, 3)
    sequence.motion = {"object_rot": rotvec, "object_trans": translation,
                       "poses": np.zeros((1, 25, 3)), "transl": translation,
                       "absolute_pose": np.asarray(True)}
    sequence.mesh = SimpleNamespace(vertices=vertices)
    sequence.original = lambda **kwargs: SimpleNamespace(vertices=zero_vertices)
    sequence.layer = SimpleNamespace(pose=lambda *args, **kwargs: {"vertices": zero_vertices})
    params = {name: np.zeros((1, 3), dtype=np.float32) for name in (
        "global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose",
        "leye_pose", "reye_pose", "expression", "transl")}
    sequence.source = {
        "body": np.asarray({"params": params}, dtype=object),
        "right_hand": {"rot": rotvec, "pose": np.zeros((1, 45)), "trans": translation},
    }
    with patch.object(viewer, "replay", return_value={"vertices": zero_vertices}):
        actual = sequence.frame(0)[2]
    rotation = batch_rodrigues(torch.from_numpy(rotvec))[0].numpy()
    # Official GRAB tools/objectmodel.py uses row vertices @ rotation.
    expected = vertices @ (rotation if kind == "GRAB" else rotation.T) + translation[0]
    error = float(np.abs(actual - expected).max())
    if error > 1e-7:
        raise AssertionError(f"{kind} object placement differs from source convention: {error:.9f} m")
    print(f"PASS {kind}: source object transform agrees within {error:.3g} m")


if __name__ == "__main__":
    check("GRAB")
    check("GraspXL")
