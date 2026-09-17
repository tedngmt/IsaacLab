# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one headless GraspXL MANO policy episode and optionally check anatomy inference.

Run with the isolated GraspXL environment's Python. No policy updates, viewer,
training logs, or motion exports are created. Use an external ``timeout`` as well
if a hard deadline must interrupt native simulator code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path


def _check_mano() -> dict[str, object]:
    """Check the CUDA MANO forward pass and anatomy loss used by the runners."""
    import torch
    from manotorch.anatomy_loss import AnatomyConstraintLossEE
    from raisimGymTorch.helper.mano_amano import PoseTrans

    pose_transform = PoseTrans()
    anatomy_loss = AnatomyConstraintLossEE(reduction="none")
    anatomy_loss.setup()
    with torch.no_grad():
        output = pose_transform.mano_layer(torch.zeros((1, 48), device="cuda"), torch.zeros((1, 10), device="cuda"))
        _, _, euler_angles = pose_transform.axisFK(output.transforms_abs)
        loss = anatomy_loss(euler_angles)
    for name, tensor in (("vertices", output.verts), ("joints", output.joints), ("anatomy_loss", loss)):
        assert tensor.is_cuda, f"MANO {name} is not on CUDA"
        assert torch.isfinite(tensor).all().item(), f"MANO {name} contains nonfinite values"
    assert tuple(output.verts.shape) == (1, 778, 3), output.verts.shape
    assert tuple(loss.shape) == (1, 15), loss.shape
    return {
        "passed": True,
        "vertices_shape": list(output.verts.shape),
        "joints_shape": list(output.joints.shape),
        "anatomy_loss_sum": float(loss.sum().item()),
    }


def _rollout(args: argparse.Namespace) -> dict[str, object]:
    """Reproduce demo initialization and 140 deterministic policy steps."""
    project = args.graspxl_root.expanduser().resolve()
    os.chdir(project / "raisimGymTorch")  # The upstream wrapper resolves meshes relative to this directory.
    sys.path.insert(0, str(project / "raisimGymTorch"))

    import numpy as np
    import raisimGymTorch.algo.ppo.module as ppo_module
    import torch
    import torch.nn as nn
    from raisimGymTorch.env.bin import ours_demo as mano
    from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
    from raisimGymTorch.helper import rotations
    from raisimGymTorch.helper.initial_pose_final import get_initial_pose
    from ruamel.yaml import YAML, RoundTripDumper, dump

    assert torch.cuda.is_available(), "The upstream GraspXL observation wrapper requires CUDA"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda")

    config_path = project / "raisimGymTorch/raisimGymTorch/env/envs/ours_demo/cfgs/cfg_reg.yaml"
    with config_path.open() as stream:
        cfg = YAML().load(stream)
    cfg["environment"].update(
        {"render": False, "visualize": False, "num_envs": 1, "num_threads": 1, "load_set": "mixed_train"}
    )
    object_dir = project / "rsc/mixed_train"
    candidates = sorted(path for path in object_dir.iterdir() if path.is_dir())
    assert candidates, f"No objects found in {object_dir}"
    mugs = [path for path in candidates if path.name.startswith("Mug")]
    selected = (mugs or candidates)[0]
    obj_list = [selected.name]

    raw_env = mano.RaisimGymEnv(str(project / "rsc"), dump(cfg["environment"], Dumper=RoundTripDumper))
    try:
        dimensions = {
            "num_envs": raw_env.getNumOfEnvs(),
            "raw_observation_right": raw_env.getRightObDim(),
            "raw_observation_left": raw_env.getLeftObDim(),
            "action": raw_env.getActionDim(),
            "global_state": raw_env.getGSDim(),
        }
        assert dimensions == {
            "num_envs": 1,
            "raw_observation_right": 178,
            "raw_observation_left": 1,
            "action": 51,
            "global_state": 196,
        }, dimensions
        env = VecEnv(obj_list, raw_env, cfg["environment"], cat_name="mixed_train")
        env.seed(args.seed)
        env.load_multi_articulated([f"{selected.name}/{selected.name}.urdf"])

        # Match demo.py's actor and deterministic action path, without constructing PPO.
        actor = ppo_module.Actor(
            ppo_module.MLP(cfg["architecture"]["policy_net"], nn.LeakyReLU, 304, 51),
            ppo_module.MultivariateGaussianDiagonalCovariance(51, 1, 1.0, mano.NormalSampler(51)),
            device,
        )
        checkpoint_path = project / "raisimGymTorch/data_all/floating_mixed/full_5600_r.pt"
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        actor.architecture.load_state_dict(checkpoint["actor_architecture_state_dict"])
        actor.distribution.load_state_dict(checkpoint["actor_distribution_state_dict"])
        actor.architecture.eval()
        env.load_scaling(str(checkpoint_path.parent), "5600")

        # Exact object, finger, wrist, and goal initialization from ours_demo/demo.py.
        pose_mean = np.loadtxt(project / "rsc/mano_double/right_pose_mean.txt")
        qpos_right = np.zeros((1, 51), dtype="float32")
        qpos_left = np.zeros((1, 51), dtype="float32")
        obj_pose = np.zeros((1, 8), dtype="float32")
        target_center = np.zeros_like(env.affordance_center)
        object_center = np.zeros_like(env.affordance_center)
        contain_non_aff = np.zeros((1, 1), dtype="float32")
        lowest_point = float((selected / "lowest_point_new.txt").read_text())
        obj_pose[0, :] = [1.0, -0.0, 0.502, 1.0, -0.0, -0.0, 0.0, 0.0]
        obj_pose[0, 2] -= lowest_point
        qpos_right[0, 6:] = pose_mean.copy() / 5
        qpos_right[0, -9:-6] *= 5
        qpos_right[0, -8] += 0.4
        qpos_right[0, -7] += 0.6
        fake_non_aff_center = np.array([0.346408, 0.346408, 0.346408])
        if np.linalg.norm(env.non_aff_mesh[0].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
        else:
            non_aff_mesh = env.non_aff_mesh[0]
            contain_non_aff[0, 0] = 1.0
        rot, pos, bias = get_initial_pose(env.aff_mesh[0], non_aff_mesh)
        obj_mat = rotations.quat2mat(obj_pose[0, 3:7])
        wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
        wrist_mat = rotations.euler2mat(wrist_pose_obj)
        wrist_pose = rotations.mat2euler(np.matmul(obj_mat, wrist_mat))
        qpos_right[0, :3] = obj_pose[0, :3] + np.matmul(obj_mat, pos[0, :])
        qpos_right[0, 3:6] = wrist_pose[0, :]
        target_center[0, :] = bias[:]
        object_center[0, :] = env.affordance_center[0]
        assert np.isfinite(qpos_right).all(), "Nonfinite initial hand pose"
        assert np.isfinite(target_center).all(), "Nonfinite grasp target"
        env.reset_state(qpos_right, qpos_left, np.zeros_like(qpos_right), np.zeros_like(qpos_left), obj_pose)
        env.set_goals(target_center, object_center, *(np.zeros((1, 1), dtype="float32") for _ in range(8)))
        # reset_state caches observations before set_goals initializes hand_center.
        # Repeat the same reset after goals to refresh the initial grasp axis;
        # otherwise global-state entries 132:135 retain zero-normalization NaNs.
        # This refresh does not integrate physics or change the requested pose.
        env.reset_state(qpos_right, qpos_left, np.zeros_like(qpos_right), np.zeros_like(qpos_left), obj_pose)
        observation, distances = env.observe(contain_non_aff, partial_obs=False)
        initial_state = env.get_global_state().copy()
        assert observation.shape == (1, 304), observation.shape
        for name, value in (("initial_observation", observation), ("initial_state", initial_state)):
            assert np.isfinite(value).all(), f"Nonfinite {name} at indices {np.argwhere(~np.isfinite(value)).tolist()}"

        reward_sum = 0.0
        max_wrist_displacement = 0.0
        max_joint_change = 0.0
        max_object_displacement = 0.0
        max_action_abs = 0.0
        with torch.no_grad():
            for step in range(140):
                if step == 60:
                    env.switch_root_guidance(True)
                action = (
                    actor.architecture.architecture(torch.from_numpy(observation.astype("float32")).to(device))
                    .cpu()
                    .numpy()
                )
                assert action.shape == (1, 51) and np.isfinite(action).all(), f"Invalid action at step {step}"
                reward, _, done = env.step(action.astype("float32"), np.zeros_like(action))
                observation, distances = env.observe(contain_non_aff, partial_obs=False)
                state = env.get_global_state()
                for name, value in (
                    ("observation", observation),
                    ("distances", distances),
                    ("state", state),
                    ("reward", reward),
                ):
                    assert np.isfinite(value).all(), f"Nonfinite {name} at step {step}"
                # This environment terminates on numerical instability, not grasp success.
                assert not done.any(), f"Simulator reported instability at step {step}"
                reward_sum += float(reward.sum())
                max_action_abs = max(max_action_abs, float(np.abs(action).max()))
                max_wrist_displacement = max(
                    max_wrist_displacement, float(np.linalg.norm(state[0, 136:139] - initial_state[0, 136:139]))
                )
                max_joint_change = max(
                    max_joint_change, float(np.abs(state[0, 143:188] - initial_state[0, 143:188]).max())
                )
                max_object_displacement = max(
                    max_object_displacement, float(np.linalg.norm(state[0, 188:191] - initial_state[0, 188:191]))
                )
        assert max_wrist_displacement > 1e-6 or max_joint_change > 1e-6, "The simulated hand did not move"
        summary = {
            "passed": True,
            "object": selected.name,
            "checkpoint": str(checkpoint_path),
            "seed": args.seed,
            "policy_steps": 140,
            "simulation_duration_s": 140 * float(cfg["environment"]["control_dt"]),
            "headless": True,
            "initial_state_refreshed_after_goals": True,
            "policy_updates": 0,
            "dimensions": dict(dimensions, policy_observation=304),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "reward_sum": reward_sum,
            "max_action_abs": max_action_abs,
            "max_wrist_displacement_m": max_wrist_displacement,
            "max_joint_change_rad": max_joint_change,
            "max_object_displacement_m": max_object_displacement,
            "object_height_change_m": float(state[0, 190] - initial_state[0, 190]),
            "mano_check": _check_mano() if args.mano_check else {"skipped": True},
        }
        return summary
    finally:
        raw_env.close()


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graspxl_root", type=Path, default=Path("/home/nmt/Projects/GraspXL"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mano_check", action="store_true")
    parser.add_argument("--timeout_seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, help="Also write the JSON summary to this file")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout_seconds must be positive")
    output = args.output.expanduser().resolve() if args.output else None

    def timeout_handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"Smoke test exceeded {args.timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout_seconds)
    started = time.monotonic()
    try:
        summary = _rollout(args)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        summary = {"passed": False, "error_type": type(error).__name__, "error": str(error)}
    finally:
        signal.alarm(0)
    summary["elapsed_seconds"] = time.monotonic() - started
    serialized = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized, flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(_main())
