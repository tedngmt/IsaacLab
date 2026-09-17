# GraspXL and RaiSim on this computer

Restored and verified on 17 September 2026 after moving the project to
`/home/nmt/Projects/GraspXL` in WSL Ubuntu.

## Installed environment

- Conda environment: `graspxl`, Python 3.8.20.
- PyTorch: 2.3.0+cu118, using the NVIDIA GeForce RTX 3070 Ti Laptop GPU.
- Simulator: existing RaiSim 1.1.6 at `/home/nmt/raisim_build`.
- Existing activation file: `/home/nmt/.raisim/activation.raisim`, linked from
  `GraspXL/rsc/activation.raisim`. The link is ignored by Git.
- MANO: existing `manotorch` 0.0.2 at `/home/nmt/manotorch`, with the licensed
  models under `/home/nmt/manotorch/assets/mano`.
- GraspXL's 12 C++ environment modules were rebuilt for this location and Python.

PyTorch's CUDA runtime packages are inside the `graspxl` environment. This setup
did not install a system CUDA toolkit or Linux NVIDIA driver. The existing
`text2hoi`, `soma-x`, and `env_isaaclab` environments were left unchanged.

Conda activation hooks set `LD_LIBRARY_PATH` for RaiSim and WSL, and supply the
default `MANO_ASSETS_ROOT`. Deactivation restores previous values, including
variables that were originally unset. No shell startup or HOME changes were made.

## Use the environment

```bash
conda activate graspxl
cd /home/nmt/Projects/GraspXL/raisimGymTorch
```

Use this working directory for the original scripts because they contain relative
resource paths.

## Run the short headless mug check

```bash
conda activate graspxl
timeout 180s python /home/nmt/Projects/IsaacLab/_local/graspxl_setup/smoke_test.py \
  --mano_check \
  --output /home/nmt/Projects/IsaacLab/_local/graspxl_setup/smoke_result.json
```

This loads `data_all/floating_mixed/full_5600_r.pt`, chooses one existing mug,
and runs 140 policy steps without opening the viewer or updating the policy.
The check refreshes the initial state after setting grasp goals because the
upstream reset otherwise leaves three cached grasp-axis entries undefined.
It does not change the simulator source or advance physics during that refresh.

The final check passed: all observations, actions, rewards, and states were finite;
the hand moved and the mug rose approximately 0.447 m. CUDA MANO forward inference,
axis conversion, and anatomy loss also passed. This is an installation check,
not a grasp success-rate evaluation.

## Run the interactive demo

In Windows File Explorer, open:

```text
\\wsl.localhost\Ubuntu\home\nmt\Projects\GraspXL\raisimUnity\win32\RaiSimUnity.exe
```

Keep the viewer's server address `127.0.0.1`, port `8080`, and enable
**Auto-connect**. Then run in Ubuntu:

```bash
conda activate graspxl
cd /home/nmt/Projects/GraspXL/raisimGymTorch
python raisimGymTorch/env/envs/ours_demo/demo.py
```

The stock demo selects an object and runs 25 episodes, saving motion recordings.
It is different from the one-mug headless check above.

All 61 Windows viewer resource paths were migrated from `C:\Linux\GraspXL` to
the WSL location. Windows can read the replacement directories. Original registry
values are backed up in
`build/viewer-resource-settings-before-migration-20260917T071859746Z.json`.
The existing viewer DLL retains the earlier disconnect fix. The GUI itself was
not launched during this setup; graphical display remains to be checked when
opening the viewer. The previous installation used the Windows viewer because
the bundled Linux viewer had rendering problems on this machine.

## Rebuild after changing simulator code

```bash
conda activate graspxl
cd /home/nmt/Projects/GraspXL/raisimGymTorch
python setup.py develop --no-deps --CMAKE_PREFIX_PATH /home/nmt/raisim_build
```

The old CMake cache under `/home/nmt/.cache/graspxl/raisim-build` still refers to
the former Windows location; this installation uses the fresh build under
`GraspXL/raisimGymTorch/build`.

## Validation and next training work

The following passed:

- Separate imports of all 12 simulator modules.
- Existing RaiSim activation and 10 basic physics steps.
- CUDA tensor computation.
- The pretrained 140-step headless mug rollout and MANO anatomy check.
- Demo and both MANO training scripts' `--help` entry points.
- `pip check` and Conda activation/deactivation path restoration.

Validation files and installed dependency versions are in
`/home/nmt/Projects/IsaacLab/_local/graspxl_setup/`:
`installation_checks.json`, `smoke_result.json`, `requirements_installed.txt`,
and `build.log`. The original setup notes are preserved as
`GraspXL/build/WSL_SETUP.before_migration.md`.

The original GraspXL training entry points are available. The GRAB reference-motion
adapter, Text2HOI connection, and table/obstacle constraints still require task
implementation. No GRAB/Text2HOI reinforcement-learning training was started here.
The existing hand collision mask also needs adaptation before claiming that the
hand is constrained by the table.
