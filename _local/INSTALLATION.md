# Isaac Lab in WSL Ubuntu

Environment: `/home/nmt/miniconda3/envs/env_isaaclab` (Python 3.12).
Checkout: `/home/nmt/Projects/IsaacLab` on WSL's native Linux filesystem.
Windows Explorer: `\\wsl.localhost\Ubuntu\home\nmt\Projects\IsaacLab`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/nmt/Projects/IsaacLab
```

Installed with:

```bash
bash isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
```

This installs Newton GPU physics and RSL-RL, without Isaac Sim. CUDA detects the
RTX 3070 Ti Laptop GPU with 8 GB VRAM. WSL currently has approximately 15 GB RAM.
Mesa Dozen was subsequently installed for basic Vulkan-over-DirectX support;
see `VULKAN.md`. It does not provide the Vulkan ray tracing extensions needed
for Isaac Sim RTX rendering.

Small training example:

```bash
bash isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct-v0 \
  --num_envs 16 --max_iterations 2 --headless --visualizer none physics=newton_mjwarp
```

For MANO/SOMA dataset playback, see `VIEWER.md`. That viewer uses the separate
existing Python 3.10 `soma-x` environment, preserving its model dependencies.

Installation output: `install.log`. Training verification output: `cartpole-check.log`.

Verified on 2026-09-16: dependency check passed; 16 Cartpole environments completed
two RSL-RL training iterations (512 steps) on the Newton MuJoCo-Warp backend, with
exit code 0 and a saved `model_1.pt` checkpoint. First-run compilation/setup adds
substantial startup time.

## Linux storage migration (2026-09-17)

The repository and sibling datasets were moved into `/home/nmt/Projects`.
All 13 previously installed IsaacLab editable packages and the SOMA-X editable
package were reinstalled from their new source directories with `--no-deps`,
`--no-build-isolation`, and `--no-index`. Their package versions and dependency
versions were preserved; IsaacLab's package consistency check passed.
The executable permission on `isaaclab.sh` was restored after the copy.

Viewer and Text2HOI launchers now use this checkout. The Python helpers locate
sibling repositories/datasets relative to themselves. Eight Text2HOI asset links
that became empty files during the copy were backed up and rebuilt as relative
Linux symlinks; all 12 downloaded assets matched their original hashes.

Migration reports and logs are in `migration/`; Text2HOI link backups and asset
validation are in `text2hoi/migration_backup/`. Historical output and audit files
retain their original paths as records of where those earlier runs took place.

Post-migration validation passed: the viewer reconstructed first/middle/last frames
for GraspXL and GRAB, including GRAB hand-only mode; Text2HOI generated and exported
a complete 147-frame mug demo; and IsaacLab completed the 16-environment,
two-iteration Cartpole GPU training check (512 steps, exit code 0). See
`migration/viewer_check.log`, `migration/text2hoi_mug.log`, and
`migration/cartpole_check.log`. This verifies the migrated runtime, not a GraspXL
training implementation or its model performance.
