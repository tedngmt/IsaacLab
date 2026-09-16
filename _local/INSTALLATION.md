# Isaac Lab in WSL Ubuntu

Environment: `/home/nmt/miniconda3/envs/env_isaaclab` (Python 3.12).
Checkout: `/mnt/c/Linux/IsaacLab`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /mnt/c/Linux/IsaacLab
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
