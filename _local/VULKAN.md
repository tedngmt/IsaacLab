# WSL Vulkan and OpenGL setup

Verified on 2026-09-16 in Ubuntu 22.04:

- Official Ubuntu Vulkan runtime and Mesa packages were already installed.
  Added `mesa-utils` for OpenGL diagnostics.
- Built Mesa 26.2.2 Dozen from https://archive.mesa3d.org/mesa-26.2.2.tar.xz,
  with Microsoft DirectX-Headers v1.619.1, using `_local/build_dzn.sh`.
- Installed only this separate driver under `/home/nmt/.local/mesa-dzn-26.2.2`.
  System Mesa and Windows NVIDIA drivers were not replaced.
- `vulkaninfo` detects Vulkan 1.2.354 on the NVIDIA RTX 3070 Ti Laptop GPU and
  AMD integrated GPU. `vkcube --gpu_number 0 --c 120 --suppress_popups` rendered
  successfully on NVIDIA and exited with code 0.
- The driver reports itself as non-conformant and intended for testing.
- `VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`, and
  `VK_KHR_ray_query` are absent. This does not enable Isaac Sim RTX rendering.

The `env_isaaclab` Conda environment now sets:

```bash
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
VK_ICD_FILENAMES=/home/nmt/.local/mesa-dzn-26.2.2/share/vulkan/icd.d/dzn_icd.x86_64.json
VK_DRIVER_FILES=/home/nmt/.local/mesa-dzn-26.2.2/share/vulkan/icd.d/dzn_icd.x86_64.json
```

Reactivate it in an existing terminal to apply these settings:

```bash
conda deactivate
conda activate env_isaaclab
vulkaninfo --summary
glxinfo -B
```

OpenGL and Vulkan are separate APIs; they do not require a global switch.
The MESA variable selects NVIDIA for OpenGL. The VK variables select the
Dozen Vulkan driver; Vulkan applications choose among the adapters it exposes.
These overrides are scoped to `env_isaaclab`, not all Ubuntu applications.

To undo the environment overrides while keeping the isolated driver installed:

```bash
conda env config vars unset -n env_isaaclab MESA_D3D12_DEFAULT_ADAPTER_NAME VK_ICD_FILENAMES VK_DRIVER_FILES
conda deactivate
conda activate env_isaaclab
```

Logs: `opengl-nvidia.log`, `dzn-build.log`, `vulkan-dzn-summary.log`,
`vulkan-dzn-full.log`, `vkcube-check.log`.
