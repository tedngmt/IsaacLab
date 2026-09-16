#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/isaaclab-vulkan-build/mesa-26.2.2"
export PATH="$HOME/isaaclab-vulkan-build/tools/bin:$PATH"
meson setup build-dzn --prefix="$HOME/.local/mesa-dzn-26.2.2" --libdir=lib \
  --buildtype=release -Dgallium-drivers=[] -Dvulkan-drivers=microsoft-experimental \
  -Dplatforms=x11 -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dopengl=false \
  -Dgles1=disabled -Dgles2=disabled -Dllvm=disabled -Dmicrosoft-clc=disabled \
  -Dspirv-tools=disabled -Dbuild-tests=false
ninja -C build-dzn -j4
meson install -C build-dzn
