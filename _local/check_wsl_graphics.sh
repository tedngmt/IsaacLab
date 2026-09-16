#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
uname -a
ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/lib/wsl/lib 2>/dev/null
dpkg -l | grep -E 'vulkan|mesa|nvidia'
printenv | grep -E 'VULKAN|VK_|MESA|GALLIUM|DISPLAY|WAYLAND|LD_LIBRARY'
vulkaninfo --summary 2>&1 | tail -65
glxinfo -B 2>&1
apt-cache policy mesa-vulkan-drivers libvulkan1 vulkan-tools
grep -nE 'VULKAN|VK_|MESA|GALLIUM|LIBGL|DISPLAY' ~/.bashrc ~/.profile /etc/environment /etc/profile.d/* 2>/dev/null
ls -l /dev/dxg /dev/dri 2>/dev/null
