<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# OpenHOI: first mug test

Status checked 2026-09-16. The immediate goal is pretrained inference on a mug,
followed by a GraspXL mug input adapter and validation if the initial run works.
Training and the larger OpenHOI/Text2HOI comparison come later.

## Prepared locally

- Official source cloned into `repo/` at revision
  `4603d56e2cc18d2e838c58aab2e61fbf3dc4447a`; source code is unmodified.
- [mug_case.json](mug_case.json) records the existing GraspXL mug, its 15 motion
  pairs (2,325 frames), and source hashes. All paired motion hashes, finite MANO
  arrays, frame counts, and object trajectories passed checks. The mesh hash
  also matches. This is source-data validation, not model/contact validation.
- The mug UID is `12652c6e0aaf44dda32aab816f433baf`, size `large`. It belongs to
  the validation split of the prepared comparison, appropriate for development.
  Testing only this mug does not establish unseen-object generalization.
- The official cached `afford/data_grab.pkl` is included in the source checkout.
  Inspection of its pickle text opcodes finds `Lift mug with right hand.` among
  the mug prompts. The cache was not deserialized or geometrically validated.
- WSL detects the RTX 3070 Ti Laptop GPU with 8 GB VRAM. No OpenHOI environment
  or trained model has been installed, and no generated motion exists yet.

## Checkpoint download blocker

The [official README](https://github.com/Zhenhao-Zhang/OpenHOI) lists these
separately:

| Asset | Official location | Observed status |
|---|---|---|
| Trained OpenHOI motion/diffusion weights | [Baidu](https://pan.baidu.com/s/1nlj8u__3Z018F2XzyZ8UFw?pwd=ue54), code `ue54` | Download-page title returned `百度网盘-链接不存在` (link does not exist). No weights downloaded. |
| OpenHOI affordance MLLM weights | [Baidu](https://pan.baidu.com/s/13yP3ihztAcBF35JYMvHD8w?pwd=q6z3), code `q6z3` | Extraction-code page responds; checkpoint contents and download not verified. |
| Supporting pretrained weights | [Google Drive](https://drive.google.com/drive/folders/1bfYF94-dVy-mA0n4cIRb_wI4ohPC6KK5) | Folder responds, but this is also the exact checkpoint URL in the Text2HOI README. It does not establish possession of final OpenHOI weights. |

An existing [mirror request](https://github.com/Zhenhao-Zhang/OpenHOI/issues/5)
also concerns access to the diffusion weights. A working author-provided mirror
or existing local copy is needed. No message or issue has been sent to the author.
The public download-page responses are saved in ignored `downloads/` files.

The GRAB inference configuration expects:

```text
checkpoints/grab/texthom.pth
checkpoints/grab/refiner.pth
checkpoints/grab/contact_estimator.pth
checkpoints/grab/pointfeat.pth
checkpoints/grab/seq_cvae.pth
```

Each is loaded through its `model` state dictionary. Validate provenance,
architecture, and state-dictionary compatibility when the files are available;
do not silently substitute Text2HOI weights for a missing OpenHOI component.
Additional dependencies include CLIP, MPNet, licensed MANO models, mesh/point
data, and matching affordance data. The existing licensed MANO files are at
`C:/Linux/SOMA-X/assets/MANO/`; they have not been copied or redistributed.

## Test order

1. Obtain the trained weights, then create a separate motion-inference Conda
   environment. Start at batch size one. The authors document Python 3.8,
   PyTorch 1.13.0/CUDA 11.6, and PyTorch3D 0.7.2 for this component. Dependency
   compatibility and peak memory must be measured locally.
2. Reproduce a right-hand mug example using the original GRAB object and its
   matching sampled points/affordance map. Fix local paths and invoke the motion
   generator without the full dataset loop. This is a component smoke test with
   a cached affordance prediction, not complete end-to-end OpenHOI evaluation.
3. Prepare the actual GraspXL mug mesh/point cloud with its native metric scale.
   Obtain an affordance map aligned to those exact points. Do not reuse a GRAB
   mug's map by point index. Begin with pretrained inference; an input-format
   adapter does not itself require retraining.
4. Export generated MANO parameters and the object trajectory, then inspect
   geometry and motion in the viewer. Compare contact distance, penetration,
   smoothness, and reference accuracy. Treat a dynamic lift/hold test separately
   from animation playback. Decide whether fine-tuning is needed from results.

## Adapter checks identified in the released code

- `start/inference.py` has an author-specific absolute affordance path and builds
  a full GRAB dataloader. A single-example launcher should avoid that dependency.
- `create_hoi` expects a batch/list of prompt strings; the released loop passes
  a single string. Supply `["Lift mug with right hand."]` for a one-item test.
- The original GRAB configuration uses `flat_hand=True`; GraspXL's source MANO
  convention uses `flat_hand_mean=False`. Any later reference-motion conversion
  must preserve reconstructed geometry across that difference. Object rotations
  also require checking the GRAB-specific transform convention.
- Motion generation uses 1,024 sampled object points, while the MLLM-facing
  preparation documents 2,048. Preserve the point correspondence at each stage.
- The active `create_hoi` path uses ordinary diffusion sampling followed by a
  learned refiner; a separate loss-guided contact/penetration sampling call is
  commented out. Record the actual inference path before attributing paper-level
  physical-refinement results to a local run.
- The full affordance MLLM is a separate, much larger setup. Its unmodified
  feasibility on this GPU is unverified; cached-map tests isolate the motion
  component while that is assessed.

To repeat the source checks from WSL:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /mnt/c/Linux/IsaacLab
./isaaclab.sh -p _local/openhoi/prepare_mug_case.py
```

The check updates only `mug_case.json`; it does not change the datasets or viewer.
