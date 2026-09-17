# GraspXL mug: Text2HOI refinement experiment

This local experiment adapts the released Text2HOI GRAB **refiner** to all 18
selected GraspXL mug clips already used in the MANO/SOMA viewer: 15 matched clips
and 3 supplemental tabletop clips, containing 2,625 native frames in total.
The model updates the original right-hand MANO motion. The mug geometry and
trajectory are fixed. This does not train the text-to-motion generator or an
Isaac reinforcement-learning controller, and does not train a SOMA model.

Every clip is used for training and visual comparison. Results measure
**training-set refinement**, not performance on unseen clips or objects.
All original frames are retained; 30 FPS is a chosen playback rate because the
physical source frame rate is unknown.

## Geometry and measurements

- The before hand must reproduce the original viewer geometry within 0.001 mm.
- Only the right hand exists. Dummy left-hand inputs required by Text2HOI are
  masked and excluded from all losses, measurements, and videos.
- The original mug has nonmanifold shared edges. A separate collision mesh
  splits vertex fans without moving, adding, or deleting any triangle. Its
  ordered triangle coordinates must exactly equal those of the original mug.
- A signed-distance grid approximates depth within mug material. Empty space
  inside the cup and handle opening does not count as penetration.
- Counts summed across a sequence are **vertex-frame samples**: the same vertex
  can be counted repeatedly over time. These are not counts of visible holes.
- Independent nearest-triangle and ray checks sample fixed frames from every
  clip. Neither these nor vertex counts certify a physically stable grasp or
  detect every possible triangle intersection.

The objective penalizes penetration while preserving nearby surface contacts,
source hand geometry, and smooth corrections. Reduced penetration can still
come with lost contact or worse individual clips; the reports retain both.

## Reproduce

Run from the IsaacLab repository root. The existing Text2HOI Python 3.8
environment supplies its compatible model dependencies; the existing soma-x
environment supplies Warp for building the distance grid.

```bash
/home/nmt/miniconda3/envs/text2hoi/bin/python _local/graspxl_mug_refinement/prepare_data.py
/home/nmt/miniconda3/envs/soma-x/bin/python _local/graspxl_mug_refinement/collision_topology.py
LD_LIBRARY_PATH=/usr/lib/wsl/lib /home/nmt/miniconda3/envs/soma-x/bin/python _local/graspxl_mug_refinement/build_mug_sdf.py
LD_LIBRARY_PATH=/usr/lib/wsl/lib /home/nmt/miniconda3/envs/text2hoi/bin/python _local/graspxl_mug_refinement/train_refinement.py \
  --output_dir /home/nmt/Projects/IsaacLab/_local/graspxl_mug_refinement/runs/all18_v1 --epochs 16
MUG_EGL_PLATFORM=x11 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  /home/nmt/miniconda3/envs/text2hoi/bin/python _local/graspxl_mug_refinement/render_comparisons.py \
  --manifest _local/graspxl_mug_refinement/runs/all18_v1/geometry_manifest.json \
  --output_dir _local/graspxl_mug_refinement/runs/all18_v1/share --stage all
```

Use a new output directory for another experiment. The downloaded checkpoint,
source datasets, existing GRAB results, and original MANO/SOMA videos are kept.
