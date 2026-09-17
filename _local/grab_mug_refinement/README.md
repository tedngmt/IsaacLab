# GRAB mug: recorded MANO versus Text2HOI refinement

This personal experiment fine-tunes the released **Text2HOI GRAB refiner** on all
44 locally available GRAB mug clips. It adjusts recorded hand poses while leaving
the mug trajectory unchanged. It is a source-motion cleanup experiment, not
text-to-motion generation, Isaac physics training, or a held-out benchmark.

The exported comparisons include both hands and every fourth source frame:
13,171 frames at 30 FPS, preserving the complete duration of the 120 FPS sources.
All displayed clips are training examples. Results cannot establish generalization
to unseen subjects, motions, or objects.

## Representation

Both columns use each subject's original personalized **standalone MANO** template
and source MANO pose parameters. Additional shape coefficients are zero because
personalization is already present in the template. Input and output have the
same anatomy, topology, timing, object geometry, object poses, and camera.

Earlier MANO-versus-SOMA videos used personalized **SMPL-X hand crops** on their
original side. These differ from standalone MANO by about 1.0–1.2 mm on average,
with some vertex differences up to 9.52 mm. Those earlier meshes are not the
baseline for this training comparison. The complete discrepancy and encoding
checks are recorded in `prepared/manifest.json`.

GRAB transforms canonical row-vector object vertices with its source Rodrigues
matrix directly: `world = canonical @ source_R + translation`. Geometry exports
store the equivalent active rotation `obj_rot = source_R.T`.

## Training adaptation

The actual pretrained 8-layer Text2HOI refiner is fine-tuned. Text2HOI's motion
generator is unchanged. Conditioning comes from observed source poses and their
geometric proximity to the mug, so it is not free text-conditioned generation.

The refiner's residual is scaled by 0.1 and smoothly limited to 15 mm root
translation and 0.15 per rotation-6D component. The same wrapper is used for the
initial and final model evaluations. This conservative wrapper is an explicit
local adaptation, not an upstream Text2HOI setting.

Losses penalize penetration into the native mug material, deviation from original
near-surface contacts, hand displacement, and rapid changes in the correction.
The recorded mug trajectory never changes. Original dataset files and downloaded
checkpoints are never overwritten.

A signed-distance grid at 0.75 mm spacing is constructed from all 61,168 native
mug triangles. Negative values indicate solid material; the empty cup cavity and
handle opening remain outside. The grid was independently checked against exact
triangle distances and two-direction ray tests. Grid interpolation is approximate;
`audit_results.py` measures selected frames independently against native triangles.

## Outputs and interpretation

Each run has its own directory under `runs/`, including a checkpoint, training
log, initial/final per-clip measurements, geometry, and rendered comparisons.
The selected run is `runs/all44_v2`: four initial passes at learning rate 1e-5,
followed by twelve passes at 5e-5, totaling 3,200 optimizer updates. All 44 clips
and every exported source frame were visited in all sixteen passes.
`share/` contains the videos and offline HTML browser, without raw licensed model
or dataset assets. Open `index.html` or the compilation MP4.

Read penetration together with contact retention and motion displacement. Lower
penetration alone could result from moving the hand away from the mug. Vertex
sampling does not measure complete triangle-intersection volume. This experiment
does not test forces, friction, or whether the mug can actually be held under
gravity. Residual overlap or contact loss may remain.

## Reproduction

Use the existing `text2hoi` Conda environment for preparation, training, and
rendering. The SDF builder and independent triangle audit use `soma-x` because
that environment already contains Warp, Trimesh, and R-tree. GPU and WSLg access
may require execution outside the restricted agent sandbox.

```bash
cd /home/nmt/Projects/IsaacLab
/home/nmt/miniconda3/envs/text2hoi/bin/python _local/grab_mug_refinement/prepare_data.py
/home/nmt/miniconda3/envs/soma-x/bin/python _local/grab_mug_refinement/build_mug_sdf.py
LD_LIBRARY_PATH=/usr/lib/wsl/lib /home/nmt/miniconda3/envs/text2hoi/bin/python \
  _local/grab_mug_refinement/train_refinement.py \
  --output_dir _local/grab_mug_refinement/runs/all44_v1 --epochs 4
LD_LIBRARY_PATH=/usr/lib/wsl/lib /home/nmt/miniconda3/envs/text2hoi/bin/python \
  _local/grab_mug_refinement/train_refinement.py \
  --output_dir _local/grab_mug_refinement/runs/all44_v2 --epochs 12 \
  --learning_rate 0.00005 \
  --resume /home/nmt/Projects/IsaacLab/_local/grab_mug_refinement/runs/all44_v1/checkpoint.pth
/home/nmt/miniconda3/envs/text2hoi/bin/python \
  _local/grab_mug_refinement/annotate_geometry.py \
  --geometry_manifest _local/grab_mug_refinement/runs/all44_v2/geometry_manifest.json
MUG_EGL_PLATFORM=x11 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  /home/nmt/miniconda3/envs/text2hoi/bin/python \
  _local/grab_mug_refinement/render_comparisons.py \
  --manifest _local/grab_mug_refinement/runs/all44_v2/geometry_manifest.json \
  --output_dir _local/grab_mug_refinement/runs/all44_v2/share --stage all
```

Record the exact command, checkpoint hash, completed epochs, and final checks in
the chosen run's report. Use a new run directory when changing experiment settings.
