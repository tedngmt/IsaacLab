# GRAB converted to SOMA-X

Full reference conversion: 51 object identities, 10 participants, 1,335 sequences,
1,623,695 frames at the original 120 FPS. Check `reports/progress.json` for status;
the package is complete only when `status` is `validated`.

## Layout

- `motions/<subject>/<sequence>.npz`: SOMA poses, root translation, object and table
  trajectories, original expression coefficients and fitting errors.
- `objects/<object>/mesh.ply`: original GRAB contact mesh, vertex order preserved.
- `scene/table.ply`: original table mesh.
- `subjects/<subject>/`: personalized body template, rig snapshot and metadata.
- `contacts/<subject>/<sequence>.npz`: original GRAB contact annotations.
- `reports/`: per-sequence fit statistics, progress, validation and SHA-256 hashes.
- `manifest.json`: sequence index after final validation.

## Pose conventions and replay

Coordinates remain in the original GRAB world frame; distances are meters.
`poses` has shape `(frames, 78, 3)`: axis-angle radians, including virtual Root.
Use `poses[:, 1:]` with SOMALayer.pose, `absolute_pose=True`, `pose2rot=True`,
`apply_correctives=False`, and `transl` for the Hips translation.
Here “absolute” is SOMA's API convention for rotations without applying joint
orientation offsets; it does not mean that every joint rotation is world-space.

Replay requires the personalized template in `subject_template_path`, the indicated
gender and zero additional SMPL-X betas. Loading a generic zero-beta person loses
the subject's original proportions. The saved public rig is a useful skeleton
snapshot; use the SOMA layer for the full procedural deformation behavior.

From the SOMA-X repository, in its configured Python environment:

```sh
python -m tools.replay_grab_soma --package ../GRAB_SOMA_51Objects \
  --motion motions/s1/mug_drink_1.npz --output out/grab-replay.npz
```

This CPU example reconstructs the first, middle and last frames. Its helper
functions can also replay selected frame batches. The licensed SMPL-X assets
remain in the repository assets directory.

Object/table rotations are original axis-angle radians. For mesh vertices stored
as row vectors, world coordinates are `vertices @ rotation_matrix.T + translation`.
Frame `i` occurs at `i / 120` seconds. Do not change object scale independently
of the body motion.

## Training considerations

The body motion is an approximate surface fit; inspect the per-sequence errors
and replay before using it as a target. A mean surface error measures the average
distance between corresponding body vertices, not a guarantee of finger contact.
Original body contacts refer to SMPL-X
vertex indices, **not SOMA vertex indices**. Object contacts retain the original
object mesh indexing. Facial expression coefficients are preserved as source
metadata, not converted into SOMA facial animation.

This package is a motion reference, not a finished Isaac simulation task.
Collision meshes, mass/inertia, friction, avatar retargeting, contact validation
and the training environment still need to be configured. Respect the original
GRAB and body-model licenses when using or sharing these derived files.

## Validation scope

`reports/validation.json` summarizes checks of every motion and contact file:
array integrity and shape, finite motion values, frame counts, identity metadata,
asset references and consistency with the conversion reports. Original object
meshes, the table mesh and each personalized template are copied byte for byte.
`reports/data_sha256.json` records hashes of the motion and contact files.

`reports/source_and_replay_validation.json`, when present, additionally records
independent CPU replays of first/middle/last frames from one clip per subject.
For these ten complete clips, object/table trajectories, expression coefficients
and both contact arrays are compared exactly with the original GRAB archives.
`reports/replay_samples/` contains the replayed and reference surfaces for those
30 sampled frames. This sampling checks reconstruction and source preservation;
it does not constitute a physical collision or contact-quality test of all frames.

`reports/regional_replay_validation.json` breaks down the sampled surface error
for left and right hands. `previews/soma-mug-replay-validation.png` visualizes one
replayed mug-drinking sequence and its surface errors. Its initial/final arms-out
poses are present in the source sequence; the table is omitted from the preview.
