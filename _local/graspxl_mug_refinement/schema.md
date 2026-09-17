# GraspXL selected-mug refinement data

The numeric NPZ layout follows `../grab_mug_refinement/schema.md` where applicable.
This preparation includes all **18 existing selected-mug comparison clips**:
15 matched clips of 155 frames and 3 supplemental tabletop clips of 100 frames,
totalling 2,625 frames. The source's physical frame rate is unknown. All native
frames are retained, with **30 FPS assumed for playback only**.

Run `prepare_data.py` using the existing Text2HOI Conda environment. The process
does not train, fit or convert hands, change object shape, or modify datasets.
Every reconstructed hand frame must match the existing original blue MANO
comparison geometry within 0.000001 m. Per-clip errors and hashes are recorded
in `prepared/manifest.json`.

## Differences from the GRAB preparation

- Only the **right hand exists**. `valid_mask_lhand` and `lhand_contact_mask`
  are entirely false; `is_lhand` is false. Left vertices/joints and proximity
  arrays are dummy zeros. `x_lhand` uses zero translation plus valid identity
  6D rotations so upstream code can safely decode it before masking.
  **Never include the dummy left hand in losses, metrics, rendering, or contacts.**
- `valid_mask_rhand` and `valid_mask_obj` are entirely true.
- `source_frame_indices` is `arange(T)`, with stride 1. JSON `source_fps` is
  `null`; the scalar NPZ value is `NaN`. This is the only nonfinite field.
- No original contact annotations are available, so no `*_source_contact`
  arrays are invented. The supplied joint proximity masks are geometric
  distances below 0.02 m, not verified contact labels.
- A single `subjects/mano_zero.npz` contains standard left/right MANO templates,
  topology, and zero betas for API compatibility. Only the right model is used.
- Matched source poses use the non-flat MANO mean: the adapter adds the mean
  to their 45 pose values before encoding for `flat_hand_mean=True`.
- Supplemental tabletop poses are already flat-mean. Their translation must
  subtract the official fixed wrist bias `[0.09566994, 0.00638343, 0.0061863]` m.
  No hand shape fitting or source pose optimization is performed.

## Files and conventions

`prepared/clips/<clip_id>.npz` contains `x_lhand`, `x_rhand` (T × 99), `x_obj`
(T × 9), `lhand_vertices` / `rhand_vertices` (T × 778 × 3 metres), hand joints
(T × 21 × 3 metres), joint contact point indices/distances/masks, validity masks,
`is_lhand` / `is_rhand`, object active rotations and translations, frame indices,
and the two frame-rate fields.

The native mug mesh and deterministic 1024-point subset are stored in
`prepared/object.npz`: `object_vertices_canonical`, `object_faces`,
`object_points`, `object_normals`, `object_point_indices`,
`object_points_normalized`, `object_centroid`, and `object_scale`.
The original object mesh is retained; the point encoder alone uses normalized
points. This is the GraspXL mug geometry, **not** Text2HOI's original GRAB mug.

World object vertices are `canonical @ object_rotation.T + object_translation`.
For compatibility with Text2HOI's `dataset_name="grab"` decoder, `x_obj` instead
encodes `object_rotation.T`. The decoder name does not change the source dataset.

All 18 clips are available to training. Outputs trained on this full set must
be labelled **training-set refinement**, not evidence of held-out generalization.

## Collision topology

The native mug has 1,489 vertices and 3,000 faces, including 15 edges shared by
four faces. Its original data remain in `object.npz`. `collision_topology.py`
creates a separate `collision_proxy.npz` by pairing oriented half-edges and
splitting their vertex fans. The proxy has 1,500 vertices and the same 3,000
faces; every ordered triangle coordinate is bitwise identical to its source.
No vertex position changes and no face is removed, added, filled, or reversed.
The proxy is one oriented watertight component. The accompanying JSON records
the 15 pairings and proves geometry identity.

Independent native-versus-proxy tests cover 18,892 actual hand points plus 2,000
random points. Distances and both ray-direction signs are identical between
native and proxy. One source rim point, 0.213 mm from its nearest triangle,
has conflicting signs between the two ray directions in both representations;
such disagreements should be reported and excluded from confirmed penetration.

`build_mug_sdf.py` uses the topology proxy to construct the distance grid while
retaining native `canonical_vertices` and `canonical_faces` in `sdf.npz` for
source-geometry checks. Additional `collision_proxy_vertices` and
`collision_proxy_faces` identify the query topology. Empty-space and solid probes
use this mug's native Y-up coordinates; the cup cavity and handle opening must
remain outside the solid. Mesh sign, exact-distance, grid-interpolation, and
gradient validations remain mandatory before training.
