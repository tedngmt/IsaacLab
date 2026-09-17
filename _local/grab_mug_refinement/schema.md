# GRAB mug refinement preparation

Run `prepare_data.py` in the existing `text2hoi` Conda environment. All source
files remain unchanged. Prepared numeric NPZ files require no pickle loading.
Lengths vary by clip; all 44 clips retain every fourth original frame (120 FPS
to 30 FPS), including approach/release and noncontact portions. No train/test
split is imposed: using every clip to train must be called **training-set
refinement**, not held-out evaluation.

`prepared/manifest.json` records all clip paths, source archive members, SHA-256
hashes, frame counts, actions, subjects, and reconstruction comparisons.

## Subject assets

`prepared/subjects/<subject>.npz`:

| Key | Shape | Meaning |
|---|---|---|
| `lhand_v_template`, `rhand_v_template` | 778 × 3 | Native personalized GRAB MANO templates, metres |
| `lhand_source_betas`, `rhand_source_betas` | 10 | Original hand shape coefficients, provenance only |
| `lhand_model_betas`, `rhand_model_betas` | 10 | Zeros: shape is already in the personalized template |
| `lhand_faces`, `rhand_faces` | 1538 × 3 | MANO triangle topology |
| `lhand_smplx_vertex_ids`, `rhand_smplx_vertex_ids` | 778 | Native correspondence for comparison with SMPL-X |

Build the upstream `build_mano_aa(is_rhand=..., flat_hand=True)` model, then
copy the subject template into `model.v_template` before reconstruction or
loss calculation. Keep model betas zero, matching upstream
`get_hand_layer_out()`. Source `fullpose` already includes the non-flat hand
mean; adding it again is wrong. Native MANO and cropped SMPL-X do not exactly
match because their skinning models differ. All-frame mean, p95 and maximum
differences are recorded; five representative frames also compare standard
zero-beta MANO. Before/after model comparisons must use the **same personalized
MANO representation**, avoiding confusion with representation changes.

## Shared object

`prepared/object.npz` contains `object_vertices_canonical` (30584 × 3 metres),
`object_faces` (61168 × 3), `object_points` (1024 × 3 metres), `object_normals`
(1024 × 3), `object_points_normalized`, `object_centroid`, and scalar
`object_scale`. Points/normals come from Text2HOI's downloaded `obj.pkl`.

`pretrained_decimated_point_indices` references the upstream decimated mesh;
**do not use these indices on the full native mesh**. The independent
`nearest_native_vertex_indices` maps each point to its nearest full-mesh
vertex and is provided only for inspection. Point-cloud proximity is not
exact triangle distance and should not replace full-mesh collision evaluation.

## Per-clip motions

`prepared/clips/<subject>__<motion>.npz`, with full clip length `T`:

| Key | Shape | Meaning |
|---|---|---|
| `x_lhand`, `x_rhand` | T × 99 | Native MANO translation [m], followed by 16 rotations as upstream 6D representations |
| `x_obj` | T × 9 | Object translation [m] and row-vector rotation in upstream GRAB convention |
| `object_rotation` | T × 3 × 3 | Active rotation used by corrected existing geometry caches |
| `object_translation` | T × 3 | Original object translation [m] |
| `lhand_vertices`, `rhand_vertices` | T × 778 × 3 | Personalized MANO **before refinement**, world coordinates [m] |
| `lhand_joints`, `rhand_joints` | T × 21 × 3 | Upstream MANO joints including its ordered five tips, world coordinates [m] |
| `lhand_contact_point_indices`, `rhand_contact_point_indices` | T × 21 | Nearest shared object point per joint |
| `lhand_contact_distances`, `rhand_contact_distances` | T × 21 | Original nearest-point joint distance [m] |
| `lhand_contact_mask`, `rhand_contact_mask` | T × 21 | Joint distance less than 0.02 m |
| `lhand_source_contact`, `rhand_source_contact` | T × 778 | Nonzero native GRAB body-contact annotation at corresponding SMPL-X vertex |
| `valid_mask_lhand`, `valid_mask_rhand`, `valid_mask_obj` | T | All true; both original hands exist throughout the full clip |
| `source_frame_indices` | T | Original 120 FPS frame indices |
| `source_fps`, `playback_fps` | scalar | 120 and 30 FPS |

6D rotations use upstream first-two-columns, interleaved row-major layout.
Object world vertices are `canonical @ object_rotation.T + translation`.
The upstream GRAB decoder instead uses `canonical @ rot6d_to_rotmat(x_obj[3:9])`,
so `x_obj` stores the transpose of the active cache matrix. No object rescaling,
centering, pose correction, or hand fitting occurs during preparation.

The nearest object point positions used for a contact target can be recovered
by transforming `object_points` through `x_obj`, then gathering with each
hand's `contact_point_indices`. These masks are proximity heuristics and do
not guarantee real contact. Keeping the original joint-to-surface separation
is preferable to indiscriminately pulling all joint centres onto the surface.
