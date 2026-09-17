# Mug dataset comparison exports

Open `share/index.html` to browse the clips, or use the two combined MP4 files
inside `share/`. The complete `share/` folder is portable and works offline.

The collection covers all 44 GRAB mug sequences and the selected GraspXL mug
from the local 51-object matched dataset: 15 existing paired sequences plus
three separately labeled tabletop motions with new local SOMA conversions.
It does not cover other mug shapes from the larger raw GraspXL archives.

Every video shows the original hand surface on the left and SOMA on the right.
The top row preserves complete hand motion in world coordinates. In GRAB's
bottom row, the camera follows the handle in world space while the mug retains
its original trajectory and tilt. The mug is not rotated into a stationary
object frame. GraspXL's bottom row keeps its existing object-stabilized close-up,
with the mug stationary to inspect the grip and handle. Corresponding original
and SOMA panels always share the same camera, scale, object, and source frame.

Both close-ups deliberately crop to the grip and handle; hands away from the
mug may leave the close-up but remain in the overview. GRAB retains both hands,
cropped from its original personalized SMPL-X surface. GraspXL uses the original
right MANO hand. All native mesh triangles are drawn.

GRAB object rotations now follow the official GRAB row-vector convention.
Earlier GRAB exports used the opposite rotation and are superseded by these
corrected views. This corrects playback; the source dataset and its contacts
were not edited. GraspXL playback is unchanged. Captured and fitted surfaces
can still overlap, and playback does not apply collision response.

GRAB uses every fourth frame from its 120 FPS source at 30 FPS playback.
GraspXL retains every frame and explicitly assumes 30 FPS because its source
frame rate is unknown. These are source motions, not Text2HOI outputs or
IsaacLab training results.

The current GRAB exports correct a mug-rotation error in the earlier viewer.
Official GRAB uses row vertices `@ Rodrigues(source_rotvec) + translation`.
The cache stores its transpose as the active rotation used by the renderer.
All 52,639 source object rotations were checked against the official convention;
hand geometry and source motions were preserved. Older GRAB videos are
superseded. GraspXL videos are unchanged. Playback does not resolve collisions,
and residual overlaps can remain in the original capture and SOMA fitting.

## Recreate the exports

Run from the IsaacLab repository root. These helpers use the existing SOMA
and Text2HOI environments independently of IsaacLab's Python environment.

```bash
# Reconstruct the original 59 paired sequences; checked caches are reused.
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /home/nmt/miniconda3/envs/soma-x/bin/python \
  _local/mug_comparisons/extract_geometry.py --batch_size 16

# Reconstruct and fit the three supplemental source tabletop motions.
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /home/nmt/miniconda3/envs/soma-x/bin/python \
  _local/mug_comparisons/prepare_tabletop.py --stage all

# Render from the validated geometry caches using WSLg's NVIDIA OpenGL path.
MUG_EGL_PLATFORM=x11 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  /home/nmt/miniconda3/envs/text2hoi/bin/python \
  _local/mug_comparisons/render_videos.py

# Decode every completed video, create compilations and the offline index.
/usr/bin/python3 _local/mug_comparisons/package_videos.py --decode_jobs 4
```

The supplemental tabletop caches and their source-convention checks are
recorded separately in `supplemental/conversion_summary.json`. Source motion
files and model assets are not modified by this export workflow.

If WSLg is unavailable, omit the two graphics environment variables for the
slower Mesa software renderer. The EGL renderer uses libraries already
installed in Ubuntu. `--backend pytorch3d` selects the slower CUDA alternative.

Use `--dataset GRAB` or `--dataset GraspXL` to render one dataset; repeat
`--clip_id` to select clips. `--preview_only` saves the first, middle, and last
comparison frames in `qa/`. Successful exports resume automatically. A failed
`.partial.mp4` is preserved for inspection before a retry.

## Verification records

- `inventory.json`: complete clip list and source provenance.
- `extraction_summary.json`: validation of the 59 existing paired sequences.
- `supplemental/conversion_summary.json`: all-frame checks and fitting errors
  for the three newly converted tabletop sequences.
- `cache/`: reconstructed hand vertices, original object meshes, and source
  object transforms. Cache SHA-256 hashes are checked before rendering.
- `qa/`: first/middle/last image checks and encoder logs.
- `qa/object_rotation_correction.json`: independent official-convention checks
  and preserved geometry checks for all corrected GRAB clips.
- `qa/contact_audit.json`: residual penetration checks at 18 sampled frames
  from six clips; this is not an exhaustive contact benchmark.
- `qa/grab_follow_camera_validation.json`: complete corrected mug framing and
  camera trajectory checks for all 44 GRAB clips.
- `share/validation.json`: exact output frame counts, full-file decoding,
  compilation chapters, and SHA-256 checksums.

Only the viewing outputs and reports belong in the shareable package; the
geometry caches and licensed source/model assets remain outside it.
