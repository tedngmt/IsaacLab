# Local MANO / SOMA comparison

Double-click `start_viewer.bat`, keep the terminal open, and visit http://localhost:8080.
If the viewer is already running, use that page without starting another instance.
Stop the viewer with Ctrl+C in its terminal.

The viewer uses the existing `soma-x` Conda environment and reads the sibling
datasets in `C:\Linux`. It does not change the datasets.

- Choose GraspXL or GRAB, select a sequence, then click **Load sequence**.
- For a loaded GRAB sequence, use **GRAB display** to switch between **Full body**
  and **Hands only**. Hands only shows both hands and the object in both reconstructions.
  Switching preserves the current frame and playback state and reframes the camera.
- Blue is the original mesh; orange is the SOMA reconstruction.
- Use **Frame**, **Play**, and **View** for synchronized comparison.
- GraspXL original geometry uses MANO. Original GRAB geometry uses personalized
  SMPL-X bodies, including the hands. The appropriate original model is used for each.
- GRAB's hands-only view crops the existing body meshes at the wrists. It preserves
  their geometry and hand/object alignment; it does not convert or export standalone
  MANO/SOMA hand motion. GraspXL is already hand-only.
- Side-by-side mode translates the complete SOMA scene for display. Overlay mode
  preserves the shared coordinates. Object scales are unchanged.
- GraspXL source FPS is unknown. Playback frames/s is a display setting, not an
  assertion about source timing. GRAB source timing is 120 FPS; reconstruction can
  limit playback speed. GRAB's table and contact annotations are not shown.

Recheck first/middle/last-frame reconstruction for one clip from each dataset,
including full-body/hand-only mesh indices and unchanged hand/object coordinates:

```bash
~/miniconda3/envs/soma-x/bin/python /mnt/c/Linux/IsaacLab/_local/mano_soma_viewer.py --check
```

The viewer listens only on localhost. Browser rendering has not been visually
verified by the installation assistant; HTTP access and GPU mesh reconstruction
were checked.
