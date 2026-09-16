# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local paired GraspXL and GRAB mesh viewer; run in the soma-x environment."""

import argparse
import io
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation

ROOT = Path('/mnt/c/Linux')
sys.path.insert(0, str(ROOT / 'SOMA-X'))
from soma._smpl_family_loader import ensure_chumpy_compat
from soma.hand import SOMAHandLayer
from tools.replay_grab_soma import load_personalized_layer, replay

ASSETS = ROOT / 'SOMA-X/assets'
MANO_ROOT = ROOT / 'GraspXL_MANO_51Objects_GRABMatched'
SOMA_ROOT = ROOT / 'GraspXL_SOMA_51Objects_GRABMatched'
GRAB_ROOT = ROOT / 'GRAB_SOMA_51Objects'


class Sequence:
    def __init__(self, kind, entry, device):
        self.kind, self.device = kind, device
        if kind == 'GraspXL':
            self.motion = dict(np.load(SOMA_ROOT / entry['soma_path'], allow_pickle=False))
            self.source = np.load(MANO_ROOT / entry['mano_path'], allow_pickle=True).item()
            self.layer = SOMAHandLayer(data_root=str(ASSETS), hand_type='right',
                                       identity_model_type='mano', device=device).to(device)
            self.layer.prepare_identity(torch.zeros(1, 10, device=device))
            self.original = smplx.MANO(str(ASSETS / 'MANO/MANO_RIGHT.pkl'), is_rhand=True,
                                      use_pca=False, flat_hand_mean=False, create_transl=False).to(device)
            mesh_path = MANO_ROOT / entry['mesh_path']
        else:
            self.motion = dict(np.load(GRAB_ROOT / entry, allow_pickle=False))
            self.layer = load_personalized_layer(GRAB_ROOT, self.motion, ASSETS, device)
            subject = Path(entry).parts[1]
            archive = ROOT / 'Grab Dataset' / f'grab__{subject}.zip'
            with zipfile.ZipFile(archive) as zf:
                candidates = [n for n in zf.namelist() if Path(n).name == Path(entry).name]
                if len(candidates) != 1:
                    raise ValueError(f'Expected one original clip, found {candidates}')
                self.source = dict(np.load(io.BytesIO(zf.read(candidates[0])), allow_pickle=True))
            gender = str(self.motion['gender'].item())
            self.original = smplx.SMPLX(str(ASSETS / 'SMPLX' / f'SMPLX_{gender.upper()}.npz'),
                gender=gender, use_pca=True, num_pca_comps=24, flat_hand_mean=False,
                v_template=self.layer.identity_model.identity_model.v_template, batch_size=1).to(device)
            mesh_path = GRAB_ROOT / str(self.motion['package_mesh_path'].item())
        self.mesh = trimesh.load(mesh_path, process=False, force='mesh')
        self.count = len(self.motion['poses'])
        self.faces = self.layer.faces.detach().cpu().numpy()
        self.original_faces = self.original.faces

    def tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def frame(self, index):
        m = self.motion
        if self.kind == 'GraspXL':
            hand = self.source['right_hand']
            source = self.original(global_orient=self.tensor(hand['rot'][index:index+1]),
                hand_pose=self.tensor(hand['pose'][index:index+1]),
                betas=torch.zeros(1, 10, device=self.device)).vertices
            source = source + self.tensor(hand['trans'][index:index+1])[:, None]
            fitted = self.layer.pose(self.tensor(m['poses'][index:index+1]),
                pose2rot=m['poses'].ndim == 3, absolute_pose=bool(m['absolute_pose']),
                global_translation=self.tensor(m['transl'][index:index+1]))['vertices']
        else:
            params = self.source['body'].item()['params']
            keys = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
                    'jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'transl']
            source = self.original(**{k: self.tensor(params[k][index:index+1]) for k in keys},
                                   betas=torch.zeros(1, 10, device=self.device)).vertices
            fitted = replay(self.layer, m, np.array([index]))['vertices']
        rot = Rotation.from_rotvec(m['object_rot'][index].reshape(3)).as_matrix()
        obj = np.asarray(self.mesh.vertices) @ rot.T + m['object_trans'][index].reshape(3)
        return source[0].cpu().numpy(), fitted[0].cpu().numpy(), obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    ensure_chumpy_compat()
    pairs = json.loads((MANO_ROOT / 'metadata/pairs.json').read_text())
    grasp = {f"{p['grab_object']} / {Path(p['mano_path']).parent.name} / {Path(p['mano_path']).stem}": p for p in pairs}
    grab = {str(p.relative_to(GRAB_ROOT)): str(p.relative_to(GRAB_ROOT))
            for p in sorted((GRAB_ROOT / 'motions').glob('*/*.npz'))}
    if args.check:
        for kind, entry in [('GraspXL', pairs[0]), ('GRAB', next(iter(grab.values())))]:
            sequence = Sequence(kind, entry, args.device)
            for i in [0, sequence.count // 2, sequence.count - 1]:
                arrays = sequence.frame(i)
                assert all(np.isfinite(a).all() for a in arrays)
                print(kind, i, [a.shape for a in arrays], flush=True)
        return
    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## MANO / SOMA comparison\nBlue: original · Orange: SOMA. Object scale is preserved.')
    dataset = server.gui.add_dropdown('Dataset', ('GraspXL', 'GRAB'))
    options = list(grasp)
    select = server.gui.add_dropdown('Sequence', options, initial_value=next(k for k in options if k.startswith('mug /')))
    load = server.gui.add_button('Load sequence')
    frame = server.gui.add_slider('Frame', min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox('Play', False)
    speed = server.gui.add_slider('Playback frames/s', min=1, max=120, step=1, initial_value=30)
    mode = server.gui.add_dropdown('View', ('Side by side', 'Overlay'))
    status = server.gui.add_markdown('Loading…')
    state = {'reload': True}

    @dataset.on_update
    def _(_event):
        select.options = list(grasp if dataset.value == 'GraspXL' else grab)
        select.value = select.options[0]
        playing.value = False

    @load.on_click
    def _(_event):
        state['reload'] = True
        playing.value = False

    @server.on_client_connect
    def _(client):
        client.camera.position = (0.65, -0.85, 0.65)
        client.camera.look_at = (0.0, 0.0, 0.0)

    sequence = None
    handles = []
    previous = None
    tick = time.monotonic()
    while True:
        try:
            if state['reload']:
                state['reload'] = False
                status.content = 'Reconstructing selected sequence…'
                sequence = Sequence(dataset.value, (grasp if dataset.value == 'GraspXL' else grab)[select.value], args.device)
                frame.max = sequence.count - 1
                frame.value = 0
                previous = None
                for h in handles:
                    h.remove()
                handles = []
                initial = sequence.frame(0)
                center = initial[2].mean(axis=0)
                span = max(np.ptp(np.concatenate(initial), axis=0).max(), 0.2)
                shift = span * 1.15
                label = 'MANO' if sequence.kind == 'GraspXL' else 'SMPL-X (original GRAB)'
                for side, color, vertices, faces in [('original', (65, 145, 245), initial[0], sequence.original_faces),
                                                    ('soma', (245, 150, 55), initial[1], sequence.faces)]:
                    handles.append(server.scene.add_mesh_simple(f'/{side}/body', vertices - center, faces, color=color))
                    handles.append(server.scene.add_mesh_simple(f'/{side}/object', initial[2] - center, sequence.mesh.faces,
                                                                color=(155, 175, 165)))
                for client in server.get_clients().values():
                    client.camera.look_at = (shift / 2, 0, 0)
                    client.camera.position = (shift / 2 + span, -span * 2, span)
                note = 'Source FPS unknown; playback speed is adjustable.' if sequence.kind == 'GraspXL' else 'Source: 120 FPS. Playback may be slower during reconstruction.'
                status.content = f'**{label} ↔ SOMA** · {sequence.count} frames\n\n{note}'
                print(f'Loaded {sequence.kind}: {select.value}', flush=True)
            now = time.monotonic()
            if sequence is not None and playing.value and now - tick >= 1 / speed.value:
                frame.value = (frame.value + 1) % sequence.count
                tick = now
            key = (frame.value, mode.value)
            if sequence is not None and key != previous:
                original, soma, obj = sequence.frame(frame.value)
                offset = np.array([shift if mode.value == 'Side by side' else 0, 0, 0])
                with server.atomic():
                    for h, vertices in zip(handles, [original - center, obj - center, soma - center + offset, obj - center + offset]):
                        h.vertices = vertices.astype(np.float32)
                previous = key
            time.sleep(0.005)
        except Exception as exc:
            traceback.print_exc()
            status.content = f'Error: {exc}'
            playing.value = False
            sequence = None
            time.sleep(0.2)


if __name__ == '__main__':
    main()
