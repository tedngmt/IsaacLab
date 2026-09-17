# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split shared vertex fans into a manifold collision topology without moving triangles.

The selected native GraspXL mug has balanced four-face edges. Pairing oriented
half-edges by local face-normal continuity yields a closed topology with exactly
the same ordered triangle coordinates. Source geometry is never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def split_vertex_fans(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Return reindexed vertices/faces with bitwise-identical triangle positions [m]."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    unique_edges, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    parents = np.arange(faces.size)
    pair_report = []

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(first: int, second: int) -> None:
        parents[find(first)] = find(second)

    for edge_index in range(len(unique_edges)):
        selected = np.flatnonzero(inverse == edge_index)
        forward = selected[edges[selected, 0] < edges[selected, 1]]
        reverse = selected[edges[selected, 0] > edges[selected, 1]]
        if len(forward) != len(reverse) or len(forward) not in (1, 2):
            raise ValueError("Only balanced two-face or four-face edges are supported")
        if len(forward) == 1:
            pairs = list(zip(forward, reverse))
        else:
            same_score = sum(
                np.dot(mesh.face_normals[a // 3], mesh.face_normals[b // 3]) for a, b in zip(forward, reverse)
            )
            flip_score = sum(
                np.dot(mesh.face_normals[a // 3], mesh.face_normals[b // 3]) for a, b in zip(forward, reverse[::-1])
            )
            pairs = list(zip(forward, reverse if same_score >= flip_score else reverse[::-1]))
            pair_report.append(
                {
                    "source_edge": unique_edges[edge_index].tolist(),
                    "incidence": int(counts[edge_index]),
                    "paired_source_faces": [[int(a // 3), int(b // 3)] for a, b in pairs],
                    "same_order_normal_dot_score": float(same_score),
                    "reverse_order_normal_dot_score": float(flip_score),
                }
            )
        for first, second in pairs:
            face_a, corner_a = divmod(int(first), 3)
            face_b, corner_b = divmod(int(second), 3)
            union(face_a * 3 + corner_a, face_b * 3 + (corner_b + 1) % 3)
            union(face_a * 3 + (corner_a + 1) % 3, face_b * 3 + corner_b)
    roots = np.asarray([find(index) for index in range(faces.size)])
    _, representative, remapped = np.unique(roots, return_index=True, return_inverse=True)
    source_indices = faces.reshape(-1)[representative]
    new_vertices = vertices[source_indices]
    new_faces = remapped.reshape(-1, 3).astype(np.int32)
    if not np.array_equal(new_vertices[new_faces], vertices[faces]):
        raise ValueError("Reindexing moved, reordered, or reversed source triangles")
    return new_vertices, new_faces, source_indices.astype(np.int64), pair_report


def main() -> None:
    """Save a separate collision proxy and prove unchanged surface triangles [m]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_file", type=Path, default=Path(__file__).with_name("prepared") / "object.npz")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("prepared") / "collision_proxy.npz")
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.object_file.read_bytes()).hexdigest()
    arrays = dict(np.load(args.object_file, allow_pickle=False))
    vertices, faces = arrays["object_vertices_canonical"], arrays["object_faces"]
    native = trimesh.Trimesh(vertices, faces, process=False)
    new_vertices, new_faces, source_indices, pair_report = split_vertex_fans(vertices, faces)
    proxy = trimesh.Trimesh(new_vertices, new_faces, process=False)
    if not proxy.is_watertight or not proxy.is_winding_consistent or proxy.body_count != 1:
        raise ValueError("Reindexed mesh is not one consistently oriented closed body")
    if abs(proxy.volume - native.volume) > 1e-12:
        raise ValueError("Reindexing changed oriented volume")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        object_vertices_canonical=new_vertices,
        object_faces=new_faces,
        vertex_source_indices=source_indices,
        face_source_indices=np.arange(len(faces), dtype=np.int64),
    )
    report = {
        "completed": True,
        "native_object_file": str(args.object_file.resolve()),
        "native_object_sha256": source_hash,
        "collision_proxy": str(args.output.resolve()),
        "collision_proxy_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "method": "Oriented half-edge pairing by face-normal continuity; split disconnected vertex fans",
        "native_vertices": len(vertices),
        "proxy_vertices": len(new_vertices),
        "native_faces": len(faces),
        "proxy_faces": len(new_faces),
        "ordered_triangle_coordinates_bitwise_identical": True,
        "triangle_coordinates_sha256": hashlib.sha256(np.ascontiguousarray(vertices[faces]).tobytes()).hexdigest(),
        "faces_deleted": 0,
        "faces_added": 0,
        "vertices_moved": 0,
        "surface_positions_changed": False,
        "native_watertight": bool(native.is_watertight),
        "proxy_watertight": bool(proxy.is_watertight),
        "proxy_winding_consistent": bool(proxy.is_winding_consistent),
        "proxy_connected_bodies": int(proxy.body_count),
        "proxy_euler_number": int(proxy.euler_number),
        "native_volume_m3": float(native.volume),
        "proxy_volume_m3": float(proxy.volume),
        "nonmanifold_edge_pairing": pair_report,
        "limitations": (
            "Reindexing does not remove self-intersections or zero-thickness folded triangles; "
            "geometric solid-sign validation remains necessary"
        ),
    }
    if hashlib.sha256(args.object_file.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source mesh file changed")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "nonmanifold_edge_pairing"}))


if __name__ == "__main__":
    main()
