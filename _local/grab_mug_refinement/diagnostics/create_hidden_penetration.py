# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explain measured but occluded hand penetration using unchanged native meshes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as PlotPath

LOCAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCAL))
sys.path.insert(0, str(LOCAL.parent / "mug_comparisons"))
from audit_results import surface_distances
from render_gl import GLRenderer


def digest(path: Path) -> str:
    """Return a source file's SHA-256 digest."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def project(points: np.ndarray, camera: dict, width: int, height: int) -> np.ndarray:
    """Project unchanged world points [m] into top-left-origin image pixels."""
    homogeneous = np.column_stack([points, np.ones(len(points))])
    clip = homogeneous @ GLRenderer._view_projection(camera).T
    normalized = clip[:, :3] / clip[:, 3:4]
    return np.column_stack([(normalized[:, 0] + 1) * width / 2, (1 - normalized[:, 1]) * height / 2])


def plane_segments(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Intersect original mesh triangles with a plane and return plane coordinates [mm]."""
    segments = trimesh.intersections.mesh_plane(mesh, normal, origin)
    return (segments - origin) @ basis * 1000


def material_mask(segments: np.ndarray, minimum: float, maximum: float, resolution: int = 500) -> np.ndarray:
    """Rasterize exact closed cross-section contours with even-odd solid occupancy [mm]."""
    # Join identical endpoints locally instead of requiring optional NetworkX.
    # The 1 nm matching tolerance is far below any displayed mesh precision.
    points_by_key = {}
    neighbors = {}
    for segment in segments:
        keys = [tuple(np.rint(point / 1e-6).astype(np.int64)) for point in segment]
        if keys[0] == keys[1]:
            continue
        for point, key, other in zip(segment, keys, keys[::-1]):
            points_by_key.setdefault(key, point)
            neighbors.setdefault(key, set()).add(other)
    if not neighbors or any(len(value) != 2 for value in neighbors.values()):
        raise ValueError("Native mug cross-section does not form closed two-neighbor contours")
    unused = set(neighbors)
    loops = []
    while unused:
        start = min(unused)
        previous, current = None, start
        loop = []
        while True:
            loop.append(points_by_key[current])
            unused.remove(current)
            following = next(value for value in sorted(neighbors[current]) if value != previous)
            previous, current = current, following
            if current == start:
                loop.append(points_by_key[start])
                break
        loops.append(np.asarray(loop))
    coordinates = np.linspace(minimum, maximum, resolution)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    occupied = np.zeros(len(points), dtype=bool)
    for loop in loops:
        occupied ^= PlotPath(loop, closed=True).contains_points(points)
    return occupied.reshape(resolution, resolution)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=LOCAL / "runs/all44_v2")
    parser.add_argument("--clip_id", default="s1__mug_drink_1")
    parser.add_argument("--frame", type=int, default=166)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = args.run / "geometry" / (args.clip_id + ".npz")
    video_report_path = args.run / "share/clips" / (args.clip_id + ".json")
    before_hash = digest(geometry_path)
    with np.load(geometry_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    frame = args.frame
    if not 0 <= frame < len(arrays["obj_rot"]):
        raise ValueError("Frame is outside the clip")
    rotation = arrays["obj_rot"][frame].astype(np.float64)
    translation = arrays["obj_trans"][frame].astype(np.float64)
    mug = trimesh.Trimesh(arrays["object_vertices_canonical"], arrays["object_faces"], process=False)
    if not mug.is_watertight or not mug.is_winding_consistent or mug.volume <= 0:
        raise ValueError("Diagnostic requires the native closed mug solid")
    world = {
        stage: np.concatenate([arrays[f"vertices_l_{stage}"][frame], arrays[f"vertices_r_{stage}"][frame]])
        for stage in ("before", "after")
    }
    local = {stage: (points.astype(np.float64) - translation) @ rotation for stage, points in world.items()}
    metrics = {}
    for stage, points in local.items():
        distance, inside, disagreement = surface_distances(mug, points)
        depth = np.where(inside, distance, 0)
        metrics[stage] = {"distance": distance, "inside": inside, "disagreement": disagreement, "depth": depth}
    inside_indices = np.flatnonzero(metrics["before"]["inside"])
    selected = np.flatnonzero(metrics["before"]["depth"] > 0.002)
    deepest = int(np.argmax(metrics["before"]["depth"]))
    deepest_point = local["before"][deepest]
    nearest, distance, triangle_index = trimesh.proximity.closest_point(mug, deepest_point[None])
    nearest = nearest[0]
    depth_mm = float(distance[0] * 1000)
    nearest_direction = (nearest - deepest_point) / distance[0]
    plane_normal = np.cross(nearest_direction, [0.0, 0.0, 1.0])
    plane_normal /= np.linalg.norm(plane_normal)
    vertical = np.cross(plane_normal, nearest_direction)
    basis = np.column_stack([nearest_direction, vertical])
    object_segments = plane_segments(mug, deepest_point, plane_normal, basis)
    hand_faces = np.concatenate([arrays["faces_l"], arrays["faces_r"] + 778])
    hand = trimesh.Trimesh(local["before"], hand_faces, process=False)
    hand_segments = plane_segments(hand, deepest_point, plane_normal, basis)
    limit = 11.0
    occupied = material_mask(object_segments, -limit, limit)

    video_report = json.loads(video_report_path.read_text())
    camera = copy.deepcopy(video_report["cameras"]["detail_frames"][frame])
    # Same viewing direction as the exported video, centered more closely on the
    # unchanged mug. Both image panels use this identical orthographic camera.
    target = mug.bounds.mean(axis=0) @ rotation.T + translation
    direction = np.asarray(camera["view_direction"])
    eye_distance = np.linalg.norm(np.asarray(camera["eye"]) - np.asarray(camera["target"]))
    camera.update(target=target.tolist(), eye=(target + direction * eye_distance).tolist(), height=0.18, width=0.18)
    camera["coordinate_frame"] = "world_follow_handle"
    specs = {
        "canonical_to_z_up": np.eye(3).tolist(), "overview": copy.deepcopy(camera),
        "detail": copy.deepcopy(camera), "detail_frames": [camera],
    }
    data = {
        "original_vertices": world["before"][None], "soma_vertices": world["before"][None],
        "original_faces": hand_faces, "soma_faces": hand_faces,
        "object_vertices_canonical": np.asarray(mug.vertices), "object_faces": np.asarray(mug.faces),
        "object_rotation": rotation[None], "object_translation": translation[None],
    }
    renderer = GLRenderer(width=900, height=900)
    opaque = renderer.render_clip(data, specs, 0, 1)[0, 2]
    pixels = project(world["before"][selected], camera, 900, 900)
    deep_pixel = project(world["before"][deepest : deepest + 1], camera, 900, 900)[0]
    if np.any(pixels < 0) or np.any(pixels >= 900):
        raise ValueError("A measured penetration marker falls outside the close-up")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold"})
    figure = plt.figure(figsize=(19.5, 8.0), facecolor="#f8fafc")
    grids = figure.add_gridspec(1, 3, left=0.022, right=0.978, bottom=0.22, top=0.80, wspace=0.09)
    axes = [figure.add_subplot(grids[0, i]) for i in range(3)]
    figure.text(0.025, 0.935, "Why a hand can look fine while intersecting the mug", fontsize=23, weight="bold",
                color="#182d40")
    source_frame = int(arrays["source_frame_indices"][frame])
    figure.text(0.025, 0.888,
                f"GRAB {args.clip_id}  |  source frame {source_frame} (zero-based), {source_frame / 120:.2f} s"
                "  |  unchanged personalized MANO + native mug", fontsize=12, color="#475569")
    for axis in axes[:2]:
        axis.imshow(opaque)
        axis.set_xlim(0, 900)
        axis.set_ylim(900, 0)
        axis.axis("off")
    axes[0].set_title("A  Normal opaque view", loc="left", pad=13, fontsize=15)
    axes[1].set_title("B  Same view + x-ray markers", loc="left", pad=13, fontsize=15)
    axes[1].scatter(pixels[:, 0], pixels[:, 1], s=64, c="#ef3340", edgecolors="white", linewidths=1.1, zorder=3)
    axes[1].scatter(*deep_pixel, s=190, marker="*", c="#b80020", edgecolors="white", linewidths=1.0, zorder=4)
    axes[1].annotate(
        f"Deepest: {depth_mm:.2f} mm\nright thumb vertex {deepest - 778}",
        xy=deep_pixel, xytext=(0.06, 0.09), textcoords="axes fraction", fontsize=11,
        color="#9d1026", weight="bold", arrowprops={"arrowstyle": "->", "color": "#b80020", "lw": 1.5},
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#ecc8d0", "alpha": 0.96},
    )
    axes[0].text(0, -0.075, "The mug surface hides vertices inside its material.", transform=axes[0].transAxes,
                 fontsize=11, color="#475569")
    axes[1].text(0, -0.075, f"{len(selected)} vertices >2 mm inside; markers drawn through surfaces.",
                 transform=axes[1].transAxes, fontsize=11, color="#475569")
    axes[1].text(0, -0.126, "Diagnostic markers, not visible holes. Marker size is enlarged.",
                 transform=axes[1].transAxes, fontsize=10.3, color="#9d1026")

    axis = axes[2]
    axis.set_title("C  Exact mesh cross-section", loc="left", pad=13, fontsize=15)
    axis.imshow(occupied, origin="lower", extent=(-limit, limit, -limit, limit),
                cmap=ListedColormap(["#ffffff", "#b6cbc3"]), interpolation="nearest", vmin=0, vmax=1, zorder=0)
    axis.add_collection(LineCollection(object_segments, colors="#3c6558", linewidths=1.3, zorder=1))
    axis.add_collection(LineCollection(hand_segments, colors="#286fcb", linewidths=1.35, zorder=2))
    axis.plot([], [], color="#3c6558", lw=5, label="Mug material (filled)")
    axis.plot([], [], color="#286fcb", lw=1.6, label="Hand surface slice")
    axis.scatter([0], [0], s=125, c="#b80020", marker="*", edgecolors="white", linewidths=0.7, zorder=6)
    axis.scatter([depth_mm], [0], s=52, c="#3c6558", marker="o", edgecolors="white", linewidths=0.9, zorder=6)
    axis.annotate("", xy=(depth_mm, 0), xytext=(0, 0),
                  arrowprops={"arrowstyle": "<->", "color": "#b80020", "lw": 1.8}, zorder=5)
    axis.text(depth_mm / 2, 1.3, f"{depth_mm:.2f} mm", color="#a01227", fontsize=13, weight="bold", ha="center",
              bbox={"fc": "white", "ec": "none", "alpha": 0.88, "pad": 2})
    axis.annotate("Hand vertex\ninside solid", (0, 0), xytext=(-9.5, -7.5), fontsize=10, color="#a01227",
                  arrowprops={"arrowstyle": "->", "color": "#a01227"})
    axis.annotate("Nearest mug\nsurface point", (depth_mm, 0), xytext=(2.5, 6.0), fontsize=10, color="#31594d",
                  arrowprops={"arrowstyle": "->", "color": "#31594d"})
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.set_xlabel("Distance toward nearest mug surface (mm)", fontsize=10)
    axis.set_ylabel("Position within slice (mm)", fontsize=10)
    axis.tick_params(labelsize=9)
    axis.grid(color="#d6dde2", alpha=0.55, linewidth=0.55)
    axis.legend(loc="upper left", fontsize=8.8, framealpha=0.96, borderpad=0.5)
    axis.spines[["top", "right"]].set_visible(False)

    after_deep = metrics["after"]["depth"][deepest] * 1000
    after_count = int((metrics["after"]["depth"] > 0.002).sum())
    figure.text(0.025, 0.095,
                f"At this frame: {len(inside_indices)} original vertices are inside mug material; "
                f"{len(selected)} are deeper than 2 mm. After refinement: {after_count} are deeper than 2 mm.",
                fontsize=12, color="#263d50")
    figure.text(0.025, 0.052,
                f"The highlighted vertex changes from {depth_mm:.2f} to {after_deep:.2f} mm deep. "
                "Exact triangle distances + two ray checks; empty cup/handle space is not counted as solid.",
                fontsize=11, color="#475569")
    image_path = args.output_dir / "hidden_penetration.png"
    figure.savefig(image_path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)

    records = []
    for index in selected:
        nearest_point, nearest_distance, face_index = trimesh.proximity.closest_point(mug, local["before"][index][None])
        records.append({
            "hand": "left" if index < 778 else "right", "vertex_id": int(index % 778),
            "canonical_point_m": local["before"][index].tolist(),
            "nearest_surface_point_m": nearest_point[0].tolist(), "nearest_triangle": int(face_index[0]),
            "exact_depth_before_mm": float(nearest_distance[0] * 1000),
            "exact_depth_after_mm": float(metrics["after"]["depth"][index] * 1000),
        })
    if digest(geometry_path) != before_hash:
        raise ValueError("Source geometry changed during diagnostic")
    report = {
        "completed": True, "image": str(image_path.resolve()), "image_sha256": digest(image_path),
        "geometry": str(geometry_path.resolve()), "geometry_sha256_unchanged": before_hash,
        "clip_id": args.clip_id, "export_frame_zero_based": frame, "source_frame_zero_based": source_frame,
        "method": "Native triangle nearest points; containment confirmed by two deterministic ray directions",
        "source_geometry_modified": False,
        "diagnostic_markers": (
            "Projected original vertex positions drawn on top of opaque image; enlarged size is not geometry"
        ),
        "before_inside_vertices": len(inside_indices), "before_vertices_over_2_mm": len(selected),
        "after_inside_vertices": int(metrics["after"]["inside"].sum()), "after_vertices_over_2_mm": after_count,
        "before_ray_disagreements": int(metrics["before"]["disagreement"].sum()),
        "after_ray_disagreements": int(metrics["after"]["disagreement"].sum()),
        "highlighted_hand": "right" if deepest >= 778 else "left", "highlighted_vertex_id": deepest % 778,
        "highlighted_depth_before_mm": depth_mm, "highlighted_depth_after_mm": float(after_deep),
        "highlighted_nearest_triangle": int(triangle_index[0]),
        "camera": camera, "renderer": renderer.renderer_info,
        "slice": {"origin_m": deepest_point.tolist(), "normal": plane_normal.tolist(),
                  "basis_xyz_columns": basis.tolist(), "boundary_source": "Exact original triangle-plane intersections",
                  "fill": "Even-odd occupancy of closed triangle-section contours; rasterized solely for display"},
        "vertices": records,
        "limitations": ["One explanatory frame, not an aggregate benchmark or proof of collision-free motion.",
                        "Point penetration does not quantify intersecting mesh volume or physical grasp stability."],
    }
    (args.output_dir / "hidden_penetration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"completed": True, "image": str(image_path), "deepest_mm": depth_mm,
                      "before_over_2_mm": len(selected), "after_over_2_mm": after_count}), flush=True)


if __name__ == "__main__":
    main()
