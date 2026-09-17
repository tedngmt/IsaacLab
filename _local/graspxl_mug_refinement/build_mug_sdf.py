# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build and independently validate a signed distance grid for the GraspXL mug.

Distances are negative in mug material and positive in empty space, including
the cup cavity and handle opening. The saved grid is ordered Z, Y, X; use XYZ
coordinates normalized by ``grid_min`` and ``grid_max`` with PyTorch's
``grid_sample(..., align_corners=True, mode="bilinear")`` for trilinear values
and differentiable point positions. Outside the padded grid, penetration is
zero; do not interpret border-clamped distances as true exterior distances.

Run using the existing soma-x environment, which includes Warp and the R-tree
dependency required for independent Trimesh validation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import warp as wp
from scipy.ndimage import map_coordinates
from trimesh.ray.ray_util import contains_points


@wp.kernel
def query_grid(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    spacing: float,
    nx: int,
    ny: int,
    output: wp.array(dtype=wp.float32),
):
    """Evaluate signed nearest-triangle distances [m] at regular grid nodes."""
    index = wp.tid()
    x = index % nx
    y = (index // nx) % ny
    z = index // (nx * ny)
    point = lower + spacing * wp.vec3(float(x), float(y), float(z))
    query = wp.mesh_query_point_sign_winding_number(mesh_id, point, 1.0)
    if query.result:
        nearest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        output[index] = query.sign * wp.length(point - nearest)
    else:
        output[index] = 1000.0


@wp.kernel
def query_points(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=wp.float32),
):
    """Evaluate signed nearest-triangle distances [m] at arbitrary points [m]."""
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_sign_winding_number(mesh_id, point, 1.0)
    if query.result:
        nearest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        output[index] = query.sign * wp.length(point - nearest)
    else:
        output[index] = 1000.0


def interpolate(grid: np.ndarray, points: np.ndarray, lower: np.ndarray, spacing: float) -> np.ndarray:
    """Interpolate a ZYX signed distance grid [m] at XYZ points [m]."""
    xyz = ((points - lower) / spacing).T
    return map_coordinates(grid, xyz[[2, 1, 0]], order=1, mode="nearest", prefilter=False)


def validate(
    mesh: trimesh.Trimesh,
    warp_mesh: wp.Mesh,
    grid: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spacing: float,
    device: str,
    sample_count: int,
    seed: int,
) -> tuple[dict, dict]:
    """Validate distances [m] and solid signs against independent triangle/ray queries."""
    random = np.random.default_rng(seed)
    surface_count = sample_count // 2
    face_indices = random.choice(len(mesh.faces), surface_count, p=mesh.area_faces / mesh.area)
    barycentric = random.dirichlet([1.0, 1.0, 1.0], size=surface_count)
    surface = np.einsum("ni,nij->nj", barycentric, mesh.triangles[face_indices])
    offsets = random.uniform(0.0001, 0.010, size=surface_count)
    offsets[::2] *= -1.0
    near_surface = surface + offsets[:, None] * mesh.face_normals[face_indices]
    uniform = random.uniform(lower, upper, size=(sample_count - surface_count, 3))
    points = np.concatenate([near_surface, uniform]).astype(np.float32)
    nearest, distances, faces = trimesh.proximity.closest_point(mesh, points)
    inside = mesh.contains(points)
    inside_second_ray = contains_points(mesh.ray, points, check_direction=np.array([0.38, 0.72, 0.57]))
    if np.any(inside != inside_second_ray):
        raise ValueError("Independent Trimesh ray directions disagree on validation signs")
    reference = np.where(inside, -distances, distances)
    query_output = wp.empty(len(points), dtype=wp.float32, device=device)
    wp.launch(
        query_points,
        len(points),
        [warp_mesh.id, wp.array(points, dtype=wp.vec3, device=device)],
        outputs=[query_output],
        device=device,
    )
    direct = query_output.numpy()
    interpolated = interpolate(grid, points, lower, spacing)
    direct_error = np.abs(direct - reference)
    interpolation_error = np.abs(interpolated - reference)
    # Float32 GPU triangle predicates differ slightly from Trimesh float64 for
    # the native scan's very small/thin triangles. Require sub-0.1 mm accuracy.
    if direct_error.max() > 1e-4:
        raise ValueError(f"Warp signed distance differs from independent triangles: {direct_error.max()} m")
    if np.any((direct < 0) != inside):
        raise ValueError("Warp winding sign differs from independent Trimesh ray parity")
    # A 0.75 mm grid cannot resolve every arbitrarily close surface point.
    # Outside a two-voxel uncertainty band, its solid classifications must agree.
    away_from_surface = distances > 2.0 * spacing
    if np.any((interpolated[away_from_surface] < 0) != inside[away_from_surface]):
        raise ValueError("Interpolated grid has incorrect signs away from the surface")
    if interpolation_error.max() > 2.0 * spacing:
        raise ValueError("Interpolation error exceeds two grid spacings")

    special_points = np.array(
        [
            [-0.018, 0.0, 0.0],  # Center of the Y-up cup cavity.
            [-0.018, 0.040, 0.0],  # Cup cavity near the open upper rim.
            [0.043, 0.0, 0.0],  # Handle opening.
            [0.057, 0.0, 0.0],  # Solid outer handle.
            [-0.018, -0.050, 0.0],  # Solid cup base.
        ],
        dtype=np.float32,
    )
    special_names = [
        "cup_cavity_center",
        "cup_cavity_near_rim",
        "handle_opening",
        "handle_material",
        "cup_base_material",
    ]
    special_inside = mesh.contains(special_points)
    expected_inside = np.array([False, False, False, True, True])
    special_values = interpolate(grid, special_points, lower, spacing)
    if not np.array_equal(special_inside, expected_inside) or not np.array_equal(special_values < 0, expected_inside):
        raise ValueError("Mug cavity, handle opening, or solid handle validation failed")

    import torch
    import torch.nn.functional as functional

    tensor_points = torch.tensor(points[:32], dtype=torch.float32, requires_grad=True)
    tensor_lower = torch.as_tensor(lower, dtype=torch.float32)
    tensor_upper = torch.as_tensor(upper, dtype=torch.float32)
    normalized = 2 * (tensor_points - tensor_lower) / (tensor_upper - tensor_lower) - 1
    values = functional.grid_sample(
        torch.from_numpy(grid)[None, None],
        normalized[None, :, None, None],
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(-1)
    values.sum().backward()
    gradient = tensor_points.grad.detach().numpy()
    if not np.isfinite(gradient).all() or np.max(np.linalg.norm(gradient, axis=1)) < 0.1:
        raise ValueError("PyTorch trilinear distance gradients are invalid")
    np.testing.assert_allclose(values.detach().numpy(), interpolated[:32], atol=2e-7, rtol=1e-4)
    arrays = {
        "probe_points": points,
        "probe_reference_sdf": reference.astype(np.float32),
        "probe_warp_sdf": direct,
        "probe_interpolated_sdf": interpolated,
        "probe_reference_nearest": nearest.astype(np.float32),
        "probe_reference_faces": faces.astype(np.int32),
        "special_probe_points": special_points,
        "special_probe_sdf": special_values,
        "special_probe_expected_inside": expected_inside,
    }
    report = {
        "samples": sample_count,
        "seed": seed,
        "inside_samples": int(inside.sum()),
        "outside_samples": int((~inside).sum()),
        "independent_reference": "Trimesh exact closest triangles and two ray-parity containment directions",
        "warp_direct_max_abs_error_mm": float(direct_error.max() * 1000),
        "warp_direct_sign_mismatches": int(((direct < 0) != inside).sum()),
        "grid_interpolation_max_abs_error_mm": float(interpolation_error.max() * 1000),
        "grid_interpolation_mean_abs_error_mm": float(interpolation_error.mean() * 1000),
        "grid_interpolation_p95_abs_error_mm": float(np.quantile(interpolation_error, 0.95) * 1000),
        "grid_sign_mismatches": int(((interpolated < 0) != inside).sum()),
        "grid_sign_mismatches_farther_than_two_voxels": int(
            ((interpolated[away_from_surface] < 0) != inside[away_from_surface]).sum()
        ),
        "torch_grid_sample_values_match_scipy": True,
        "torch_position_gradients_finite": True,
        "torch_gradient_norm_range": [
            float(x) for x in (np.linalg.norm(gradient, axis=1).min(), np.linalg.norm(gradient, axis=1).max())
        ],
        "special_probes": [
            {"name": name, "point_m": point.tolist(), "inside_solid": bool(solid), "sdf_mm": float(sdf * 1000)}
            for name, point, solid, sdf in zip(special_names, special_points, special_inside, special_values)
        ],
    }
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    local = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=local / "prepared/collision_proxy.npz")
    parser.add_argument("--native_source", type=Path, default=local / "prepared/object.npz")
    parser.add_argument("--output", type=Path, default=local / "prepared/sdf.npz")
    parser.add_argument("--spacing_mm", type=float, default=0.75)
    parser.add_argument("--padding_mm", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.spacing_mm <= 0 or args.padding_mm < 10 or args.validation_samples < 100:
        parser.error("Spacing must be positive, padding at least 10 mm, and validation at least 100 samples")
    started = time.perf_counter()
    with np.load(args.source, allow_pickle=False) as source:
        vertices = source["object_vertices_canonical"].astype(np.float64)
        faces = source["object_faces"].astype(np.int32)
    with np.load(args.native_source, allow_pickle=False) as source:
        native_vertices = source["object_vertices_canonical"].astype(np.float64)
        native_faces = source["object_faces"].astype(np.int32)
    if not np.array_equal(vertices[faces], native_vertices[native_faces]):
        raise ValueError("Collision topology must preserve every ordered native triangle position exactly")
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Collision topology must be an outward-oriented watertight solid")
    spacing = args.spacing_mm / 1000.0
    lower = (mesh.bounds[0] - args.padding_mm / 1000).astype(np.float32)
    shape_xyz = np.ceil((mesh.bounds[1] + args.padding_mm / 1000 - lower) / spacing).astype(np.int32) + 1
    upper = (lower + (shape_xyz - 1) * spacing).astype(np.float32)
    point_count = int(np.prod(shape_xyz))
    wp.config.kernel_cache_dir = str(local / "prepared/warp_cache")
    wp.init()
    warp_mesh = wp.Mesh(
        wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=args.device),
        wp.array(faces.reshape(-1), dtype=wp.int32, device=args.device),
        support_winding_number=True,
    )
    output = wp.empty(point_count, dtype=wp.float32, device=args.device)
    print(json.dumps({"status": "building", "shape_xyz": shape_xyz.tolist(), "nodes": point_count}), flush=True)
    wp.launch(
        query_grid,
        point_count,
        [warp_mesh.id, wp.vec3(*lower), spacing, int(shape_xyz[0]), int(shape_xyz[1])],
        outputs=[output],
        device=args.device,
    )
    grid = output.numpy().reshape(tuple(shape_xyz[::-1]))
    if not np.isfinite(grid).all() or grid.max() >= 1:
        raise ValueError("A grid query failed")
    print(json.dumps({"status": "validating", "grid_range_m": [float(grid.min()), float(grid.max())]}), flush=True)
    validation, probe_arrays = validate(
        mesh, warp_mesh, grid, lower, upper, spacing, args.device, args.validation_samples, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".building.npz")
    np.savez_compressed(
        temporary,
        sdf_grid=grid,
        grid_min=lower,
        grid_max=upper,
        spacing_m=np.float32(spacing),
        shape_xyz=shape_xyz,
        canonical_vertices=native_vertices,
        canonical_faces=native_faces,
        collision_proxy_vertices=vertices,
        collision_proxy_faces=faces,
        axis_x=lower[0] + np.arange(shape_xyz[0], dtype=np.float32) * spacing,
        axis_y=lower[1] + np.arange(shape_xyz[1], dtype=np.float32) * spacing,
        axis_z=lower[2] + np.arange(shape_xyz[2], dtype=np.float32) * spacing,
        source_cache=np.asarray(str(args.source.resolve())),
        native_source_cache=np.asarray(str(args.native_source.resolve())),
        sign_convention=np.asarray("negative_inside_solid_positive_empty"),
        **probe_arrays,
    )
    temporary.replace(args.output)
    report = {
        "completed": True,
        "source": str(args.source.resolve()),
        "native_source": str(args.native_source.resolve()),
        "native_source_sha256": hashlib.sha256(args.native_source.read_bytes()).hexdigest(),
        "collision_proxy_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "native_vertices": len(native_vertices),
        "native_faces": len(native_faces),
        "ordered_native_triangle_positions_bitwise_identical": True,
        "geometry_change": (
            "None: 11 shared vertex indices duplicated to represent oriented manifold fans; all triangles identical"
        ),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "vertices": len(vertices),
        "faces": len(faces),
        "mesh_geometry_sha256": hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "solid_volume_m3": float(mesh.volume),
        "shape_xyz": shape_xyz.tolist(),
        "grid_layout": "zyx",
        "grid_min_m": lower.tolist(),
        "grid_max_m": upper.tolist(),
        "spacing_mm": args.spacing_mm,
        "padding_mm": args.padding_mm,
        "sign_convention": "negative_inside_solid_positive_empty",
        "method": "Warp closest triangle with fast winding-number solid sign",
        "validation": validation,
        "limitations": [
            "Trilinear distance approximations near mesh edges can differ by a fraction of a voxel.",
            "Outside padded grid, penetration is zero; clamped grid values are not exact exterior distances.",
            "Vertex penetration checks do not detect all possible triangle-triangle intersections.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
