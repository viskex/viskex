# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Tests for viskex.utils.pyvista.add_point_markers module."""

import typing
import zlib

import numpy as np
import numpy.typing as npt
import pytest
import pyvista

import viskex.utils.dtype
import viskex.utils.pyvista


def get_seed_from_points(points: npt.NDArray[viskex.utils.dtype.RealType]) -> int:
    """
    Compute a deterministic 32-bit integer seed based on the byte content of the points array.

    This function converts the given points array to bytes and computes a 32-bit checksum
    using the Adler-32 algorithm. The result can be used as a seed for random number
    generators to produce consistent randomness dependent on the exact point coordinates.

    Parameters
    ----------
    points
        An (N, 3) array of point coordinates as a numpy array.

    Returns
    -------
    int
        A 32-bit integer seed derived from the input points data.
    """
    points_bytes = points.tobytes()
    seed = zlib.adler32(points_bytes) & 0xFFFFFFFF
    return seed


def create_wireframe_poly_data(points: npt.NDArray[viskex.utils.dtype.RealType]) -> pyvista.PolyData:
    """
    Construct a wireframe polydata representing the edges of a regular grid from a flat point array.

    The points in the polydata are randomly permuted.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        Array of 3D coordinates assumed to be sampled from a regular grid and flattened
        in row-major (C-style) order. The function will attempt to infer the grid shape
        and connect adjacent points to form the wireframe.

    Returns
    -------
    pyvista.PolyData
        A polydata mesh containing line segments between adjacent grid points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be an (N, 3) array of 3D points")

    n_points = len(points)

    # Infer structured shape from unique coordinate counts
    x_vals, y_vals, z_vals = points[:, 0], points[:, 1], points[:, 2]
    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)
    z_unique = np.unique(z_vals)
    shape = (len(x_unique), len(y_unique), len(z_unique))

    if np.prod(shape) != n_points:
        raise ValueError(f"Inferred grid shape {shape} does not match number of points {n_points}")

    def idx(i: int, j: int, k: int) -> int:
        return i * shape[1] * shape[2] + j * shape[2] + k

    lines = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                index = idx(i, j, k)
                if i + 1 < shape[0]:
                    lines.append([2, index, idx(i + 1, j, k)])
                if j + 1 < shape[1]:
                    lines.append([2, index, idx(i, j + 1, k)])
                if k + 1 < shape[2]:
                    lines.append([2, index, idx(i, j, k + 1)])
    lines_np = np.array(lines, dtype=np.int32)

    # Define a random number generator
    seed = get_seed_from_points(points) + 1
    rng = np.random.default_rng(seed)

    # Apply a random permutation to the points
    perm = rng.permutation(n_points)
    points_permuted = points[perm]

    # Update the lines connectivity using the inverse permutation
    inv_perm = np.argsort(perm)
    lines_permuted = np.empty_like(lines_np)
    lines_permuted[:, 0] = lines_np[:, 0]
    lines_permuted[:, 1] = inv_perm[lines_np[:, 1]]
    lines_permuted[:, 2] = inv_perm[lines_np[:, 2]]

    return pyvista.PolyData(points_permuted, lines=lines_permuted)  # type: ignore[arg-type]


def create_points_poly_data(points: npt.NDArray[viskex.utils.dtype.RealType]) -> pyvista.PolyData:
    """
    Create a pyvista polydata object consisting only of points (vertices).

    This function constructs a polydata mesh from the given array of points,
    explicitly setting the vertices connectivity so that each point corresponds
    to a single vertex cell. Points and cells are permuted independently.

    Parameters
    ----------
    points
        An (N, 3) array of point coordinates.

    Returns
    -------
    pyvista.PolyData
        A polydata mesh with one vertex cell per point, where the vertex connectivity
        array (`vertices`) is set to sequentially reference all points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be an (N, 3) array of 3D points")

    n_points = len(points)

    # Define a random number generator
    seed = get_seed_from_points(points) + 2
    rng = np.random.default_rng(seed)

    # Apply a random permutation to the points
    perm = rng.permutation(n_points)
    points_permuted = points[perm]

    # Apply a different random permutation to the cells
    cells_permuted = np.empty(n_points * 2, dtype=np.int64)
    cells_permuted[0::2] = 1  # cell size equal to 1
    cells_permuted[1::2] = rng.permutation(n_points)

    return pyvista.PolyData(points_permuted, verts=cells_permuted)  # type: ignore[arg-type]


def create_unstructured_grid(points: npt.NDArray[viskex.utils.dtype.RealType]) -> pyvista.UnstructuredGrid:
    """
    Create an unstructured grid consisting solely of vertex cells from an array of points.

    Each point corresponds to a single vertex cell. The resulting unstructured grid
    has one cell per point, with cell type VTK_VERTEX.

    Parameters
    ----------
    points
        An (N, 3) array of point coordinates defining the vertex locations.

    Returns
    -------
    pyvista.UnstructuredGrid
        An unstructured grid where each cell is a vertex referencing one point from
        the input array. The cell connectivity array is constructed such that each
        cell size is 1.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be an (N, 3) array of 3D points")

    n_points = len(points)

    # Define a random number generator
    seed = get_seed_from_points(points) + 3
    rng = np.random.default_rng(seed)

    # Apply a random permutation to the points
    points_permuted = rng.permutation(points)

    # Apply a different random permutation to the cells
    cells_permuted = np.empty(n_points * 2, dtype=np.int64)
    cells_permuted[0::2] = 1  # cell size equal to 1
    cells_permuted[1::2] = rng.permutation(n_points)

    # Cell types for vertex cells is VTK_VERTEX = 1
    cell_types = np.full(n_points, 1, dtype=np.uint8)

    return pyvista.UnstructuredGrid(cells_permuted, cell_types, points_permuted)


@pytest.mark.parametrize("dim, point_markers_dim, create_data_set", [
    (0, 1, create_points_poly_data), (0, 2, create_points_poly_data), (0, 3, create_points_poly_data),
    (0, 1, create_unstructured_grid), (0, 2, create_unstructured_grid), (0, 3, create_unstructured_grid),
    (1, 1, create_wireframe_poly_data), (1, 2, create_wireframe_poly_data), (1, 3, create_wireframe_poly_data),
    (2, 2, create_wireframe_poly_data), (2, 3, create_wireframe_poly_data), (3, 3, create_wireframe_poly_data)
])
@pytest.mark.parametrize("with_data", [False, True])
def test_add_point_markers(
    dim: int, point_markers_dim: int,
    create_data_set: typing.Callable[[npt.NDArray[viskex.utils.dtype.RealType]], pyvista.DataSet],
    with_data: bool
) -> None:
    """
    Visually compare `add_point_markers` with PyVista's built-in `show_vertices=True`.

    This test runs across 0D (point cloud), 1D, 2D, and 3D structured point configurations.
    It optionally includes scalar data for colormapping, coming from either:
    - cell data for dim = 0
    - point data for dim > 0

    In each subplot:
    - Left: custom point markers rendered with `add_point_markers`
    - Right: pyvista native vertex rendering using `show_vertices=True`

    Glyphs should appear consistent across both, but `add_point_markers` should yield solid shapes
    that do not vary in size with zoom (screen-space scaling), while `show_vertices=True` may produce
    inconsistent appearance when zooming (e.g., partially filled squares or outlines).

    Parameters
    ----------
    dim
        Topological dimension of the point set (0, 1, 2, or 3).
    point_markers_dim
        Topological dimension of the points marker.
    with_data
        Whether to attach scalar data and use colormapping instead of fixed color.
    """
    plotter = pyvista.Plotter(shape=(1, 2), window_size=[1200, 600])

    if dim == 0:
        # A polydata formed only by points
        x = np.linspace(0, 1, 10, dtype=viskex.utils.dtype.RealType)
        points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    elif dim == 1:
        # Line along x-axis with 10 points
        x = np.linspace(0, 1, 10, dtype=viskex.utils.dtype.RealType)
        points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    elif dim == 2:
        # 5 x 5 grid in XY plane
        x, y = np.meshgrid(
            np.linspace(0, 1, 5, dtype=viskex.utils.dtype.RealType),
            np.linspace(0, 1, 5, dtype=viskex.utils.dtype.RealType)
        )
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros_like(x).ravel()))
    elif dim == 3:
        # 3 x 3 x 3 grid in 3D
        x, y, z = np.meshgrid(  # type: ignore[assignment,unused-ignore]
            np.linspace(0, 1, 3, dtype=viskex.utils.dtype.RealType),
            np.linspace(0, 1, 3, dtype=viskex.utils.dtype.RealType),
            np.linspace(0, 1, 3, dtype=viskex.utils.dtype.RealType),
            indexing="ij"
        )
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    else:
        raise ValueError(f"Unsupported dimension {dim}")

    mesh = create_data_set(points)

    if with_data:
        if dim == 0:
            if isinstance(mesh, pyvista.PolyData):
                cell_point_indices = mesh.verts[1::2]
            elif isinstance(mesh, pyvista.UnstructuredGrid):
                cell_point_indices = mesh.cells[1::2]
            else:
                raise TypeError(f"Unsupported mesh type for cell data: {type(mesh)}")
            values = np.linalg.norm(mesh.points[cell_point_indices], axis=1)
            mesh.cell_data["distance"] = values
        else:
            values = np.linalg.norm(mesh.points, axis=1)
            mesh.point_data["distance"] = values
        mesh.set_active_scalars("distance")


    # Set a large point size to be able to visualize the points
    point_size = 25

    # Left subplot: custom point markers
    plotter.subplot(0, 0)
    if dim > 0:
        plotter.add_mesh(mesh, color="black", style="wireframe", line_width=1)
    if dim < 3:
        plotter.camera_position = "xy"
    if dim > 0:
        plotter.reset_camera()  # type: ignore[call-arg]
    else:
        viskex.utils.pyvista.update_camera_with_mesh(plotter, mesh)
    viskex.utils.pyvista.add_point_markers(
        plotter, mesh, dim=point_markers_dim, point_size=point_size,
        point_color=None if with_data else "red"
    )

    # Right subplot: show_vertices=True
    plotter.subplot(0, 1)
    if dim > 0:
        plotter.add_mesh(mesh, color="black", style="wireframe", line_width=1)
    plotter.add_mesh(
        mesh, style="points", point_size=point_size,
        color=None if with_data else "red", scalars="distance" if with_data else None
    )
    if dim < 3:
        plotter.camera_position = "xy"
    if with_data:
        plotter.add_scalar_bar()  # type: ignore[call-arg]

    if not pyvista.OFF_SCREEN:
        plotter.link_views()
        plotter.show()
