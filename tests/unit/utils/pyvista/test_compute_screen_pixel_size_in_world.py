# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Tests for viskex.utils.pyvista.compute_screen_pixel_size_in_world module."""

import numpy as np
import pytest
import pyvista

import viskex.utils.dtype
import viskex.utils.pyvista


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_compute_screen_pixel_size_in_world(parallel: bool, dim: int) -> None:
    """
    Test whether a manually sized square matches pyvista's screen-sized point rendering.

    The test considers both orthographic and perspective projections.

    This function:
    - Adds points in a simple 1D line, 2D grid, or 3D grid at the origin (in red)
    - Adds a manually sized square/cube using `compute_screen_pixel_size_in_world` sized to match point size (in blue)

    A VTK render screen will open, and you should compare the visual size of the red and blue squares.
    When looking at the default camera view, the two squares should look of similar size.
    Note however that when zooming in or out the blue square will increase or decrease its size,
    while the red one will keep its size: this is expected, as only add_points fully supports
    zooming in or out.

    Parameters
    ----------
    parallel
        Whether to use orthographic (parallel) projection. If False, uses perspective.
    dim
        The spatial dimension of the points (1, 2, or 3).
    """
    plotter = pyvista.Plotter(window_size=(600, 600))  # type: ignore[no-untyped-call]

    # Create a single point at origin
    point = np.array([[0.0, 0.0, 0.0]], dtype=viskex.utils.dtype.RealType)
    cloud = pyvista.PolyData(point)  # type: ignore[arg-type, unused-ignore]

    # Add the point using PyVista's point size
    point_size = 100.0
    plotter.add_points(cloud, point_size=point_size, color="red", render_points_as_spheres=False)

    # Set parallel projection
    plotter.camera.SetParallelProjection(parallel)

    # Set camera position
    if dim < 3:
        plotter.camera_position = "xy"

    # Reset camera in preparation of call to compute_screen_pixel_size_in_world
    plotter.reset_camera()  # type: ignore[call-arg]

    # Compute world-space size equivalent to screen pixels
    pixel_size = viskex.utils.pyvista.compute_screen_pixel_size_in_world(plotter, point=point[0])
    square_size = point_size * pixel_size

    # Add a world-space geometry structure at the same location
    if dim == 1:
        # Use a line segment as "square"
        line = pyvista.Line(
            point[0] - np.array([square_size / 2, 0, 0], dtype=viskex.utils.dtype.RealType),
            point[0] + np.array([square_size / 2, 0, 0], dtype=viskex.utils.dtype.RealType)
        )
        plotter.add_mesh(line, color="blue", line_width=3)
    elif dim == 2:
        square = pyvista.Plane(center=point[0], i_size=square_size, j_size=square_size)  # type: ignore[arg-type]
        plotter.add_mesh(square, color="blue", style="wireframe", line_width=2)
    else:  # dim == 3
        cube = pyvista.Cube(
            center=point[0], x_length=square_size, y_length=square_size, z_length=square_size, clean=False)
        plotter.add_mesh(cube, color="blue", style="wireframe", line_width=2)

    mode = "Orthographic" if parallel else "Perspective"
    plotter.add_text(f"{mode} projection, dim={dim}", font_size=10)  # type: ignore[no-untyped-call]
    if not pyvista.OFF_SCREEN:
        plotter.show()  # type: ignore[no-untyped-call]
