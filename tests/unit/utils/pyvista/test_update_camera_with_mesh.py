# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Tests for viskex.utils.pyvista.update_camera_with_mesh module."""

import numpy as np
import pyvista

import viskex.utils.dtype
import viskex.utils.pyvista


def test_update_camera_with_mesh() -> None:
    """
    Test that `update_camera_with_mesh` updates camera parameters correctly without permanently adding the mesh.

    The test compares the camera parameters obtained after `update_camera_with_mesh` to the ones that are obtained
    after adding the mesh permanently to a new plotter.
    """
    # Create initial plotter without any mesh
    plotter_1 = pyvista.Plotter(off_screen=True)  # type: ignore[no-untyped-call]
    initial_position_1 = plotter_1.camera.position
    initial_focal_point_1 = plotter_1.camera.focal_point
    initial_clipping_range_1 = plotter_1.camera.clipping_range

    # Create mesh with points offset in space
    points = np.array([[10, 0, 0], [10, 1, 1], [11, 1, 0]], dtype=viskex.utils.dtype.RealType)
    mesh = pyvista.PolyData(points)  # type: ignore[arg-type, unused-ignore]

    # Call the update function (temporary add + camera update)
    viskex.utils.pyvista.update_camera_with_mesh(plotter_1, mesh)
    updated_position_1 = plotter_1.camera.position
    updated_focal_point_1 = plotter_1.camera.focal_point
    updated_clipping_range_1 = plotter_1.camera.clipping_range

    # Verify camera parameters changed after update
    assert not np.allclose(updated_position_1, initial_position_1), "Camera position did not update"
    assert not np.allclose(updated_focal_point_1, initial_focal_point_1), "Camera focal point did not update"
    assert not np.allclose(updated_clipping_range_1, initial_clipping_range_1), "Camera clipping range did not update"

    # Create a second plotter and permanently add the mesh
    plotter_2 = pyvista.Plotter(off_screen=True)  # type: ignore[no-untyped-call]
    plotter_2.add_mesh(mesh)
    plotter_2.reset_camera()  # type: ignore[call-arg]

    updated_position_2 = plotter_2.camera.position
    updated_focal_point_2 = plotter_2.camera.focal_point
    updated_clipping_range_2 = plotter_2.camera.clipping_range

    # The two plotters should now have similar camera parameters
    assert np.allclose(updated_position_1, updated_position_2), "Camera positions differ"
    assert np.allclose(updated_focal_point_1, updated_focal_point_2), "Camera focal points differ"
    assert np.allclose(updated_clipping_range_1, updated_clipping_range_2), "Camera clipping ranges differ"

    # Clean up
    plotter_1.close()  # type: ignore[no-untyped-call]
    plotter_2.close()  # type: ignore[no-untyped-call]
