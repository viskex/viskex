# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Scale from pixels to plotter world-space size."""

import typing

import numpy as np
import numpy.typing as npt
import pyvista
import vtkmodules.vtkRenderingCore


def compute_screen_pixel_size_in_world(
    plotter: pyvista.Plotter, point: typing.Optional[npt.NDArray[typing.Any]] = None
) -> float:
    """
    Compute the world-space size that corresponds to 1 screen pixel at a given point.

    Parameters
    ----------
    plotter
        The plotter with camera and viewport settings.
    point
        World-space point to compute size at (only used for perspective projection).
        Defaults to camera focal point.

    Returns
    -------
    float
        World-space length per screen pixel.
    """
    camera = plotter.camera
    height_px = plotter.window_size[1]

    if camera.GetParallelProjection():
        # Orthographic projection: world height = 2 * parallel scale
        world_height = 2 * camera.GetParallelScale()
        return world_height / height_px
    else:
        # Perspective projection
        if point is None:
            point = np.array(camera.GetFocalPoint())

        # Convert the world point to screen (display) coordinates
        world_to_display = vtkmodules.vtkRenderingCore.vtkCoordinate()
        world_to_display.SetCoordinateSystemToWorld()
        world_to_display.SetValue(*point)
        screen_point_1 = np.array(world_to_display.GetComputedDisplayValue(plotter.renderer))

        # Add a 10-pixel vertical offset in screen space
        offset_px = 10
        screen_point_2 = screen_point_1 + np.array([0, offset_px])

        # Convert both screen points back to world coordinates
        display_to_world = vtkmodules.vtkRenderingCore.vtkCoordinate()
        display_to_world.SetCoordinateSystemToDisplay()
        display_to_world.SetValue(*screen_point_1, 0)  # type: ignore[call-overload]
        world_1 = np.array(display_to_world.GetComputedWorldValue(plotter.renderer))
        display_to_world.SetValue(*screen_point_2, 0)  # type: ignore[call-overload]
        world_2 = np.array(display_to_world.GetComputedWorldValue(plotter.renderer))

        world_distance = np.linalg.norm(world_2 - world_1)
        return world_distance / offset_px  # type: ignore[return-value]
