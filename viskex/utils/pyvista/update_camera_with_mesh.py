# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Update the camera of a pyvista plotter without adding the object to it."""

import pyvista


def update_camera_with_mesh(plotter: pyvista.Plotter, mesh: pyvista.DataSet) -> None:
    """
    Update the camera of a pyvista plotter as if the given mesh was added.

    This works by temporarily adding an actor for the mesh to the renderer,
    updating the camera bounds accordingly, then removing the actor immediately.

    Parameters
    ----------
    plotter
        The pyvista plotter whose camera should be updated.
    mesh
        The mesh to simulate adding, used for computing bounds affecting the camera.
    """
    # Create a VTK mapper and actor for the mesh
    mapper = pyvista._vtk.vtkDataSetMapper()
    mapper.SetInputData(mesh)

    temp_actor = pyvista._vtk.vtkActor()
    temp_actor.SetMapper(mapper)

    # Add the actor temporarily
    plotter.renderer.AddActor(temp_actor)

    # Update the camera to fit the mesh bounds
    plotter.reset_camera()  # type: ignore[call-arg]

    # Remove the actor immediately
    plotter.renderer.RemoveActor(temp_actor)
