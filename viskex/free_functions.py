# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Free functions to be imported in main module."""

import typing

import panel.pane.vtk.vtk
import plotly.graph_objects as go
import plum
import pyvista.trame.jupyter

try:
    import dolfinx  # noqa: F401
except ImportError:
    has_dolfinx = False
else:
    has_dolfinx = True
    from viskex.dolfinx_plotter import DolfinxPlotter
    _dolfinx_plotter_instance = DolfinxPlotter()

try:
    import firedrake  # noqa: F401
except ImportError:
    has_firedrake = False
else:
    has_firedrake = True
    from viskex.firedrake_plotter import FiredrakePlotter
    _firedrake_plotter_instance = FiredrakePlotter()

# We need to introduce a dependency on a multiple dispatch library and cannot use functools.singledispatch
# because the latter would ask to decorate a base implementation with @functools.singledispatch.
# No such base implementation is available in our case.

plot_mesh_dispatcher = plum.Dispatcher()

if has_dolfinx:
    @plot_mesh_dispatcher
    def _plot_mesh(mesh: dolfinx.mesh.Mesh, dim: typing.Optional[int] = None) -> typing.Union[
            go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]:
        """
        Plot a mesh stored in dolfinx.mesh.Mesh object.

        Parameters
        ----------
        mesh
            A dolfinx mesh to be plotted.
        dim
            Plot entities associated to this dimension. If not provided, the topological dimension is used.

        Returns
        -------
        :
            A widget representing a plot of the mesh.
        """
        return _dolfinx_plotter_instance.plot_mesh(mesh, dim)

if has_firedrake:
    @plot_mesh_dispatcher  # type: ignore[no-redef]
    def _plot_mesh(mesh: firedrake.MeshGeometry, dim: typing.Optional[int] = None)  -> typing.Union[  # noqa: F811
            go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]:
        """
        Plot a mesh stored in firedrake.MeshGeometry object.

        Parameters
        ----------
        mesh
            A firedrake mesh to be plotted.
        dim
            Plot entities associated to this dimension. If not provided, the topological dimension is used.

        Returns
        -------
        :
            A widget representing a plot of the mesh.
        """
        return _firedrake_plotter_instance.plot_mesh(mesh, dim)

if has_dolfinx:
    @typing.overload
    def plot_mesh(mesh: dolfinx.mesh.Mesh, dim: typing.Optional[int] = None) -> typing.Union[
            go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]:
        """Stub of plot_mesh for type checking. See the concrete implementation above."""
        ...  # pragma: no cover

if has_firedrake:
    @typing.overload
    def plot_mesh(mesh: firedrake.MeshGeometry, dim: typing.Optional[int] = None)  -> typing.Union[
            go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]:
        """Stub of plot_mesh for type checking. See the concrete implementation above."""
        ...  # pragma: no cover


def plot_mesh(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Dispatcher of plot_mesh for type checking. See the concrete implementation above."""
    return _plot_mesh(*args, **kwargs)
