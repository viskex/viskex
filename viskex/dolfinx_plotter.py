# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing dolfinx."""

import typing

import dolfinx.mesh
import dolfinx.plot
import numpy as np
import panel.pane.vtk.vtk
import plotly.graph_objects as go
import pyvista.trame.jupyter

from viskex.base_plotter import BasePlotter
from viskex.plotly_plotter import PlotlyPlotter
from viskex.pyvista_plotter import PyvistaPlotter


class DolfinxPlotter(BasePlotter):
    """viskex plotter interfacing dolfinx."""

    _plotly_plotter = PlotlyPlotter()
    _pyvista_plotter = PyvistaPlotter()

    def plot_mesh(self, mesh: dolfinx.mesh.Mesh) -> typing.Union[
            go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]:
        """
        Plot a mesh stored in dolfinx.mesh.Mesh object.

        Parameters
        ----------
        mesh
            A dolfinx mesh to be plotted.

        Returns
        -------
        :
            A widget representing a plot of the mesh.
        """
        if mesh.topology.dim == 1:
            vertices = mesh.geometry.x[:, 0]
            assert np.all(vertices[1:] >= vertices[:-1])
            mesh.topology.create_connectivity(1, 0)
            cells = mesh.topology.connectivity(1, 0).array
            expected_cells = np.repeat(np.arange(vertices.shape[0], dtype=np.int32), 2)
            expected_cells = np.delete(np.delete(expected_cells, 0), -1)
            assert np.array_equal(cells, expected_cells)
            return self._plotly_plotter.plot_mesh(vertices)
        else:
            pyvista_grid = self._dolfinx_mesh_to_pyvista_grid(mesh, mesh.topology.dim)
            return self._pyvista_plotter.plot_mesh(pyvista_grid)

    def plot_mesh_entities(self, mesh, dim: int, entities) -> None:
        """Plot a mesh, highlighting the provided `dim`-dimensional entities."""
        pass  # pragma: no cover

    def plot_scalar_field(
        self, scalar_field, name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> None:
        """Plot a scalar field."""
        pass  # pragma: no cover

    def plot_vector_field(
        self, vector_field, name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0,
        part: str = "real"
    ) -> None:
        """Plot a vector field."""
        pass  # pragma: no cover

    @staticmethod
    def _dolfinx_mesh_to_pyvista_grid(mesh: dolfinx.mesh.Mesh, dim: int) -> pyvista.UnstructuredGrid:
        """Helper method to convert a dolfinx.mesh.Mesh to a pyvista.UnstructuredGrid."""
        mesh.topology.create_connectivity(dim, dim)
        num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
        cell_entities = np.arange(num_cells, dtype=np.int32)
        pyvista_cells, cell_types, coordinates = dolfinx.plot.create_vtk_mesh(mesh, dim, cell_entities)
        return pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)
