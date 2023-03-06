# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing firedrake."""

import typing

import firedrake
import numpy as np
import panel.pane.vtk.vtk
import plotly.graph_objects as go
import pyvista.trame.jupyter

from viskex.base_plotter import BasePlotter
from viskex.plotly_plotter import PlotlyPlotter
from viskex.pyvista_plotter import PyvistaPlotter


class FiredrakePlotter(BasePlotter):
    """viskex plotter interfacing firedrake."""

    _plotly_plotter = PlotlyPlotter()
    _pyvista_plotter = PyvistaPlotter()
    _ufl_cellname_to_vtk_celltype = {
        "triangle": 5,
        "quadrilateral": 9,
        "tetrahedron": 10,
        "hexahedron": 12
    }

    def plot_mesh(self, mesh: firedrake.MeshGeometry) -> typing.Union[
            go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]:
        """
        Plot a mesh stored in firedrake.MeshGeometry object.

        Parameters
        ----------
        mesh
            A firedrake mesh to be plotted.

        Returns
        -------
        :
            A widget representing a plot of the mesh.
        """
        if mesh.topological_dimension() == 1:
            vertices = mesh.coordinates.dat.data_ro
            assert len(vertices.shape) == 1
            assert np.all(vertices[1:] >= vertices[:-1])
            cells = mesh.coordinates.cell_node_map().values
            expected_cells = np.repeat(np.arange(vertices.shape[0], dtype=np.int32), 2)
            expected_cells = np.delete(np.delete(expected_cells, 0), -1)
            expected_cells = expected_cells.reshape(-1, 2)
            expected_cells[:, [0, 1]] = expected_cells[:, [1, 0]]
            assert np.array_equal(cells, expected_cells)
            return self._plotly_plotter.plot_mesh(vertices)
        else:
            pyvista_grid = self._firedrake_mesh_to_pyvista_grid(mesh)
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

    @classmethod
    def _firedrake_mesh_to_pyvista_grid(cls, mesh: firedrake.MeshGeometry) -> pyvista.UnstructuredGrid:
        """Helper method to convert a firedrake.MeshGeometry to a pyvista.UnstructuredGrid."""
        vertices = mesh.coordinates.dat.data_ro
        if mesh.topological_dimension() == 2:
            vertices = np.insert(vertices, 2, values=0.0, axis=1)
        cells = mesh.coordinates.cell_node_map().values
        cellname = mesh.ufl_cell().cellname()
        if cellname in ("triangle", "tetrahedron"):
            pass
        elif cellname == "quadrilateral":
            cells[:, [2, 3]] = cells[:, [3, 2]]
        elif cellname == "hexahedron":
            cells[:, [2, 3]] = cells[:, [3, 2]]
            cells[:, [6, 7]] = cells[:, [7, 6]]
        else:
            raise RuntimeError("Unsupported cellname")
        cells = np.insert(cells, 0, values=cells.shape[1], axis=1)
        cell_types = np.full(cells.shape[0], cls._ufl_cellname_to_vtk_celltype[cellname])
        return pyvista.UnstructuredGrid(cells.reshape(-1), cell_types, vertices)
