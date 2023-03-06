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
        "point": 1,
        "interval": 3,
        "triangle": 5,
        "quadrilateral": 9,
        "tetrahedron": 10,
        "hexahedron": 12
    }
    _tdim_cellname_to_dim_cellname = {
        ("point", 0): "point",
        ("interval", 1): "interval",
        ("interval", 0): "point",
        ("triangle", 2): "triangle",
        ("triangle", 1): "interval",
        ("triangle", 0): "point",
        ("quadrilateral", 2): "quadrilateral",
        ("quadrilateral", 1): "interval",
        ("quadrilateral", 0): "point",
        ("tetrahedron", 3): "tetrahedron",
        ("tetrahedron", 2): "triangle",
        ("tetrahedron", 1): "interval",
        ("tetrahedron", 0): "point",
        ("hexahedron", 3): "hexahedron",
        ("hexahedron", 2): "quadrilateral",
        ("hexahedron", 1): "interval",
        ("hexahedron", 0): "point"
    }

    def plot_mesh(self, mesh: firedrake.MeshGeometry, dim: typing.Optional[int] = None) -> typing.Union[
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
        if dim is None:
            dim = mesh.topological_dimension()
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
            return self._plotly_plotter.plot_mesh(vertices, dim)
        else:
            pyvista_grid = self._firedrake_mesh_to_pyvista_grid(mesh, dim)
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
    def _firedrake_mesh_to_pyvista_grid(cls, mesh: firedrake.MeshGeometry, dim: int) -> pyvista.UnstructuredGrid:
        """Helper method to convert a firedrake.MeshGeometry to a pyvista.UnstructuredGrid."""
        tdim = mesh.topological_dimension()
        tdim_cellname = mesh.ufl_cell().cellname()
        vertices = mesh.coordinates.dat.data_ro
        if tdim == 2:
            vertices = np.insert(vertices, 2, values=0.0, axis=1)
        if dim == tdim:
            connectivity = mesh.coordinates.cell_node_map().values.copy()
        elif dim == 0:
            connectivity = np.arange(vertices.shape[0]).reshape(-1, 1)
        else:
            topology = mesh.coordinates.function_space().finat_element.cell.get_topology()
            interior_facet_local_ids = mesh.interior_facets.local_facet_dat.data_ro[:, :1].reshape(-1)
            exterior_facet_local_ids = mesh.exterior_facets.local_facet_dat.data_ro
            facet_local_ids = np.concatenate((interior_facet_local_ids, exterior_facet_local_ids))
            interior_facet_node_map = mesh.coordinates.interior_facet_node_map()
            interior_facet_node_map = interior_facet_node_map.values[:, :interior_facet_node_map.arity//2]
            exterior_facet_node_map = mesh.coordinates.exterior_facet_node_map().values
            facet_node_map = np.concatenate((interior_facet_node_map, exterior_facet_node_map))
            mask = np.zeros(facet_node_map.shape, dtype=bool)
            for mask_row, facet_local_id in enumerate(facet_local_ids):
                mask[mask_row, topology[tdim - 1][facet_local_id]] = True
            connectivity = facet_node_map[mask].reshape(-1, len(topology[tdim - 1][0]))
            if dim == tdim - 1:
                pass
            elif dim == tdim - 2:
                cls._reorder_connectivity(connectivity, cls._tdim_cellname_to_dim_cellname[tdim_cellname, tdim - 1])
                repeated_connectivity = np.roll(np.repeat(connectivity, repeats=2, axis=1), shift=-1, axis=1)
                connectivity = repeated_connectivity.reshape(-1, 2)
            else:
                raise RuntimeError("Invalid values of dim and tdim")
        dim_cellname = cls._tdim_cellname_to_dim_cellname[tdim_cellname, dim]
        cls._reorder_connectivity(connectivity, dim_cellname)
        connectivity = np.insert(connectivity, 0, values=connectivity.shape[1], axis=1)
        pyvista_types = np.full(connectivity.shape[0], cls._ufl_cellname_to_vtk_celltype[dim_cellname])
        return pyvista.UnstructuredGrid(connectivity.reshape(-1), pyvista_types, vertices)

    @staticmethod
    def _reorder_connectivity(connectivity: np.typing.NDArray[np.int32], cellname: str) -> None:
        """Helper method to reorder in-place a connectivity array according to vtk ordering."""
        if cellname in ("point", "interval", "triangle", "tetrahedron"):
            pass
        elif cellname == "quadrilateral":
            connectivity[:, [2, 3]] = connectivity[:, [3, 2]]
        elif cellname == "hexahedron":
            connectivity[:, [2, 3]] = connectivity[:, [3, 2]]
            connectivity[:, [6, 7]] = connectivity[:, [7, 6]]
        else:
            raise RuntimeError("Unsupported cellname")
