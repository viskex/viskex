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
import ufl

from viskex.base_plotter import BasePlotter
from viskex.plotly_plotter import PlotlyPlotter
from viskex.pyvista_plotter import PyvistaPlotter
from viskex.utils import extract_part

if dolfinx.mesh.CellType.point not in dolfinx.plot._first_order_vtk:
    dolfinx.plot._first_order_vtk[dolfinx.mesh.CellType.point] = 1


class DolfinxPlotter(BasePlotter[  # type: ignore[no-any-unimported]
    dolfinx.mesh.Mesh,
    typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
    typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
    typing.Union[go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]
]):
    """viskex plotter interfacing dolfinx."""

    @classmethod
    def plot_mesh(  # type: ignore[no-any-unimported]
        cls, mesh: dolfinx.mesh.Mesh, dim: typing.Optional[int] = None
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
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
        tdim = mesh.topology.dim
        if dim is None:
            dim = tdim
        assert dim <= tdim
        if tdim == 1:
            plotly_grid = cls._dolfinx_mesh_to_plotly_grid(mesh, dim)
            return PlotlyPlotter.plot_mesh(plotly_grid, dim)
        else:
            pyvista_grid = cls._dolfinx_mesh_to_pyvista_grid(mesh, dim)
            return PyvistaPlotter.plot_mesh((pyvista_grid, tdim))

    @classmethod
    def plot_mesh_entities(  # type: ignore[no-any-unimported]
        cls, mesh: dolfinx.mesh.Mesh, dim: int, name: str, indices: np.typing.NDArray[np.int32],
        values: typing.Optional[np.typing.NDArray[np.int32]] = None
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot `dim`-dimensional mesh entities of a given dolfinx mesh.

        Parameters
        ----------
        mesh
            A dolfinx mesh from which to extract mesh entities.
        dim
            Extract entities associated to this dimension.
        name
            Name to be assigned to the field containing the mesh entities values.
        indices
            Array containing the IDs of the entities to be plotted.
        values
            Array containing the value to be associated to each of the entities in `indices`.
            Values are assumed to be greater than zero, because every entity not part of `indices`
            will be automatically assigned value equal to zero.
            If not provided, every entity part of `indices` will be marked with value one.

        Returns
        -------
        :
            A widget representing a plot of the mesh entities.
        """
        tdim = mesh.topology.dim
        assert dim <= tdim
        if values is None:
            values = np.ones_like(indices)
        if tdim == 1:
            plotly_grid = cls._dolfinx_mesh_to_plotly_grid(mesh, dim)
            return PlotlyPlotter.plot_mesh_entities(plotly_grid, dim, name, indices, values)
        else:
            pyvista_grid = cls._dolfinx_mesh_to_pyvista_grid(mesh, dim)
            return PyvistaPlotter.plot_mesh_entities((pyvista_grid, tdim), dim, name, indices, values)

    @classmethod
    def plot_mesh_tags(  # type: ignore[no-any-unimported]
        cls, mesh: dolfinx.mesh.Mesh, mesh_tags: dolfinx.mesh.MeshTags, name: str
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot dolfinx.mesh.MeshTags.

        Parameters
        ----------
        mesh
            The dolfinx mesh associated to the provided mesh tags.
        mesh_tags
            A dolfinx mesh tags from which to extract mesh entities and their values.
        name
            Name to be assigned to the field containing the mesh entities values.

        Returns
        -------
        :
            A widget representing a plot of the mesh entities.
        """
        return cls.plot_mesh_entities(mesh, mesh_tags.dim, name, mesh_tags.indices, mesh_tags.values)

    @classmethod
    def plot_scalar_field(  # type: ignore[no-any-unimported]
        cls, scalar_field: typing.Union[
            dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]
        ], name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a scalar field stored in a dolfinx Function, or a pair of UFL Expression and dolfinx FunctionSpace.

        Parameters
        ----------
        scalar_field
            Expression to be plotted, which contains a scalar field.
            If the expression is provided as a dolfinx Function, such function will be plotted.
            If the expression is provided as a tuple containing UFL expression and a dolfinx FunctionSpace,
            the UFL expression will first be interpolated on the function space and then plotted.
        name
            Name of the quantity stored in the scalar field.
        warp_factor
            This argument is ignored for a field on 1D or 3D meshes.
            For a 2D mesh, if provided then the factor is used to produce a warped representation
            the field; if not provided then the scalar field will be plotted on the mesh.
        part
            Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.

        Returns
        -------
        :
            A widget representing a plot of the scalar field.
        """
        scalar_field = cls._interpolate_if_ufl_expression(scalar_field)
        mesh = scalar_field.function_space.mesh
        values = scalar_field.x.array
        (values, name) = extract_part(values, name, part)
        tdim = mesh.topology.dim
        if tdim == 1:
            coordinates = scalar_field.function_space.tabulate_dof_coordinates()
            coordinates = coordinates[:, 0]
            argsort = coordinates.argsort()
            coordinates = coordinates[argsort]
            values = values[argsort]
            return PlotlyPlotter.plot_scalar_field((coordinates, values), name, warp_factor, part)
        else:
            pyvista_cells, cell_types, coordinates = dolfinx.plot.create_vtk_mesh(scalar_field.function_space)
            pyvista_grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)
            pyvista_grid.point_data[name] = values
            pyvista_grid.set_active_scalars(name)
            return PyvistaPlotter.plot_scalar_field((pyvista_grid, tdim), name, warp_factor, part)

    @classmethod
    def plot_vector_field(  # type: ignore[no-any-unimported]
        cls, vector_field: typing.Union[
            dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]
        ], name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a vector field stored in a dolfinx Function, or a pair of UFL Expression and dolfinx FunctionSpace.

        Parameters
        ----------
        vector_field
            Expression to be plotted, which contains a vector field.
            If the expression is provided as a dolfinx Function, such function will be plotted.
            If the expression is provided as a tuple containing UFL expression and a dolfinx FunctionSpace,
            the UFL expression will first be interpolated on the function space and then plotted.
        name
            Name of the quantity stored in the vector field.
        glyph_factor
            If provided, the vector field is represented using a gylph, scaled by this factor.
        warp_factor
            If provided then the factor is used to produce a warped representation of the field.
            If not provided then the magnitude of the vector field will be plotted on the mesh.
            The argument cannot be used if `glyph_factor` is also provided.
        part
            Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.

        Returns
        -------
        :
            A widget representing a plot of the vector field.
        """
        vector_field = cls._interpolate_if_ufl_expression(vector_field)
        mesh = vector_field.function_space.mesh
        values = vector_field.x.array
        (values, name) = extract_part(values, name, part)
        tdim = mesh.topology.dim
        assert tdim > 1, "Cannot call plot_vector_field for 1D meshes"
        pyvista_cells, cell_types, coordinates = dolfinx.plot.create_vtk_mesh(vector_field.function_space)
        pyvista_grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)
        values = values.reshape(coordinates.shape[0], vector_field.function_space.dofmap.index_map_bs)
        if tdim == 2:
            values = np.insert(values, values.shape[1], 0.0, axis=1)
        pyvista_grid.point_data[name] = values
        pyvista_grid.set_active_vectors(name)
        pyvista_grid_edges = cls._dolfinx_mesh_to_pyvista_grid(mesh, 1)
        return PyvistaPlotter.plot_vector_field(
            (pyvista_grid, pyvista_grid_edges, tdim), name, glyph_factor, warp_factor, part)

    @staticmethod
    def _dolfinx_mesh_to_plotly_grid(mesh: dolfinx.mesh.Mesh, dim: int) -> np.typing.NDArray[np.float64]:
        """Convert a 1D dolfinx.mesh.Mesh to an array of coordinates."""
        vertices = mesh.geometry.x[:, 0]
        argsort = vertices.argsort()
        return vertices[argsort]  # type: ignore[no-any-return]

    @staticmethod
    def _dolfinx_mesh_to_pyvista_grid(mesh: dolfinx.mesh.Mesh, dim: int) -> pyvista.UnstructuredGrid:
        """Convert a 2D or 3D dolfinx.mesh.Mesh to a pyvista.UnstructuredGrid."""
        mesh.topology.create_connectivity(dim, dim)
        num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
        cell_entities = np.arange(num_cells, dtype=np.int32)
        pyvista_cells, cell_types, coordinates = dolfinx.plot.create_vtk_mesh(mesh, dim, cell_entities)
        return pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)

    @staticmethod
    def _interpolate_if_ufl_expression(  # type: ignore[no-any-unimported]
        field: typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]]
    ) -> dolfinx.fem.Function:
        """Interpolate a UFL Expression in a dolfinx Function."""
        if isinstance(field, tuple):
            expression, function_space = field
            interpolated_field = dolfinx.fem.Function(function_space)
            interpolated_field.interpolate(
                dolfinx.fem.Expression(expression, function_space.element.interpolation_points()))
            return interpolated_field
        else:
            assert isinstance(field, dolfinx.fem.Function)
            return field
