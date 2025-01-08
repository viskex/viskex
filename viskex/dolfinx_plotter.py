# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing dolfinx."""

import typing

import dolfinx.fem
import dolfinx.mesh
import dolfinx.plot
import pyvista
import ufl

from viskex.base_plotter import BasePlotter
from viskex.dolfinx_converter import DolfinxConverter
from viskex.pyvista_plotter import PyvistaPlotter


class DolfinxPlotter(BasePlotter[  # type: ignore[no-any-unimported]
    dolfinx.mesh.Mesh,
    typing.Union[dolfinx.fem.Function, tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
    typing.Union[dolfinx.fem.Function, tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
    pyvista.UnstructuredGrid,
    pyvista.Plotter
]):
    """viskex plotter interfacing dolfinx."""

    @classmethod
    def plot_mesh(
        cls, mesh: dolfinx.mesh.Mesh, dim: typing.Optional[int] = None,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a mesh stored in dolfinx.mesh.Mesh object.

        Parameters
        ----------
        mesh
            A dolfinx mesh to be plotted.
        dim
            Plot entities associated to this dimension. If not provided, the topological dimension is used.
        grid_filter
            A filter to be applied to the grid representing the mesh before it is passed to pyvista.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to pyvista.

        Returns
        -------
        :
            A pyvista plotter representing a plot of the mesh.
        """
        pyvista_grid = DolfinxConverter.convert_mesh(mesh, dim)
        return PyvistaPlotter.plot_mesh((pyvista_grid, mesh.topology.dim), dim, grid_filter, **kwargs)

    @classmethod
    def plot_mesh_tags(
        cls, mesh: dolfinx.mesh.Mesh, mesh_tags: dolfinx.mesh.MeshTags, name: str = "mesh tags",
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
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
        grid_filter
            A filter to be applied to the grid representing the mesh before it is passed to pyvista.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to pyvista.

        Returns
        -------
        :
            A pyvista plotter representing a plot of the mesh entities.
        """
        pyvista_grid = DolfinxConverter.convert_mesh_tags(mesh, mesh_tags, name)
        return PyvistaPlotter.plot_mesh((pyvista_grid, mesh.topology.dim), mesh_tags.dim, grid_filter, **kwargs)

    @classmethod
    def plot_scalar_field(  # type: ignore[no-any-unimported]
        cls, scalar_field: typing.Union[
            dolfinx.fem.Function, tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]
        ], name: str = "scalar", part: str = "real", warp_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a scalar field stored in a dolfinx function, or a pair of UFL expression and dolfinx function space.

        Parameters
        ----------
        scalar_field
            Expression to be plotted, which contains a scalar field.
            If the expression is provided as a dolfinx function, such function will be plotted.
            If the expression is provided as a tuple containing UFL expression and a dolfinx function space,
            the UFL expression will first be interpolated on the function space and then plotted.
        name
            Name of the quantity stored in the scalar field.
        part
            Part of the field (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.
        warp_factor
            If provided then the factor is used to produce a warped representation
            the field; if not provided then the scalar field will be plotted on the mesh.
        grid_filter
            A filter to be applied to the field representing the field before it is passed to pyvista.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to pyvista.

        Returns
        -------
        :
            A pyvista plotter representing a plot of the scalar field.
        """
        if isinstance(scalar_field, tuple):
            tdim = scalar_field[1].mesh.topology.dim
        else:
            tdim = scalar_field.function_space.mesh.topology.dim
        pyvista_grid = DolfinxConverter.convert_field(scalar_field, name, part)
        return PyvistaPlotter.plot_scalar_field((pyvista_grid, tdim), name, part, warp_factor, grid_filter, **kwargs)

    @classmethod
    def plot_vector_field(  # type: ignore[no-any-unimported]
        cls, vector_field: typing.Union[
            dolfinx.fem.Function, tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]
        ], name: str = "vector", part: str = "real", warp_factor: float = 0.0, glyph_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a vector field stored in a dolfinx function, or a pair of UFL expression and dolfinx function space.

        Parameters
        ----------
        vector_field
            Expression to be plotted, which contains a vector field.
            If the expression is provided as a dolfinx function, such function will be plotted.
            If the expression is provided as a tuple containing UFL expression and a dolfinx function space,
            the UFL expression will first be interpolated on the function space and then plotted.
        name
            Name of the quantity stored in the vector field.
        part
            Part of the field (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.
        warp_factor
            If provided then the factor is used to produce a warped representation of the field.
            If not provided then the magnitude of the vector field will be plotted on the mesh.
            The argument cannot be used if `glyph_factor` is also provided.
        glyph_factor
            If provided, the vector field is represented using a gylph, scaled by this factor.
            The argument cannot be used if `warp_factor` is also provided.
        grid_filter
            A filter to be applied to the field representing the field before it is passed to pyvista.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to pyvista.

        Returns
        -------
        :
            A pyvista plotter representing a plot of the vector field.
        """
        if isinstance(vector_field, tuple):
            tdim = vector_field[1].mesh.topology.dim
        else:
            tdim = vector_field.function_space.mesh.topology.dim
        assert tdim in (2, 3)
        pyvista_grid = DolfinxConverter.convert_field(vector_field, name, part)
        return PyvistaPlotter.plot_vector_field(
            (pyvista_grid, tdim), name, part, warp_factor, glyph_factor, grid_filter, **kwargs)
