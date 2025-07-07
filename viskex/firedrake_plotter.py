# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing firedrake."""

import typing

import firedrake
import pyvista
import ufl

from viskex.base_plotter import BasePlotter
from viskex.firedrake_converter import FiredrakeConverter
from viskex.pyvista_plotter import PyvistaPlotter


class FiredrakePlotter(BasePlotter[  # type: ignore[no-any-unimported]
    firedrake.MeshGeometry,
    typing.Union[firedrake.Function, tuple[ufl.core.expr.Expr, ufl.FunctionSpace]],
    typing.Union[firedrake.Function, tuple[ufl.core.expr.Expr, ufl.FunctionSpace]],
    pyvista.UnstructuredGrid,
    pyvista.Plotter
]):
    """viskex plotter interfacing firedrake."""

    @classmethod
    def plot_mesh(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: typing.Optional[int] = None,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a mesh stored in firedrake.MeshGeometry object.

        Parameters
        ----------
        mesh
            A firedrake mesh to be plotted.
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
        tdim = mesh.topological_dimension()
        if tdim == 3 and dim == 1:
            # Firedrake does not offer edge (dim = 1) to vertices connectivity. Convert the case with
            # facet (dim = 2) to vertices connectivity, and then ask vtk to extract edges.
            pyvista_grid = FiredrakeConverter.convert_mesh(mesh, dim + 1).extract_all_edges()
        else:
            pyvista_grid = FiredrakeConverter.convert_mesh(mesh, dim)
        plotter = PyvistaPlotter.plot_mesh((pyvista_grid, tdim), dim, grid_filter, **kwargs)
        return plotter

    @classmethod
    def plot_mesh_sets(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: int, name: str = "mesh sets",
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a cell set or a face sets of a given firedrake mesh.

        Parameters
        ----------
        mesh
            A firedrake mesh from which to extract mesh entities and their values.
        dim
            Extract entities associated to this dimension.
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
        pyvista_grid = FiredrakeConverter.convert_mesh_sets(mesh, dim, name)
        return PyvistaPlotter.plot_mesh((pyvista_grid, mesh.topological_dimension()), dim, grid_filter, **kwargs)

    @classmethod
    def plot_scalar_field(  # type: ignore[no-any-unimported]
        cls, scalar_field: typing.Union[
            firedrake.Function, tuple[ufl.core.expr.Expr, ufl.FunctionSpace]
        ], name: str = "scalar", part: str = "real", warp_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a scalar field stored in a firedrake Function, or a pair of UFL Expression and firedrake FunctionSpace.

        Parameters
        ----------
        scalar_field
            Expression to be plotted, which contains a scalar field.
            If the expression is provided as a firedrake Function, such function will be plotted.
            If the expression is provided as a tuple containing UFL expression and a firedrake FunctionSpace,
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
            tdim = scalar_field[1].mesh().topological_dimension()
        else:
            tdim = scalar_field.function_space().mesh().topological_dimension()
        pyvista_grid = FiredrakeConverter.convert_field(scalar_field, name, part)
        return PyvistaPlotter.plot_scalar_field((pyvista_grid, tdim), name, part, warp_factor, grid_filter, **kwargs)

    @classmethod
    def plot_vector_field(  # type: ignore[no-any-unimported]
        cls, vector_field: typing.Union[
            firedrake.Function, tuple[ufl.core.expr.Expr, ufl.FunctionSpace]
        ], name: str = "vector", part: str = "real", warp_factor: float = 0.0, glyph_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a vector field stored in a firedrake Function, or a pair of UFL Expression and firedrake FunctionSpace.

        Parameters
        ----------
        vector_field
            Expression to be plotted, which contains a vector field.
            If the expression is provided as a firedrake Function, such function will be plotted.
            If the expression is provided as a tuple containing UFL expression and a firedrake FunctionSpace,
            the UFL expression will first be interpolated on the function space and then plotted.
            Notice that the field will be interpolated to a P1 space before plotting.
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
            tdim = vector_field[1].mesh().topological_dimension()
        else:
            tdim = vector_field.function_space().mesh().topological_dimension()
        assert tdim in (2, 3)
        pyvista_grid = FiredrakeConverter.convert_field(vector_field, name, part)
        return PyvistaPlotter.plot_vector_field(
            (pyvista_grid, tdim), name, part, warp_factor, glyph_factor, grid_filter, **kwargs)
