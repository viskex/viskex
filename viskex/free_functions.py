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
import ufl

try:
    import dolfinx  # noqa: F401
except ImportError:
    has_dolfinx = False
else:
    has_dolfinx = True
    import dolfinx.fem
    import dolfinx.mesh

try:
    import firedrake  # noqa: F401
except ImportError:
    has_firedrake = False
else:
    has_firedrake = True

if not typing.TYPE_CHECKING:
    if has_dolfinx:
        from viskex.dolfinx_plotter import DolfinxPlotter

    if has_firedrake:
        from viskex.firedrake_plotter import FiredrakePlotter
else:
    # CI only ships either dolfinx or firedrake, but not both.
    # Avoid importing the actual plotters on CI, because the lines
    #   from viskex.dolfinx_plotter import DolfinxPlotter
    #   from viskex.firedrake_plotter import FiredrakePlotter
    # would force type checking also for the plotter of the other library,
    # regardless of:
    # * the fact that we add it to the exclude of the [mypy] section
    #   in the setup.cfg file
    # * the imports are guarded or not with the appropriate if has_*
    from viskex.base_plotter import BasePlotter

    class DolfinxPlotter(BasePlotter[typing.Any, typing.Any, typing.Any, typing.Any]):
        """Stub of DolfinxPlotter for type checking. See the concrete implementation instead."""

        @classmethod
        def plot_mesh_tags(  # type: ignore[no-any-unimported]
            cls, mesh_tags: typing.Any, name: str  # noqa: ANN401
        ) -> typing.Union[
            go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
        ]:
            """Stub of plot_mesh_tags for type checking. See the concrete implementation instead."""
            ...

    class FiredrakePlotter(BasePlotter[typing.Any, typing.Any, typing.Any, typing.Any]):
        """Stub of FiredrakePlotter for type checking. See the concrete implementation instead."""

        @classmethod
        def plot_mesh_sets(  # type: ignore[no-any-unimported]
            cls, mesh: typing.Any, dim: int, name: str  # noqa: ANN401
        ) -> typing.Union[
            go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
        ]:
            """Stub of plot_mesh_sets for type checking. See the concrete implementation instead."""
            ...

plot_mesh_dispatcher = plum.Dispatcher()


@plot_mesh_dispatcher.abstract
def _plot_mesh(
    mesh: typing.Any, dim: typing.Optional[int] = None  # noqa: ANN401
) -> typing.Union[
    go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
]:
    """Plot a mesh."""
    raise NotImplementedError("The abstract case has not been implemented")  # pragma: no cover


if typing.TYPE_CHECKING or has_dolfinx:
    @plot_mesh_dispatcher  # type: ignore[no-redef]
    def _plot_mesh(  # noqa: F811  # type: ignore[no-any-unimported]
        mesh: dolfinx.mesh.Mesh, dim: typing.Optional[int] = None
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
        return DolfinxPlotter.plot_mesh(mesh, dim)

if typing.TYPE_CHECKING or has_firedrake:
    @plot_mesh_dispatcher  # type: ignore[no-redef]
    def _plot_mesh(  # noqa: F811
        mesh: firedrake.MeshGeometry, dim: typing.Optional[int] = None
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
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
        return FiredrakePlotter.plot_mesh(mesh, dim)

if typing.TYPE_CHECKING or has_dolfinx:
    @typing.overload
    def plot_mesh(  # type: ignore[no-any-unimported]
        mesh: dolfinx.mesh.Mesh, dim: typing.Optional[int] = None
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Stub of plot_mesh for type checking. See the concrete implementation above."""
        ...  # pragma: no cover

if typing.TYPE_CHECKING or has_firedrake:
    @typing.overload
    def plot_mesh(  # type: ignore[no-any-unimported, misc]
        mesh: firedrake.MeshGeometry, dim: typing.Optional[int] = None
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Stub of plot_mesh for type checking. See the concrete implementation above."""
        ...  # pragma: no cover


def plot_mesh(  # type: ignore[no-any-unimported]
    mesh: typing.Any, dim: typing.Optional[int] = None  # noqa: ANN401
) -> typing.Union[
    go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
]:
    """Dispatcher of plot_mesh for type checking. See the concrete implementation above."""
    return _plot_mesh(mesh, dim)


plot_mesh.__doc__ = _plot_mesh.__doc__

plot_mesh_entities_dispatcher = plum.Dispatcher()


@plot_mesh_entities_dispatcher.abstract
def _plot_mesh_entities(
    mesh: typing.Any, dim: int, name: str, indices: typing.Any,  # noqa: ANN401 # TODO plum issue #74
    values: typing.Optional[typing.Any] = None  # noqa: ANN401 # TODO plum issue #74
) -> typing.Union[
    go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
]:
    """Plot `dim`-dimensional mesh entities of a given mesh."""
    raise NotImplementedError("The abstract case has not been implemented")  # pragma: no cover


if typing.TYPE_CHECKING or has_dolfinx:
    @plot_mesh_entities_dispatcher  # type: ignore[no-redef]
    def _plot_mesh_entities(  # noqa: F811
        mesh: dolfinx.mesh.Mesh, dim: int, name: str, indices: typing.Any,  # noqa: ANN401 # TODO plum issue #74
        values: typing.Optional[typing.Any] = None  # noqa: ANN401 # TODO plum issue #74
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
        return DolfinxPlotter.plot_mesh_entities(mesh, dim, name, indices, values)

if typing.TYPE_CHECKING or has_firedrake:
    @plot_mesh_entities_dispatcher  # type: ignore[no-redef]
    def _plot_mesh_entities(  # noqa: F811
        mesh: firedrake.MeshGeometry, dim: int, name: str, indices: typing.Any,  # noqa: ANN401 # TODO plum issue #74
        values: typing.Optional[typing.Any] = None  # noqa: ANN401 # TODO plum issue #74
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot `dim`-dimensional mesh entities of a given firedrake mesh.

        Parameters
        ----------
        mesh
            A firedrake mesh to be plotted.
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
        return FiredrakePlotter.plot_mesh_entities(mesh, dim, name, indices, values)

if typing.TYPE_CHECKING or has_dolfinx:
    @typing.overload
    def plot_mesh_entities(  # type: ignore[no-any-unimported]
        mesh: dolfinx.mesh.Mesh, dim: int, name: str, indices: typing.Any,  # noqa: ANN401 # TODO plum issue #74
        values: typing.Optional[typing.Any] = None  # noqa: ANN401 # TODO plum issue #74
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Stub of plot_mesh_entities for type checking. See the concrete implementation above."""
        ...  # pragma: no cover

if typing.TYPE_CHECKING or has_firedrake:
    @typing.overload
    def plot_mesh_entities(  # type: ignore[no-any-unimported, misc]
        mesh: firedrake.MeshGeometry, dim: int, name: str, indices: typing.Any,  # noqa: ANN401 # TODO plum issue #74
        values: typing.Optional[typing.Any] = None  # noqa: ANN401 # TODO plum issue #74
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Stub of plot_mesh_entities for type checking. See the concrete implementation above."""
        ...  # pragma: no cover


def plot_mesh_entities(  # type: ignore[no-any-unimported]
    mesh: typing.Any, dim: int, name: str, indices: typing.Any,  # noqa: ANN401 # TODO plum issue #74
    values: typing.Optional[typing.Any] = None  # noqa: ANN401 # TODO plum issue #74
) -> typing.Union[
    go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
]:
    """Dispatcher of plot_mesh_entities for type checking. See the concrete implementation above."""
    return _plot_mesh_entities(mesh, dim, name, indices, values)


plot_mesh_entities.__doc__ = _plot_mesh_entities.__doc__

if typing.TYPE_CHECKING or has_dolfinx:
    def plot_mesh_tags(  # type: ignore[no-any-unimported]
        mesh_tags: dolfinx.mesh.MeshTags, name: str
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot dolfinx.mesh.MeshTags.

        Parameters
        ----------
        mesh_tags
            A dolfinx mesh tags from which to extract mesh entities and their values.
        name
            Name to be assigned to the field containing the mesh entities values.

        Returns
        -------
        :
            A widget representing a plot of the mesh entities.
        """
        return DolfinxPlotter.plot_mesh_tags(mesh_tags, name)

if typing.TYPE_CHECKING or has_firedrake:
    def plot_mesh_sets(  # type: ignore[no-any-unimported]
        mesh: firedrake.MeshGeometry, dim: int, name: str
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
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

        Returns
        -------
        :
            A widget representing a plot of the mesh entities.
        """
        return FiredrakePlotter.plot_mesh_sets(mesh, dim, name)


plot_scalar_field_dispatcher = plum.Dispatcher()


@plot_scalar_field_dispatcher.abstract
def _plot_scalar_field(
    scalar_field: typing.Any, name: str, warp_factor: float = 0.0, part: str = "real"  # noqa: ANN401
) -> typing.Union[
    go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
]:
    """Plot a scalar field."""
    raise NotImplementedError("The abstract case has not been implemented")  # pragma: no cover


if typing.TYPE_CHECKING or has_dolfinx:
    @plot_scalar_field_dispatcher  # type: ignore[no-redef]
    def _plot_scalar_field(  # noqa: F811
        scalar_field: typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
        name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a scalar field stored in a dolfinx Function, or a pair of ufl Expression and dolfinx FunctionSpace.

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
        return DolfinxPlotter.plot_scalar_field(scalar_field, name, warp_factor, part)

if typing.TYPE_CHECKING or has_firedrake:
    @plot_scalar_field_dispatcher  # type: ignore[no-redef]
    def _plot_scalar_field(  # noqa: F811
        scalar_field: typing.Union[
            firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]
        ], name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
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
        return FiredrakePlotter.plot_scalar_field(scalar_field, name, warp_factor, part)

if typing.TYPE_CHECKING or has_dolfinx:
    @typing.overload
    def plot_scalar_field(  # type: ignore[no-any-unimported]
        scalar_field: typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
        name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Stub of plot_scalar_field for type checking. See the concrete implementation above."""
        ...  # pragma: no cover

if typing.TYPE_CHECKING or has_firedrake:
    @typing.overload
    def plot_scalar_field(  # type: ignore[no-any-unimported, misc]
        scalar_field: typing.Union[
            firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]
        ], name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Stub of plot_scalar_field for type checking. See the concrete implementation above."""
        ...  # pragma: no cover


def plot_scalar_field(  # type: ignore[no-any-unimported]
    scalar_field: typing.Any, name: str, warp_factor: float = 0.0, part: str = "real"  # noqa: ANN401
) -> typing.Union[
    go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
]:
    """Dispatcher of plot_scalar_field for type checking. See the concrete implementation above."""
    return _plot_scalar_field(scalar_field, name, warp_factor, part)


plot_scalar_field.__doc__ = _plot_scalar_field.__doc__

plot_vector_field_dispatcher = plum.Dispatcher()


@plot_vector_field_dispatcher.abstract
def _plot_vector_field(
    vector_field: typing.Any, name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0,  # noqa: ANN401
    part: str = "real"
) -> typing.Union[
    go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
]:
    """Plot a vector field."""
    raise NotImplementedError("The abstract case has not been implemented")  # pragma: no cover


if typing.TYPE_CHECKING or has_dolfinx:
    @plot_vector_field_dispatcher  # type: ignore[no-redef]
    def _plot_vector_field(  # noqa: F811
        vector_field: typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
        name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0, part: str = "real"
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
        return DolfinxPlotter.plot_vector_field(vector_field, name, glyph_factor, warp_factor, part)

if typing.TYPE_CHECKING or has_firedrake:
    @plot_vector_field_dispatcher  # type: ignore[no-redef]
    def _plot_vector_field(  # noqa: F811
        vector_field: typing.Union[firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]],
        name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
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
        return FiredrakePlotter.plot_vector_field(vector_field, name, glyph_factor, warp_factor, part)

if typing.TYPE_CHECKING or has_dolfinx:
    @typing.overload
    def plot_vector_field(  # type: ignore[no-any-unimported]
        vector_field: typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
        name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Stub of plot_vector_field for type checking. See the concrete implementation above."""
        ...  # pragma: no cover

if typing.TYPE_CHECKING or has_firedrake:
    @typing.overload
    def plot_vector_field(  # type: ignore[no-any-unimported, misc]
        vector_field: typing.Union[firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]],
        name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Stub of plot_vector_field for type checking. See the concrete implementation above."""
        ...  # pragma: no cover


def plot_vector_field(  # type: ignore[no-any-unimported]
    vector_field: typing.Any, name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0,  # noqa: ANN401
    part: str = "real"
) -> typing.Union[
    go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
]:
    """Dispatcher of plot_vector_field for type checking. See the concrete implementation above."""
    return _plot_vector_field(vector_field, name, glyph_factor, warp_factor, part)


plot_vector_field.__doc__ = _plot_vector_field.__doc__
