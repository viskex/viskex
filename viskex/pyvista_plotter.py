# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing pyvista."""

import importlib.util
import os
import typing

import numpy as np
import pyvista

from viskex.base_plotter import BasePlotter

pyvista.set_plot_theme("document")  # type: ignore[no-untyped-call]
pyvista.global_theme.cmap = "jet"
pyvista.global_theme.color = "red"
pyvista.global_theme.edge_color = "black"
pyvista.global_theme.line_width = 2.0
pyvista.global_theme.nan_color = "lightgrey"
pyvista.global_theme.point_size = 10.0
pyvista.global_theme.show_edges = False
pyvista.global_theme.show_vertices = False

_jupyter_backend = "client"
_jupyter_notebook = None
if importlib.util.find_spec("google") and importlib.util.find_spec("google.colab"):  # pragma: no cover
    _jupyter_backend = "html"
    _jupyter_notebook = True
if "kaggle" in os.environ.get("KAGGLE_URL_BASE", ""):  # pragma: no cover
    _jupyter_backend = "html"
    _jupyter_notebook = True
_jupyter_backend = os.getenv("VISKEX_PYVISTA_BACKEND", _jupyter_backend)
assert _jupyter_backend in (
    "client", "html", "server", "trame",  # trame backends
    "static"  # static backend
)
pyvista.global_theme.jupyter_backend = _jupyter_backend
if _jupyter_notebook is not None:
    pyvista.global_theme.notebook = _jupyter_notebook
del _jupyter_backend
del _jupyter_notebook


class PyvistaPlotter(BasePlotter[
    tuple[pyvista.UnstructuredGrid, int], tuple[pyvista.UnstructuredGrid, int],
    tuple[pyvista.UnstructuredGrid, int], pyvista.UnstructuredGrid,
    pyvista.Plotter
]):
    """viskex plotter interfacing pyvista."""

    @classmethod
    def plot_mesh(
        cls, mesh: tuple[pyvista.UnstructuredGrid, int], dim: typing.Optional[int] = None,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a mesh stored in a pyvista.UnstructuredGrid.

        Parameters
        ----------
        mesh
            A pair containing the pyvista unstructured grid to be plotted and its topological dimension.
        dim
            Plot entities associated to this dimension. If not provided, the topological dimension is used.
        grid_filter
            A filter to be applied to the grid representing the mesh before it is passed to pyvista.Plotter.add_mesh.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to pyvista.Plotter.add_mesh.

        Returns
        -------
        :
            A pyvista plotter representing a plot of the mesh.
        """
        grid, tdim = mesh
        if dim is None:
            dim = tdim

        # Filter the input grid
        if grid_filter is not None:
            grid = grid_filter(grid)

        # Create plotter
        plotter = pyvista.Plotter()

        # Terminate early if the grid is empty
        if not grid.n_points:
            return plotter

        # Determine whether cell data or point data have active scalars associated to the mesh
        active_scalars_name = {
            data_key: data.active_scalars_name
            for (data_key, data) in (("cell", grid.cell_data), ("point", grid.point_data))
        }

        # Reset to default cell color if scalar data have a single value
        if "cmap" not in kwargs:
            for data in (grid.cell_data, grid.point_data):
                if data.active_scalars is not None and not np.isnan(data.active_scalars).all():
                    if np.nanmin(data.active_scalars) == np.nanmax(data.active_scalars):
                        kwargs["cmap"] = [pyvista.global_theme.color.name]
                        break

        # Determine whether to show vertices: this happens by default in 1D, or when plotting points.
        show_vertices = kwargs.get("show_vertices", None)
        if show_vertices is None:
            if dim == 0:
                # No need to show vertices, since the grid itself is already composed by them
                show_vertices = False
            elif tdim < 2:
                show_vertices = True
            else:
                show_vertices = False
        if show_vertices:
            kwargs.pop("show_vertices", None)
            grid_points = pyvista.PolyData(grid.points)
            if dim == 0:
                if active_scalars_name["cell"] is not None:
                    # Do not override colors provided by cell data
                    vertex_color = None
                else:
                    vertex_color = kwargs.pop("color", pyvista.global_theme.color)
            else:
                if active_scalars_name["point"] is not None:
                    grid_points.point_data[active_scalars_name["point"]] = grid.point_data[
                        active_scalars_name["point"]]
                    # Do not override colors provided by point data
                    vertex_color = None
                else:
                    vertex_color = kwargs.pop("edge_color", pyvista.global_theme.edge_color)

        # Determine whether to show edges: this happens by default in 2D when plotting cells or in 3D
        # when plotting cells or faces.
        show_edges = kwargs.get("show_edges", None)
        if show_edges is None:
            if tdim > 1 and dim > 1:
                show_edges = True
            else:
                show_edges = False
        if show_edges:
            kwargs.pop("show_edges", None)
            edge_color = kwargs.pop("edge_color", pyvista.global_theme.edge_color)

        # Add grids to the plotter
        # Vertices and edges are manually added to plot, rather than using show_vertices and show_edges properties
        # because they lack support for high order meshes
        plotter.add_mesh(grid, **kwargs)
        if show_vertices:
            plotter.add_mesh(grid_points, color=vertex_color)
        if show_edges:
            plotter.add_mesh(grid.extract_all_edges(), color=edge_color)
        plotter.add_axes()  # type: ignore[call-arg]

        # Reset camera position in 1D and 2D
        if tdim < 3:
            plotter.camera_position = "xy"

        return plotter

    @classmethod
    def plot_scalar_field(
        cls, scalar_field: tuple[pyvista.UnstructuredGrid, int], name: str = "scalar", part: str = "real",
        warp_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a scalar field.

        Parameters
        ----------
        scalar_field
            A pair containing the pyvista unstructured grid to be plotted and its topological dimension.
            The grid must already have the scalar field to be plotted set as the active scalar.
        name
            This optional argument is never used.
        part
            This optional argument is never used.
        warp_factor
            If provided then the factor is used to produce a warped representation
            the field; if not provided then the scalar field will be plotted on the mesh.
        grid_filter
            A filter to be applied to the field representing the field before it is passed to pyvista.Plotter.add_mesh.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to pyvista.Plotter.add_mesh.

        Returns
        -------
        :
            A pyvista plotter representing a plot of the scalar field.
        """
        grid, tdim = scalar_field

        # Filter the input grid
        if grid_filter is not None:
            grid = grid_filter(grid)

        # Warp the grid, if requested
        if warp_factor != 0.0:
            assert warp_factor > 0.0
            if tdim == 1:
                normal = [0, 1, 0]
            elif tdim == 2:
                normal = [0, 0, 1]
            else:
                normal = None
            warped_grid = grid.warp_by_scalar(factor=warp_factor, normal=normal)
            warped_tdim = tdim + 1
        else:
            warped_grid = grid
            warped_tdim = tdim

        # Call mesh plotter, with grid_filter=None because it was already applied in this function
        return cls.plot_mesh((warped_grid, warped_tdim), tdim, None, **kwargs)

    @classmethod
    def plot_vector_field(
        cls, vector_field: tuple[pyvista.UnstructuredGrid, int], name: str = "vector", part: str = "real",
        warp_factor: float = 0.0, glyph_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a vector field.

        Parameters
        ----------
        vector_field
            A pair containing the pyvista unstructured grid to be plotted and its topological dimension.
            The grid must already have the vector field to be plotted set as the active vector.
        name
            This optional argument is never used.
        part
            This optional argument is never used.
        warp_factor
            If provided then the factor is used to produce a warped representation of the field.
            If not provided then the magnitude of the vector field will be plotted on the mesh.
            The argument cannot be used if `glyph_factor` is also provided.
        glyph_factor
            If provided, the vector field is represented using a gylph, scaled by this factor.
            The argument cannot be used if `warp_factor` is also provided.
        grid_filter
            A filter to be applied to the field representing the field before it is passed to pyvista.Plotter.add_mesh.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to pyvista.Plotter.add_mesh.

        Returns
        -------
        :
            A pyvista plotter representing a plot of the vector field.
        """
        grid, tdim = vector_field

        # Filter the input grid
        if grid_filter is not None:
            grid = grid_filter(grid)

        # Warp the grid, if requested
        if warp_factor != 0.0:
            assert warp_factor > 0.0
            assert glyph_factor == 0.0
            warped_grid = grid.warp_by_vector(factor=warp_factor)
            warped_tdim = tdim
            warped_dim = tdim
        else:
            if glyph_factor != 0.0:
                # Show just mesh edges when adding glyphs
                warped_grid = grid.extract_all_edges()
                warped_dim = 1
            else:
                warped_grid = grid
                warped_dim = tdim
            warped_tdim = tdim

        # Call mesh plotter, with grid_filter=None because it was already applied in this function
        plotter = cls.plot_mesh((warped_grid, warped_tdim), warped_dim, None, **kwargs)

        # Add glyphs to the plot, if request
        if glyph_factor != 0.0:
            assert glyph_factor > 0.0
            assert warp_factor == 0.0
            glyphed_grid = grid.glyph(factor=glyph_factor)
            plotter.add_mesh(glyphed_grid)

        return plotter
