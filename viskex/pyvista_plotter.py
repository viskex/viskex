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
from viskex.utils import add_point_markers, update_camera_with_mesh

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
    "static",  # static backend
    "none"  # open VTK render window
)
pyvista.global_theme.jupyter_backend = _jupyter_backend
if _jupyter_backend == "none":
    assert _jupyter_notebook is None
    _jupyter_notebook = False
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
        plotter: typing.Optional[pyvista.Plotter] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> pyvista.Plotter:
        """
        Plot a mesh stored in a pyvista.UnstructuredGrid.

        Parameters
        ----------
        mesh
            A pair containing the pyvista unstructured grid to be plotted and its topological dimension.
        dim
            The pyvista unstructured grid provided in the first input argument is contains entities of maximum
            topological dimension equal to dim. If not provided, the topological dimension is assumed.
        grid_filter
            A filter to be applied to the grid representing the mesh before it is passed to pyvista.Plotter.add_mesh.
            If not provided, no filter will be applied.
        plotter
            The pyvista plotter to which the mesh will be added. If not provided, a new plotter will be created.
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
        if plotter is None:
            plotter = pyvista.Plotter()  # type: ignore[no-untyped-call]

        # Terminate early if the grid is empty
        if not grid.n_points:
            return plotter

        # Determine whether cell/point data have active scalars/vectors/tensors associated to the mesh
        active_data: list[dict[str, typing.Any]] = []
        field_types = ("scalars", "vectors")
        for location, location_data in (("cell", grid.cell_data), ("point", grid.point_data)):
            for field_type in field_types:
                field_name = getattr(location_data, f"active_{field_type}_name")
                if field_name is not None:
                    array = getattr(location_data, f"active_{field_type}")
                    if np.isnan(array).all():
                        array_range: typing.Optional[tuple[typing.Any, ...]] = None
                    else:
                        array_min = np.nanmin(array)
                        array_max = np.nanmax(array)
                        if array_min == array_max:
                            array_range = (array_min, )
                        else:
                            array_range = (array_min, array_max)
                    active_data.append({
                        "location": location, "field_type": field_type,
                        "name": field_name, "array": array, "range": array_range
                    })

        # Reset to default cell color if active data have a single value
        if (
            "cmap" not in kwargs and len(active_data) > 0
            and all(active_data_["range"] is None or len(active_data_["range"]) == 1 for active_data_ in active_data)
        ):
            kwargs["cmap"] = [pyvista.global_theme.color.name]

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
            line_width = kwargs.pop("line_width", pyvista.global_theme.line_width)

        # Determine whether to show vertices: this happens by default in 1D, or when plotting points.
        show_vertices = kwargs.get("show_vertices", None)
        if show_vertices is None:
            if tdim < 2 or dim == 0:
                show_vertices = True
            else:
                show_vertices = False
        if show_vertices:
            kwargs.pop("show_vertices", None)
            # Extract the mesh points
            if dim > 0:
                # Create a poly data grid which contains only the mesh points
                grid_points: pyvista.DataSet = pyvista.PolyData(grid.points)
            else:
                # Simply reuse the provided grid, since it already contains only points
                grid_points = grid
            # Determine the vertex color and cmap, depending on wheter array values are being plotted
            if dim > 0:
                vertex_color = kwargs.pop("edge_color", pyvista.global_theme.edge_color)
                vertex_cmap = None
            else:
                if len(active_data) > 0 and any(active_data_["location"] == "cell" for active_data_ in active_data):
                    vertex_color = None
                    vertex_cmap = kwargs.get("cmap", None)  # use get and not pop since cmap is used by other plots too
                else:
                    vertex_color = kwargs.pop("color", pyvista.global_theme.color)
                    vertex_cmap = None
            # Determine the dimension of the point markers, either 1 (segments), 2 (squares) or 3 (cubes)
            if dim > 0:
                point_markers_dim = tdim
            else:
                if tdim == 1:
                    # Plotting vertices of a 1D mesh without the cells: use squares intead of segments
                    # to make the markers more visibile
                    point_markers_dim = 2
                else:
                    point_markers_dim = tdim
            # Determine the default point size.
            point_size = kwargs.pop("point_size", pyvista.global_theme.point_size)

        # Add grid to the plotter, except for the case dim == 0 that will be handled later
        if dim > 0:
            plotter.add_mesh(grid, **kwargs)

        # Vertices and edges are manually added to plot, rather than using show_vertices and show_edges properties
        # because the properties lack support for high order meshes.
        # First of all, we add edges following pyvista discussion #5777.
        if show_edges:
            order = cls._infer_lagrange_cell_order(grid)
            grid_edges = grid.separate_cells().extract_surface(
                nonlinear_subdivision=order - 1).extract_feature_edges()
            plotter.add_mesh(grid_edges, line_width=line_width, color=edge_color)
        # Then we also add vertices using a custom implementation in viskex.utils.add_point_markers which allows
        # to give them a shape depending on the topological dimension.
        if show_vertices:
            # The custom implementation in viskex.utils.add_point_markers requires the camera to be up to date.
            if dim > 0:
                # There was surely a previous call to plotter.add_mesh(grid, ...)
                plotter.reset_camera()  # type: ignore[call-arg]
            else:
                # There may not have been a previous call to add_mesh: update the camera as if grid was added
                update_camera_with_mesh(plotter, grid)
            add_point_markers(
                plotter, grid_points, dim=point_markers_dim, point_size=point_size, point_color=vertex_color,
                point_cmap=vertex_cmap)
            # Force a further camera update after grid points have been added.
            plotter.reset_camera()  # type: ignore[call-arg]

        # Add coordinate axes
        plotter.add_axes()  # type: ignore[call-arg]

        # Set camera position in 1D and 2D
        if tdim < 3:
            plotter.camera_position = "xy"
            plotter.enable_parallel_projection()  # type: ignore[call-arg]

        return plotter

    @classmethod
    def plot_scalar_field(
        cls, scalar_field: tuple[pyvista.UnstructuredGrid, int], name: str = "scalar", part: str = "real",
        warp_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        plotter: typing.Optional[pyvista.Plotter] = None,
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
        plotter
            The pyvista plotter to which the scalar field will be added.
            If not provided, a new plotter will be created.
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
        return cls.plot_mesh((warped_grid, warped_tdim), tdim, None, plotter, **kwargs)

    @classmethod
    def plot_vector_field(
        cls, vector_field: tuple[pyvista.UnstructuredGrid, int], name: str = "vector", part: str = "real",
        warp_factor: float = 0.0, glyph_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]] = None,
        plotter: typing.Optional[pyvista.Plotter] = None,
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
        plotter
            The pyvista plotter to which the vector field will be added.
            If not provided, a new plotter will be created.
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
        plotter = cls.plot_mesh((warped_grid, warped_tdim), warped_dim, None, plotter, **kwargs)

        # Add glyphs to the plot, if requested
        if glyph_factor != 0.0:
            assert glyph_factor > 0.0
            assert warp_factor == 0.0
            glyphed_grid = grid.glyph(factor=glyph_factor)
            plotter.add_mesh(glyphed_grid, show_scalar_bar=False)

        return plotter

    @classmethod
    def _infer_lagrange_cell_order(cls, grid: pyvista.UnstructuredGrid) -> int:
        """
        Infers the polynomial order of a pyvista.UnstructuredGrid.

        Parameters
        ----------
        grid
            A pyvista unstructured grid.

        Returns
        -------
        :
            An integer representing the polynomial order of the grid.
        """
        # Get cell type information from the first cell, assuming that every cell has the same type
        cell0 = grid.get_cell(0)
        cell_type = cell0.type
        num_points = cell0.n_points

        # Determine order based on the number of points and the cell type
        if cell_type in (1, 3, 5, 9, 10, 12):  # linear elements
            order = 1
        elif cell_type == 68:  # Lagrange curve
            # num_points is equal to order + 1
            order = num_points - 1
        elif cell_type == 69:  # Lagrange triangle
            # num_points is equal to (order + 1) * (order + 2) / 2
            order = int((np.sqrt(8 * num_points + 1) - 3) / 2)
        elif cell_type == 70:  # Lagrange quadrilateral
            # num_points is equal to (order + 1) ** 2
            order = int(np.sqrt(num_points)) - 1
        elif cell_type == 71:  # Lagrange tetrahedron
            # num_points is equal to (order + 1) * (order + 2) * (order + 3) / 6
            order = 0
            while ((order + 1) * (order + 2) * (order + 3)) // 6 < num_points:
                order += 1
        elif cell_type == 72:  # Lagrange hexahedron
            # num_points is equal to (order + 1) ** 3
            order = int(num_points ** (1 / 3)) - 1
        else:
            raise ValueError(f"Invalid cell type {cell_type}")

        return order
