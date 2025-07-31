# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Add point markers to the plotter."""

import typing

import numpy as np
import pyvista

from viskex.utils.compute_screen_pixel_size_in_world import compute_screen_pixel_size_in_world


def add_point_markers(
    plotter: pyvista.Plotter, mesh: pyvista.DataSet, dim: int, point_size: float,
    point_color: typing.Optional[str] = None,
    point_cmap: typing.Optional[typing.Union[str, list[str], pyvista.LookupTable]] = None
) -> None:
    """
    Add point markers (lines, squares, or cubes) at specified points in the plotter.

    The added markers are sized roughly to the given screen-space point size in pixels.

    Parameters
    ----------
    plotter
        The pyvista plotter to add glyphs to.
    mesh
        A pyvista dataset object containing point coordinates and optional point or cell data.
    dim
        Topological dimension of glyphs: 1 for vertical line segment, 2 for square plane, 3 for cube.
    point_size
        Screen-space point size in pixels (like PyVista's `point_size`).
    point_color
        If `point_color` is None and `mesh` has point scalars, the glyphs will be colored
        by that scalar data. Otherwise, the color provided in `point_color` will be used.
    point_cmap
        If `point_color` is None and `mesh` has point scalars, colormap to be used when plotting
        scalar data. Otherwise, the value is ignored.

    Notes
    -----
    This method creates actual 3D glyph meshes (lines, planes, or cubes) positioned
    at the given points and sized to approximately match a desired screen-space pixel size.

    This differs from using `show_vertices=True` in `plotter.add_mesh(mesh, show_vertices=True)`,
    which renders simple square points handled internally by VTK, typically always aligned to
    the screen and potentially more efficient.

    Advantages of this approach:
      - Custom glyph shapes with dimensionality control (line, square or cube).
      - Precise control over glyph size in world coordinates computed from screen pixels.

    Disadvantages compared to `show_vertices=True`:
      - Requires more geometry (glyph meshes) and may be less efficient for large point sets.
      - Glyphs are world-aligned and do not always behave exactly like VTK points (which
        stay square on screen regardless of camera orientation, and stay always of the same size).
    """
    # Estimate world size for the glyph from screen pixels
    px_to_world = compute_screen_pixel_size_in_world(plotter)
    glyph_size = point_size * px_to_world

    # Build glyph source geometry based on dim
    if dim == 1:
        source = pyvista.Line((0, -glyph_size / 2, 0), (0, glyph_size / 2, 0))
    elif dim == 2:
        source = pyvista.Plane(center=(0, 0, 0), i_size=glyph_size, j_size=glyph_size)  # type: ignore[arg-type]
    elif dim == 3:
        source = pyvista.Cube(
            center=(0, 0, 0), x_length=glyph_size, y_length=glyph_size, z_length=glyph_size, clean=False)
    else:
        raise ValueError(f"Unsupported dimension {dim}")

    # Create glyphs at mesh points
    glyphs = mesh.glyph(orient=False, scale=False, geom=source)

    # Determine coloring
    if point_color is not None:
        # Use provided point color
        color_opts: dict[str, typing.Any] = {
            "color": point_color,
            "smooth_shading": True,
            "ambient": 1.0,
            "specular": 0.0,
            "show_edges": False,
            "lighting": True
        }
    else:
        # Use scalar coloring if available
        scalars = mesh.active_scalars_name
        if scalars is None:
            raise RuntimeError("The mesh PolyData must have active scalar data if point_color is None")

        in_point_data = scalars in mesh.point_data
        in_cell_data = scalars in mesh.cell_data

        if in_point_data and in_cell_data:
            raise ValueError(f"Scalar '{scalars}' exists in both point_data and cell_data; please disambiguate.")

        # Repeat scalar values to match glyph points
        if in_point_data:
            n_original_points = mesh.n_points
            n_glyph_points = glyphs.n_points
            if n_glyph_points % n_original_points != 0:
                raise RuntimeError(
                    f"Glyph points ({n_glyph_points}) not multiple of original points ({n_original_points}).")
            points_per_glyph = n_glyph_points // n_original_points
            original_scalars = mesh.point_data[scalars]
            if len(original_scalars) != n_original_points:
                raise RuntimeError("Original point scalar length mismatch.")
            repeated_scalars = np.repeat(original_scalars, points_per_glyph)
            glyphs.point_data[scalars] = repeated_scalars
        elif in_cell_data:
            n_original_cells = mesh.n_cells
            n_glyph_cells = glyphs.n_cells
            if n_glyph_cells % n_original_cells != 0:
                raise RuntimeError(
                    f"Glyph cells ({n_glyph_cells}) not multiple of original cells ({n_original_cells}).")
            cells_per_glyph = n_glyph_cells // n_original_cells
            original_scalars = mesh.cell_data[scalars]
            if len(original_scalars) != n_original_cells:
                raise RuntimeError("Original cell scalar length mismatch.")
            # Reorder cell scalars to match mesh.point order, if possible
            if isinstance(mesh, pyvista.PolyData) and mesh.verts.size > 0:
                assert np.all(mesh.verts[0::2] == 1)  # cells formed by a single vertex
                cell_to_point = mesh.verts[1::2]
            elif isinstance(mesh, pyvista.UnstructuredGrid) and mesh.cells.size > 0:
                assert np.all(mesh.cells[0::2] == 1)  # cells formed by a single vertex
                cell_to_point = mesh.cells[1::2]
            else:
                raise RuntimeError("Cannot deduce cell-to-point mapping for scalar reordering.")
            point_to_cell_index = np.argsort(cell_to_point)
            reordered_scalars = original_scalars[point_to_cell_index]
            repeated_scalars = np.repeat(reordered_scalars, cells_per_glyph)
            glyphs.cell_data[scalars] = repeated_scalars
        else:
            raise ValueError(f"Scalar '{scalars}' not found in point_data or cell_data.")

        color_opts = {
            "scalars": scalars,
            "show_edges": False,
            "lighting": False,
            "cmap": point_cmap
        }

    # Add glyphs to the plotter
    plotter.add_mesh(glyphs, **color_opts)
