# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing pyvista."""

import os
import typing

import numpy as np
import numpy.typing
import panel
import panel.pane.vtk.vtk
import pyvista.trame.jupyter

from viskex.base_plotter import BasePlotter

panel.extension("vtk")

pyvista.set_plot_theme("document")  # type: ignore[no-untyped-call]
pyvista.global_theme.cmap = "jet"


class PyvistaPlotter(BasePlotter[
    typing.Tuple[pyvista.UnstructuredGrid, int],
    typing.Tuple[pyvista.UnstructuredGrid, int],
    typing.Tuple[pyvista.UnstructuredGrid, pyvista.UnstructuredGrid, int],
    typing.Union[panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]
]):
    """viskex plotter interfacing pyvista."""

    _jupyter_backend = "client"
    try:
        import google.colab  # noqa: F401, I2000
    except ImportError:
        pass
    else:
        _jupyter_backend = "panel"
    if "kaggle" in os.environ.get("KAGGLE_URL_BASE", ""):
        _jupyter_backend = "panel"
    _jupyter_backend = os.getenv("VISKEX_PYVISTA_BACKEND", _jupyter_backend)
    assert _jupyter_backend in (
        "client", "server", "trame",  # trame backends
        "panel",  # panel backends
        "static"  # static backends
    )

    @classmethod
    def plot_mesh(
        cls, mesh_tdim: typing.Tuple[pyvista.UnstructuredGrid, int], dim: typing.Optional[int] = None
    ) -> typing.Union[
        panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a 2D or 3D mesh, stored in a pyvista.UnstructuredGrid.

        Parameters
        ----------
        mesh
            A pyvista unstructured grid to be plotted.
        dim
            Plot entities associated to this dimension. If not provided, the topological dimension is used.

        Returns
        -------
        :
            A pyvista widget representing a plot of the 2D or 3D mesh.
        """
        (mesh, tdim) = mesh_tdim
        assert dim is None
        plotter = pyvista.Plotter(notebook=True)  # type: ignore[no-untyped-call]
        plotter.add_mesh(mesh, color="red", edge_color="black", show_edges=True)  # type: ignore[no-untyped-call]
        if cls._jupyter_backend != "client":
            plotter.add_axes()  # type: ignore[no-untyped-call]
        if tdim == 2:
            plotter.camera_position = "xy"
        return cls._show_plotter(plotter)

    @classmethod
    def plot_mesh_entities(
        cls, mesh_tdim: typing.Tuple[pyvista.UnstructuredGrid, int], dim: int, name: str,
        indices: np.typing.NDArray[np.int32], values: np.typing.NDArray[np.int32]
    ) -> typing.Union[
        panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot `dim`-dimensional mesh entities of a 2D or 3D mesh.

        Parameters
        ----------
        mesh
            A pyvista unstructured grid to be plotted, alongisde its topological dimension.
            Entities in the grid should already be `dim`-dimensional, e.g. if the entities to be plotted
            are facets of a triangular mesh, then cells in the mesh must be segments.
        dim
            Plot entities associated to this dimension.
        name
            Name to be assigned to the field containing the mesh entities values.
        indices
            Array containing the IDs of the entities to be plotted.
        values
            Array containing the value to be associated to each of the entities in `indices`.
            Values are assumed to be greater than zero, because every entity not part of `indices`
            will be automatically assigned value equal to zero.

        Returns
        -------
        :
            A pyvista widget representing a plot of the mesh entities of the 2D or 3D mesh.
        """
        (mesh, tdim) = mesh_tdim
        all_values = np.zeros(mesh.n_cells)
        if values.shape[0] != all_values.shape[0]:
            assert np.all(values != 0), "Zero is used as a placeholder for non-provided entities"
        for (index, value) in zip(indices, values):
            all_values[index] = value

        mesh.cell_data[name] = all_values
        mesh.set_active_scalars(name)
        plotter = pyvista.Plotter(notebook=True)  # type: ignore[no-untyped-call]
        plotter.add_mesh(mesh, edge_color="black", show_edges=True)  # type: ignore[no-untyped-call]
        if cls._jupyter_backend != "client":
            plotter.add_axes()  # type: ignore[no-untyped-call]
        if tdim == 2:
            plotter.camera_position = "xy"
        return cls._show_plotter(plotter)

    @classmethod
    def plot_scalar_field(
        cls, mesh_tdim: typing.Tuple[pyvista.UnstructuredGrid, int], name: str, warp_factor: float = 0.0,
        part: str = "real"
    ) -> typing.Union[
        panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a 2D or 3D scalar field.

        Parameters
        ----------
        scalar_field
            A pyvista unstructured grid to be plotted, alongisde its topological dimension.
            The grid must already have the scalar field to be plotted set as the active scalar.
        name
            Name of the scalar field.
        warp_factor
            This argument is ignored for a field on 3D meshes.
            For a 2D mesh, if provided then the factor is used to produce a warped representation
            the field; if not provided then the scalar field will be plotted on the mesh.
        part
            Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.

        Returns
        -------
        :
            A pyvista widget representing a plot of the 2D or 3D scalar field.
        """
        (mesh, tdim) = mesh_tdim
        plotter = pyvista.Plotter(notebook=True)  # type: ignore[no-untyped-call]
        if warp_factor != 0.0:
            assert warp_factor > 0.0
            assert tdim == 2
            warped = mesh.warp_by_scalar(factor=warp_factor)  # type: ignore[no-untyped-call]
            plotter.add_mesh(warped)  # type: ignore[no-untyped-call]
        else:
            plotter.add_mesh(mesh)  # type: ignore[no-untyped-call]
            if tdim == 2:
                plotter.camera_position = "xy"
        if cls._jupyter_backend != "client":
            plotter.add_axes()  # type: ignore[no-untyped-call]
        return cls._show_plotter(plotter)

    @classmethod
    def plot_vector_field(
        cls, mesh_edgemesh_tdim: typing.Tuple[pyvista.UnstructuredGrid, pyvista.UnstructuredGrid, int],
        name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a 2D or 3D vector field.

        Parameters
        ----------
        mesh_edgemesh_tdim
            A triplet containing the pyvista unstructured grid to be plotted, a further unstructured
            grid representing its edges, and the topological dimension of the former grid.
            The grid must already have the vector field to be plotted set as the active vector.
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
            A pyvista widget representing a plot of the 2D or 3D vector field.
        """
        (mesh, edgemesh, tdim) = mesh_edgemesh_tdim
        plotter = pyvista.Plotter(notebook=True)  # type: ignore[no-untyped-call]
        if glyph_factor == 0.0 and warp_factor == 0.0:
            plotter.add_mesh(mesh)  # type: ignore[no-untyped-call]
        elif glyph_factor == 0.0 and warp_factor != 0.0:
            assert warp_factor > 0.0
            warped = mesh.warp_by_vector(factor=warp_factor)  # type: ignore[no-untyped-call]
            plotter.add_mesh(warped)  # type: ignore[no-untyped-call]
        else:
            assert glyph_factor > 0.0
            assert warp_factor == 0.0
            glyphs = mesh.glyph(orient=name, factor=glyph_factor)  # type: ignore[no-untyped-call]
            glyphs.rename_array("GlyphScale", name)
            plotter.add_mesh(glyphs)  # type: ignore[no-untyped-call]
            plotter.add_mesh(edgemesh)  # type: ignore[no-untyped-call]
        if tdim == 2:
            plotter.camera_position = "xy"
        if cls._jupyter_backend != "client":
            plotter.add_axes()  # type: ignore[no-untyped-call]
        return cls._show_plotter(plotter)

    @classmethod
    def _show_plotter(cls, plotter: pyvista.Plotter) -> typing.Union[
        panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """Show pyvista Plotter using the requested backend."""
        if cls._jupyter_backend == "panel":
            # Set up camera
            plotter._on_first_render_request()  # type: ignore[no-untyped-call]
            # Only set window size if explicitly set within the plotter
            if not plotter._window_size_unset:
                width, height = plotter.window_size
                sizing = {"width": width, "height": height}
            else:
                sizing = {}
            # Pass plotter render window to panel
            return panel.panel(
                plotter.render_window, orientation_widget=plotter.renderer.axes_enabled,
                enable_keybindings=False, sizing_mode="stretch_width", **sizing
            )
        else:
            return plotter.show(  # type: ignore[no-any-return, no-untyped-call]
                jupyter_backend=cls._jupyter_backend, return_viewer=True)
