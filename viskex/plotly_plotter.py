# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing plotly."""

import typing

import numpy as np
import numpy.typing
import petsc4py.PETSc
import plotly.colors
import plotly.graph_objects as go

from viskex.base_plotter import BasePlotter


class PlotlyPlotter(BasePlotter):
    """viskex plotter interfacing plotly."""

    def plot_mesh(self, coordinates: np.typing.NDArray[np.float64], dim: typing.Optional[int] = None) -> go.Figure:
        """
        Plot a 1D mesh, described by a set of coordinates.

        Parameters
        ----------
        coordinates
            Array of mesh coordinates. The coordinates are assumed to be sorted in increasing order.
        dim
            Plot entities associated to this dimension. If not provided, the topological dimension is used.

        Returns
        -------
        :
            A plotly figure representing a plot of the 1D mesh.
        """
        if dim is None:
            dim = 1
        assert dim in (0, 1)
        if dim == 0:
            mode = "markers"
        elif dim == 1:
            mode = "lines+markers"
        else:
            raise RuntimeError("Invalid dimension")
        fig = go.Figure()
        fig.add_scatter(
            x=coordinates, y=np.zeros_like(coordinates),
            line=dict(color="blue", width=2, dash="solid"), marker=dict(color="blue", size=10),
            mode=mode
        )
        fig.update_xaxes(title_text="x")
        return fig

    def plot_mesh_entities(
        self, coordinates: np.typing.NDArray[np.float64], dim: int, name: str,
        indices: np.typing.NDArray[np.int32], values: np.typing.NDArray[np.int32]
    ) -> go.Figure:
        """
        Plot `dim`-dimensional mesh entities of a 1D mesh.

        Parameters
        ----------
        coordinates
            Array of mesh coordinates. The coordinates are assumed to be sorted in increasing order.
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
            A plotly figure representing a plot of the mesh entities of the 1D mesh.
        """
        assert dim in (0, 1)

        all_values = np.zeros(coordinates.shape[0] if dim == 0 else coordinates.shape[0] - 1, dtype=np.int32)
        if values.shape[0] != all_values.shape[0]:
            assert np.all(values != 0), "Zero is used as a placeholder for non-provided entities"
        for (index, value) in zip(indices, values):
            all_values[index] = value

        def consecutive(data):
            """Helper function to determine consecutive indices in an array."""
            return np.split(data, np.where(np.diff(data) != 1)[0] + 1)

        colors = plotly.colors.qualitative.Set1
        fig = go.Figure()
        for value in np.unique(all_values):
            indices_value = np.where(all_values == value)[0]
            if dim == 0:
                mode = "markers"
                marker_color = colors[value]
                line_color = None
                coordinates_value = [coordinates[indices_value]]
            elif dim == 1:
                mode = "lines+markers"
                marker_color = "black"
                line_color = colors[value]
                coordinates_value = [
                    coordinates[consecutive_indices[0]:consecutive_indices[-1] + 2]
                    for consecutive_indices in consecutive(indices_value)
                ]
            else:
                raise RuntimeError("Invalid dimension")
            for s, coordinates_value_s in enumerate(coordinates_value):
                if s == 0:
                    name_value_s = str(name + " == " + str(value))
                else:
                    name_value_s = None
                fig.add_scatter(
                    x=coordinates_value_s, y=np.zeros_like(coordinates_value_s),
                    line=dict(color=line_color, width=2, dash="solid"), marker=dict(color=marker_color, size=10),
                    mode=mode, name=name_value_s
                )
        fig.update_xaxes(title_text="x")
        fig.update_layout(showlegend=True)
        return fig

    def plot_scalar_field(
        self, scalar_field: typing.Tuple[np.typing.NDArray[np.float64], np.typing.NDArray[petsc4py.PETSc.ScalarType]],
        name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> go.Figure:
        """
        Plot a 1D scalar field.

        Parameters
        ----------
        scalar_field
            A pair containing the array of mesh coordinates, and the corresponding values for the scalar field.
            The coordinates are assumed to be sorted in increasing order.
        name
            Name of the quantity stored in the scalar field.
        warp_factor
            This argument is ignored for a field on a 1D mesh.
        part
            Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.

        Returns
        -------
        :
            A plotly figure representing a plot of the 1D scalar field.
        """
        (coordinates, values) = scalar_field
        fig = go.Figure()
        fig.add_scatter(
            x=coordinates, y=values,
            line=dict(color="blue", width=2, dash="solid"), marker=dict(color="blue", size=10),
            mode="lines+markers"
        )
        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text=name)
        return fig

    def plot_vector_field(
        self, vector_field: typing.Tuple[np.typing.NDArray[np.float64], np.typing.NDArray[petsc4py.PETSc.ScalarType]],
        name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0, part: str = "real"
    ) -> None:
        """Cannot plot a 1D vector field: no such field exists."""
        raise RuntimeError("Cannot call plot_vector_field for 1D meshes")
