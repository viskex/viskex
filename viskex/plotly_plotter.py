# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing plotly."""

import typing

import numpy as np
import numpy.typing
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
        fig = go.Figure(data=go.Scatter(
            x=coordinates, y=np.zeros_like(coordinates),
            line=dict(color="blue", width=2, dash="solid"), marker=dict(color="blue", size=10),
            mode=mode))
        fig.update_xaxes(title_text="x")
        return fig

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
