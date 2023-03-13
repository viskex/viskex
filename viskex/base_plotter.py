# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex base plotter."""

import abc
import typing

import numpy as np
import numpy.typing

Mesh = typing.TypeVar("Mesh")
ScalarFunction = typing.TypeVar("ScalarFunction")
VectorFunction = typing.TypeVar("VectorFunction")
PlotterWidget = typing.TypeVar("PlotterWidget")


class BasePlotter(abc.ABC, typing.Generic[Mesh, ScalarFunction, VectorFunction, PlotterWidget]):
    """viskex base plotter."""

    @classmethod
    @abc.abstractmethod
    def plot_mesh(cls, mesh: Mesh, dim: typing.Optional[int] = None) -> PlotterWidget:
        """Plot a mesh."""
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def plot_mesh_entities(
        cls, mesh: Mesh, dim: int, name: str, indices: np.typing.NDArray[np.int32], values: np.typing.NDArray[np.int32]
    ) -> PlotterWidget:
        """Plot `dim`-dimensional entities."""
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def plot_scalar_field(
        cls, scalar_field: ScalarFunction, name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> PlotterWidget:
        """Plot a scalar field."""
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def plot_vector_field(
        cls, vector_field: VectorFunction, name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0,
        part: str = "real"
    ) -> PlotterWidget:
        """Plot a vector field."""
        pass  # pragma: no cover
