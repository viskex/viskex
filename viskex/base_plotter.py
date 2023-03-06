# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex base plotter."""

import abc
import typing

Function = typing.TypeVar("Function")
Mesh = typing.TypeVar("Mesh")
MeshEntities = typing.TypeVar("MeshEntities")
PlotterWidget = typing.TypeVar("PlotterWidget")


class BasePlotter(abc.ABC):
    """viskex base plotter."""

    @abc.abstractmethod
    def plot_mesh(self, mesh: Mesh, dim: typing.Optional[int] = None) -> PlotterWidget:
        """Plot a mesh."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def plot_mesh_entities(self, mesh: Mesh, dim: int, entities: MeshEntities) -> PlotterWidget:
        """Plot a mesh, highlighting the provided `dim`-dimensional entities."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def plot_scalar_field(
        self, scalar_field: Function, name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> PlotterWidget:
        """Plot a scalar field."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def plot_vector_field(
        self, vector_field: Function, name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0,
        part: str = "real"
    ) -> PlotterWidget:
        """Plot a vector field."""
        pass  # pragma: no cover
