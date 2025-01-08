# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex base plotter."""

import abc
import typing

Mesh = typing.TypeVar("Mesh")
ScalarFunction = typing.TypeVar("ScalarFunction")
VectorFunction = typing.TypeVar("VectorFunction")
OutputMesh = typing.TypeVar("OutputMesh")
OutputPlotter = typing.TypeVar("OutputPlotter")


class BasePlotter(abc.ABC, typing.Generic[Mesh, ScalarFunction, VectorFunction, OutputMesh, OutputPlotter]):
    """viskex base plotter."""

    @classmethod
    @abc.abstractmethod
    def plot_mesh(
        cls, mesh: Mesh, dim: typing.Optional[int] = None,
        grid_filter: typing.Optional[typing.Callable[[OutputMesh], OutputMesh]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> OutputPlotter:
        """
        Plot a mesh.

        Parameters
        ----------
        mesh
            A mesh to be plotted.
        dim
            Plot entities associated to this dimension. If not provided, the topological dimension is used.
        grid_filter
            A filter to be applied to the grid representing the mesh before it is passed to the plotting backend.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to the plotting backend.

        Returns
        -------
        :
            An output plotter representing the plot of the mesh.
        """
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def plot_scalar_field(
        cls, scalar_field: ScalarFunction, name: str = "scalar", part: str = "real", warp_factor: float = 0.0,
        grid_filter: typing.Optional[typing.Callable[[OutputMesh], OutputMesh]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> OutputPlotter:
        """
        Plot a scalar field.

        Parameters
        ----------
        scalar_field
            A field to be plotted. The field must be a scalar field.
        name
            Name of the quantity stored in the scalar field.
        part
            Part of the field (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.
        warp_factor
            If provided then the factor is used to produce a warped representation
            the field; if not provided then the scalar field will be plotted on the mesh.
        grid_filter
            A filter to be applied to the grid representing the field before it is pass to the plotting backend.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to the plotting backend.

        Returns
        -------
        :
            A output plotter representing a plot of the scalar field.
        """
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def plot_vector_field(
        cls, vector_field: VectorFunction, name: str = "vector", part: str = "real", warp_factor: float = 0.0,
        glyph_factor: float = 0.0, grid_filter: typing.Optional[typing.Callable[[OutputMesh], OutputMesh]] = None,
        **kwargs: typing.Any  # noqa: ANN401
    ) -> OutputPlotter:
        """
        Plot a vector field stored in a dolfinx function, or a pair of UFL expression and dolfinx function space.

        Parameters
        ----------
        vector_field
            A field to be plotted. The field must be a vector field.
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
            A filter to be applied to the grid representing the field before it is pass to the plotting backend.
            If not provided, no filter will be applied.
        kwargs
            Additional keyword arguments to be passed to the plotting backend.

        Returns
        -------
        :
            A output plotter representing a plot of the vector field.
        """
        pass  # pragma: no cover
