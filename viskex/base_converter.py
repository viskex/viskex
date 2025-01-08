# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex base converter."""

import abc
import typing

Mesh = typing.TypeVar("Mesh")
Field = typing.TypeVar("Field")
OutputMesh = typing.TypeVar("OutputMesh")


class BaseConverter(abc.ABC, typing.Generic[Mesh, Field, OutputMesh]):
    """viskex base converter."""

    @classmethod
    @abc.abstractmethod
    def convert_mesh(cls, mesh: Mesh, dim: typing.Optional[int] = None) -> OutputMesh:
        """
        Convert a mesh.

        Parameters
        ----------
        mesh
            A mesh to be converted.
        dim
            Convert entities associated to this dimension. If not provided, the topological dimension is used.

        Returns
        -------
        :
            An unstructured grid representing the mesh.
        """
        pass  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def convert_field(cls, field: Field, name: str, part: str = "real") -> OutputMesh:
        """
        Convert a field.

        Parameters
        ----------
        field
            Field to be converted.
        name
            Name of the quantity stored in the field.
        part
            Part of the field (real or imag) to be converted. By default, the real part is converted.
            The argument is ignored when converting a real field.

        Returns
        -------
        :
            An unstructured grid representing the field.
        """
        pass  # pragma: no cover
