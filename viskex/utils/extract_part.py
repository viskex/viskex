# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex utils module."""

import typing

import numpy as np
import numpy.typing
import petsc4py.PETSc


def extract_part(  # type: ignore[no-any-unimported]
    values: np.typing.NDArray[petsc4py.PETSc.ScalarType], name: str, part: str
) -> typing.Tuple[np.typing.NDArray[np.float64], str]:
    """Extract real or complex part from an array, and update the name to reflect this."""
    if np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):  # pragma: no cover
        if part == "real":
            values = values.real
            name = "real(" + name + ")"
        elif part == "imag":
            values = values.imag
            name = "imag(" + name + ")"
    return values, name
