# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex utils module."""


import numpy as np
import numpy.typing as npt

from viskex.utils.scalar_type import ScalarType


def extract_part(
    values: npt.NDArray[ScalarType], name: str, part: str
) -> tuple[npt.NDArray[np.float64], str]:
    """Extract real or complex part from an array, and update the name to reflect this."""
    if np.issubdtype(ScalarType, np.complexfloating):
        if part == "real":
            values = values.real
            name = "real(" + name + ")"
        elif part == "imag":
            values = values.imag
            name = "imag(" + name + ")"
        else:
            raise RuntimeError(f"Invalid part {part}")
    else:
        if part != "real":
            raise RuntimeError(f"Invalid part {part}")
    return values, name
