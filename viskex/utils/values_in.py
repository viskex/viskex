# Copyright (C) 2023-2024 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Limit cell data in the pyvista grid to specific values."""

import typing

import numpy as np
import numpy.typing
import pyvista


def values_in(
    values: typing.Union[typing.List[typing.Any], np.typing.NDArray[typing.Any]]
) -> typing.Callable[[pyvista.UnstructuredGrid], pyvista.UnstructuredGrid]:
    """Limit cell data in the pyvista grid to specific values."""
    def _(grid: pyvista.UnstructuredGrid) -> pyvista.UnstructuredGrid:
        """Limit cell data in the pyvista grid to specific values: internal implementation."""
        name = grid.cell_data.active_scalars_name
        assert name is not None
        data = grid.cell_data.active_scalars
        assert data is not None
        not_equal_to_values, = np.where(~np.isin(data, values))
        data[not_equal_to_values] = np.nan
        grid.cell_data[name] = data
        return grid

    return _
