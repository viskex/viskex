# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex converter interfacing pyvista."""

import pyvista

from viskex.base_converter import BaseConverter, Field, Mesh


class PyvistaConverter(BaseConverter[Mesh, Field, pyvista.UnstructuredGrid]):
    """viskex converter interfacing pyvista."""

    pass
