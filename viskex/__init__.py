# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex main module."""

import importlib

from viskex.base_converter import BaseConverter
from viskex.base_plotter import BasePlotter
from viskex.pyvista_converter import PyvistaConverter
from viskex.pyvista_plotter import PyvistaPlotter

if importlib.util.find_spec("dolfinx"):
    import viskex.dolfinx
    from viskex.dolfinx_converter import DolfinxConverter
    from viskex.dolfinx_plotter import DolfinxPlotter

if importlib.util.find_spec("firedrake"):
    import viskex.firedrake
    from viskex.firedrake_converter import FiredrakeConverter
    from viskex.firedrake_plotter import FiredrakePlotter
