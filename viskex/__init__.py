# Copyright (C) 2023-2024 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex main module."""

from viskex.base_plotter import BasePlotter
from viskex.plotly_plotter import PlotlyPlotter
from viskex.pyvista_plotter import PyvistaPlotter

try:
    import dolfinx as dolfinx_check_availability
except ImportError:
    pass
else:
    del dolfinx_check_availability
    from viskex.dolfinx_plotter import DolfinxPlotter
    dolfinx = DolfinxPlotter

try:
    import firedrake as firedrake_check_availability
except ImportError:
    pass
else:
    del firedrake_check_availability
    from viskex.firedrake_plotter import FiredrakePlotter
    firedrake = FiredrakePlotter
