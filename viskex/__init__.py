# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex main module."""

import typing

try:
    import dolfinx as dolfinx_check_availability
except ImportError:
    pass
else:
    del dolfinx_check_availability
    from viskex.dolfinx_plotter import DolfinxPlotter as dolfinx  # noqa: N813

try:
    import firedrake as firedrake_check_availability
except ImportError:
    pass
else:
    del firedrake_check_availability
    from viskex.firedrake_plotter import FiredrakePlotter as firedrake  # noqa: N813
