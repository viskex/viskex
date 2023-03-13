# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex main module."""

from viskex.free_functions import (
    has_dolfinx, has_firedrake, plot_mesh, plot_mesh_entities, plot_scalar_field, plot_vector_field)

if has_dolfinx:
    from viskex.free_functions import plot_mesh_tags

if has_firedrake:
    from viskex.free_functions import plot_mesh_sets
