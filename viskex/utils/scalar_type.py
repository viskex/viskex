# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Definition of default scalar type."""

import numpy as np

try:
    import petsc4py.PETSc
except ImportError:
    ScalarType = np.float64
else:
    ScalarType = petsc4py.PETSc.ScalarType  # type: ignore[attr-defined, misc, unused-ignore]
