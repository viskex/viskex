# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Definition of default real type."""

import numpy as np

try:
    import petsc4py.PETSc
except ImportError:
    RealType = np.float64
else:
    RealType = petsc4py.PETSc.RealType  # type: ignore[attr-defined, misc, unused-ignore]
