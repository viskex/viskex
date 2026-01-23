# Copyright (C) 2023-2026 by the viskex authors
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
    RealType = petsc4py.PETSc.RealType  # type: ignore[assignment, attr-defined, misc, unused-ignore]
