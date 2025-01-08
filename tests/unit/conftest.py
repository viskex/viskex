# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for unit tests."""

import nbvalx.pytest_hooks_unit_tests

pytest_runtest_setup = nbvalx.pytest_hooks_unit_tests.runtest_setup
pytest_runtest_teardown = nbvalx.pytest_hooks_unit_tests.runtest_teardown
