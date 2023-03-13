# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for tutorials tests."""

import os

import nbvalx.pytest_hooks_notebooks
import pytest
import pyvista

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file
pytest_runtest_makereport = nbvalx.pytest_hooks_notebooks.runtest_makereport
pytest_runtest_teardown = nbvalx.pytest_hooks_notebooks.runtest_teardown


def pytest_sessionstart(session: pytest.Session) -> None:
    """Start xvfb depending on the value of the VISKEX_PYVISTA_BACKEND environment variable."""
    # Do the session start as in nbvalx
    nbvalx.pytest_hooks_notebooks.sessionstart(session)
    # Start xfvb for pyvista backends that require so
    jupyter_backend = os.getenv("VISKEX_PYVISTA_BACKEND", "client")
    display = os.getenv("DISPLAY", None)
    if jupyter_backend in ("panel", "server", "static") and display is None:
        pyvista.start_xvfb()


def pytest_runtest_setup(item: pytest.File) -> None:
    """Check backend availability."""
    # Do the setup as in nbvalx
    nbvalx.pytest_hooks_notebooks.runtest_setup(item)
    # Get notebook name
    notebook_name = item.parent.name
    # Check backend availability depending on the item name
    if notebook_name.endswith("dolfinx.ipynb"):
        pytest.importorskip("dolfinx")
    elif notebook_name.endswith("firedrake.ipynb"):
        pytest.importorskip("firedrake")
    else:
        raise ValueError("Invalid notebook name " + notebook_name)
