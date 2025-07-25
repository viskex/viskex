# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for tutorials tests."""

import re

import nbvalx.pytest_hooks_notebooks
import pytest

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file


def pytest_sessionstart(session: pytest.Session) -> None:
    """Automatically mark common files as data to be linked in the work directory."""
    # Add common files as data to be linked
    link_data_in_work_dir = session.config.option.link_data_in_work_dir
    assert len(link_data_in_work_dir) == 0
    link_data_in_work_dir.append("**/common_*.py")
    # Start session as in nbvalx
    nbvalx.pytest_hooks_notebooks.sessionstart(session)


def pytest_runtest_setup(item: nbvalx.pytest_hooks_notebooks.IPyNbFile) -> None:
    """Check backend availability."""
    # Get notebook name, ignore nbvalx parametrization
    notebook_name = re.sub(r"\[.*?\]", "", item.parent.name)
    # Check backend availability depending on the item name
    if notebook_name.endswith("dolfinx.ipynb"):
        pytest.importorskip("dolfinx")
    elif notebook_name.endswith("firedrake.ipynb"):
        pytest.importorskip("firedrake")
    else:
        raise ValueError("Invalid notebook name " + notebook_name)
