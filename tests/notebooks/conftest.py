# Copyright (C) 2023-2024 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for notebooks tests."""

import nbvalx.pytest_hooks_notebooks
import pytest

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file
pytest_runtest_makereport = nbvalx.pytest_hooks_notebooks.runtest_makereport
pytest_runtest_teardown = nbvalx.pytest_hooks_notebooks.runtest_teardown


def pytest_sessionstart(session: pytest.Session) -> None:
    """Automatically mark common files as data to be copied to the work directory."""
    # Add common files as data to be copied
    copy_data_to_work_dir = session.config.option.copy_data_to_work_dir
    assert len(copy_data_to_work_dir) == 0
    copy_data_to_work_dir.append("**/common_*.py")
    # Start session as in nbvalx
    nbvalx.pytest_hooks_notebooks.sessionstart(session)


def pytest_runtest_setup(item: nbvalx.pytest_hooks_notebooks.IPyNbFile) -> None:
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
    elif notebook_name.endswith("firedrake_netgen.ipynb"):
        pytest.importorskip("firedrake")
        pytest.importorskip("netgen")
    else:
        raise ValueError("Invalid notebook name " + notebook_name)
