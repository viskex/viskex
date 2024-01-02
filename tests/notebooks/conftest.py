# Copyright (C) 2023-2024 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for notebooks tests."""

import fnmatch
import os
import shutil

import _pytest
import nbvalx.pytest_hooks_notebooks
import pytest

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_sessionstart = nbvalx.pytest_hooks_notebooks.sessionstart
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file
pytest_runtest_makereport = nbvalx.pytest_hooks_notebooks.runtest_makereport
pytest_runtest_teardown = nbvalx.pytest_hooks_notebooks.runtest_teardown


def pytest_runtest_setup(item: nbvalx.pytest_hooks_notebooks.IPyNbFile) -> None:
    """Check backend availability."""
    # Do the setup as in nbvalx
    nbvalx.pytest_hooks_notebooks.runtest_setup(item)
    # Get notebook name
    notebook_name = item.parent.name
    # Determine backend names from the item name
    if notebook_name.endswith("dolfinx.ipynb"):
        backends = ["dolfinx"]
    elif notebook_name.endswith("firedrake.ipynb"):
        backends = ["firedrake"]
    elif notebook_name.endswith("firedrake_netgen.ipynb"):
        backends = ["firedrake", "netgen"]
    else:
        raise ValueError("Invalid notebook name " + notebook_name)
    # Copy common files
    if item.name == "Cell 0":
        work_dir = item.parent.config.option.work_dir
        notebook_original_dir = os.path.dirname(notebook_name).replace(work_dir, "")
        if notebook_original_dir == "":
            notebook_original_dir = "."
        for dir_entry in _pytest.pathlib.visit(notebook_original_dir, lambda _: True):
            if dir_entry.is_file():
                source_path = str(dir_entry.path)
                if (
                    any(fnmatch.fnmatch(source_path, f"**/common_{backend}.py") for backend in backends)
                        and
                    work_dir not in source_path
                ):
                    destination_path = os.path.join(
                        notebook_original_dir, work_dir, os.path.relpath(source_path, notebook_original_dir))
                    if not os.path.exists(destination_path):
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        shutil.copyfile(source_path, destination_path)
    # Skip if backend is not available
    for backend in backends:
        pytest.importorskip(backend)
