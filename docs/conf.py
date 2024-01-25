# Copyright (C) 2023-2024 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex documentation configuration."""

# Project information
project = "viskex"
copyright = "2023-2024, the viskex authors"
author = "Francesco Ballarin"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode"
]

# Extensions configuration
autodoc_default_options = {
    "exclude-members": ",".join([  # noqa: FLY002
        "__abstractmethods__", "__annotations__", "__dict__", "__init__", "__module__", "__orig_bases__",
         "__parameters__", "__weakref__"
    ]),
    "imported-members": True,
    "members": True,
    "show-inheritance": True,
    "special-members": True,
    "undoc-members": True
}

# Options for HTML output
html_theme = "nature"
