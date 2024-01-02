# Copyright (C) 2023-2024 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Common functions used across firedrake notebooks tests."""

import typing

import firedrake
import numpy as np
import ufl


def mark_subdomains(mesh: firedrake.MeshGeometry) -> firedrake.MeshGeometry:  # type: ignore[no-any-unimported]
    """Mark left and right subdomains in a given mesh with values 1 and 2, respectively."""
    cellname = mesh.ufl_cell().cellname()
    if cellname in ("interval", "triangle", "tetrahedron"):
        subdomains_function_space = firedrake.FunctionSpace(mesh, "DP", 0)
    elif cellname in ("quadrilateral", "hexahedron"):
        subdomains_function_space = firedrake.FunctionSpace(mesh, "DQ", 0)
    else:
        raise RuntimeError("Invalid cellname")
    x = firedrake.SpatialCoordinate(mesh)
    left_subdomain = firedrake.Function(subdomains_function_space).interpolate(
        firedrake.conditional(x[0] <= 1.0 / 3.0, 1.0, 0.0))
    right_subdomain = firedrake.Function(subdomains_function_space).interpolate(
        firedrake.conditional(x[0] >= 2.0 / 3.0, 1.0, 0.0))
    mesh_with_subdomains = firedrake.RelabeledMesh(mesh, [left_subdomain, right_subdomain], [1, 2])
    mesh_with_subdomains.init()
    return mesh_with_subdomains


def mark_boundaries(mesh: firedrake.MeshGeometry) -> firedrake.MeshGeometry:  # type: ignore[no-any-unimported]
    """
    Mark internal and boundary facets in a given mesh with four different values.

    Internal facets of left and right subdomains are associated with values 1 and 2, respectively.
    Furthermore, boundary facets on the left and right boundaries are associated with values 3 and 4,
    respectively.
    """
    cellname = mesh.ufl_cell().cellname()
    if cellname in ("interval", ):
        boundaries_function_space = firedrake.FunctionSpace(mesh, "P", 1)
    elif cellname in ("triangle", "quadrilateral", "tetrahedron"):
        boundaries_function_space = firedrake.FunctionSpace(mesh, "HDiv Trace", 0)
    elif cellname in ("hexahedron", ):
        boundaries_function_space = firedrake.FunctionSpace(mesh, "Q", 2)
    else:
        raise RuntimeError("Invalid cellname")
    x = firedrake.SpatialCoordinate(mesh)
    left_boundary = firedrake.Function(boundaries_function_space).interpolate(
        firedrake.conditional(abs(x[0] - 0.) < np.sqrt(np.finfo(float).eps), 1.0, 0.0))
    right_boundary = firedrake.Function(boundaries_function_space).interpolate(
        firedrake.conditional(abs(x[0] - 1.) < np.sqrt(np.finfo(float).eps), 1.0, 0.0))
    left_subdomain = firedrake.Function(boundaries_function_space).interpolate(
        firedrake.conditional(
            firedrake.And(x[0] <= 1.0 / 3.0, abs(x[0] - 0.) > np.sqrt(np.finfo(float).eps)), 1.0, 0.0))
    right_subdomain = firedrake.Function(boundaries_function_space).interpolate(
        firedrake.conditional(
            firedrake.And(x[0] >= 2.0 / 3.0, abs(x[0] - 1.) > np.sqrt(np.finfo(float).eps)), 1.0, 0.0))
    mesh.topology_dm.removeLabel(firedrake.cython.dmcommon.FACE_SETS_LABEL)
    mesh_with_boundaries = firedrake.RelabeledMesh(
        mesh, [left_boundary, right_boundary, left_subdomain, right_subdomain], [3, 4, 1, 2])
    mesh_with_boundaries.init()
    return mesh_with_boundaries


def prepare_scalar_field_cases(  # type: ignore[no-any-unimported]
    mesh: firedrake.Mesh,
    expression: typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr]
) -> typing.Tuple[firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]]:
    """Prepare scalar field cases."""
    scalar_function_space = firedrake.FunctionSpace(mesh, "CG", 2)
    scalar_field_ufl = expression(ufl.SpatialCoordinate(mesh))
    scalar_field = firedrake.interpolate(scalar_field_ufl, scalar_function_space)
    return scalar_field, (scalar_field_ufl, scalar_function_space)


def prepare_vector_field_cases(  # type: ignore[no-any-unimported]
    mesh: firedrake.Mesh,
    expression: typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr]
) -> typing.Tuple[firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]]:
    """Prepare vector field cases."""
    vector_function_space = firedrake.VectorFunctionSpace(mesh, "CG", 2)
    vector_field_ufl = ufl.as_vector(expression(ufl.SpatialCoordinate(mesh)))
    vector_field = firedrake.interpolate(vector_field_ufl, vector_function_space)
    return vector_field, (vector_field_ufl, vector_function_space)
