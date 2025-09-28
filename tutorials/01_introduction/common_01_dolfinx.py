# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Common functions used across dolfinx notebooks tests."""

import typing

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import ufl


def mark_subdomains(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.MeshTags:
    """Mark left and right subdomains in a given mesh with values 1 and 2, respectively."""
    def left_subdomain(x: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Condition that defines the left subdomain."""
        return x[0] <= 1.0 / 3.0  # type: ignore[no-any-return]

    def right_subdomain(x: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Condition that defines the right subdomain."""
        return x[0] >= 2.0 / 3.0  # type: ignore[no-any-return]

    subdomains_entities = dict()
    subdomains_values = dict()
    for (subdomain, subdomain_id) in zip((left_subdomain, right_subdomain), (1, 2)):
        subdomains_entities[subdomain_id] = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, subdomain)
        subdomains_values[subdomain_id] = np.full(subdomains_entities[subdomain_id].shape, subdomain_id, dtype=np.int32)

    subdomains_entities_unsorted = np.hstack(list(subdomains_entities.values()))
    subdomains_values_unsorted = np.hstack(list(subdomains_values.values()))
    subdomains_entities_argsort = np.argsort(subdomains_entities_unsorted)
    subdomains_entities_sorted = subdomains_entities_unsorted[subdomains_entities_argsort]
    subdomains_values_sorted = subdomains_values_unsorted[subdomains_entities_argsort]
    subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, subdomains_entities_sorted, subdomains_values_sorted)
    return subdomains


def mark_boundaries(mesh: dolfinx.mesh.Mesh, subdomains: dolfinx.mesh.MeshTags) -> dolfinx.mesh.MeshTags:
    """
    Mark internal and boundary facets in a given mesh with four different values.

    Internal facets of left and right subdomains are associated with values 1 and 2, respectively.
    Furthermore, boundary facets on the left and right boundaries are associated with values 3 and 4,
    respectively.
    """
    def left(x: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Condition that defines the left boundary."""
        return abs(x[0] - 0.) < np.finfo(float).eps  # type: ignore[no-any-return]

    def right(x: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Condition that defines the right boundary."""
        return abs(x[0] - 1.) < np.finfo(float).eps  # type: ignore[no-any-return]

    subdomain_to_boundary_function = {1: left, 2: right}

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
    cell_to_facets_connectivity = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    cell_to_vertices_connectivity = mesh.topology.connectivity(mesh.topology.dim, 0)
    facet_to_vertices_connectivity = mesh.topology.connectivity(mesh.topology.dim - 1, 0)
    cells_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cells_map.size_local + cells_map.num_ghosts
    vertex_map = {
        topology_index: geometry_index for cell in range(num_cells) for (topology_index, geometry_index) in zip(
            cell_to_vertices_connectivity.links(cell), mesh.geometry.dofmap[cell])
    }
    facets_indices = list()
    facets_values = list()
    for cell in range(cell_to_facets_connectivity.num_nodes):
        facets = cell_to_facets_connectivity.links(cell)
        subdomain_index = np.where(subdomains.indices == cell)[0]
        if subdomain_index.size > 0:
            assert subdomain_index.size == 1
            subdomain_value = subdomains.values[subdomain_index[0]]
            if subdomain_value in (1, 2):
                boundary_function = subdomain_to_boundary_function[subdomain_value]
                for facet in facets:
                    if facet not in facets_indices:
                        vertices = [vertex_map[vertex] for vertex in facet_to_vertices_connectivity.links(facet)]
                        facets_indices.append(facet)
                        if all(boundary_function(mesh.geometry.x[vertex]) for vertex in vertices):
                            facets_values.append(subdomain_value + 2)
                        else:
                            facets_values.append(subdomain_value)

    boundaries_and_interfaces_entities_unsorted = np.array(facets_indices, dtype=np.int32)
    boundaries_and_interfaces_values_unsorted = np.array(facets_values, dtype=np.int32)
    boundaries_and_interfaces_entities_argsort = np.argsort(boundaries_and_interfaces_entities_unsorted)
    boundaries_and_interfaces_entities_sorted = boundaries_and_interfaces_entities_unsorted[
        boundaries_and_interfaces_entities_argsort]
    boundaries_and_interfaces_values_sorted = boundaries_and_interfaces_values_unsorted[
        boundaries_and_interfaces_entities_argsort]
    boundaries_and_interfaces = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1,
        boundaries_and_interfaces_entities_sorted, boundaries_and_interfaces_values_sorted)

    return boundaries_and_interfaces


def prepare_scalar_field_cases(  # type: ignore[no-any-unimported]
    mesh: dolfinx.mesh.Mesh,
    expression: typing.Callable[
        [npt.NDArray[np.float64] | ufl.core.expr.Expr],
        npt.NDArray[np.float64] | ufl.core.expr.Expr
    ]
) -> tuple[
    dolfinx.fem.Function, tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]
]:
    """Prepare scalar field cases."""
    scalar_function_space = dolfinx.fem.functionspace(mesh, ("CG", 2))
    scalar_field = dolfinx.fem.Function(scalar_function_space)
    scalar_field.interpolate(expression)
    scalar_field_ufl = expression(ufl.SpatialCoordinate(mesh))
    return scalar_field, (scalar_field_ufl, scalar_function_space)


def prepare_vector_field_cases(  # type: ignore[no-any-unimported]
    mesh: dolfinx.mesh.Mesh,
    expression: typing.Callable[
        [npt.NDArray[np.float64] | ufl.core.expr.Expr],
        npt.NDArray[np.float64] | ufl.core.expr.Expr
    ]
) -> tuple[
    dolfinx.fem.Function, tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]
]:
    """Prepare vector field cases."""
    vector_function_space = dolfinx.fem.functionspace(mesh, ("CG", 2, (mesh.geometry.dim, )))
    vector_field = dolfinx.fem.Function(vector_function_space)
    vector_field.interpolate(expression)
    vector_field_ufl = ufl.as_vector(expression(ufl.SpatialCoordinate(mesh)))
    return vector_field, (vector_field_ufl, vector_function_space)
