# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex converter interfacing firedrake."""

import typing

import firedrake
import firedrake.output
import numpy as np
import pyvista
import ufl

from viskex.pyvista_converter import PyvistaConverter
from viskex.utils import extract_part


class FiredrakeConverter(PyvistaConverter[  # type: ignore[no-any-unimported]
    firedrake.MeshGeometry,
    typing.Union[firedrake.Function, tuple[ufl.core.expr.Expr, ufl.FunctionSpace]]
]):
    """viskex converter interfacing firedrake."""

    # Conversion from UFL cell name and is_linear (True/False) to vtk cell type. Replicates the cells attribute
    # at the beginning of firedrake/output.py, with the exception that the boolean attribute is reversed
    # (this code uses is_linear, firedrake uses not is_linear)
    _ufl_cellname_to_vtk_celltype: typing.ClassVar[dict[tuple[str, bool], int]] = {
        ("point", True): 1,
        ("point", False): 1,
        ("interval", True): 3,
        ("interval", False): 68,
        ("triangle", True): 5,
        ("triangle", False): 69,
        ("quadrilateral", True): 9,
        ("quadrilateral", False): 70,
        ("tetrahedron", True): 10,
        ("tetrahedron", False): 71,
        ("hexahedron", True): 12,
        ("hexahedron", False): 72
    }

    # Conversion from UFL cell name and is_linear to vtk reordering lambda function.
    # The lambda function takes as input mesh element, and returns either a permutation, or None if
    # no permutation is required. Replicates part of the implementation in the function get_topology
    # in firedrake/output.py
    _ufl_cellname_to_vtk_permutation: typing.ClassVar[
        dict[tuple[str, bool], typing.Callable[[typing.Any], typing.Optional[list[int]]]]
    ] = {
        ("point", True): lambda ufl_element: None,
        ("point", False): lambda ufl_element: None,
        ("interval", True): lambda ufl_element: None,
        ("interval", False): firedrake.output.paraview_reordering.vtk_lagrange_interval_reorder,
        ("triangle", True): lambda ufl_element: None,
        ("triangle", False): firedrake.output.paraview_reordering.vtk_lagrange_triangle_reorder,
        ("quadrilateral", True): lambda ufl_element: [0, 2, 3, 1],
        ("quadrilateral", False): firedrake.output.paraview_reordering.vtk_lagrange_quad_reorder,
        ("tetrahedron", True): lambda ufl_element: None,
        ("tetrahedron", False): firedrake.output.paraview_reordering.vtk_lagrange_tet_reorder,
        ("hexahedron", True): lambda ufl_element: [0, 2, 3, 1, 4, 6, 7, 5],
        ("hexahedron", False): firedrake.output.paraview_reordering.vtk_lagrange_hex_reorder
    }

    # Conversion from UFL cell name of a mesh and topological dimension of an entity, to the UFL cell name
    # of the entity. Replicates the _sub_entity_celltypes attriute in ufl/cell.py.
    _tdim_cellname_to_dim_cellname: typing.ClassVar[dict[tuple[str, int], str]] = {
        ("point", 0): "point",
        ("interval", 1): "interval",
        ("interval", 0): "point",
        ("triangle", 2): "triangle",
        ("triangle", 1): "interval",
        ("triangle", 0): "point",
        ("quadrilateral", 2): "quadrilateral",
        ("quadrilateral", 1): "interval",
        ("quadrilateral", 0): "point",
        ("tetrahedron", 3): "tetrahedron",
        ("tetrahedron", 2): "triangle",
        ("tetrahedron", 1): "interval",
        ("tetrahedron", 0): "point",
        ("hexahedron", 3): "hexahedron",
        ("hexahedron", 2): "quadrilateral",
        ("hexahedron", 1): "interval",
        ("hexahedron", 0): "point"
    }

    @classmethod
    def convert_mesh(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: typing.Optional[int] = None
    ) -> pyvista.UnstructuredGrid:
        """
        Convert a mesh stored in firedrake.MeshGeometry object.

        Parameters
        ----------
        mesh
            A firedrake mesh to be converted.
        dim
            Convert entities associated to this dimension. If not provided, the topological dimension is used.

        Returns
        -------
        :
            A pyvista unstructured grid representing the mesh.
        """
        tdim = mesh.topological_dimension()
        if dim is None:
            dim = tdim
        assert dim <= tdim

        # Determine connectivity from entites of dimension dim to vertices
        if dim == tdim:
            # The connectivity from cells to vertices is tabulated by firedrake
            connectivity = mesh.coordinates.cell_node_map().values_with_halo.copy()
        elif dim == 0 and tdim >= 2:
            # The connectivity from vertices to vertices is the identity, except for the 1D case
            # in which vertices are facets, and which will be handled in the next elif.
            vertices = mesh.coordinates.dat.data_ro_with_halos
            connectivity = np.arange(vertices.shape[0]).reshape(-1, 1)
        elif dim == tdim - 1:
            # Determine connectivity from facets to vertices: part of this code is inspired by
            # the triplot function in firedrake/plot.py
            topology = mesh.coordinates.function_space().finat_element.cell.get_topology()
            exterior_facet_local_ids = mesh.exterior_facets.local_facet_dat.data_ro_with_halos
            interior_facet_local_ids = mesh.interior_facets.local_facet_dat.data_ro_with_halos[:, :1].reshape(-1)
            facet_local_ids = np.concatenate((exterior_facet_local_ids, interior_facet_local_ids), dtype=np.int32)
            exterior_facet_node_map = mesh.coordinates.exterior_facet_node_map().values_with_halo
            interior_facet_node_map = mesh.coordinates.interior_facet_node_map()
            interior_facet_node_map = interior_facet_node_map.values_with_halo[:, :interior_facet_node_map.arity // 2]
            facet_node_map = np.concatenate((exterior_facet_node_map, interior_facet_node_map), dtype=np.int32)
            mask = np.zeros(facet_node_map.shape, dtype=bool)
            for mask_row, facet_local_id in enumerate(facet_local_ids):
                mask[mask_row, topology[tdim - 1][facet_local_id]] = True
            connectivity = facet_node_map[mask].reshape(-1, len(topology[tdim - 1][0]))
        else:
            raise NotImplementedError(f"The case dim={dim} and tdim={tdim} has not been implemented yet")

        # Permute connectivity information according to vtk ordering. The implementation is inspired by
        # the function get_topology in firedrake/output.py
        mesh_is_linear = firedrake.output.vtk_output.is_linear(mesh.coordinates.function_space())
        tdim_cellname = mesh.ufl_cell().cellname()
        dim_cellname = cls._tdim_cellname_to_dim_cellname[tdim_cellname, dim]
        permutation = cls._ufl_cellname_to_vtk_permutation[dim_cellname, mesh_is_linear](
            mesh.coordinates.function_space().ufl_element())
        if permutation is not None:
            connectivity = connectivity[:, permutation]

        # Convert the firedrake mesh to a pyvista unstructured grid
        connectivity = np.insert(connectivity, 0, values=connectivity.shape[1], axis=1)
        pyvista_types = np.full(connectivity.shape[0], cls._ufl_cellname_to_vtk_celltype[dim_cellname, mesh_is_linear])
        vertices = mesh.coordinates.dat.data_ro_with_halos
        (vertices, _) = extract_part(vertices, "vertices", "real")
        if tdim < 2:
            assert len(vertices.shape) == 1
            vertices = np.insert(vertices[:, np.newaxis], 1, values=0.0, axis=1)
        if tdim < 3:
            vertices = np.insert(vertices, 2, values=0.0, axis=1)
        return pyvista.UnstructuredGrid(connectivity.reshape(-1), pyvista_types, vertices)

    @classmethod
    def convert_mesh_sets(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: int, name: str
    ) -> pyvista.UnstructuredGrid:
        """
        Convert cell sets or face sets of a given firedrake mesh.

        Parameters
        ----------
        mesh
            A firedrake mesh from which to extract mesh entities and their values.
        dim
            Extract entities associated to this dimension.
        name
            Name to be assigned to the field containing the mesh entities values.

        Returns
        -------
        :
            A pyvista unstructured grid representing the mesh sets.
        """
        tdim = mesh.topological_dimension()
        assert dim <= tdim

        # Convert the firedrake mesh to a pyvista unstructured grid, extracting entities of dimension dim
        pyvista_grid = cls.convert_mesh(mesh, dim)

        # Extract markers, assigning NaN to missing entities
        if dim == tdim:
            cells = mesh.coordinates.cell_node_map().values_with_halo
            assert cells.shape[0] == pyvista_grid.n_cells
            cell_markers = np.full(pyvista_grid.n_cells, np.nan)
            unique_cell_markers = mesh.topology_dm.getLabelIdIS(
                firedrake.cython.dmcommon.CELL_SETS_LABEL).indices.tolist()
            if len(unique_cell_markers) > 0:
                assert np.invert(np.isnan(unique_cell_markers)).all(), (
                    "NaN is used as a placeholder for unmarked cells")
                for cm in unique_cell_markers:
                    cell_markers[mesh.cell_subset(cm).indices] = cm
        elif dim == tdim - 1:
            facet_sizes = {
                facet_set_name: facet_set.measure_set(facet_set_name, "everywhere").total_size
                for (facet_set, facet_set_name) in zip(
                    (mesh.exterior_facets, mesh.interior_facets), ("exterior_facet", "interior_facet")
                )
            }
            assert sum(facet_sizes.values()) == pyvista_grid.n_cells
            facet_markers = np.full(pyvista_grid.n_cells, np.nan)
            unique_facet_markers = mesh.topology_dm.getLabelIdIS(
                firedrake.cython.dmcommon.FACE_SETS_LABEL).indices.tolist()
            if len(unique_facet_markers) > 0:
                assert np.invert(np.isnan(unique_facet_markers)).all(), (
                    "NaN is used as a placeholder for unmarked facets")
                for (facet_set, facet_set_name, offset) in zip(
                    (mesh.exterior_facets, mesh.interior_facets), ("exterior_facet", "interior_facet"),
                    (0, facet_sizes["exterior_facet"])
                ):
                    for fm in unique_facet_markers:
                        facet_indices_fm = offset + facet_set.measure_set(facet_set_name, fm).indices
                        facet_markers[facet_indices_fm] = fm
        else:
            raise NotImplementedError(f"The case dim={dim} and tdim={tdim} has not been implemented yet")

        # Attach mesh tags to a cell field of the pyvista unstructured grid
        if dim == tdim:
            pyvista_grid.cell_data[name] = cell_markers
        elif dim == tdim - 1:
            pyvista_grid.cell_data[name] = facet_markers
        pyvista_grid.set_active_scalars(name)
        return pyvista_grid

    @classmethod
    def convert_field(  # type: ignore[no-any-unimported]
        cls, field: typing.Union[
            firedrake.Function, tuple[ufl.core.expr.Expr, ufl.FunctionSpace]
        ], name: str, part: str = "real"
    ) -> pyvista.UnstructuredGrid:
        """
        Convert a field stored in a firedrake Function, or a pair of UFL Expression and firedrake FunctionSpace.

        Parameters
        ----------
        field
            Expression to be converted.
            If the expression is provided as a firedrake Function, such function will be converted.
            If the expression is provided as a tuple containing UFL expression and a firedrake FunctionSpace,
            the UFL expression will first be interpolated on the function space and then converted.
        name
            Name of the quantity stored in the field.
        part
            Part of the field (real or imag) to be converted. By default, the real part is converted.
            The argument is ignored when converting a real field.

        Returns
        -------
        :
            A pyvista unstructured grid representing the field.
        """
        # Interpolate the field if it is provided as an UFL expression
        if isinstance(field, tuple):
            expression, function_space = field
            interpolated_field = firedrake.Function(function_space)
            interpolated_field.interpolate(expression)
        else:
            interpolated_field = field
            function_space = field.function_space()

        # Get a firedrake mesh which has the same degree as the field. Note that the mesh attached to the function
        # space cannot be directly used, since it may be of a lower degree than the one associated to the field
        # (e.g. when the field is P^2, but the mesh is only P^1)
        assert function_space.rank in (0, 1)
        if function_space.rank == 0:
            field_vector_function_space = firedrake.VectorFunctionSpace(
                function_space.mesh(), function_space.ufl_element())
        else:
            field_vector_function_space = function_space
        mesh_coordinates = firedrake.Function(field_vector_function_space)
        mesh_coordinates.interpolate(function_space.mesh().coordinates)
        mesh = firedrake.Mesh(mesh_coordinates)

        # Convert the firedrake mesh to a pyvista unstructured grid
        pyvista_grid = cls.convert_mesh(mesh, mesh.topological_dimension())

        # Attach the field to the pyvista unstructured grid
        values = interpolated_field.dat.data_ro_with_halos
        (values, name) = extract_part(values, name, part)
        if function_space.rank == 1 and values.shape[1] == 2:
            values = np.insert(values, values.shape[1], 0.0, axis=1)
        pyvista_grid.point_data[name] = values
        if function_space.rank == 0:
            pyvista_grid.set_active_scalars(name)
        else:
            pyvista_grid.set_active_vectors(name)
        return pyvista_grid
