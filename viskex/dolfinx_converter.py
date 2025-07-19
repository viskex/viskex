# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex converter interfacing dolfinx."""

import typing

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import packaging.version
import pyvista
import ufl

from viskex.pyvista_converter import PyvistaConverter
from viskex.utils import extract_part

assert dolfinx.mesh.CellType.point not in dolfinx.plot._first_order_vtk
dolfinx.plot._first_order_vtk[dolfinx.mesh.CellType.point] = 1


class DolfinxConverter(PyvistaConverter[  # type: ignore[no-any-unimported]
    dolfinx.mesh.Mesh,
    typing.Union[dolfinx.fem.Function, tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]]
]):
    """viskex converter interfacing dolfinx."""

    @classmethod
    def convert_mesh(cls, mesh: dolfinx.mesh.Mesh, dim: typing.Optional[int] = None) -> pyvista.UnstructuredGrid:
        """
        Convert a mesh stored in dolfinx.mesh.Mesh object.

        Parameters
        ----------
        mesh
            A dolfinx mesh to be converted.
        dim
            Convert entities associated to this dimension. If not provided, the topological dimension is used.

        Returns
        -------
        :
            A pyvista unstructured grid representing the mesh.
        """
        if dim is None:
            dim = mesh.topology.dim
        assert dim <= mesh.topology.dim

        # Convert the dolfinx mesh to a pyvista unstructured grid
        mesh.topology.create_connectivity(dim, dim)
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
        cell_entities = np.arange(num_cells, dtype=np.int32)
        pyvista_cells, cell_types, coordinates = dolfinx.plot.vtk_mesh(mesh, dim, cell_entities)
        return pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)

    @classmethod
    def convert_mesh_tags(
        cls, mesh: dolfinx.mesh.Mesh, mesh_tags: dolfinx.mesh.MeshTags, name: str
    ) -> pyvista.UnstructuredGrid:
        """
        Convert dolfinx.mesh.MeshTags.

        Parameters
        ----------
        mesh
            The dolfinx mesh associated to the provided mesh tags.
        mesh_tags
            A dolfinx mesh tags from which to extract mesh entities and their values.
        name
            Name to be assigned to the field containing the mesh entities values.

        Returns
        -------
        :
            A pyvista unstructured grid representing the mesh tags.
        """
        # Convert the dolfinx mesh to a pyvista unstructured grid, extracting entities of dimension mesh_tags.dim
        pyvista_grid = cls.convert_mesh(mesh, mesh_tags.dim)

        # Assign NaN values to missing indices
        assert np.invert(np.isnan(mesh_tags.values)).all(), "NaN is used as a placeholder for non-provided entities"
        values = np.full(pyvista_grid.n_cells, np.nan)
        for (index, value) in zip(mesh_tags.indices, mesh_tags.values):
            values[index] = value

        # Attach mesh tags to a cell field of the pyvista unstructured grid
        pyvista_grid.cell_data[name] = values
        pyvista_grid.set_active_scalars(name)
        return pyvista_grid

    @classmethod
    def convert_field(  # type: ignore[no-any-unimported]
        cls, field: typing.Union[
            dolfinx.fem.Function, tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]
        ], name: str, part: str = "real"
    ) -> pyvista.UnstructuredGrid:
        """
        Convert a field stored in a dolfinx function, or a pair of UFL expression and dolfinx function space.

        Parameters
        ----------
        field
            Expression to be converted.
            If the expression is provided as a dolfinx function, such function will be converted.
            If the expression is provided as a tuple containing UFL expression and a dolfinx function space,
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
            interpolated_field = dolfinx.fem.Function(function_space)
            if packaging.version.Version(dolfinx.__version__) >= packaging.version.Version("0.10.0"):
                interpolation_points = function_space.element.interpolation_points
            else:
                interpolation_points = function_space.element.interpolation_points()  # type: ignore[operator, unused-ignore]
            interpolated_field.interpolate(dolfinx.fem.Expression(expression, interpolation_points))
        else:
            interpolated_field = field
            function_space = field.function_space

        # Get a pyvista unstructured grid based on the DOFs coordinates
        mesh = function_space.mesh
        dim = mesh.topology.dim
        mesh.topology.create_connectivity(dim, dim)
        num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
        cell_entities = np.arange(num_cells, dtype=np.int32)
        pyvista_cells, cell_types, coordinates = dolfinx.plot.vtk_mesh(
            function_space, cell_entities)  # type: ignore[arg-type]
        pyvista_grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)

        # Attach the field to the pyvista unstructured grid
        with interpolated_field.x.petsc_vec.localForm() as values:
            (values, name) = extract_part(values.array.copy(), name, part)
            if function_space.dofmap.index_map_bs > 1:
                values = values.reshape(-1, function_space.dofmap.index_map_bs)
                if values.shape[1] == 2:
                    values = np.insert(values, values.shape[1], 0.0, axis=1)
            pyvista_grid.point_data[name] = values
            if function_space.dofmap.index_map_bs == 1:
                pyvista_grid.set_active_scalars(name)
            else:
                pyvista_grid.set_active_vectors(name)
            return pyvista_grid
