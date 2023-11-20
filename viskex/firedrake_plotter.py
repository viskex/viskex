# Copyright (C) 2023 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""viskex plotter interfacing firedrake."""

import typing

import firedrake
import numpy as np
import panel.pane.vtk.vtk
import plotly.graph_objects as go
import pyvista.trame.jupyter
import ufl

from viskex.base_plotter import BasePlotter
from viskex.plotly_plotter import PlotlyPlotter
from viskex.pyvista_plotter import PyvistaPlotter
from viskex.utils import extract_part


class FiredrakePlotter(BasePlotter[  # type: ignore[no-any-unimported]
    firedrake.MeshGeometry,
    typing.Union[firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]],
    typing.Union[firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]],
    typing.Union[go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget]
]):
    """viskex plotter interfacing firedrake."""

    _ufl_cellname_to_vtk_celltype = {
        "point": 1,
        "interval": 3,
        "triangle": 5,
        "quadrilateral": 9,
        "tetrahedron": 10,
        "hexahedron": 12
    }
    _tdim_cellname_to_dim_cellname = {
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
    def plot_mesh(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: typing.Optional[int] = None
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a mesh stored in firedrake.MeshGeometry object.

        Parameters
        ----------
        mesh
            A firedrake mesh to be plotted.
        dim
            Plot entities associated to this dimension. If not provided, the topological dimension is used.

        Returns
        -------
        :
            A widget representing a plot of the mesh.
        """
        tdim = mesh.topological_dimension()
        if dim is None:
            dim = tdim
        assert dim <= tdim
        if tdim == 1:
            plotly_grid = cls._firedrake_mesh_to_plotly_grid(mesh, dim)
            return PlotlyPlotter.plot_mesh(plotly_grid, dim)
        else:
            pyvista_grid = cls._firedrake_mesh_to_pyvista_grid(mesh, dim)
            return PyvistaPlotter.plot_mesh((pyvista_grid, tdim))

    @classmethod
    def plot_mesh_entities(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: int, name: str, indices: np.typing.NDArray[np.int32],
        values: typing.Optional[np.typing.NDArray[np.int32]] = None
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot `dim`-dimensional mesh entities of a given firedrake mesh.

        Parameters
        ----------
        mesh
            A firedrake mesh to be plotted.
        dim
            Extract entities associated to this dimension.
        name
            Name to be assigned to the field containing the mesh entities values.
        indices
            Array containing the IDs of the entities to be plotted.
        values
            Array containing the value to be associated to each of the entities in `indices`.
            Values are assumed to be greater than zero, because every entity not part of `indices`
            will be automatically assigned value equal to zero.
            If not provided, every entity part of `indices` will be marked with value one.

        Returns
        -------
        :
            A widget representing a plot of the mesh entities.
        """
        tdim = mesh.topological_dimension()
        assert dim <= tdim
        if values is None:
            values = np.ones_like(indices)
        if tdim == 1:
            plotly_grid = cls._firedrake_mesh_to_plotly_grid(mesh, dim)
            return PlotlyPlotter.plot_mesh_entities(plotly_grid, dim, name, indices, values)
        else:
            pyvista_grid = cls._firedrake_mesh_to_pyvista_grid(mesh, dim)
            return PyvistaPlotter.plot_mesh_entities((pyvista_grid, tdim), dim, name, indices, values)

    @classmethod
    def plot_mesh_sets(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: int, name: str
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a cell set or a face sets of a given firedrake mesh.

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
            A widget representing a plot of the mesh entities.
        """
        tdim = mesh.topological_dimension()
        assert dim in (tdim, tdim - 1)
        if dim == tdim:
            cells = mesh.coordinates.cell_node_map().values
            unique_cell_markers = tuple(mesh.topology_dm.getLabelIdIS(
                firedrake.cython.dmcommon.CELL_SETS_LABEL).indices.tolist())
            cell_indices = np.arange(cells.shape[0], dtype=np.int32)
            cell_markers = np.full(cells.shape[0], 0, dtype=np.int32)
            for cm in unique_cell_markers:
                cell_markers[mesh.cell_subset(cm).indices] = cm
            return cls.plot_mesh_entities(mesh, dim, name, cell_indices, cell_markers)
        elif dim == tdim - 1:
            unique_face_markers = tuple(mesh.topology_dm.getLabelIdIS(
                firedrake.cython.dmcommon.FACE_SETS_LABEL).indices.tolist())
            facet_sizes = {
                facet_set_name: facet_set.measure_set(facet_set_name, "everywhere").size
                for (facet_set, facet_set_name) in zip(
                    (mesh.exterior_facets, mesh.interior_facets), ("exterior_facet", "interior_facet")
                )
            }
            all_facet_size = sum(facet_sizes.values())
            all_facet_indices = np.arange(all_facet_size, dtype=np.int32)
            all_facet_markers = np.full(all_facet_size, 0, dtype=np.int32)
            for (facet_set, facet_set_name, offset) in zip(
                (mesh.exterior_facets, mesh.interior_facets), ("exterior_facet", "interior_facet"),
                (0, facet_sizes["exterior_facet"])
            ):
                for fm in unique_face_markers:
                    facet_indices_fm = offset + facet_set.measure_set(facet_set_name, fm).indices
                    all_facet_markers[facet_indices_fm] = fm
            return cls.plot_mesh_entities(mesh, dim, name, all_facet_indices, all_facet_markers)
        else:
            raise RuntimeError("Invalid mesh set dimension")

    @classmethod
    def plot_scalar_field(  # type: ignore[no-any-unimported]
        cls, scalar_field: typing.Union[
            firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]
        ], name: str, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a scalar field stored in a firedrake Function, or a pair of UFL Expression and firedrake FunctionSpace.

        Parameters
        ----------
        scalar_field
            Expression to be plotted, which contains a scalar field.
            If the expression is provided as a firedrake Function, such function will be plotted.
            If the expression is provided as a tuple containing UFL expression and a firedrake FunctionSpace,
            the UFL expression will first be interpolated on the function space and then plotted.
            Notice that the field will be interpolated to a P1 space before plotting.
        name
            Name of the quantity stored in the scalar field.
        warp_factor
            This argument is ignored for a field on 1D or 3D meshes.
            For a 2D mesh, if provided then the factor is used to produce a warped representation
            the field; if not provided then the scalar field will be plotted on the mesh.
        part
            Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.

        Returns
        -------
        :
            A widget representing a plot of the scalar field.
        """
        scalar_field = cls._interpolate_to_P1_space(
            scalar_field, lambda mesh: firedrake.FunctionSpace(mesh, "CG", 1))
        values = scalar_field.vector().array()
        (values, name) = extract_part(values, name, part)
        mesh = scalar_field.function_space().mesh()
        tdim = mesh.topological_dimension()
        if tdim == 1:
            coordinates = cls._firedrake_mesh_to_plotly_grid(mesh, tdim)
            return PlotlyPlotter.plot_scalar_field((coordinates, values), name, warp_factor, part)
        else:
            pyvista_grid = cls._firedrake_mesh_to_pyvista_grid(mesh, tdim)
            pyvista_grid.point_data[name] = values
            pyvista_grid.set_active_scalars(name)
            return PyvistaPlotter.plot_scalar_field((pyvista_grid, tdim), name, warp_factor, part)

    @classmethod
    def plot_vector_field(  # type: ignore[no-any-unimported]
        cls, vector_field: typing.Union[
            firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]
        ], name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0, part: str = "real"
    ) -> typing.Union[
        go.Figure, panel.pane.vtk.vtk.VTKRenderWindowSynchronized, pyvista.trame.jupyter.Widget
    ]:
        """
        Plot a vector field stored in a firedrake Function, or a pair of UFL Expression and firedrake FunctionSpace.

        Parameters
        ----------
        vector_field
            Expression to be plotted, which contains a vector field.
            If the expression is provided as a firedrake Function, such function will be plotted.
            If the expression is provided as a tuple containing UFL expression and a firedrake FunctionSpace,
            the UFL expression will first be interpolated on the function space and then plotted.
            Notice that the field will be interpolated to a P1 space before plotting.
        name
            Name of the quantity stored in the vector field.
        glyph_factor
            If provided, the vector field is represented using a gylph, scaled by this factor.
        warp_factor
            If provided then the factor is used to produce a warped representation of the field.
            If not provided then the magnitude of the vector field will be plotted on the mesh.
            The argument cannot be used if `glyph_factor` is also provided.
        part
            Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
            The argument is ignored when plotting a real field.

        Returns
        -------
        :
            A widget representing a plot of the vector field.
        """
        vector_field = cls._interpolate_to_P1_space(
            vector_field, lambda mesh: firedrake.VectorFunctionSpace(mesh, "CG", 1))
        mesh = vector_field.function_space().mesh()
        tdim = mesh.topological_dimension()
        assert tdim > 1, "Cannot call plot_vector_field for 1D meshes"
        values = vector_field.vector().array().reshape(-1, tdim)
        (values, name) = extract_part(values, name, part)
        if tdim == 2:
            values = np.insert(values, values.shape[1], 0.0, axis=1)
        pyvista_grid = cls._firedrake_mesh_to_pyvista_grid(mesh, tdim)
        pyvista_grid.point_data[name] = values
        pyvista_grid.set_active_vectors(name)
        pyvista_grid_edges = cls._firedrake_mesh_to_pyvista_grid(mesh, 1)
        return PyvistaPlotter.plot_vector_field(
            (pyvista_grid, pyvista_grid_edges, tdim), name, glyph_factor, warp_factor, part)

    @classmethod
    def _firedrake_mesh_to_plotly_grid(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: int
    ) -> np.typing.NDArray[np.float64]:
        """Convert a 1D firedrake.MeshGeometry to an array of coordinates."""
        vertices = mesh.coordinates.dat.data_ro
        assert len(vertices.shape) == 1
        assert np.all(vertices[1:] >= vertices[:-1])
        if dim == 1:
            cells = cls._determine_connectivity(mesh, dim)
            expected_cells = np.repeat(np.arange(vertices.shape[0], dtype=np.int32), 2)
            expected_cells = np.delete(np.delete(expected_cells, 0), -1)
            expected_cells = expected_cells.reshape(-1, 2)
            expected_cells[:, [0, 1]] = expected_cells[:, [1, 0]]
            assert np.array_equal(cells, expected_cells)
            return vertices  # type: ignore[no-any-return]
        elif dim == 0:
            vertices_reordering = cls._determine_connectivity(mesh, dim).reshape(-1)
            return vertices[vertices_reordering]  # type: ignore[no-any-return]
        else:
            raise RuntimeError("Invali dimension")

    @classmethod
    def _firedrake_mesh_to_pyvista_grid(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: int
    ) -> pyvista.UnstructuredGrid:
        """Convert a 2D or 3D firedrake.MeshGeometry to a pyvista.UnstructuredGrid."""
        tdim = mesh.topological_dimension()
        connectivity = cls._determine_connectivity(mesh, dim)
        tdim_cellname = mesh.ufl_cell().cellname()
        dim_cellname = cls._tdim_cellname_to_dim_cellname[tdim_cellname, dim]
        cls._reorder_connectivity(connectivity, dim_cellname)
        connectivity = np.insert(connectivity, 0, values=connectivity.shape[1], axis=1)
        pyvista_types = np.full(connectivity.shape[0], cls._ufl_cellname_to_vtk_celltype[dim_cellname])
        vertices = mesh.coordinates.dat.data_ro
        if tdim == 2:
            vertices = np.insert(vertices, 2, values=0.0, axis=1)
        return pyvista.UnstructuredGrid(connectivity.reshape(-1), pyvista_types, vertices)

    @classmethod
    def _determine_connectivity(  # type: ignore[no-any-unimported]
        cls, mesh: firedrake.MeshGeometry, dim: int
    ) -> np.typing.NDArray[np.int32]:
        """Determine connectivity of a given dimension."""
        tdim = mesh.topological_dimension()
        if dim == tdim:
            connectivity = mesh.coordinates.cell_node_map().values.copy()
        elif dim == 0 and tdim > 1:
            vertices = mesh.coordinates.dat.data_ro
            connectivity = np.arange(vertices.shape[0]).reshape(-1, 1)
        else:
            topology = mesh.coordinates.function_space().finat_element.cell.get_topology()
            exterior_facet_local_ids = mesh.exterior_facets.local_facet_dat.data_ro
            interior_facet_local_ids = mesh.interior_facets.local_facet_dat.data_ro[:, :1].reshape(-1)
            facet_local_ids = np.concatenate((exterior_facet_local_ids, interior_facet_local_ids), dtype=np.int32)
            exterior_facet_node_map = mesh.coordinates.exterior_facet_node_map().values
            interior_facet_node_map = mesh.coordinates.interior_facet_node_map()
            interior_facet_node_map = interior_facet_node_map.values[:, :interior_facet_node_map.arity // 2]
            facet_node_map = np.concatenate((exterior_facet_node_map, interior_facet_node_map), dtype=np.int32)
            mask = np.zeros(facet_node_map.shape, dtype=bool)
            for mask_row, facet_local_id in enumerate(facet_local_ids):
                mask[mask_row, topology[tdim - 1][facet_local_id]] = True
            connectivity = facet_node_map[mask].reshape(-1, len(topology[tdim - 1][0]))
            if dim == tdim - 1:
                pass
            elif dim == tdim - 2:
                tdim_cellname = mesh.ufl_cell().cellname()
                cls._reorder_connectivity(connectivity, cls._tdim_cellname_to_dim_cellname[tdim_cellname, tdim - 1])
                repeated_connectivity = np.roll(np.repeat(connectivity, repeats=2, axis=1), shift=-1, axis=1)
                connectivity = repeated_connectivity.reshape(-1, 2)
            else:
                raise RuntimeError("Invalid values of dim and tdim")
        return connectivity  # type: ignore[no-any-return]

    @staticmethod
    def _reorder_connectivity(connectivity: np.typing.NDArray[np.int32], cellname: str) -> None:
        """Reorder in-place a connectivity array according to vtk ordering."""
        if cellname in ("point", "interval", "triangle", "tetrahedron"):
            pass
        elif cellname == "quadrilateral":
            connectivity[:, [2, 3]] = connectivity[:, [3, 2]]
        elif cellname == "hexahedron":
            connectivity[:, [2, 3]] = connectivity[:, [3, 2]]
            connectivity[:, [6, 7]] = connectivity[:, [7, 6]]
        else:
            raise RuntimeError("Unsupported cellname")

    @staticmethod
    def _interpolate_to_P1_space(  # type: ignore[no-any-unimported]
        field: typing.Union[
            firedrake.Function, typing.Tuple[ufl.core.expr.Expr, ufl.FunctionSpace]
        ], function_space_generator: typing.Callable[[firedrake.MeshGeometry], ufl.FunctionSpace]
    ) -> firedrake.Function:
        """Interpolate a firedrake Function or UFL Expression to a P1 space."""
        if isinstance(field, tuple):
            expression, function_space = field
            mesh = function_space.mesh()
        else:
            assert isinstance(field, firedrake.Function)
            expression = field
            mesh = field.function_space().mesh()
        return firedrake.interpolate(expression, function_space_generator(mesh))
