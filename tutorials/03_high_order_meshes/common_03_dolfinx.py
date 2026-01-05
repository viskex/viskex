# Copyright (C) 2023-2026 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Common functions used across dolfinx tutorials for high order meshes."""


import dolfinx.io
import dolfinx.mesh
import gmsh
import mpi4py.MPI
import numpy as np
import packaging.version

from common_03_none import get_key, key_to_int  # isort: skip


def create_reference_interval(comm: mpi4py.MPI.Comm, order: int, num_segments: int) -> tuple[
    dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags
]:
    """
    Create a mesh of the reference interval [-1, 1] using gmsh.

    This function will also tag sub entitites from all co-dimensions. See the examples in the key_to_int
    function in the common_03_none module for the numerical value of each tag.

    Parameters
    ----------
    comm
        MPI communicator to create the mesh on.
    order
        Order of the mesh. If order > 1 than an high order mesh is created, e.g. quadratric
        if order is 2.
    num_segments
        Number of subdivisions of the interval. Must be an even number.

    Returns
    -------
    mesh
        The created mesh.
    cell_tags
        The mesh tags for cells.
    facet_tags
        The mesh tags for facets.
    """
    if num_segments % 2 != 0:
        raise ValueError("num_segments must be even")

    # Add new model
    gmsh.model.add(f"interval_{order}")

    # Add points
    p_left = gmsh.model.geo.addPoint(-1.0, 0, 0)
    p_mid = gmsh.model.geo.addPoint(0.0, 0, 0)
    p_right = gmsh.model.geo.addPoint(1.0, 0, 0)

    # Add lines
    l1 = gmsh.model.geo.addLine(p_left, p_mid)
    l2 = gmsh.model.geo.addLine(p_mid, p_right)

    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # Assign physical groups for cells
    gmsh.model.addPhysicalGroup(1, [l1], 1)
    gmsh.model.addPhysicalGroup(1, [l2], 2)

    # Assign physical groups for facets
    gmsh.model.addPhysicalGroup(0, [p_left], 1)
    gmsh.model.addPhysicalGroup(0, [p_mid], 0)
    gmsh.model.addPhysicalGroup(0, [p_right], 2)

    # Set a coarse mesh size
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, num_segments // 2 + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, num_segments // 2 + 1)

    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # Generate the mesh
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    # Convert to dolfinx format
    if packaging.version.Version(dolfinx.__version__) >= packaging.version.Version("0.10.0"):
        mesh, cell_tags, facet_tags, _ridge_tags, _peak_tags, _physical_groups = dolfinx.io.gmsh.model_to_mesh(  # type: ignore[attr-defined, misc, unused-ignore]
            gmsh.model, comm, rank=0, gdim=1)
    else:
        mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(  # type: ignore[attr-defined, misc, unused-ignore]
            gmsh.model, comm, rank=0, gdim=1)

    assert cell_tags is not None
    assert facet_tags is not None

    # Clean up gmsh model
    gmsh.clear()

    return mesh, cell_tags, facet_tags

def create_unit_disk(comm: mpi4py.MPI.Comm, order: int, num_segments: int) -> tuple[
    dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags,
    dolfinx.mesh.MeshTags | None
]:
    """
    Create a mesh of the unit disk using gmsh.

    This function will also tag sub entitites from all co-dimensions. See the examples in the key_to_int
    function in the common_03_none module for the numerical value of each tag.

    Parameters
    ----------
    comm
        MPI communicator to create the mesh on.
    order
        Order of the mesh. If order > 1 than an high order mesh is created, e.g. quadratric
        if order is 2.
    num_segments
        Number of subdivisions of the unit interval.

    Returns
    -------
    mesh
        The created mesh.
    cell_tags
        The mesh tags for cells.
    facet_tags
        The mesh tags for facets.
    vertex_tags
        The mesh tags for vertices. Only returned with dolfinx >= 0.10; older versions return None.
    """
    mesh, cell_tags, facet_tags, vertex_tags, _ = create_unit_ball(comm, order, num_segments, 2)
    return mesh, cell_tags, facet_tags, vertex_tags


def create_unit_ball(comm: mpi4py.MPI.Comm, order: int, num_segments: int, dimension: int = 3) -> tuple[
    dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags,
    dolfinx.mesh.MeshTags | None, dolfinx.mesh.MeshTags | None
]:
    """
    Create a mesh of the unit ball of the provided dimension (default: 3) using gmsh.

    This function will also tag sub entitites from all co-dimensions. See the examples in the key_to_int
    function in the common_03_none module for the numerical value of each tag.

    Parameters
    ----------
    comm
        MPI communicator to create the mesh on.
    order
        Order of the mesh. If order > 1 than an high order mesh is created, e.g. quadratric
        if order is 2.
    num_segments
        Number of subdivisions of the unit interval.
    dimension
        The ambient geometric dimension of the domain (e.g., 2 or 3). Defaults to 3.

    Returns
    -------
    mesh
        The created mesh.
    cell_tags
        The mesh tags for cells.
    facet_tags
        The mesh tags for facets.
    ridge_tags
        The mesh tags for ridges. Only returned with dolfinx >= 0.10; older versions return None.
    peak_tags
        The mesh tags for peaks. Only returned with dolfinx >= 0.10; older versions return None.
    """
    # Add new model
    gmsh.model.add(f"ball_{order}_{dimension}")

    # Add ball of radius r=1
    if dimension not in (2, 3):
        raise ValueError("Dimension must be 2 or 3.")
    if dimension == 2:
        ball = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)
    else:
        ball = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)

    # Split the ball into four (in 2d) or eight (in 3d) slices, one for each quadrant of the plane (in 2d)
    # or each octant of the space (in 3d)
    fragmented_ball = fragment_by_coordinate_directions([(dimension, ball)], num_segments)

    # Assign physical groups
    for dim, key_to_entities in fragmented_ball.items():
        if packaging.version.Version(dolfinx.__version__) < packaging.version.Version("0.10.0"):
            # Old dolfinx does not support ridge and peak tags, so don't bother marking them
            if dim <= dimension - 2:
                continue
        for key, entity_ids in key_to_entities.items():
            physical_group_int = key_to_int(key)
            gmsh.model.addPhysicalGroup(dim, entity_ids, physical_group_int)

    # Set a coarse mesh size
    h = 1.0 / num_segments
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    # Generate the mesh
    gmsh.model.mesh.generate(dimension)
    gmsh.model.mesh.setOrder(order)

    # Convert to dolfinx format
    if packaging.version.Version(dolfinx.__version__) >= packaging.version.Version("0.10.0"):
        mesh, cell_tags, facet_tags, ridge_tags, peak_tags, _physical_groups = dolfinx.io.gmsh.model_to_mesh(  # type: ignore[attr-defined, misc, unused-ignore]
            gmsh.model, comm, rank=0, gdim=dimension)
    else:
        mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(  # type: ignore[attr-defined, misc, unused-ignore]
            gmsh.model, comm, rank=0, gdim=dimension)
        ridge_tags = None
        peak_tags = None

    assert cell_tags is not None
    assert facet_tags is not None

    # Clean up gmsh model
    gmsh.clear()

    return mesh, cell_tags, facet_tags, ridge_tags, peak_tags


def fragment_by_coordinate_directions(
    domains_to_fragment: list[tuple[int, int]], num_segments: int
) -> dict[int, dict[tuple[tuple[str, str], ...], list[int]]]:
    """
    Insert axis-aligned points, edges, or surfaces depending on dimension, and fragment the input domains.

    Parameters
    ----------
    domains_to_fragment
        List of CAD entities (dimension, entity_id) to be fragmented by the edges.
        All CAD entities in the list are assumed to be of the same dimension.
    num_segments
        Number of segments per side on each axis.

    Returns
    -------
    :
        Nested dictionary classifying connected entities by their barycenter position keys,
        grouped by dimension.
    """
    # Input checks
    if len(domains_to_fragment) == 0:
        raise ValueError("domains_to_fragment cannot be empty.")

    dimension = domains_to_fragment[0][0]
    if dimension not in (2, 3):
        raise ValueError("Domain dimension must be 2 or 3.")
    if any(entity[0] != dimension for entity in domains_to_fragment):
        raise ValueError("Domain dimension is inconsistent.")

    points_cache: dict[tuple[float, float, float], int] = {}
    if dimension == 2:
        # Create lines for each axis and each side, resulting in 4 combinations
        created_edges = []

        for axis in ("x", "y"):
            for side in ("negative", "positive"):
                lines = _create_subdivided_line_in_2d(axis, side, num_segments, points_cache)
                created_edges.extend(lines)

        # Fragment original domain based on created edges
        fragmented_entities, _ = gmsh.model.occ.fragment(
            domains_to_fragment,
            [(1, edge) for edge in created_edges]
        )
        gmsh.model.occ.synchronize()
    else:  # dimension == 3
        # Create planes for each axis pair and each side pair, resulting in 8 combinations
        created_surfaces: list[int] = []
        lines_cache: dict[tuple[int, int], int] = {}
        for axes in (("x", "y"), ("x", "z"), ("y", "z")):
            for sides in (
                ("negative", "negative"), ("negative", "positive"), ("positive", "negative"), ("positive", "positive")
            ):
                planes = _create_plane_in_3d_with_subdivided_boundary(
                    axes, sides, num_segments, points_cache, lines_cache)
                created_surfaces.extend(planes)

        # Intersect the planes with the sphere
        intersected_surfaces: list[int] = []
        for surface in created_surfaces:
            intersected_surface, _ = gmsh.model.occ.intersect(
                [(2, surface)], domains_to_fragment,
                removeTool=False
            )
            intersected_surfaces.extend(surface for _, surface in intersected_surface)
        gmsh.model.occ.synchronize()

        # Fragment original domain based on created surfaces
        fragmented_entities, _ = gmsh.model.occ.fragment(
            domains_to_fragment, [(2, surface) for surface in intersected_surfaces]
        )
        gmsh.model.occ.synchronize()

    # Tolerance for entity classification
    tol = 1.0 / num_segments * 1e-3
    return _classify_connected_entities_by_key(fragmented_entities, dimension, tol)


def _create_subdivided_line_in_2d(
    axis: str, side: str, num_segments: int,
    points_cache: dict[tuple[float, float, float], int]
) -> list[int]:
    """
    Create a subdivided unit interval along a specified axis in 2D, limited to the given side.

    Parameters
    ----------
    axis
        "x" or "y", axis along which the line extends.
    side
        "negative" or "positive", where negative corresponds to the interval [-1, 0] and positive to [0,1].
    num_segments
        Number of segments to create.
    points_cache
        Dictionary mapping coordinates to gmsh points to avoid duplicates.

    Returns
    -------
    :
        List of gmsh lines created.

    Examples
    --------
    Create 3 segments on the positive x axis:

        lines = _create_subdivided_line_in_2d("x", "positive", 3, points_cache)

    Create 2 segments on the negative y axis:

        lines = _create_subdivided_line_in_2d("y", "negative", 2, points_cache)
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y' in 2D.")
    if side not in ("negative", "positive"):
        raise ValueError("side must be 'negative' or 'positive'.")

    coords = _side_range(side, num_segments)

    points: list[int] = []
    for c in coords:
        coord = (c, 0.0, 0.0) if axis == "x" else (0.0, c, 0.0)
        points.append(_get_or_add_point(coord, points_cache))

    lines: list[int] = []
    for i in range(len(points) - 1):
        line = gmsh.model.occ.addLine(points[i], points[i + 1])
        lines.append(line)

    return lines


def _create_plane_in_3d_with_subdivided_boundary(
    axes: tuple[str, str], sides: tuple[str, str], num_segments: int,
    points_cache: dict[tuple[float, float, float], int], lines_cache: dict[tuple[int, int], int]
) -> list[int]:
    """
    Create a unit square in 3D on specified axes and sides, subdividing its edges into num_segments.

    Parameters
    ----------
    axes
        Tuple of two distinct axes from ("x", "y", "z") defining the plane.
    sides
        Tuple of two sides ("negative" or "positive") corresponding to each axis.
    num_segments
        Number of segments per axis.
    points_cache
        Dictionary mapping coordinates to gmsh points to avoid duplicates.
    lines_cache
        Dictionary mapping point pairs to gmsh lines to avoid duplicates.

    Returns
    -------
    :
        List of gmsh surfaces created.

    Examples
    --------
    Create a subdivided plane in the (x, y) plane on negative x and positive y sides with 3 segments per axis:

        planes = _create_plane_in_3d_with_subdivided_boundary(
            ("x", "y"), ("negative", "positive"), 3, points_cache, lines_cache
        )

    Create a subdivided plane in the (y, z) plane on positive y and positive z sides with 2 segments per axis:

        planes = _create_plane_in_3d_with_subdivided_boundary(
            ("y", "z"), ("positive", "positive"), 2, points_cache, lines_cache
        )
    """
    all_axes = {"x", "y", "z"}
    if len(axes) != 2 or any(a not in all_axes for a in axes) or axes[0] == axes[1]:
        raise ValueError(f"axes must be two distinct elements from {all_axes}")
    if any(side not in ("negative", "positive") for side in sides):
        raise ValueError("sides must be 'negative' or 'positive' for each axis.")

    axis_idx = {"x": 0, "y": 1, "z": 2}

    coords_axis0 = _side_range(sides[0], num_segments)
    coords_axis1 = _side_range(sides[1], num_segments)

    # Add points on the bottom edge
    bottom_points = []
    for c1 in coords_axis1:
        coord = [0.0, 0.0, 0.0]
        coord[axis_idx[axes[0]]] = coords_axis0[0]
        coord[axis_idx[axes[1]]] = c1
        bottom_points.append(_get_or_add_point(tuple(coord), points_cache))  # type: ignore[arg-type]

    # Add points on the right edge (except for the corner point already included in the bottom edge)
    right_points = []
    for c0 in coords_axis0[1:]:
        coord = [0.0, 0.0, 0.0]
        coord[axis_idx[axes[0]]] = c0
        coord[axis_idx[axes[1]]] = coords_axis1[-1]
        right_points.append(_get_or_add_point(tuple(coord), points_cache))  # type: ignore[arg-type]

    # Add points on the top edge (except for the corner point already included in the right edge)
    top_points = []
    for c1 in reversed(coords_axis1[:-1]):
        coord = [0.0, 0.0, 0.0]
        coord[axis_idx[axes[0]]] = coords_axis0[-1]
        coord[axis_idx[axes[1]]] = c1
        top_points.append(_get_or_add_point(tuple(coord), points_cache))  # type: ignore[arg-type]

    # Add points on the left edge (except for the endpoints already included at the top and bottom edges)
    left_points = []
    for c0 in reversed(coords_axis0[1:-1]):
        coord = [0.0, 0.0, 0.0]
        coord[axis_idx[axes[0]]] = c0
        coord[axis_idx[axes[1]]] = coords_axis1[0]
        left_points.append(_get_or_add_point(tuple(coord), points_cache))  # type: ignore[arg-type]

    # Combine points around the boundary in counter clockwise order
    boundary_points = bottom_points + right_points + top_points + left_points

    # Create lines connecting consecutive points
    lines = []
    num_boundary_points = len(boundary_points)
    for i in range(num_boundary_points):
        p_start = boundary_points[i]
        p_end = boundary_points[(i + 1) % num_boundary_points]
        line = _get_or_add_line(p_start, p_end, lines_cache)
        lines.append(line)

    # Generate surface
    curve_loop = gmsh.model.occ.addCurveLoop(lines)
    surface = gmsh.model.occ.addPlaneSurface([curve_loop])

    return [surface]


def _side_range(side: str, num_segments: int) -> list[float]:
    """
    Return equispaced points in [-1, 0] if side is 'negative', else in [0, 1].

    Parameters
    ----------
    side
        "negative" or "positive".
    num_segments
        Number of segments to be created.

    Returns
    -------
    :
        1D array of coordinate points.
    """
    if side == "negative":
        return np.linspace(-1.0, 0.0, num_segments + 1).tolist()  # type: ignore[no-any-return]
    else:
        return np.linspace(0.0, 1.0, num_segments + 1).tolist()  # type: ignore[no-any-return]


def _get_or_add_point(coord: tuple[float, float, float], points_cache: dict[tuple[float, float, float], int]) -> int:
    """
    Add a point to the gmsh model or reuse an existing one at the same coordinates.

    Parameters
    ----------
    coord
        Coordinates (x, y, z) of the point.
    points_cache
        Cache mapping coordinates to points.

    Returns
    -------
    int
        gmsh point.
    """
    try:
        return points_cache[coord]
    except KeyError:
        point = gmsh.model.occ.addPoint(*coord)
        points_cache[coord] = point
        return point  # type: ignore[no-any-return]


def _get_or_add_line(p1: int, p2: int, lines_cache: dict[tuple[int, int], int]) -> int:
    """
    Add a line between two points to the gmsh model or reuse an existing one.

    Parameters
    ----------
    p1
        First point.
    p2
        Second point.
    lines_cache
        Cache mapping point pairs to line.

    Returns
    -------
    int
        gmsh line.
    """
    try:
        return lines_cache[p1, p2]
    except KeyError:
        line = gmsh.model.occ.addLine(p1, p2)
        lines_cache[p1, p2] = line
        lines_cache[p2, p1] = - line
        return line  # type: ignore[no-any-return]


def _classify_connected_entities_by_key(
    entities: list[tuple[int, int]], dimension: int, tol: float
) -> dict[int, dict[tuple[tuple[str, str], ...], list[int]]]:
    """
    Classify connected entities based on their barycenter position keys.

    Parameters
    ----------
    entities
        A list of (dimension, id) tuples representing the starting set of entities.
    dimension
        The ambient geometric dimension of the domain (e.g., 2 or 3).
    tol
        Tolerance for coordinate classification.

    Returns
    -------
    :
        A nested dictionary: {dim: {key: list of entity ids}}
    """
    connected_entities = _get_connected_entities(entities)
    barycenters = _get_entity_barycenters(connected_entities, dimension)

    output: dict[int, dict[tuple[tuple[str, str], ...], list[int]]] = {}
    for dim in connected_entities.keys():
        output[dim] = {}
        for entity in connected_entities[dim]:
            coords = barycenters[dim][entity]
            key = get_key(coords, tol)
            if key not in output[dim]:
                output[dim][key] = []
            output[dim][key].append(entity)
    return output


def _get_connected_entities(entities: list[tuple[int, int]]) -> dict[int, list[int]]:
    """
    Return all entities connected to the given ones through boundaries, using recursive traversal.

    The connected entities are grouped by their dimension.

    Parameters
    ----------
    entities
        A list of (dimension, id) tuples.

    Returns
    -------
    :
        A dictionary where each key is a dimension, and the value is a list of entities of that dimension.
    """
    connected = set(entities)
    queue = list(entities)

    while queue:
        dim, entity = queue.pop()
        for b in gmsh.model.getBoundary([(dim, entity)], oriented=False, combined=False, recursive=False):
            if b not in connected:
                connected.add(b)
                queue.append(b)

    grouped_entities: dict[int, list[int]] = {}
    for dim, entity in connected:
        if dim not in grouped_entities:
            grouped_entities[dim] = []
        grouped_entities[dim].append(entity)
    return grouped_entities


def _get_entity_barycenters(
    grouped_entities: dict[int, list[int]],
    dimension: int
) -> dict[int, dict[int, tuple[float, ...]]]:
    """
    Compute the barycenter of each entity in grouped_entities.

    For 0D entities (points), returns the exact coordinates.
    For higher dimensions, uses the center of mass from the OCC kernel.

    Parameters
    ----------
    grouped_entities
        Dictionary mapping topological dimension to a list of entity IDs.
    dimension
        The ambient geometric dimension of the domain (e.g., 2 or 3). Used to slice the coordinates.

    Returns
    -------
    :
        A nested dictionary of the form {dim: {id: barycenter}}, where barycenter is
        a tuple of floats of length `dimension`.

    Examples
    --------
    >>> grouped = {0: [p1, p2], 1: [l1], 2: [s1]}
    >>> _get_entity_barycenters(grouped, 2)
    {
        0: {p1: (x1, y1), p2: (x2, y2)},
        1: {l1: (x3, y3)},
        2: {s1: (x4, y4)}
    }
    """
    barycenters: dict[int, dict[int, tuple[float, ...]]] = {}
    for dim, ids in grouped_entities.items():
        barycenters_dim = {}
        for entity_id in ids:
            if dim == 0:
                coords = gmsh.model.getValue(0, entity_id, [])
            else:
                coords = gmsh.model.occ.getCenterOfMass(dim, entity_id)
            barycenters_dim[entity_id] = tuple(coords[:dimension])
        barycenters[dim] = barycenters_dim
    return barycenters
