# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Common functions user for mesh marking."""

import itertools
import math


def key_to_int(key: tuple[tuple[str, str], ...]) -> int:
    """
    Convert a multidimensional key of axis-side descriptors into a unique integer index.

    This function maps a tuple representing positions on coordinate axes to a unique integer.
    Each axis ('x', 'y' in 2D/3D, and 'z' in 3D) is paired with a descriptor indicating
    whether the position is 'negative', 'zero', or 'positive' along that axis.

    The indexing scheme partitions keys by the number of axes labeled 'zero'. Keys with
    more zero-labeled axes receive higher base offsets, accounting for combinations of zero
    axes and possible sign variations on the remaining axes. Within each partition, keys
    are sorted lexicographically by the positions of zeros and the signs of non-zero axes.

    Parameters
    ----------
    key
        A tuple of pairs, each consisting of an axis name ('x', 'y' in 2D/3D, 'z' in 3D)
        and a side descriptor ('negative', 'zero', or 'positive'). The dimension is inferred
        from the length of the input key, which must contain 1 to 3 axes.

    Returns
    -------
    int
        A unique integer index associated with the given key, ranging from 1 up to the
        total number of distinct keys in the specified dimension.

    Examples
    --------
    In 1D (dimension = 1):

        (("x", "zero"))     -> 0
        ------------------------
        (("x", "negative")) -> 1
        (("x", "positive")) -> 2

    In 2D (dimension = 2):

        (("x", "zero"),     ("y", "zero"))     -> 0
        -------------------------------------------
        (("x", "zero"),     ("y", "negative")) -> 1
        (("x", "zero"),     ("y", "positive")) -> 2
        (("x", "negative"), ("y", "zero"))     -> 3
        (("x", "positive"), ("y", "zero"))     -> 4
        -------------------------------------------
        (("x", "negative"), ("y", "negative")) -> 5
        (("x", "negative"), ("y", "positive")) -> 6
        (("x", "positive"), ("y", "negative")) -> 7
        (("x", "positive"), ("y", "positive")) -> 8

    In 3D (dimension = 3):

        (("x", "zero"),     ("y", "zero"),     ("z", "zero"))     -> 0
        ---------------------------------------------------------------
        (("x", "zero"),     ("y", "zero"),     ("z", "negative")) -> 1
        (("x", "zero"),     ("y", "zero"),     ("z", "positive")) -> 2
        (("x", "zero"),     ("y", "negative"), ("z", "zero"))     -> 3
        (("x", "zero"),     ("y", "positive"), ("z", "zero"))     -> 4
        (("x", "negative"), ("y", "zero"),     ("z", "zero"))     -> 5
        (("x", "positive"), ("y", "zero"),     ("z", "zero"))     -> 6
        ---------------------------------------------------------------
        (("x", "zero"),     ("y", "negative"), ("z", "negative")) -> 7
        (("x", "zero"),     ("y", "negative"), ("z", "positive")) -> 8
        (("x", "zero"),     ("y", "positive"), ("z", "negative")) -> 9
        (("x", "zero"),     ("y", "positive"), ("z", "positive")) -> 10
        (("x", "negative"), ("y", "zero"),     ("z", "negative")) -> 11
        (("x", "negative"), ("y", "zero"),     ("z", "positive")) -> 12
        (("x", "positive"), ("y", "zero"),     ("z", "negative")) -> 13
        (("x", "positive"), ("y", "zero"),     ("z", "positive")) -> 14
        (("x", "negative"), ("y", "negative"), ("z", "zero"))     -> 15
        (("x", "negative"), ("y", "positive"), ("z", "zero"))     -> 16
        (("x", "positive"), ("y", "negative"), ("z", "zero"))     -> 17
        (("x", "positive"), ("y", "positive"), ("z", "zero"))     -> 18
        ---------------------------------------------------------------
        (("x", "negative"), ("y", "negative"), ("z", "negative")) -> 19
        (("x", "negative"), ("y", "negative"), ("z", "positive")) -> 20
        (("x", "negative"), ("y", "positive"), ("z", "negative")) -> 21
        (("x", "negative"), ("y", "positive"), ("z", "positive")) -> 22
        (("x", "positive"), ("y", "negative"), ("z", "negative")) -> 23
        (("x", "positive"), ("y", "negative"), ("z", "positive")) -> 24
        (("x", "positive"), ("y", "positive"), ("z", "negative")) -> 25
        (("x", "positive"), ("y", "positive"), ("z", "positive")) -> 26
    """
    dimension = len(key)
    if dimension not in (1, 2, 3):
        raise ValueError("Only dimensions 1, 2, and 3 are supported.")

    if dimension == 1:
        axes = ("x", )
    elif dimension == 2:
        axes = ("x", "y")  # type: ignore[assignment]
    else:
        axes = ("x", "y", "z")  # type: ignore[assignment]
    sides = ("negative", "zero", "positive")
    side_order_map = {"zero": 0, "negative": 1, "positive": 2}

    # Verify key structure and ordering
    for i, (axis, side) in enumerate(key):
        if axes[i] != axis:
            raise ValueError("Key axes must be in order.")
        if side not in sides:
            raise ValueError(f"Invalid side '{side}' in key.")

    zero_count = sum(side == "zero" for _, side in key)

    # Calculate offset for keys with more zeros
    offset = 0
    for z in range(dimension, zero_count, -1):
        offset += math.comb(dimension, z) * (2 ** (dimension - z))

    # All zero positions with current zero_count, lex sorted
    zero_positions_choices = sorted(itertools.combinations(range(dimension), zero_count))

    candidates = []
    for zero_pos in zero_positions_choices:
        non_zero_indices = [i for i in range(dimension) if i not in zero_pos]
        # All sign combos for non-zero positions
        for signs in itertools.product(("negative", "positive"), repeat=len(non_zero_indices)):
            candidate = []
            sign_iter = iter(signs)
            for i in range(dimension):
                if i in zero_pos:
                    candidate.append((axes[i], "zero"))
                else:
                    candidate.append((axes[i], next(sign_iter)))
            candidates.append(tuple(candidate))

    def sort_key(k: tuple[tuple[str, str], ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Key to be used while sorting candidates."""
        zero_pos = tuple(i for i, (_, side) in enumerate(k) if side == "zero")
        side_order = tuple(side_order_map[side] for _, side in k)
        return (zero_pos, side_order)

    candidates.sort(key=sort_key)

    return offset + candidates.index(key)


def get_key(coordinates: tuple[float, ...], tol: float) -> tuple[tuple[str, str], ...]:
    """
    Generate a key describing the sign of each coordinate, classified by axis.

    Each coordinate "x", "y" (or "z") is labeled as "negative", "zero", or "positive" based on the given tolerance.

    Parameters
    ----------
    coordinates
        A tuple of floats representing the coordinates of a point.
    tol
        A small positive float tolerance. Coordinates with absolute value less than `tol` are classified as "zero".

    Returns
    -------
    :
        A tuple of (axis, side) pairs, one for each coordinate.

    Examples
    --------
    >>> _get_key((0.0, -1e-7, 2.0), tol=1e-8)
    (("x", "zero"), ("y", "negative"), ("z", "positive"))

    >>> _get_key((0.5, 0.0), tol=1e-6)
    (("x", "positive"), ("y", "zero"))
    """
    all_axes = ("x", "y", "z")
    key = []
    for i, coord in enumerate(coordinates):
        axis = all_axes[i]
        if coord < -tol:
            side = "negative"
        elif coord > tol:
            side = "positive"
        else:
            side = "zero"
        key.append((axis, side))
    return tuple(key)


if __name__ == "__main__":
    test_cases = {
        # 1D
        (("x", "zero"),): 0,
        (("x", "negative"),): 1,
        (("x", "positive"),): 2,

        # 2D
        (("x", "zero"),     ("y", "zero")):     0,
        (("x", "zero"),     ("y", "negative")): 1,
        (("x", "zero"),     ("y", "positive")): 2,
        (("x", "negative"), ("y", "zero")):     3,
        (("x", "positive"), ("y", "zero")):     4,
        (("x", "negative"), ("y", "negative")): 5,
        (("x", "negative"), ("y", "positive")): 6,
        (("x", "positive"), ("y", "negative")): 7,
        (("x", "positive"), ("y", "positive")): 8,

        # 3D
        (("x", "zero"),     ("y", "zero"),     ("z", "zero")):     0,
        (("x", "zero"),     ("y", "zero"),     ("z", "negative")): 1,
        (("x", "zero"),     ("y", "zero"),     ("z", "positive")): 2,
        (("x", "zero"),     ("y", "negative"), ("z", "zero")):     3,
        (("x", "zero"),     ("y", "positive"), ("z", "zero")):     4,
        (("x", "negative"), ("y", "zero"),     ("z", "zero")):     5,
        (("x", "positive"), ("y", "zero"),     ("z", "zero")):     6,
        (("x", "zero"),     ("y", "negative"), ("z", "negative")): 7,
        (("x", "zero"),     ("y", "negative"), ("z", "positive")): 8,
        (("x", "zero"),     ("y", "positive"), ("z", "negative")): 9,
        (("x", "zero"),     ("y", "positive"), ("z", "positive")): 10,
        (("x", "negative"), ("y", "zero"),     ("z", "negative")): 11,
        (("x", "negative"), ("y", "zero"),     ("z", "positive")): 12,
        (("x", "positive"), ("y", "zero"),     ("z", "negative")): 13,
        (("x", "positive"), ("y", "zero"),     ("z", "positive")): 14,
        (("x", "negative"), ("y", "negative"), ("z", "zero")):     15,
        (("x", "negative"), ("y", "positive"), ("z", "zero")):     16,
        (("x", "positive"), ("y", "negative"), ("z", "zero")):     17,
        (("x", "positive"), ("y", "positive"), ("z", "zero")):     18,
        (("x", "negative"), ("y", "negative"), ("z", "negative")): 19,
        (("x", "negative"), ("y", "negative"), ("z", "positive")): 20,
        (("x", "negative"), ("y", "positive"), ("z", "negative")): 21,
        (("x", "negative"), ("y", "positive"), ("z", "positive")): 22,
        (("x", "positive"), ("y", "negative"), ("z", "negative")): 23,
        (("x", "positive"), ("y", "negative"), ("z", "positive")): 24,
        (("x", "positive"), ("y", "positive"), ("z", "negative")): 25,
        (("x", "positive"), ("y", "positive"), ("z", "positive")): 26,
    }

    failures = 0
    for key, expected in test_cases.items():
        result = key_to_int(key)
        if result != expected:
            print(f"Failed for key {key}, expected {expected}, got {result}")
            failures += 1
    assert failures == 0, "There were some failures"
