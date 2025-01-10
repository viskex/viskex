# Copyright (C) 2023-2025 by the viskex authors
#
# This file is part of viskex.
#
# SPDX-License-Identifier: MIT
"""Tests for viskex.utils.extract_part module."""

import numpy as np
import numpy.typing
import pytest

import viskex.utils


@pytest.fixture
def array() -> np.typing.NDArray[viskex.utils.ScalarType]:
    """Create an array with dtype=viskex.utils.ScalarType."""
    array = np.zeros((3, ), dtype=viskex.utils.ScalarType)
    if np.issubdtype(array.dtype, np.complexfloating):
        array[0] = 1.0 + 2.0j
        array[1] = 3.0 + 4.0j
        array[2] = 5.0 + 6.0j
    else:
        array[0] = 1.0
        array[1] = 3.0
        array[2] = 5.0
    return array


@pytest.mark.skipif(
    np.issubdtype(viskex.utils.ScalarType, np.complexfloating),  # type: ignore[arg-type, unused-ignore]
    reason="Needs real numbers")
def test_extract_part_real_array_real_part(array: np.typing.NDArray[viskex.utils.ScalarType]) -> None:
    """Test extraction of the real part from a complex array."""
    (array_part, name_part) = viskex.utils.extract_part(array, "array", "real")
    assert np.issubdtype(array.dtype, np.floating)
    assert np.issubdtype(array_part.dtype, np.floating)
    assert np.allclose(array_part, [1.0, 3.0, 5.0])
    assert name_part == "array"


@pytest.mark.skipif(
    np.issubdtype(viskex.utils.ScalarType, np.complexfloating),  # type: ignore[arg-type, unused-ignore]
    reason="Needs real numbers")
def test_extract_part_real_array_imaginary_part(array: np.typing.NDArray[viskex.utils.ScalarType]) -> None:
    """Test extraction of the imaginary part from a complex array."""
    with pytest.raises(RuntimeError) as excinfo:
        viskex.utils.extract_part(array, "array", "imag")
    runtime_error_text = str(excinfo.value)
    assert runtime_error_text == "Invalid part imag"


@pytest.mark.skipif(not np.issubdtype(viskex.utils.ScalarType, np.complexfloating), reason="Needs complex numbers")
def test_extract_part_complex_array_real_part(array: np.typing.NDArray[viskex.utils.ScalarType]) -> None:
    """Test extraction of the real part from a complex array."""
    (array_part, name_part) = viskex.utils.extract_part(array, "array", "real")
    assert not np.issubdtype(array.dtype, np.floating)
    assert np.issubdtype(array_part.dtype, np.floating)
    assert np.allclose(array_part, [1.0, 3.0, 5.0])
    assert name_part == "real(array)"


@pytest.mark.skipif(not np.issubdtype(viskex.utils.ScalarType, np.complexfloating), reason="Needs complex numbers")
def test_extract_part_complex_array_imaginary_part(array: np.typing.NDArray[viskex.utils.ScalarType]) -> None:
    """Test extraction of the imaginary part from a complex array."""
    (array_part, name_part) = viskex.utils.extract_part(array, "array", "imag")
    assert not np.issubdtype(array.dtype, np.floating)
    assert np.issubdtype(array_part.dtype, np.floating)
    assert np.allclose(array_part, [2.0, 4.0, 6.0])
    assert name_part == "imag(array)"


def test_extract_part_real_array_wrong_part(array: np.typing.NDArray[viskex.utils.ScalarType]) -> None:
    """Test extraction of an incorrect part, which is neither real nor imag."""
    with pytest.raises(RuntimeError) as excinfo:
        viskex.utils.extract_part(array, "array", "wrong_part")
    runtime_error_text = str(excinfo.value)
    assert runtime_error_text == "Invalid part wrong_part"
