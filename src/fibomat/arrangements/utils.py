"""Helpers of the arrangements."""
from __future__ import annotations

import typing as t

import numpy as np


__all__ = ['check_lattice_vectors']


def check_lattice_vectors(u: t.Any, v: t.Any) -> None:
    """Check that two vectors can be the basis vectors of a lattice: they must not be null vectors and must not be
    collinear.

    Args:
        u (VectorLike): first lattice vector (unitless numbers or the coordinates in one unit)
        v (VectorLike): second lattice vector

    Raises:
        ValueError: Raised if a vector is the null vector or the vectors are collinear.
    """
    u_array = np.asarray(u, dtype=float)
    v_array = np.asarray(v, dtype=float)

    length_u = float(np.hypot(*u_array))
    length_v = float(np.hypot(*v_array))
    if length_u == 0. or length_v == 0.:
        raise ValueError('The lattice vectors must not be the null vector.')

    cross = u_array[0] * v_array[1] - u_array[1] * v_array[0]
    if abs(cross) <= 1e-9 * length_u * length_v:
        raise ValueError('The lattice vectors must not be collinear.')
