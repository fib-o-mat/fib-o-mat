"""Calculation of the points of lattices (without units).

All functions return the integer indices ``(i_u, i_v)`` of the lattice points and their coordinates, in the order
in which the points are numbered. The indices start at 0 (the point (0, 0) is the "upper left" point of lattices with
``u = (du, 0)`` and ``v = (0, -dv)``).
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.arrangements.lattices.fast_axis import FastAxis
from fibomat.arrangements.utils import check_lattice_vectors


__all__ = ['count_points', 'grid_points', 'boundary_points']


def _order(indices: np.ndarray, fast_axis: FastAxis) -> np.ndarray:
    """Sort lattice indices (n, 2) so that `fast_axis` changes fastest."""
    if fast_axis is FastAxis.U:
        order = np.lexsort((indices[:, 0], indices[:, 1]))
    else:
        order = np.lexsort((indices[:, 1], indices[:, 0]))
    return order


def count_points(size: float, pitch: float) -> int:
    """Number of lattice points with the distance `pitch` which fit into an interval of the length `size` (points on the
    ends of the interval are included).

    Args:
        size (float): length of the interval
        pitch (float): distance between the points

    Returns:
        int: number of points (at least 1)

    Raises:
        ValueError: Raised if size is negative or pitch is not positive (or not finite).
    """
    if not np.isfinite(size) or size < 0.:
        raise ValueError(f'The size must not be negative, got {size}.')
    if not np.isfinite(pitch) or pitch <= 0.:
        raise ValueError(f'The pitch must be positive, got {pitch}.')

    # (a small tolerance, so that e.g. 0.3 / 0.1 gives 3 intervals)
    return int(np.floor(size / pitch * (1. + 1e-9) + 1e-9)) + 1


def grid_points(
    n_u: int, n_v: int, du: float, dv: float, center: t.Tuple[float, float], fast_axis: FastAxis
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Points of a rectangular grid with the lattice vectors ``u = (du, 0)`` and ``v = (0, -dv)``, centered at `center`.

    The point with the indices (0, 0) is the upper left point, `u` points to the right and `v` down.

    Args:
        n_u (int): number of points along u (at least 1)
        n_v (int): number of points along v (at least 1)
        du (float): distance of the points along u (positive)
        dv (float): distance of the points along v (positive)
        center (Tuple[float, float]): center of the grid
        fast_axis (FastAxis): axis along which the points are numbered first

    Returns:
        Tuple[np.ndarray, np.ndarray]: indices (n, 2) and coordinates (n, 2) of the points

    Raises:
        ValueError: Raised if a number of points is less than 1 or a distance is not positive and finite.
    """
    if int(n_u) != n_u or int(n_v) != n_v or n_u < 1 or n_v < 1:
        raise ValueError(f'The numbers of lattice points must be integers which are at least 1, got {n_u}, {n_v}.')
    for name, value in (('du', du), ('dv', dv)):
        if not np.isfinite(value) or value <= 0.:
            raise ValueError(f'{name} must be positive and finite, got {value}.')

    i_u, i_v = np.meshgrid(np.arange(int(n_u)), np.arange(int(n_v)))
    indices = np.column_stack((i_u.ravel(), i_v.ravel()))
    indices = indices[_order(indices, fast_axis)]

    xy = np.column_stack((
        (indices[:, 0] - (n_u - 1) / 2.) * du + center[0],
        -(indices[:, 1] - (n_v - 1) / 2.) * dv + center[1],
    ))
    return indices, xy


def boundary_points(
    boundary: t.Any, u: t.Tuple[float, float], v: t.Tuple[float, float], origin: t.Tuple[float, float],
    fast_axis: FastAxis
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Points of the lattice with the lattice vectors `u` and `v` and a point at `origin`, which lie in a shape.

    Points on the boundary of the shape are not included (only points strictly inside). The indices of the points are
    shifted, so that the smallest index along each lattice vector is 0.

    Args:
        boundary (ArcSpline, HollowArcSpline): closed shape (it needs the methods `contains` and `bounding_box`)
        u (Tuple[float, float]): first lattice vector
        v (Tuple[float, float]): second lattice vector
        origin (Tuple[float, float]): position of a lattice point
        fast_axis (FastAxis): axis along which the points are numbered first

    Returns:
        Tuple[np.ndarray, np.ndarray]: indices (n, 2) and coordinates (n, 2) of the points (empty arrays if the shape
        does not contain any lattice point)

    Raises:
        ValueError: Raised if the lattice vectors are null vectors or collinear.
    """
    check_lattice_vectors(u, v)

    basis = np.array([u, v], dtype=float).T  # columns: u, v
    origin_array = np.asarray(origin, dtype=float)

    bbox = boundary.bounding_box
    corners = np.array([np.asarray(corner, dtype=float) for corner in bbox.corners])

    # the bounding box in the coordinates of the lattice (a parallelogram): the range of the indices
    coefficients = np.linalg.solve(basis, (corners - origin_array).T).T
    i_min = np.floor(coefficients.min(axis=0)).astype(int)
    i_max = np.ceil(coefficients.max(axis=0)).astype(int)

    i_u, i_v = np.meshgrid(np.arange(i_min[0], i_max[0] + 1), np.arange(i_min[1], i_max[1] + 1))
    candidates = np.column_stack((i_u.ravel(), i_v.ravel()))
    xy = origin_array + candidates @ basis.T

    # cheap test first
    lower_left, upper_right = np.asarray(bbox.lower_left, dtype=float), np.asarray(bbox.upper_right, dtype=float)
    in_bbox = np.all((xy > lower_left) & (xy < upper_right), axis=1)
    candidates, xy = candidates[in_bbox], xy[in_bbox]

    inside = np.array([boundary.contains(point) for point in xy], dtype=bool)
    candidates, xy = candidates[inside], xy[inside]

    if not len(candidates):
        return np.empty((0, 2), dtype=int), np.empty((0, 2))

    indices = candidates - candidates.min(axis=0)
    order = _order(indices, fast_axis)
    return indices[order], xy[order]
