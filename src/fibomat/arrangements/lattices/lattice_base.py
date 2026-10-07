"""Shared code of :class:`~fibomat.arrangements.Lattice` and :class:`~fibomat.arrangements.DimLattice`."""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.arrangements.utils import check_lattice_vectors
from fibomat.linalg import Transformable


__all__ = ['LatticeMixin', 'place_elements', 'is_callback']


ElementCallback = t.Callable[[int, t.Tuple[int, int], t.Any, int], t.Any]
"""``callback(i, uv, xy, n)``: `i` is the index of the lattice point, `uv` the indices of the point, `xy` its
coordinates and `n` the total number of lattice points. It returns the element for the point or None (no element)."""


def is_callback(element: t.Any) -> bool:
    """True if `element` is a callback which creates the elements (and not a shape, ...)."""
    return callable(element) and not isinstance(element, Transformable)


def place_elements(
    points_uv: np.ndarray,
    points_xy: np.ndarray,
    element: t.Any,
    to_vector: t.Callable[[np.ndarray], t.Any],
    to_callback_xy: t.Callable[[np.ndarray], t.Any],
    check_element: t.Callable[[t.Any], None],
    keep: t.Optional[t.Callable[[t.Any], bool]] = None,
) -> t.Tuple[t.List[t.Any], np.ndarray]:
    """Create the elements of the lattice points and move each one to its point.

    Args:
        points_uv (np.ndarray): indices of the lattice points, shape (n, 2)
        points_xy (np.ndarray): coordinates of the lattice points, shape (n, 2)
        element (Transformable, Callable): element or callback (see :data:`ElementCallback`)
        to_vector (Callable): converts coordinates (2,) to the vector (with or without unit) the elements are moved to
        to_callback_xy (Callable): converts coordinates (2,) to the value which is passed to the callback
        check_element (Callable): raises a TypeError if an element is not valid
        keep (Callable, optional): called with each placed element; elements for which it returns False are dropped

    Returns:
        Tuple[List, np.ndarray]: the elements and the indices of their lattice points (shape (m, 2))

    Raises:
        TypeError: Raised if an element is not valid.
    """
    n_points = len(points_uv)
    use_callback = is_callback(element)

    if not use_callback:
        check_element(element)

    elements: t.List[t.Any] = []
    kept: t.List[int] = []

    for i, (uv, xy) in enumerate(zip(points_uv, points_xy)):
        if use_callback:
            created = element(i, (int(uv[0]), int(uv[1])), to_callback_xy(xy), n_points)
            if created is None:
                continue
            check_element(created)
        else:
            created = element

        placed = created.translated_to(to_vector(xy))

        if keep is not None and not keep(placed):
            continue

        elements.append(placed)
        kept.append(i)

    return elements, points_uv[kept]


class LatticeMixin:
    """State and behaviour which is shared by the lattice classes: the lattice vectors and the indices of the points.

    It must come before the group class in the list of the base classes (it extends the transformations).
    """

    _elements: t.Tuple[t.Any, ...]  # (of the group)

    def _init_lattice(self, points_uv: t.Any, u: t.Any, v: t.Any) -> None:
        """Store the lattice data (the group must be initialized first).

        Args:
            points_uv (array-like): indices (i_u, i_v) of the lattice point of each element, shape (n, 2)
            u (Vector): first lattice vector
            v (Vector): second lattice vector

        Raises:
            ValueError: Raised if the indices have the wrong shape or are negative or the number of the points differs
                from the number of elements, or if the vectors are collinear.
        """
        points_uv = np.array(points_uv, dtype=int)
        if points_uv.ndim != 2 or points_uv.shape[1:] != (2,):
            raise ValueError('points_uv must have the shape (n, 2).')
        if len(points_uv) != len(self._elements):
            raise ValueError('There must be one lattice point for every element.')
        if np.any(points_uv < 0):
            raise ValueError('The indices of the lattice points must not be negative.')
        if len({tuple(point) for point in points_uv}) != len(points_uv):
            raise ValueError('The lattice points must be different.')

        check_lattice_vectors(np.asarray(u.vector if hasattr(u, 'vector') else u),
                              np.asarray(v.vector if hasattr(v, 'vector') else v))

        self._points_uv = points_uv
        self._u = u
        self._v = v

    @property
    def u(self) -> t.Any:
        """First lattice vector.

        Access:
            get
        """
        return self._u

    @property
    def v(self) -> t.Any:
        """Second lattice vector.

        Access:
            get
        """
        return self._v

    @property
    def n_points(self) -> int:
        """Number of lattice points which have an element.

        Access:
            get
        """
        return len(self._elements)

    @property
    def points_uv(self) -> np.ndarray:
        """Indices (i_u, i_v) of the lattice point of each element (a copy), shape (n, 2). The elements are ordered
        like the points: the element ``lattice.elements[i]`` is at the lattice point ``lattice.points_uv[i]``.

        Access:
            get
        """
        return self._points_uv.copy()

    @property
    def elements_by_uv(self) -> np.ndarray:
        """The elements in an array of objects, shape (n_v, n_u): ``elements_by_uv[i_v, i_u]`` is the element at the
        lattice point (i_u, i_v), or None if there is no element at this point.

        Access:
            get
        """
        array = np.empty((int(self._points_uv[:, 1].max()) + 1, int(self._points_uv[:, 0].max()) + 1), dtype=object)
        for element, (i_u, i_v) in zip(self._elements, self._points_uv):
            array[i_v, i_u] = element
        return array

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n_points={self.n_points})'

    # (the lattice vectors are transformed with the lattice, except for translations)

    def _impl_rotate(self, theta: float) -> None:
        super()._impl_rotate(theta)  # type: ignore[misc]
        self._u = self._u.rotated(theta)
        self._v = self._v.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        super()._impl_scale(fac)  # type: ignore[misc]
        self._u = self._u * fac
        self._v = self._v * fac

    def _impl_mirror(self, mirror_axis: t.Any) -> None:
        super()._impl_mirror(mirror_axis)  # type: ignore[misc]
        self._u = self._u.mirrored(mirror_axis)
        self._v = self._v.mirrored(mirror_axis)
