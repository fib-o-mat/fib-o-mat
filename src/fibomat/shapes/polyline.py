"""Provides the :class:`Polyline` class.

Example::

    from fibomat.shapes import Polyline

    polyline = Polyline([(0, 0), (1, 0), (1, 1)])
    polyline.points  # array([[0., 0.], [1., 0.], [1., 1.]])
    polyline.length, polyline.bounding_box
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import BoundingBox, DimVector, Vector, VectorLike, VectorValueError
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.shape import Shape


__all__ = ['Polyline']


def _points_to_array(points: t.Any) -> np.ndarray:
    """Convert points to an array of shape (n, 2)."""
    if not isinstance(points, np.ndarray):
        try:
            points = list(points)
        except TypeError as error:
            raise VectorValueError('points must be an iterable of vectors.') from error

        if any(isinstance(point, DimVector) for point in points):
            # np.array would silently drop the unit
            raise VectorValueError('points must not have units.')

    try:
        array = np.array(points, dtype=float)
    except (TypeError, ValueError) as error:
        raise VectorValueError('points must be an iterable of vectors.') from error

    if array.ndim != 2 or array.shape[1] != 2:
        raise VectorValueError(f'points must have shape (n, 2), got {array.shape}.')
    return array


class Polyline(Shape, ArcSplineCompatible):
    """Polyline"""

    def __init__(self, points: t.Iterable[VectorLike], description: t.Optional[str] = None, _closed: bool = False):
        """
        Args:
            points (Iterable[VectorLike], np.ndarray): polyline points (at least two), array of shape (n, 2)
            description (str, optional): description

        Raises:
            VectorValueError: Raised if points are no vectors.
            ValueError: Raised if there are less than two points or points are not finite.
        """
        super().__init__(description)

        array = _points_to_array(points)

        self._polygon = ArcSpline(np.c_[array, np.zeros(len(array))], _closed)

    def to_arc_spline(self) -> ArcSpline:
        return ArcSpline(self._polygon, description=self.description)

    @property
    def points(self) -> np.ndarray:
        """Polyline points.

        Access:
            get

        Returns:
            np.ndarray
        """
        return self._polygon.vertices[:, :2]

    @property
    def n_points(self) -> int:
        """Number of points.

        Access:
            get

        Returns:
            int
        """
        return len(self._polygon)

    @property
    def length(self) -> float:
        """Length of the polyline (including the closing segment if the polyline is closed).

        Access:
            get

        Returns:
            float
        """
        return self._polygon.length

    @property
    def boundary_length(self) -> float:
        """Length of the polyline. Same as :attr:`Polyline.length`.

        Access:
            get

        Returns:
            float
        """
        return self.length

    def __repr__(self) -> str:
        if self.n_points < 5:
            return '{}(points={!r})'.format(self.__class__.__name__, self.points)
        return '{}(points=...)'.format(self.__class__.__name__)

    @property
    def bounding_box(self) -> BoundingBox:
        return self._polygon.bounding_box

    @property
    def is_closed(self) -> bool:
        return self._polygon.is_closed

    @property
    def center(self) -> Vector:
        """Center of the bounding box of the polyline.

        Access:
            get

        Returns:
            Vector
        """
        return self._polygon.center

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._polygon._impl_translate(trans_vec)  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        self._polygon._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        self._polygon._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        self._polygon._impl_mirror(mirror_axis)  # pylint: disable=protected-access
