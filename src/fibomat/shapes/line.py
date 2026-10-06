"""Provides the :class:`Line` class.

Example::

    from fibomat.shapes import Line

    line = Line((0, 0), (3, 4))
    line.length  # 5.0
    line.start, line.end, line.center
    line.distance_to_point((0, 5))  # 3.0
    line.intersection(Line((0, 4), (3, 0)))  # Vector(x=1.5, y=2.0)
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.linalg.helpers import GeomLine
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.shape import Shape


__all__ = ['Line']


class Line(Shape, ArcSplineCompatible):
    """1-dim line (a line segment from `start` to `end`)."""

    def __init__(self, start: VectorLike, end: VectorLike, description: t.Optional[str] = None):
        """
        Args:
            start (VectorLike): start point of line
            end (VectorLike): end point of line
            description (str, optional): description

        Raises:
            VectorValueError: Raised if start or end are no vectors.
            ValueError: Raised if start or end are not finite.
        """
        super().__init__(description)

        start = Vector(start)
        end = Vector(end)

        # (the finiteness is checked by ArcSpline)
        self._line: ArcSpline = ArcSpline([(start.x, start.y, 0.0), (end.x, end.y, 0.0)], is_closed=False)

    def to_arc_spline(self) -> ArcSpline:
        return ArcSpline(self._line, description=self.description)

    @property
    def start(self) -> Vector:
        """Start point.

        Access:
            get

        Returns:
            Vector
        """
        return self._line.start

    @property
    def end(self) -> Vector:
        """End point.

        Access:
            get

        Returns:
            Vector
        """
        return self._line.end

    @property
    def length(self) -> float:
        """length of line.

        Access:
            get

        Returns:
            float
        """
        return self._line.length

    @property
    def boundary_length(self) -> float:
        """Length of the line. Same as :attr:`Line.length`.

        Access:
            get

        Returns:
            float
        """
        return self.length

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(start={self.start!r}, end={self.end!r})'

    @property
    def is_closed(self) -> bool:
        return False

    @property
    def bounding_box(self) -> BoundingBox:
        return self._line.bounding_box

    @property
    def center(self) -> Vector:
        """Midpoint of the line.

        Access:
            get

        Returns:
            Vector
        """
        return self._line.center

    def distance_to_point(self, p: VectorLike) -> float:  # pylint: disable=invalid-name
        """Returns the shortest distance between the line segment and the point p.

        Args:
            p (VectorLike): point

        Returns:
            float
        """
        p = Vector(p)
        start = self.start
        direction = self.end - start

        length_squared = direction.dot(direction)
        if length_squared == 0.:
            return (p - start).mag

        # parameter of the orthogonal projection of p onto the (infinite) line, clipped to the segment
        param = min(max((p - start).dot(direction) / length_squared, 0.), 1.)

        return (p - (start + direction * param)).mag

    def intersection(self, other: Line) -> t.Optional[Vector]:
        """Returns the intersection point of the two line segments.

        Args:
            other (Line): other line

        Returns:
            Optional[Vector]: intersection point or None if the segments do not intersect or are parallel.
        """
        if not isinstance(other, Line):
            raise TypeError('other must be a Line.')

        if self.length == 0. or other.length == 0.:
            return None

        line_1 = GeomLine(self.end - self.start, self.start)
        line_2 = GeomLine(other.end - other.start, other.start)

        if line_1.parallel_to(line_2):
            return None

        param_1 = line_1.intersect_at_param(line_2)
        param_2 = line_2.intersect_at_param(line_1)

        tol = 1e-12
        if not (-tol <= param_1 <= 1. + tol and -tol <= param_2 <= 1. + tol):
            return None

        return Vector(line_1(param_1))

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._line._impl_translate(trans_vec)  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        self._line._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        self._line._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        self._line._impl_mirror(mirror_axis)  # pylint: disable=protected-access
