"""Provides the :class:`Line` class."""

from __future__ import annotations

from typing import Optional

from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes import shape
from fibomat.shapes.arc_spline import ArcSpline, ArcSplineCompatible


class Line(shape.Shape, ArcSplineCompatible):
    """1-dim line."""

    def __init__(
        self, start: VectorLike, end: VectorLike, description: Optional[str] = None
    ):
        """
        Args:
            start (VectorLike): start point of line
            end (VectorLike): end point of line
            description (str, optional): description
        """
        super().__init__(description)

        start = Vector(start)
        end = Vector(end)

        self._line: ArcSpline = ArcSpline([(*start, 0.0), (*end, 0.0)], is_closed=False)

    def to_arc_spline(self) -> ArcSpline:
        return self._line.clone_with_new_description(self.description)

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
        return self.length

    # def clone(self) -> Line:
    #     return self.__class__(self._line.clone())
    #
    # __copy__ = clone

    def __repr__(self) -> str:
        return "{}(start={!r},end={!r}".format(
            self.__class__.__name__, self.start, self.end
        )

    @property
    def is_closed(self) -> bool:
        return False

    @property
    def bounding_box(self) -> BoundingBox:
        return self._line.bounding_box

    @property
    def center(self) -> Vector:
        return self._line.center

    def distance_to_point(self, p: Vector):
        """Returns the shortest distance between self and point p.

        Args:
            p (Vector): point

        Returns float
        """
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

        n = (self.end - self.start).normalized()
        a = self.start

        print((p - a), (p - a).dot(n) * n)

        return ((p - a) - (p - a).dot(n) * n).mag

    def intersection(self, other: Line):
        from fibomat.curve_tools.intersections import curve_intersections

        return curve_intersections(self._line, other._line)["intersections"][0]["pos"]

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._line._impl_translate(trans_vec)  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        self._line._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        self._line._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        self._line._impl_mirror(mirror_axis)  # pylint: disable=protected-access
