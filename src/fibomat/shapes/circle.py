"""Provides the :class:`Circle` class.

Example::

    from fibomat.shapes import Circle

    circle = Circle(r=2, center=(1, 1))
    circle.area, circle.boundary_length, circle.bounding_box
    Circle.from_points((1, 0), (0, 1), (-1, 0))  # circumscribed circle
"""
# pylint: disable=invalid-name
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.shape import Shape


__all__ = ['Circle']


class Circle(Shape, ArcSplineCompatible):
    """2-dim circle."""

    def __init__(self, r: float, center: t.Optional[VectorLike] = None, description: t.Optional[str] = None):
        """

        Args:
            r (float): radius
            center (:class:`~fibomat.linalg.vectortypes.VectorLike`, optional): center of circle, default to (0, 0)
            description (str, optional): description

        Raises:
            ValueError: Raised if r <= 0 or r or the center are not finite.
        """
        super().__init__(description)

        self._center = Vector(center) if center is not None else Vector()
        self._r = float(r)

        if not np.isfinite(self._r) or self._r <= 0.:
            raise ValueError('radius must be positive and finite.')
        if not np.all(np.isfinite(np.asarray(self._center))):
            raise ValueError('center must be finite.')

    @classmethod
    def from_points(
        cls, p1: VectorLike, p2: VectorLike, p3: VectorLike, description: t.Optional[str] = None
    ) -> Circle:
        """Create circumscribed circle (from three points).

        Args:
            p1 (VectorLike): first points
            p2 (VectorLike): second points
            p3 (VectorLike): third points
            description (str, optional): description

        Returns:
            Circle

        Raises:
            ValueError: Raised if two points are equal or the points are collinear.
        """
        # https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2
        a = Vector(p1)  # A
        b = Vector(p2) - a  # B
        c = Vector(p3) - a  # C

        det = 2 * b.cross(c)
        b2 = b.dot(b)
        c2 = c.dot(c)

        if b2 == 0. or c2 == 0. or (Vector(p3) - Vector(p2)).mag == 0.:
            raise ValueError('The circle through points is only defined for three different points.')
        if abs(det) <= 1e-12 * max(b2, c2):
            raise ValueError('The circle through points is only defined for points which are not collinear.')

        u = Vector((c.y * b2 - b.y * c2) / det, (b.x * c2 - c.x * b2) / det)

        return cls(u.mag, u + a, description)

    def __repr__(self) -> str:
        return '{}(r={!r}, center={!r})'.format(self.__class__.__name__, self.r, self.center)

    def to_arc_spline(self) -> ArcSpline:
        return ArcSpline(
            [
                (*(self._center + Vector(self._r, 0.0)), 1.0),
                (*(self._center - Vector(self._r, 0.0)), 1.0),
            ],
            True,
            self.description,
        )

    @property
    def r(self) -> float:  # pylint: disable=invalid-name
        """Radius.

        Access:
            get

        Returns:
            float
        """
        return self._r

    @property
    def center(self) -> Vector:
        return self._center

    @property
    def area(self) -> float:
        return float(np.pi * self._r ** 2)

    @property
    def boundary_length(self) -> float:
        """Circumference of the circle.

        Access:
            get

        Returns:
            float
        """
        return float(2 * np.pi * self._r)

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self._center - (self._r, self._r), self._center + (self._r, self._r))

    @property
    def is_closed(self) -> bool:
        return True

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._center = self._center + Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        self._center = self._center.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)
        self._center = self._center * fac
        # a negative factor is a point reflection, which maps a circle to a circle with radius abs(fac) * r
        self._r *= abs(fac)

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        self._center = self._center.mirrored(mirror_axis)
