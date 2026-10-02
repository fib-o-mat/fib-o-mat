"""Provides the :class:`Circle` class."""

# pylint: disable=invalid-name
from __future__ import annotations
from typing import Optional

import numpy as np

from fibomat.shapes.shape import Shape
from fibomat.linalg import Vector, VectorLike, BoundingBox
from fibomat.shapes.arc_spline import ArcSpline, ArcSplineCompatible


class Circle(Shape, ArcSplineCompatible):
    """2-dim circle."""

    def __init__(
        self,
        r: float,
        center: Optional[VectorLike] = None,
        description: Optional[str] = None,
    ):
        """

        Args:
            r (float): radius
            center (:class:`~fibomat.linalg.vectortypes.VectorLike`, optional): center of circle, default to (0, 0)
            description (str, optional): description

        Raises:
            ValueError: Raised if r <= 0.
        """
        super().__init__(description)

        self._center = Vector(center) if center is not None else Vector()
        self._r = float(r)
        if self._r <= 0:
            raise ValueError("radius <= 0.")

    @classmethod
    def from_points(
        cls,
        p1: VectorLike,
        p2: VectorLike,
        p3: VectorLike,
        description: Optional[str] = None,
    ):
        """Create circumscribed circle (from three points).

        Args:
            p1 (VectorLike): first points
            p2 (VectorLike): second points
            p3 (VectorLike): third points
            description (str, optional): description

        Returns:
            Circle
        """
        # https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2
        a = Vector(p1)  # A
        b = Vector(p2) - a  # B
        c = Vector(p3) - a  # C

        inv_d = 1 / (2 * (b.x * c.y - b.y * c.x))

        b2 = b.x * b.x + b.y * b.y
        c2 = c.x * c.x + c.y * c.y

        u = Vector(inv_d * (c.y * b2 - b.y * c2), inv_d * (b.x * c2 - c.x * b2))
        r = np.sqrt(u.x * u.x + u.y * u.y)

        return cls(r, u + a, description)

    def __repr__(self) -> str:
        return "{}(r={!r}, center={!r})".format(
            self.__class__.__name__, self.r, self.center
        )

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
    def r(self) -> float:
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
        return np.pi * self._r**2

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(
            self._center - (self._r, self._r), self._center + (self._r, self._r)
        )

    @property
    def is_closed(self) -> bool:
        return True

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._center += Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        self._center = self._center.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        self._center *= float(fac)
        self._r *= float(fac)

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        self._center = self._center.mirrored(mirror_axis)
