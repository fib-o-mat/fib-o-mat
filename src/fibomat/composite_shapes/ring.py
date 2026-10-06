"""Provides the :class:`Ring` class."""
# pylint: disable=invalid-name
from __future__ import annotations

import numpy as np

from fibomat.shapes.shape import Shape
from fibomat.linalg import Vector, VectorLike, BoundingBox
from fibomat.shapes.arc_spline import ArcSplineCompatible
from fibomat.shapes.circle import Circle
from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline


class Ring(Shape, ArcSplineCompatible):
    def __init__(self, r_outer: float, thickness: float, description: str = None):
        super().__init__(description)

        self._center = Vector(0, 0)
        self._r_outer = r_outer
        self._thickness = thickness

        self._theta = 0

    def __repr__(self):
        return "Ring"

    @property
    def r_outer(self):
        return self._r_outer

    @property
    def r_inner(self):
        return self._r_outer - self._thickness

    @property
    def thickness(self):
        return self._thickness

    @property
    def center(self) -> Vector:
        return self._center

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(
            self._center - (self._r_outer, self._r_outer),
            self._center + (self._r_outer, self._r_outer),
        )

    @property
    def is_closed(self) -> bool:
        return True

    def to_arc_spline(self):
        if np.isclose(self._r_outer - self._thickness, 0):
            return Circle(r=self._r_outer, center=self._center).to_arc_spline()
        else:
            return HollowArcSpline(
                boundary=Circle(r=self._r_outer, center=self._center).to_arc_spline(),
                holes=[
                    Circle(
                        r=self._r_outer - self._thickness, center=self._center
                    ).to_arc_spline()
                ],
            )

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._center += Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        self._center = self._center.rotated(theta)
        self._theta += theta

    def _impl_scale(self, fac: float) -> None:
        self._center *= float(fac)
        self._r_outer *= float(fac)
        self._thickness *= float(fac)

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        self._center = self._center.mirrored(mirror_axis)
