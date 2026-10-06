"""Provides the :class:`Ring` class.

Example::

    from fibomat.composite_shapes import Ring

    ring = Ring(r_outer=2, thickness=0.5)  # inner radius 1.5
    ring = Ring.from_radii(r_outer=2, r_inner=1, center=(1, 1))
    ring.area, ring.boundary_length, ring.contains((1.5, 1))
    ring.to_hollow_arc_spline()
"""
# pylint: disable=invalid-name
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline
from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.circle import Circle
from fibomat.shapes.shape import Shape


__all__ = ['Ring']


class Ring(Shape):
    """A ring (annulus): a circle with a concentric circular hole.

    A ring cannot be described by one arc spline. Use :meth:`Ring.to_hollow_arc_spline` for a conversion.
    """

    def __init__(
        self,
        r_outer: float,
        thickness: float,
        center: t.Optional[VectorLike] = None,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            r_outer (float): outer radius
            thickness (float): thickness of the ring (``r_outer - r_inner``). If it is equal to `r_outer`, the ring is
                               a disk.
            center (VectorLike, optional): center of the ring, default to (0, 0)
            description (str, optional): description

        Raises:
            ValueError: Raised if `r_outer` or `thickness` are not positive and finite, the thickness is larger than
                the outer radius or the center is not finite.
        """
        super().__init__(description)

        r_outer = float(r_outer)
        thickness = float(thickness)
        if not (np.isfinite(r_outer) and np.isfinite(thickness)) or r_outer <= 0. or thickness <= 0.:
            raise ValueError('r_outer and thickness must be positive and finite.')
        if thickness > r_outer:
            raise ValueError('thickness must not be larger than r_outer.')

        self._center = Vector(center) if center is not None else Vector(0, 0)
        if not np.all(np.isfinite(np.asarray(self._center))):
            raise ValueError('center must be finite.')

        self._r_outer = r_outer
        self._thickness = thickness

    @classmethod
    def from_radii(
        cls, r_outer: float, r_inner: float, center: t.Optional[VectorLike] = None, description: t.Optional[str] = None
    ) -> Ring:
        """Create a ring from the outer and the inner radius.

        Args:
            r_outer (float): outer radius
            r_inner (float): inner radius (0 <= r_inner < r_outer)
            center (VectorLike, optional): center of the ring, default to (0, 0)
            description (str, optional): description

        Returns:
            Ring

        Raises:
            ValueError: Raised if the radii are invalid.
        """
        r_inner = float(r_inner)
        if not np.isfinite(r_inner) or r_inner < 0. or r_inner >= float(r_outer):
            raise ValueError('r_inner must be finite and 0 <= r_inner < r_outer.')
        return cls(r_outer, float(r_outer) - r_inner, center, description)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(r_outer={self._r_outer!r}, thickness={self._thickness!r}, '
            f'center={self._center!r})'
        )

    @property
    def r_outer(self) -> float:
        """Outer radius.

        Access:
            get
        """
        return self._r_outer

    @property
    def r_inner(self) -> float:
        """Inner radius (0 for a disk).

        Access:
            get
        """
        inner = self._r_outer - self._thickness
        return 0. if inner <= 1e-12 * self._r_outer else inner

    @property
    def thickness(self) -> float:
        """Thickness of the ring.

        Access:
            get
        """
        return self._thickness

    @property
    def center(self) -> Vector:
        return self._center

    @property
    def area(self) -> float:
        return float(np.pi * (self._r_outer ** 2 - self.r_inner ** 2))

    @property
    def boundary_length(self) -> float:
        """Length of the outer and the inner circle.

        Access:
            get
        """
        return float(2 * np.pi * (self._r_outer + self.r_inner))

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self._center - (self._r_outer, self._r_outer), self._center + (self._r_outer, self._r_outer))

    @property
    def is_closed(self) -> bool:
        return True

    def contains(self, pos: VectorLike) -> bool:
        """Returns True if pos lies in the ring (on the boundary circles too).

        Args:
            pos (VectorLike): point to be tested

        Returns:
            bool
        """
        distance = (Vector(pos) - self._center).mag
        return bool(self.r_inner <= distance <= self._r_outer)

    def to_hollow_arc_spline(self) -> HollowArcSpline:
        """Convert the ring to a :class:`HollowArcSpline` (a disk, if the inner radius is 0, has no hole).

        Returns:
            HollowArcSpline
        """
        holes = []
        if self.r_inner > 0.:
            holes.append(Circle(r=self.r_inner, center=self._center).to_arc_spline())

        return HollowArcSpline(
            Circle(r=self._r_outer, center=self._center).to_arc_spline(),
            holes,
            self.description,
            disable_checks=True,
        )

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._center = self._center + Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        # (a ring is rotation symmetric, only the center changes)
        self._center = self._center.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)
        self._center = self._center * fac
        # a negative factor is a point reflection
        self._r_outer *= abs(fac)
        self._thickness *= abs(fac)

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        self._center = self._center.mirrored(mirror_axis)
