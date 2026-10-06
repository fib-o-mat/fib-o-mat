"""Provides the :class:`Ellipse` class.

Example::

    import numpy as np
    from fibomat.shapes import Ellipse

    ellipse = Ellipse(a=3, b=1, theta=np.pi / 4, center=(1, 1))
    ellipse.area, ellipse.boundary_length, ellipse.bounding_box
    ellipse.to_arc_spline()  # approximation by circular arcs
"""
# pylint: disable=too-many-arguments,invalid-name
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import BoundingBox, Vector, VectorLike, rotate, translate
from fibomat.shapes.arc import Arc
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.circle import Circle
from fibomat.shapes.shape import Shape


__all__ = ['Ellipse']


ARC_SPLINE_TOLERANCE = 1e-6
"""Maximal distance of the arc spline approximation of an ellipse from the ellipse in units of the smaller half axis."""


class Ellipse(Shape, ArcSplineCompatible):
    """2-dim ellipse."""

    def __init__(
        self,
        a: float,
        b: float,
        theta: float = 0,
        center: t.Optional[VectorLike] = None,
        description: t.Optional[str] = None,
    ):
        """

        Args:
            a (float): length of half axis in pos. x-direction (unrotated)
            b (float): length of half axis in pos. y-direction (unrotated)
            theta (float): rotation angle, default to 0
            center (VectorLike, optional): center of ellipse, default to (0, 0)
            description (str, optional): description

        Raises:
            ValueError: Raised if a or b are not positive and finite or theta or the center are not finite.
        """
        super().__init__(description)

        a = float(a)
        b = float(b)
        theta = float(theta)
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0. or b <= 0.:
            raise ValueError('a and b must be positive and finite.')
        if not np.isfinite(theta):
            raise ValueError('theta must be finite.')

        # unit vector of the a axis (the b axis is orthogonal to it)
        self._a_axis = Vector(1, 0.0).rotated(theta)

        self._a = a
        self._b = b

        self._center = Vector(center) if center is not None else Vector()
        if not np.all(np.isfinite(np.asarray(self._center))):
            raise ValueError('center must be finite.')

    def __repr__(self) -> str:
        return '{}(a={!r}, b={!r}, theta={!r}, center={!r})'.format(
            self.__class__.__name__, self.a, self.b, self.theta, self.center
        )

    def _arc_spline_vertices(self) -> t.List[t.Tuple[float, float, float]]:
        """Vertices (x, y, bulge) of an arc spline approximation of the unrotated ellipse around the origin."""
        tolerance = ARC_SPLINE_TOLERANCE * min(self._a, self._b)

        def point(param: float) -> Vector:
            return Vector(self._a * np.cos(param), self._b * np.sin(param))

        def arc_through(param_0: float, param_1: float) -> Arc:
            return Arc.from_points(point(param_0), point((param_0 + param_1) / 2), point(param_1))

        def is_accurate(param_0: float, param_1: float, arc: Arc) -> bool:
            # the ellipse must not deviate from the arc at the quarter points of the interval
            quarter = (param_1 - param_0) / 4
            return all(
                abs((point(param_0 + i * quarter) - arc.center).mag - arc.radius) <= tolerance for i in (1, 3)
            )

        def subdivide(param_0: float, param_1: float, depth: int = 0) -> t.List[t.Tuple[float, float]]:
            arc = arc_through(param_0, param_1)
            if depth >= 20 or is_accurate(param_0, param_1, arc):
                return [(param_0, arc.bulge)]
            middle = (param_0 + param_1) / 2
            return subdivide(param_0, middle, depth + 1) + subdivide(middle, param_1, depth + 1)

        # start with the four quadrants, which are symmetric (the ellipse is symmetric about both axes)
        vertices = []
        for i_quadrant in range(4):
            for param, bulge in subdivide(i_quadrant * np.pi / 2, (i_quadrant + 1) * np.pi / 2):
                vertices.append((*point(param), bulge))
        return vertices

    def to_arc_spline(self) -> ArcSpline:
        """Approximation of the ellipse by circular arcs.

        The maximal distance of the arc spline from the ellipse is about ``ARC_SPLINE_TOLERANCE`` times the smaller
        half axis. A (nearly) circular ellipse is converted to the exact arc spline of a circle.

        Returns:
            ArcSpline
        """
        if np.isclose(self._a, self._b):
            return Circle(r=self._a, center=self._center, description=self.description).to_arc_spline()

        return ArcSpline(self._arc_spline_vertices(), True, self.description).transformed(
            rotate(self.theta) | translate(self._center)
        )

    @property
    def a(self) -> float:
        """Length of half axis in pos. x-direction (unrotated)

        Access:
            get

        Returns:
            float
        """
        return self._a

    @property
    def b(self) -> float:
        """Length of half axis in pos. y-direction (unrotated)

        Access:
            get

        Returns:
            float
        """
        return self._b

    @property
    def theta(self) -> float:
        """Rotation angle of the a axis of the ellipse (in [0, 2pi)).

        Access:
            get

        Returns:
            float
        """
        return float(np.mod(self._a_axis.phi, 2 * np.pi))

    @property
    def area(self) -> float:
        return float(np.pi * self._a * self._b)

    @property
    def boundary_length(self) -> float:
        """Perimeter of the ellipse (calculated with the complete elliptic integral of the second kind).

        Access:
            get

        Returns:
            float
        """
        from scipy.special import ellipe  # pylint: disable=import-outside-toplevel

        major, minor = max(self._a, self._b), min(self._a, self._b)
        return float(4 * major * ellipe(1 - (minor / major) ** 2))

    @property
    def center(self) -> Vector:
        return self._center

    @property
    def bounding_box(self) -> BoundingBox:
        # half extents of a rotated ellipse:
        # https://www.iquilezles.org/www/articles/ellipses/ellipses.htm
        cos, sin = self._a_axis.x, self._a_axis.y
        half_width = np.sqrt((self._a * cos) ** 2 + (self._b * sin) ** 2)
        half_height = np.sqrt((self._a * sin) ** 2 + (self._b * cos) ** 2)
        half = Vector(half_width, half_height)

        return BoundingBox(self._center - half, self._center + half)

    @property
    def is_closed(self) -> bool:
        return True

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._center = self._center + Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        theta = float(theta)
        self._center = self._center.rotated(theta)
        self._a_axis = self._a_axis.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)
        self._center = self._center * fac
        # a negative factor is a point reflection, which maps an ellipse to an ellipse (the axes keep their directions)
        self._a *= abs(fac)
        self._b *= abs(fac)

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        # an ellipse is symmetric about its axes, hence, only the direction of the a axis has to be mirrored
        self._a_axis = self._a_axis.mirrored(mirror_axis)
        self._center = self._center.mirrored(mirror_axis)

