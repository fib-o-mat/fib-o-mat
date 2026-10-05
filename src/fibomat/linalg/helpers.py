"""Small geometric helpers: :func:`make_perp_vector` and :class:`GeomLine`.

All helpers work on plain (unitless) vectors.

Example::

    from fibomat.linalg.helpers import GeomLine, make_perp_vector

    make_perp_vector((1, 2))  # Vector(-2, 1)

    line_1 = GeomLine(direction=(1, 0), support=(0, 0))
    line_2 = GeomLine(direction=(0, 1), support=(2, -1))

    line_1(0.5)  # point (0.5, 0) of the line
    line_1.intersect_at(line_2)  # array([2., 0.])
    line_1.intersect_at_param(line_2)  # 2., so that line_1(2.) is the intersection point
    line_1.find_param((3, 0))  # 3., so that line_1(3.) == (3, 0)
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg.vectors import VectorLike, Vector


__all__ = ['make_perp_vector', 'GeomLine']


_PARALLEL_TOL = 1e-8
"""Default tolerance of the sine of the angle between two lines which are treated as parallel."""


def make_perp_vector(vec: VectorLike) -> Vector:
    """Return the vector perpendicular to `vec`, i.e. `vec` rotated by 90 degrees in math. positive direction.

    Args:
        vec (VectorLike): vector

    Returns:
        Vector: perpendicular vector with the same length

    Raises:
        VectorValueError: Raised if `vec` is no vector.
    """
    vec = Vector(vec)
    return Vector(-vec.y, vec.x)


class GeomLine:
    """A line ``support + u * direction`` in 2D with the (real) parameter `u`.

    Lines are immutable by convention. The parameter of a point depends on the length of `direction`.
    """

    def __init__(self, direction: VectorLike, support: VectorLike):
        """
        Args:
            direction (VectorLike): direction of the line
            support (VectorLike): a point of the line (the line at parameter 0)

        Raises:
            VectorValueError: Raised if `direction` or `support` are no vectors.
            ValueError: Raised if direction or support are not finite.
            RuntimeError: Raised if `direction` is the null vector.
        """
        self._direction = np.array(Vector(direction), dtype=float)
        self._support = np.array(Vector(support), dtype=float)

        if not np.all(np.isfinite(self._direction)) or not np.all(np.isfinite(self._support)):
            raise ValueError('direction and support must be finite.')

        # exact test: a short direction is a valid direction (the length only scales the parameter)
        if not np.any(self._direction):
            raise RuntimeError('direction must not be the null vector.')

    @classmethod
    def make_bisector(cls, p_0: VectorLike, p_1: VectorLike) -> GeomLine:
        """Create the line through `p_0` and `p_1`. The support point is the midpoint of `p_0` and `p_1`.

        .. note:: Despite its name, this is not the perpendicular bisector, see :meth:`GeomLine.make_perp_bisector`.

        Args:
            p_0 (VectorLike): first point
            p_1 (VectorLike): second point

        Returns:
            GeomLine

        Raises:
            RuntimeError: Raised if the points are identical.
        """
        p_0 = Vector(p_0)
        p_1 = Vector(p_1)

        return cls(p_1 - p_0, (p_0 + p_1) / 2)

    @classmethod
    def make_perp_bisector(cls, p_0: VectorLike, p_1: VectorLike) -> GeomLine:
        """Create the perpendicular bisector of the segment `p_0` - `p_1`, i.e. the line perpendicular to the segment
        through its midpoint (support point).

        Args:
            p_0 (VectorLike): first point
            p_1 (VectorLike): second point

        Returns:
            GeomLine

        Raises:
            RuntimeError: Raised if the points are identical.
        """
        p_0 = Vector(p_0)
        p_1 = Vector(p_1)

        return cls(make_perp_vector(p_1 - p_0), (p_0 + p_1) / 2)

    @property
    def direction(self) -> Vector:
        """Direction of the line (not normalized).

        Access:
            get
        """
        return Vector(self._direction)

    @property
    def support(self) -> Vector:
        """Support point of the line.

        Access:
            get
        """
        return Vector(self._support)

    def __repr__(self) -> str:
        return f'GeomLine(direction=({self._direction[0]}, {self._direction[1]}), ' \
               f'support=({self._support[0]}, {self._support[1]}))'

    def __call__(self, u: t.Union[float, t.Sequence[float], np.ndarray]) -> np.ndarray:  # pylint: disable=invalid-name
        """Return the point(s) of the line at parameter(s) `u`.

        Args:
            u (float, array-like): parameter or array of parameters

        Returns:
            numpy.ndarray: point of shape (2,) or an array of points of shape (len(u), 2)
        """
        u = np.asarray(u, dtype=float)
        return self._support + u[..., np.newaxis] * self._direction

    def _cross_with(self, other: GeomLine) -> float:
        return float(self._direction[0] * other._direction[1] - self._direction[1] * other._direction[0])

    def parallel_to(self, other: GeomLine, tol: float = _PARALLEL_TOL) -> bool:
        """Check if the lines are parallel (or antiparallel, i.e. have opposite directions). Identical lines are
        parallel.

        Args:
            other (GeomLine): other line
            tol (float): tolerance of the sine of the angle between the directions

        Returns:
            bool
        """
        if not isinstance(other, GeomLine):
            raise TypeError('other must be a GeomLine.')

        sin = self._cross_with(other) / (np.linalg.norm(self._direction) * np.linalg.norm(other._direction))
        return bool(abs(sin) <= tol)

    def intersect_at_param(self, other: GeomLine) -> float:
        """Return the parameter `u` of this line at which it intersects `other`.

        Args:
            other (GeomLine): other line

        Returns:
            float: `u`, so that ``self(u)`` is the intersection point

        Raises:
            ValueError: Raised if the lines are parallel (no or infinitely many intersection points).
        """
        if self.parallel_to(other):
            raise ValueError('Lines are parallel, there is no unique intersection point.')

        # self.support + u * self.direction = other.support + w * other.direction, solved with Cramer's rule
        diff = other._support - self._support
        return float((diff[0] * other._direction[1] - diff[1] * other._direction[0]) / self._cross_with(other))

    def intersect_at(self, other: GeomLine) -> np.ndarray:
        """Return the intersection point with `other`.

        Args:
            other (GeomLine): other line

        Returns:
            numpy.ndarray: intersection point

        Raises:
            ValueError: Raised if the lines are parallel (no or infinitely many intersection points).
        """
        return self(self.intersect_at_param(other))

    def find_param(self, point: VectorLike) -> float:
        """Return the parameter `u` of `point`, which must lie on the line.

        Args:
            point (VectorLike): point on the line

        Returns:
            float: `u`, so that ``self(u) == point``

        Raises:
            ValueError: Raised if the point does not lie on the line (up to numerical tolerances).
        """
        point = np.array(Vector(point), dtype=float)

        # orthogonal projection of the point onto the line
        param = float((point - self._support) @ self._direction / (self._direction @ self._direction))

        if not np.allclose(self(param), point):
            raise ValueError('point lies not on GeomLine.')

        return param
