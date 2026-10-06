"""Provides the :class:`Polygon` class.

Example::

    from fibomat.shapes import Polygon

    triangle = Polygon([(0, 0), (1, 0), (0, 1)])
    triangle.area  # 0.5

    Polygon.regular_ngon(6, radius=2, center=(1, 1))
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import Vector, VectorLike
from fibomat.shapes.polyline import Polyline


__all__ = ['Polygon']


class Polygon(Polyline):
    """2-dim polygon (closed polyline)."""

    def __init__(self, points: t.Iterable[VectorLike], description: t.Optional[str] = None):
        """
        .. note:: `point[0]` and `point[-1]` are automatically connected. Hence, `point[0] != point[-1]` usually.

        Args:
            points (Iterable[VectorLike], np.ndarray): polygon points (at least two), array of shape (n, 2)
            description (str, optional): description

        Raises:
            VectorValueError: Raised if points are no vectors.
            ValueError: Raised if there are less than two points or points are not finite.
        """
        super().__init__(points, description, _closed=True)

    @classmethod
    def regular_ngon(  # pylint: disable=too-many-arguments,invalid-name
        cls,
        n: int,
        radius: float = 1.0,
        circumcircle: bool = True,
        center: t.Optional[VectorLike] = None,
        description: t.Optional[str] = None,
    ) -> Polygon:
        """
        Creates an regular polygon. The polygon is symmetric about the x axis (a side is orthogonal to the x axis).

        Args:
            n (int): number of corners
            radius: circumcircle or incircle radius
            circumcircle (bool): if true, radius is treated as circumcircle radius else as incircle radius
            center (VectorLike, optional): center of ngon, default to (0, 0)
            description (str, optional): description

        Returns:
            Polygon

        Raises:
            ValueError: Raised if n < 3, n is no integer or radius <= 0 (or not finite).
        """
        if int(n) != n:
            raise ValueError('n must be an integer.')
        n = int(n)

        if n < 3:
            raise ValueError('n < 3')

        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError('radius <= 0.')

        if not circumcircle:
            radius = radius / np.cos(np.pi / n)

        center = Vector(center) if center is not None else Vector(0, 0)

        angle = 2 * np.pi / n

        points = [Vector(radius, 0.0).rotated(-angle / 2)]
        for i in range(1, n):
            points.append(points[i - 1].rotated(angle))

        return cls(np.array(points) + center, description)

    @property
    def is_closed(self) -> bool:
        return True

    @property
    def area(self) -> float:
        return self._polygon.area
