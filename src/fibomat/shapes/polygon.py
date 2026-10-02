"""Provides the :class:`Polygon` class."""

from typing import Optional, List

import numpy as np

from fibomat.shapes import polyline
from fibomat.linalg import VectorLike, Vector


class Polygon(polyline.Polyline):
    """2-dim polygon (closed polyline)."""

    def __init__(self, points: List[VectorLike], description: Optional[str] = None):
        """
        .. note:: `point[0]` and `point[-1]` are automatically connected. Hence, `point[0] != point[-1]` usually.

        Args:
            points (VectorArrayLike): polygon points.
            description (str, optional): description
        """
        super().__init__(points, description, _closed=True)

    @classmethod
    def regular_ngon(  # pylint: disable=too-many-arguments,invalid-name
        cls,
        n: int,
        radius: float = 1.0,
        circumcircle: bool = True,
        center: Optional[VectorLike] = None,
        description: Optional[str] = None,
    ):
        """
        Creates an regular polygon.

        Args:
            n (int): number of corners
            radius: cirumcircle or incircle radius
            circumcircle (bool): if true, radius is treated as circumcircle radius else as incircle radius
            center (VectorLike, optional): center of ngon, default to (0, 0)
            description (str, optional): description

        Returns:
            Polygon

        Raises:
            ValueError: Raised of n < 3 or radius <= 0.
        """
        if n < 3:
            raise ValueError("n < 3")

        if radius <= 0.0:
            raise ValueError("radius <= 0.")

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
        return self.to_arc_spline().area
