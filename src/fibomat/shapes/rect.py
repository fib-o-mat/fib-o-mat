"""Provides the :class:`Rect` class.

Example::

    import numpy as np
    from fibomat.shapes import Rect

    rect = Rect(width=4, height=2, theta=np.pi / 6, center=(1, 1))
    rect.area, rect.boundary_length, rect.corners, rect.bounding_box

    Rect.from_bounding_box(BoundingBox((0, 0), (1, 2)))
    Rect.from_line(Line((0, 0), (3, 0)), height=1)
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.line import Line
from fibomat.shapes.shape import Shape


__all__ = ['Rect']


class Rect(Shape, ArcSplineCompatible):
    """2-dim rect."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        width: float,
        height: float,
        theta: float = 0,
        center: t.Optional[VectorLike] = None,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            width (float): width of rect
            height (float): height of rect
            theta (float): rotation angle of rect, default to 0
            center (VectorLike, optional): center of rect, default to (0, 0)
            description (str, optional): description

        Raises:
            ValueError: Raised if width or height are not positive and finite or theta or the center are not finite.
        """
        super().__init__(description)

        width = float(width)
        height = float(height)
        theta = float(theta)
        if not (np.isfinite(width) and np.isfinite(height)) or width <= 0. or height <= 0.:
            raise ValueError('width and height must be positive and finite.')
        if not np.isfinite(theta):
            raise ValueError('theta must be finite.')

        self._width = width
        self._height = height
        # unit vector in the direction of the width (the height direction is orthogonal to it)
        self._width_axis = Vector(1, 0).rotated(theta)
        self._center = Vector(center) if center is not None else Vector()
        if not np.all(np.isfinite(np.asarray(self._center))):
            raise ValueError('center must be finite.')

    @classmethod
    def from_bounding_box(cls, bbox: BoundingBox) -> Rect:
        """Create a rect from a BoundingBox.

        Args:
            bbox: bounding box from which a rect is constructed

        Returns:
            Rect

        Raises:
            ValueError: Raised if the bounding box has no area.
        """
        return cls(bbox.width, bbox.height, 0.0, bbox.center)

    @classmethod
    def from_line(cls, line: Line, height: float) -> Rect:
        """Create a rect defined by a line and a certain height. The line is the center line of the rect.

        Args:
            line (Line): line (its length is the width of the rect)
            height (float): height

        Returns:
            Rect

        Raises:
            ValueError: Raised if the line has no length or height is not positive.
        """
        direction = line.end - line.start

        if direction.mag == 0.:
            raise ValueError('The line must have a length.')

        return cls(width=direction.mag, height=height, center=line.center, theta=direction.phi)

    def to_arc_spline(self) -> ArcSpline:
        vertices = np.append(np.array(self.corners), np.zeros(4)[:, np.newaxis], axis=1)

        return ArcSpline(vertices, True, self.description)

    @property
    def width(self) -> float:
        """Width of rect.

        Access:
            get

        Returns:
            float
        """
        return self._width

    @property
    def height(self) -> float:
        """Height of rect.

        Access:
            get

        Returns:
            float
        """
        return self._height

    @property
    def area(self) -> float:
        return self._width * self._height

    @property
    def boundary_length(self) -> float:
        """Perimeter of the rect.

        Access:
            get

        Returns:
            float
        """
        return 2 * (self._width + self._height)

    @property
    def theta(self) -> float:
        """Rotation angle of the width axis of the rect (in [0, 2pi)).

        Access:
            get

        Returns:
            float
        """
        return float(np.mod(self._width_axis.phi, 2 * np.pi))

    @property
    def corners(self) -> t.List[Vector]:
        """Corner points of rect in counterclockwise order, starting at the corner at +width/2, +height/2 of the
        unrotated rect.

        Access:
            get

        Returns:
            List[Vector]
        """
        half_width = self._width_axis * (self._width / 2)
        half_height = Vector(-self._width_axis.y, self._width_axis.x) * (self._height / 2)

        return [
            self._center + half_width + half_height,
            self._center - half_width + half_height,
            self._center - half_width - half_height,
            self._center + half_width - half_height,
        ]

    def __repr__(self) -> str:
        return '{}(width={!r}, height={!r}, theta={!r}, center={!r})'.format(
            self.__class__.__name__, self.width, self.height, self.theta, self.center
        )

    @property
    def center(self) -> Vector:
        return self._center

    @property
    def bounding_box(self) -> BoundingBox:
        cos, sin = abs(self._width_axis.x), abs(self._width_axis.y)
        half = Vector((cos * self._width + sin * self._height) / 2, (sin * self._width + cos * self._height) / 2)

        return BoundingBox(self._center - half, self._center + half)

    @property
    def is_closed(self) -> bool:
        return True

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._center = self._center + Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        theta = float(theta)
        self._center = self._center.rotated(theta)
        self._width_axis = self._width_axis.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)
        self._center = self._center * fac
        # a negative factor is a point reflection, which maps a rect to a rect (the axes keep their directions)
        self._width *= abs(fac)
        self._height *= abs(fac)

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        # a rect is symmetric about its axes, hence, only the direction of the width axis has to be mirrored
        self._width_axis = self._width_axis.mirrored(mirror_axis)
        self._center = self._center.mirrored(mirror_axis)
