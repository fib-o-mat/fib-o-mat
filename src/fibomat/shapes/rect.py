"""Provides the :class:`Rect` class."""

from __future__ import annotations
from typing import Optional, List

import numpy as np

from fibomat.shapes import Shape
from fibomat.shapes.arc_spline import ArcSplineCompatible, ArcSpline
from fibomat.linalg import VectorLike, Vector, BoundingBox, signed_angle_between
from fibomat.shapes.line import Line


class Rect(Shape, ArcSplineCompatible):
    """2-dim rect."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        width: float,
        height: float,
        theta: float = 0,
        center: Optional[VectorLike] = None,
        description: Optional[str] = None,
    ):
        """
        Args:
            width (float): width of rect
            height (float): height of rect
            theta (float): rotation angle of rect, default to 0
            center (VectorLike, optional): center of rect, default to (0, 0)
            description (str, optional): description
        """
        super().__init__(description)

        width_half = float(width) / 2
        height_half = float(height) / 2

        center = Vector(center)

        self._rect = ArcSpline(
            [
                (*(center + (width_half, height_half)), 0.0),
                (*(center + (-width_half, height_half)), 0.0),
                (*(center + (-width_half, -height_half)), 0.0),
                (*(center + (width_half, -height_half)), 0.0),
            ],
            True,
        ).rotated(theta, "center")

        self._rot_angle = float(theta)

        # self._axes = VectorArray([float(width) / 2, 0.], [0., float(height) / 2])
        # self._axes = self._axes.rotated(theta)
        #
        # self._center = Vector(center) if center is not None else Vector()

    @classmethod
    def from_bounding_box(cls, bbox: BoundingBox) -> Rect:
        """Create a rect from a BoundingBox.

        Args:
            bbox: bounding box from which a rect is constructed

        Returns:
            Rect
        """
        return cls(bbox.width, bbox.height, 0.0, bbox.center)

    @classmethod
    def from_line(cls, line: Line, height: float):
        """Create a rect defined by a line a certain height

        Args:
            line (Line):
            height (float):

        Returns:
            Rect
        """
        direction = line.end - line.start
        width = direction.mag

        # is theta correct?
        return cls(
            width=width,
            height=height,
            center=line.center,
            theta=direction.angle_about_x_axis,
        )

    def to_arc_spline(self) -> ArcSpline:
        vertices = np.append(self.corners, np.zeros(4)[:, np.newaxis], axis=1)

        return ArcSpline(vertices, True, self.description)

    @property
    def width(self) -> float:
        """Width of rect.

        Access:
            get

        Returns:
            float
        """
        start, end = self._rect.vertices[(0, 1), :2]
        return np.linalg.norm(start - end)

    @property
    def height(self) -> float:
        """Height of rect.

        Access:
            get

        Returns:
            float
        """
        start, end = self._rect.vertices[(1, 2), :2]
        return np.linalg.norm(start - end)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def theta(self) -> float:
        """Rotation angle of rect.

        Access:
            get

        Returns:
            float
        """
        return self._rot_angle

    @property
    def corners(self) -> List[Vector]:
        """Corner points of rect

        Access:
            get

        Returns:
            List[Vector]
        """
        return [Vector(vec) for vec in self._rect.vertices[:, :2]]

    def __repr__(self) -> str:
        return "{}(width={!r}, height={!r}, theta={!r}, center={!r})".format(
            self.__class__.__name__, self.width, self.height, self.theta, self.center
        )

    @property
    def center(self) -> Vector:
        return self._rect.center

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox.from_points(self.corners)

    @property
    def is_closed(self) -> bool:
        return True

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._rect._impl_translate(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        self._rect._impl_rotate(theta)
        self._rot_angle += float(theta)

    def _impl_scale(self, fac: float) -> None:
        self._rect._impl_scale(fac)

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        # how das mirroring affect rot angle?
        # TODO: test this carefully
        ref_vec = self._rect.vertices[0, :2]

        self._rect._impl_mirror(mirror_axis)

        rot_angle = signed_angle_between(ref_vec, self._rect.vertices[0, :2])
        self._rot_angle += rot_angle
