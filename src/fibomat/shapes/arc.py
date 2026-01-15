"""Provides the :class:`Arc` class."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from fibomat.linalg import Vector, VectorLike, BoundingBox
from fibomat.shapes.shape import Shape
from fibomat.utils import mod_2pi
from fibomat.shapes.arc_spline import ArcSpline, ArcSplineCompatible


class Arc(Shape, ArcSplineCompatible):  # pylint: disable=too-many-public-methods
    """Circular arc shape.

    Some formulas take from `here <http://www.lee-mac.com/bulgeconversion.html>`_.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        radius: float,
        start_angle: float, end_angle: float,
        sweep_dir: bool,
        center: Optional[VectorLike] = None,
        description: Optional[str] = None
    ):
        """
        Args:
            radius (float): radius
            start_angle (float): starting angle (measured from pos. x-axis, in radiants)
            end_angle (float): end angle (measured from pos. x-axis, in radiants)
            sweep_dir (bool): if True, arc direction is in mathematical positive direction and in math. negative
                              direction if False
            center (VectorLike, optional): center of completed arc, default to (0, 0)
            description (str, optional): description

        .. warning:: `center` is the center of the circle (aka. the completed arc), not centroid!

        """
        super().__init__(description)

        self._r: float = float(radius)
        # make sure that angles are in[0, 2pi]
        self._start_angle: float = mod_2pi(float(start_angle))
        self._end_angle: float = mod_2pi(float(end_angle))
        self._sweep_dir = bool(sweep_dir)
        # self._large_arc = bool(large_arc)
        self._center: Vector = Vector(center) if center is not None else Vector(0, 0)

    @classmethod
    def from_bulge(cls, start: VectorLike, end: VectorLike, bulge: float):
        """
        Construct a curve from start and end points and bulge value.
        See `here <http://www.lee-mac.com/bulgeconversion.html>`_ and `there
        <https://ezdxf.readthedocs.io/en/stable/dxfentities/lwpolyline.html#bulge-value>`_ for details concerning the
        bulge value.

        Args:
            start (VectorLike): start point
            end (VectorLike): end point
            bulge (float): bulge value

        Returns:
            Arc
        """
        start = Vector(start)
        end = Vector(end)
        bulge = float(bulge)

        theta_2 = 2 * np.arctan(bulge)
        radius = (start - end).length / 2 / np.sin(theta_2)
        phi = np.pi / 2 - theta_2 + (end - start).angle_about_x_axis
        # print(theta_2, r, phi)
        center = start+Vector(r=radius, phi=phi)

        # if bulge < 0.:
        #     start_angle = angle_about_xaxis(end - center)
        #     end_angle = angle_about_xaxis(start - center)
        # else:
        start_angle = (start - center).angle_about_x_axis
        end_angle = (end - center).angle_about_x_axis

        return cls(abs(radius), start_angle, end_angle, bool(bulge >= 0), center)

    @classmethod
    def from_points(cls, p1: VectorLike, p2: VectorLike, p3: VectorLike):  # pylint: disable=invalid-name
        """Creates an arc connecting `p1` with `p3` via `p2`.

        Args:
            p1 (VectorLike): start point
            p2 (VectorLike): intermediate point
            p3 (VectorLike): end point

        Returns:
            Arc
        """
        p1 = Vector(p1)
        p2 = Vector(p2)
        p3 = Vector(p3)

        if p1==p2 or p1==p3 or p2==p3:
            raise ValueError("Arc through points is only defined for three different points.")

        bulge = np.tan((np.pi - (p1 - p2).angle_about_x_axis + (p3 - p2).angle_about_x_axis) / 2)
        # print('bulge', bulge, (np.pi - angle_about_xaxis(p1 - p2) + angle_about_xaxis(p3 - p2)) / 2)

        return cls.from_bulge(p1, p3, bulge)

    @classmethod
    def from_points_center(cls, start: VectorLike, end: VectorLike, center: VectorLike, sweep_dir: bool):
        """Create Arc from start, end, center and sweep_dir

        Args:
            start (VectorLike): start point
            end (VectorLike): end point
            center (VectorLike): center
            sweep_dir (bool): sweep_dir

        Returns:
            Arc
        """
        start = Vector(start)
        end = Vector(end)
        center = Vector(center)

        start_angle = (start - center).angle_about_x_axis
        end_angle = (end - center).angle_about_x_axis

        radius = np.linalg.norm(start - center)

        return cls(radius=radius, start_angle=start_angle, end_angle=end_angle, sweep_dir=sweep_dir, center=center)

    @classmethod
    def from_points_center_tangent(
        cls,
        start: VectorLike,
        end: VectorLike,
        center: VectorLike,
        *, unit_tangent_start=None, unit_tangent_end=None
    ):
        """Create Arc from start, end, center and tangent at start or end. The sweep_dir is calculated automatically.

        Args:
            start (VectorLike): start point
            end (VectorLike): end point
            center (VectorLike): center
            unit_tangent_start (VectorLike, optional): unit tangent at start
            unit_tangent_end (VectorLike, optional): unit tangent at end

        Returns:
            Arc

        Raises:
            ValueError: Raised of none or both of unit_tangent_start and unit_tangent_end are defined.
        """
        start = Vector(start)
        end = Vector(end)
        center = Vector(center)

        start_angle = (start - center).angle_about_x_axis
        end_angle = (end - center).angle_about_x_axis

        radius = np.linalg.norm(start - center)

        if unit_tangent_start is not None and unit_tangent_end is None:
            unit_tangent_start = Vector(unit_tangent_start)

            sweep_matrix = np.ones(shape=(3, 3), dtype=float)
            sweep_matrix[0, 1:] = np.asarray(start)
            sweep_matrix[1, 1:] = np.asarray(start + unit_tangent_start)
            sweep_matrix[2, 1:] = np.asarray(end)
        elif unit_tangent_start is None and unit_tangent_end is not None:
            unit_tangent_end = Vector(unit_tangent_end)

            sweep_matrix = np.ones(shape=(3, 3), dtype=float)
            sweep_matrix[0, 1:] = np.asarray(start)
            sweep_matrix[1, 1:] = np.asarray(end)
            sweep_matrix[2, 1:] = np.asarray(end + unit_tangent_end)
        elif unit_tangent_start is None and unit_tangent_end is None:
            raise ValueError('Anyone of unit_tangent_start or unit_tangent_end must be defined.')
        else:
            raise ValueError('unit_tangent_start and unit_tangent_end cannot be defined both.')

        sweep_dir = bool(np.linalg.det(sweep_matrix) > 0)

        arc = cls(radius=radius, start_angle=start_angle, end_angle=end_angle, sweep_dir=sweep_dir, center=center)

        if unit_tangent_start is not None:
            if not np.allclose(arc.unit_tangent_start, unit_tangent_start):
                raise RuntimeError
        else:
            if not np.allclose(arc.unit_tangent_end, unit_tangent_end):
                raise RuntimeError

        return arc

    def to_arc_spline(self) -> ArcSpline:
        is_closed = self.is_closed

        if abs(self.bulge) > 1:
            arcs = self.split()
            vertices = [(*arcs[0].start, arcs[0].bulge), (*arcs[1].start, arcs[1].bulge)]
            if not is_closed:
                vertices.append((*arcs[1].end, 0.))
        else:
            vertices = [(*self.start, self.bulge), (*self.end, 0.)]

        return ArcSpline(vertices, is_closed, self.description)

    @property
    def start(self) -> Vector:
        """Start point of arc

        Access:
            get

        Returns:
            Vector
        """
        return Vector(r=self._r, phi=self._start_angle)+self._center

    @property
    def end(self) -> Vector:
        """End point of arc

         Access:
             get

         Returns:
             Vector
         """
        return Vector(r=self._r, phi=self._end_angle)+self._center

    @property
    def start_angle(self) -> float:
        """Start angle of arc (measured from pos. x. axis).

        Access:
            get

        Returns:
            float
        """
        return self._start_angle

    @property
    def end_angle(self):
        """End angle of arc (measured from pos. x. axis).

        Access:
            get

        Returns:
            float
        """
        return self._end_angle

    @property
    def sweep_dir(self) -> bool:
        """If `True`, arc direction is in mathematical positive direction and in math. negative direction if `False`

        Access:
            get

        Returns:
            bool
        """
        return self._sweep_dir

    @property
    def radius(self) -> float:
        """Arc radius

        Access:
            get

        Returns:
            float
        """
        return self._r

    @property
    def theta(self) -> float:
        """Absolute value of  enclosed angle

        Access:
            get

        Returns:
            float
        """
        angle = np.abs(self._end_angle - self._start_angle)
        if (self._sweep_dir and self._start_angle >= self._end_angle) \
           or (not self._sweep_dir and self._end_angle >= self._start_angle):
            angle = 2*np.pi - angle

        return angle

    @property
    def bulge(self) -> float:
        """Bulge value = tan(theta/4). bulge > 0 arc goes in math. positve direction and for bulge < 0 in math.
        neg. direction

        Access:
            get

        Returns:
            float
        """
        if self._sweep_dir:
            sign = 1
        else:
            sign = -1
        return sign * np.tan(self.theta / 4.)

    @property
    def center(self) -> Vector:
        """Center of enclosing circle.

        Access:
            get

        Returns:
            Vector
        """
        return self._center

    @property
    def midpoint(self) -> Vector:
        """Midpoint of arc.

        Access:
            get

        Returns:
            Vector
        """
        sign = 1. if self._sweep_dir else -1.
        return self.start.rotated(sign * self.theta/2, origin=self._center)

    @property
    def unit_tangent_start(self) -> Vector:
        """Unit tangent at start.

        Access:
            get

        Returns:
            Vector
        """
        sign = 1. if self._sweep_dir else -1.
        return sign * Vector(-np.sin(self._start_angle), np.cos(self._start_angle))

    @property
    def unit_tangent_end(self) -> Vector:
        """Unit tangent at end.

        Access:
            get

        Returns:
            Vector
        """
        sign = 1. if self._sweep_dir else -1.
        return sign * Vector(-np.sin(self._end_angle), np.cos(self._end_angle))

    def unit_tangent_at(self, angle):
        """Unit tangent at specific angle.

        Args:
            angle (float): 0 < angle < self.theta

        Returns:
            Vector
        """
        if angle > self.theta:
            raise ValueError('angle > self.theta')

        sign = 1. if self._sweep_dir else -1.
        phi = self._start_angle + sign * angle

        return sign * Vector(-np.sin(phi), np.cos(phi))

    @property
    def length(self):
        """Length of arc.

        Access:
            get

        Returns:
            float
        """
        return self.radius * self.theta

    def split(self) -> List[Arc]:
        """Return two arcs if theta > np.pi (hence the abs(bulge) value of new arcs will be smaller than 1).

        Returns:
            List[Arc]: if theta > np.pi list contains two elements and one otherwise.
        """
        # pylint: disable=protected-access
        theta = self.theta
        if theta > np.pi:
            arc_1 = self.clone()
            arc_2 = self.clone()

            if self._sweep_dir:
                arc_1._end_angle = mod_2pi(arc_1._start_angle + theta / 2)
                arc_2._start_angle = arc_1._end_angle
            else:
                arc_1._end_angle = mod_2pi(arc_1._start_angle - theta / 2)
                arc_2._start_angle = arc_1._end_angle
            return [arc_1, arc_2]

        return [self.clone()]

    def split_at(self, angle: float) -> List[Arc]:
        """Split the arc in two halves ([0, angle], [angle, theta]).

        Args:
            angle (float): split angle.

        Returns:
            List[Arc]

        Raises:
            ValueError: Raised if angle < 0 or angle > self.theta
        """
        # pylint: disable=protected-access

        if angle < 0 or angle > self.theta:
            raise ValueError('angle < 0 or angle > self.theta')

        arc_1 = self.clone()
        arc_2 = self.clone()

        if self._sweep_dir:
            arc_1._end_angle = mod_2pi(arc_1._start_angle + angle)
            arc_2._start_angle = arc_1._end_angle
        else:
            arc_1._end_angle = mod_2pi(arc_1._start_angle - angle)
            arc_2._start_angle = arc_1._end_angle
        return [arc_1, arc_2]

    # def reverse(self) -> None:
    #     """Reverse the arc in-place.
    #
    #     Returns:
    #         None
    #     """
    #     self._start_angle, self._end_angle = self._end_angle, self._start_angle
    #     self._sweep_dir = not self._sweep_dir

    def reversed(self) -> Arc:
        """Return a reversed version of the arc

        Returns:
            Arc
        """
        # pylint: disable=protected-access
        copy = self.clone()
        copy._start_angle, copy._end_angle = copy._end_angle, copy._start_angle
        copy._sweep_dir = not copy._sweep_dir
        return copy

    # def clone(self) -> Arc:
    #     return self.__class__(self._r, self._start_angle, self._end_angle, self._sweep_dir, self._center.clone())
    #
    # __copy__ = clone

    def __repr__(self) -> str:
        return 'Arc(r={!r}, start_angle={!r}, end_angle={!r}, sweep_dir={!r}, center={!r})'.format(
            self.radius, self.start_angle, self.end_angle, self.sweep_dir, self.center
        )

    @property
    def bounding_box(self) -> BoundingBox:
        # https://stackoverflow.com/questions/1336663/2d-bounding-box-of-a-sector
        def axis_intersection(angle: float):
            return self._center+Vector(self._r, 0).rotated(angle)

        if np.isclose(self.theta, 2*np.pi):
            r_vec = Vector(self._r, self._r)
            return BoundingBox(self.center-r_vec, self.center+r_vec)

        points = [self.start, self.end]

        for axis_angle in [0., np.pi/2, np.pi, 3/2 * np.pi, 2*np.pi]:
            if self._sweep_dir:
                if self._end_angle >= self._start_angle:
                    if self._end_angle >= axis_angle >= self._start_angle:
                        points.append(axis_intersection(axis_angle))
                else:  # self._end_angle < self._start_angle
                    if self._end_angle >= axis_angle or self._start_angle <= axis_angle:
                        points.append(axis_intersection(axis_angle))
            else:
                if self._start_angle >= self._end_angle:
                    if self._start_angle >= axis_angle >= self._end_angle:
                        points.append(axis_intersection(axis_angle))
                else:  # self._start_angle < self._end_angle
                    if self._end_angle <= axis_angle or self._start_angle >= axis_angle:
                        points.append(axis_intersection(axis_angle))

        return BoundingBox.from_points(points)

    @property
    def is_closed(self) -> bool:
        return np.isclose(self.theta, 2*np.pi)

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._center += Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        theta = float(theta)

        self._start_angle = mod_2pi(self._start_angle + theta)
        self._end_angle = mod_2pi(self._end_angle + theta)

    def _impl_scale(self, fac: float) -> None:
        self._r *= float(fac)

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        # pylint: disable=protected-access

        mirrored_arc = self.__class__.from_points_center(
            self.start.mirrored(mirror_axis),
            self.end.mirrored(mirror_axis),
            self._center.mirrored(mirror_axis),
            not self._sweep_dir
        )
        self._start_angle = mirrored_arc._start_angle
        self._end_angle = mirrored_arc._end_angle
        self._sweep_dir = mirrored_arc._sweep_dir
        self._center = mirrored_arc._center
