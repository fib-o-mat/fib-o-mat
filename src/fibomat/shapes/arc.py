"""Provides the :class:`Arc` class.

Example::

    import numpy as np
    from fibomat.shapes import Arc

    # counterclockwise quarter circle with radius 2 around (1, 1) from angle 0 to angle pi / 2
    arc = Arc(radius=2, start_angle=0, end_angle=np.pi / 2, sweep_dir=True, center=(1, 1))
    arc.start, arc.end  # Vector(3, 1), Vector(1, 3)
    arc.theta, arc.length, arc.bulge  # pi / 2, pi, tan(pi / 8)

    Arc.from_bulge((0, 0), (1, 0), 1)  # half circle
    Arc.from_points((1, 0), (0, 1), (-1, 0))  # arc through three points
    Arc.from_points_center((1, 0), (0, 1), (0, 0), sweep_dir=True)
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import Vector, VectorLike, BoundingBox
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.shape import Shape
from fibomat.utils import mod_2pi


__all__ = ['Arc']


_ANGLE_TOL = 1e-8
"""Angles which differ less than this value are treated as equal."""


class Arc(Shape, ArcSplineCompatible):  # pylint: disable=too-many-public-methods
    """Circular arc shape.

    Some formulas take from `here <http://www.lee-mac.com/bulgeconversion.html>`_.

    The arc goes from the start angle to the end angle in the direction `sweep_dir`. If the start and end angle are
    equal, the arc is a full circle.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        radius: float,
        start_angle: float,
        end_angle: float,
        sweep_dir: bool,
        center: t.Optional[VectorLike] = None,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            radius (float): radius, must be positive
            start_angle (float): starting angle (measured from pos. x-axis, in radians)
            end_angle (float): end angle (measured from pos. x-axis, in radians)
            sweep_dir (bool): if True, arc direction is in mathematical positive direction and in math. negative
                              direction if False
            center (VectorLike, optional): center of completed arc, default to (0, 0)
            description (str, optional): description

        Raises:
            ValueError: Raised if radius is not positive and finite or the angles or the center are not finite.

        .. warning:: `center` is the center of the circle (aka. the completed arc), not centroid!

        """
        super().__init__(description)

        radius = float(radius)
        start_angle = float(start_angle)
        end_angle = float(end_angle)
        if not np.isfinite(radius) or radius <= 0.:
            raise ValueError(f'radius must be positive and finite, got {radius}.')
        if not np.isfinite(start_angle) or not np.isfinite(end_angle):
            raise ValueError('start_angle and end_angle must be finite.')

        self._r: float = radius
        # make sure that angles are in [0, 2pi]
        self._start_angle: float = float(mod_2pi(start_angle))
        self._end_angle: float = float(mod_2pi(end_angle))
        self._sweep_dir: bool = bool(sweep_dir)
        self._center: Vector = Vector(center) if center is not None else Vector(0, 0)

        if not np.all(np.isfinite(np.asarray(self._center))):
            raise ValueError('center must be finite.')

    # construction

    @classmethod
    def from_bulge(cls, start: VectorLike, end: VectorLike, bulge: float) -> Arc:
        """
        Construct a curve from start and end points and bulge value.
        See `here <http://www.lee-mac.com/bulgeconversion.html>`_ and `there
        <https://ezdxf.readthedocs.io/en/stable/dxfentities/lwpolyline.html#bulge-value>`_ for details concerning the
        bulge value.

        Args:
            start (VectorLike): start point
            end (VectorLike): end point
            bulge (float): bulge value, positive for counterclockwise arcs (|bulge| > 1: more than a half circle)

        Returns:
            Arc

        Raises:
            ValueError: Raised if start and end are equal, the bulge is zero (a line) or not finite.
        """
        start = Vector(start)
        end = Vector(end)
        bulge = float(bulge)

        if not np.isfinite(bulge) or bulge == 0.:
            raise ValueError('bulge must be finite and nonzero.')

        chord = end - start
        chord_length = chord.mag
        if chord_length == 0.:
            raise ValueError('start and end must be different points.')

        # distance of the center from the midpoint of the chord along the left normal of the chord
        # (negative if the center lies on the right hand side)
        center = (start + end) / 2 + Vector(-chord.y, chord.x) / chord_length * (
            chord_length * (1. - bulge ** 2) / (4. * bulge)
        )
        radius = chord_length * (1. + bulge ** 2) / (4. * abs(bulge))

        return cls(radius, (start - center).angle_about_x_axis, (end - center).angle_about_x_axis, bulge > 0., center)

    @classmethod
    def from_points(cls, p1: VectorLike, p2: VectorLike, p3: VectorLike) -> Arc:  # pylint: disable=invalid-name
        """Creates an arc connecting `p1` with `p3` via `p2`.

        Args:
            p1 (VectorLike): start point
            p2 (VectorLike): intermediate point
            p3 (VectorLike): end point

        Returns:
            Arc

        Raises:
            ValueError: Raised if two points are equal or the points are collinear.
        """
        p1 = Vector(p1)
        p2 = Vector(p2)
        p3 = Vector(p3)

        if p1 == p2 or p1 == p3 or p2 == p3:
            raise ValueError('Arc through points is only defined for three different points.')

        # circumcenter of the triangle
        a_vec, b_vec = p2 - p1, p3 - p1
        det = 2. * a_vec.cross(b_vec)
        if abs(det) <= 1e-12 * max(a_vec.dot(a_vec), b_vec.dot(b_vec)):
            raise ValueError('Arc through points is only defined for points which are not collinear.')

        center = p1 + Vector(
            b_vec.y * a_vec.dot(a_vec) - a_vec.y * b_vec.dot(b_vec),
            a_vec.x * b_vec.dot(b_vec) - b_vec.x * a_vec.dot(a_vec),
        ) / det

        # p1 -> p2 -> p3 is counterclockwise on the circle if the triangle is counterclockwise
        return cls.from_points_center(p1, p3, center, bool((p2 - p1).cross(p3 - p2) > 0.))

    @classmethod
    def from_points_center(cls, start: VectorLike, end: VectorLike, center: VectorLike, sweep_dir: bool) -> Arc:
        """Create Arc from start, end, center and sweep_dir

        Args:
            start (VectorLike): start point
            end (VectorLike): end point (must have the same distance to `center` as `start`)
            center (VectorLike): center
            sweep_dir (bool): sweep_dir

        Returns:
            Arc

        Raises:
            ValueError: Raised if start or end coincide with the center or their distances to the center differ.
        """
        start = Vector(start)
        end = Vector(end)
        center = Vector(center)

        radius = (start - center).mag
        if radius == 0. or not np.isclose((end - center).mag, radius):
            raise ValueError('start and end must have the same nonzero distance to center.')

        return cls(
            radius=radius,
            start_angle=(start - center).angle_about_x_axis,
            end_angle=(end - center).angle_about_x_axis,
            sweep_dir=sweep_dir,
            center=center,
        )

    @classmethod
    def from_points_center_tangent(
        cls,
        start: VectorLike,
        end: VectorLike,
        center: VectorLike,
        *,
        unit_tangent_start: t.Optional[VectorLike] = None,
        unit_tangent_end: t.Optional[VectorLike] = None,
    ) -> Arc:
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
            ValueError: Raised if none or both of unit_tangent_start and unit_tangent_end are defined or start and end
                have different distances to the center.
            RuntimeError: Raised if the tangent is not a tangent of the arc.
        """
        start = Vector(start)
        end = Vector(end)
        center = Vector(center)

        if unit_tangent_start is not None and unit_tangent_end is None:
            unit_tangent_start = Vector(unit_tangent_start)
            sweep_dir = bool((start + unit_tangent_start - start).cross(end - start) > 0.)
        elif unit_tangent_start is None and unit_tangent_end is not None:
            unit_tangent_end = Vector(unit_tangent_end)
            sweep_dir = bool((end - start).cross(unit_tangent_end) > 0.)
        elif unit_tangent_start is None and unit_tangent_end is None:
            raise ValueError('Anyone of unit_tangent_start or unit_tangent_end must be defined.')
        else:
            raise ValueError('unit_tangent_start and unit_tangent_end cannot be defined both.')

        arc = cls.from_points_center(start, end, center, sweep_dir)

        if unit_tangent_start is not None:
            if not np.allclose(arc.unit_tangent_start, unit_tangent_start):
                raise RuntimeError(
                    f'unit_tangent_start {unit_tangent_start} is not the tangent of the arc at its start '
                    f'({arc.unit_tangent_start}).'
                )
        else:
            if not np.allclose(arc.unit_tangent_end, unit_tangent_end):
                raise RuntimeError(
                    f'unit_tangent_end {unit_tangent_end} is not the tangent of the arc at its end '
                    f'({arc.unit_tangent_end}).'
                )

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

    # properties

    @property
    def start(self) -> Vector:
        """Start point of arc

        Access:
            get

        Returns:
            Vector
        """
        return Vector(r=self._r, phi=self._start_angle) + self._center

    @property
    def end(self) -> Vector:
        """End point of arc

         Access:
             get

         Returns:
             Vector
         """
        return Vector(r=self._r, phi=self._end_angle) + self._center

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
    def end_angle(self) -> float:
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
        """Absolute value of the enclosed angle (in (0, 2pi]). Arcs with equal start and end angle are full circles.

        Access:
            get

        Returns:
            float
        """
        delta = (self._end_angle - self._start_angle) if self._sweep_dir else (self._start_angle - self._end_angle)
        angle = delta % (2 * np.pi)

        if angle < _ANGLE_TOL or angle > 2 * np.pi - _ANGLE_TOL:
            return 2 * np.pi
        return float(angle)

    @property
    def bulge(self) -> float:
        """Bulge value = tan(theta/4). bulge > 0 arc goes in math. positive direction and for bulge < 0 in math.
        neg. direction

        Access:
            get

        Returns:
            float
        """
        sign = 1. if self._sweep_dir else -1.
        return float(sign * np.tan(self.theta / 4.))

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
        return self.start.rotated(sign * self.theta / 2, origin=self._center)

    @property
    def unit_tangent_start(self) -> Vector:
        """Unit tangent at start.

        Access:
            get

        Returns:
            Vector
        """
        return self._unit_tangent_at_angle(self._start_angle)

    @property
    def unit_tangent_end(self) -> Vector:
        """Unit tangent at end.

        Access:
            get

        Returns:
            Vector
        """
        return self._unit_tangent_at_angle(self._end_angle)

    def _unit_tangent_at_angle(self, phi: float) -> Vector:
        sign = 1. if self._sweep_dir else -1.
        return sign * Vector(-np.sin(phi), np.cos(phi))

    def unit_tangent_at(self, angle: float) -> Vector:
        """Unit tangent at specific angle.

        Args:
            angle (float): 0 <= angle <= self.theta, measured from the start of the arc in the sweep direction

        Returns:
            Vector

        Raises:
            ValueError: Raised if angle < 0 or angle > self.theta
        """
        if angle < 0 or angle > self.theta:
            raise ValueError('angle < 0 or angle > self.theta')

        sign = 1. if self._sweep_dir else -1.
        return self._unit_tangent_at_angle(self._start_angle + sign * angle)

    @property
    def length(self) -> float:
        """Length of arc.

        Access:
            get

        Returns:
            float
        """
        return self.radius * self.theta

    @property
    def boundary_length(self) -> float:
        """Length of the arc. Same as :attr:`Arc.length`.

        Access:
            get

        Returns:
            float
        """
        return self.length

    @property
    def area(self) -> float:
        """Area of the circle. Only defined for closed arcs (full circles).

        Access:
            get

        Returns:
            float

        Raises:
            NotImplementedError: Raised if the arc is not closed.
        """
        if not self.is_closed:
            raise NotImplementedError('The area is only defined for closed arcs (full circles).')
        return float(np.pi * self._r ** 2)

    @property
    def is_closed(self) -> bool:
        """True if the arc is a full circle.

        Access:
            get

        Returns:
            bool
        """
        return bool(np.isclose(self.theta, 2 * np.pi))

    # splitting and reversing

    def _with_angles(self, start_angle: float, end_angle: float) -> Arc:
        arc = self.clone()
        arc._start_angle = float(mod_2pi(start_angle))  # pylint: disable=protected-access
        arc._end_angle = float(mod_2pi(end_angle))  # pylint: disable=protected-access
        return arc

    def split(self) -> t.List[Arc]:
        """Return two arcs if theta > np.pi (hence the abs(bulge) value of new arcs will be smaller than 1).

        Returns:
            List[Arc]: if theta > np.pi list contains two elements and one otherwise.
        """
        theta = self.theta
        if theta > np.pi:
            return self.split_at(theta / 2)

        return [self.clone()]

    def split_at(self, angle: float) -> t.List[Arc]:
        """Split the arc in two parts ([0, angle], [angle, theta]).

        Args:
            angle (float): split angle, measured from the start of the arc in the sweep direction.

        Returns:
            List[Arc]

        Raises:
            ValueError: Raised if angle <= 0 or angle >= self.theta (the parts would be empty; an empty part would be
                interpreted as full circle).
        """
        if angle <= 0 or angle >= self.theta:
            raise ValueError('The split angle must be larger than 0 and smaller than self.theta.')

        sign = 1. if self._sweep_dir else -1.
        split_angle = self._start_angle + sign * angle

        return [
            self._with_angles(self._start_angle, split_angle),
            self._with_angles(split_angle, self._end_angle),
        ]

    def reversed(self) -> Arc:
        """Return a reversed version of the arc

        Returns:
            Arc
        """
        reversed_arc = self._with_angles(self._end_angle, self._start_angle)
        reversed_arc._sweep_dir = not self._sweep_dir  # pylint: disable=protected-access
        return reversed_arc

    def __repr__(self) -> str:
        return 'Arc(r={!r}, start_angle={!r}, end_angle={!r}, sweep_dir={!r}, center={!r})'.format(
            self.radius, self.start_angle, self.end_angle, self.sweep_dir, self.center
        )

    @property
    def bounding_box(self) -> BoundingBox:
        # https://stackoverflow.com/questions/1336663/2d-bounding-box-of-a-sector
        theta = self.theta

        if theta >= 2 * np.pi - _ANGLE_TOL:
            r_vec = Vector(self._r, self._r)
            return BoundingBox(self._center - r_vec, self._center + r_vec)

        sign = 1. if self._sweep_dir else -1.
        points = [self.start, self.end]

        # the extreme points are located at the axis angles which are passed by the arc
        for axis_angle in (0., np.pi / 2, np.pi, 3 * np.pi / 2):
            if (axis_angle - self._start_angle) * sign % (2 * np.pi) <= theta:
                points.append(self._center + Vector(r=self._r, phi=axis_angle))

        return BoundingBox.from_points(points)

    # transformations

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._center = self._center + Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        theta = float(theta)

        self._center = self._center.rotated(theta)
        self._start_angle = float(mod_2pi(self._start_angle + theta))
        self._end_angle = float(mod_2pi(self._end_angle + theta))

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)

        self._center = self._center * fac
        self._r *= abs(fac)

        if fac < 0.:
            # a negative factor is a scaling with abs(fac) followed by a rotation by pi
            self._start_angle = float(mod_2pi(self._start_angle + np.pi))
            self._end_angle = float(mod_2pi(self._end_angle + np.pi))

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        mirror_axis = Vector(mirror_axis)
        axis_angle = mirror_axis.phi

        # a reflection on a line with angle `axis_angle` maps the angle phi to 2 * axis_angle - phi and reverses the
        # direction of rotation
        self._center = self._center.mirrored(mirror_axis)
        self._start_angle = float(mod_2pi(2 * axis_angle - self._start_angle))
        self._end_angle = float(mod_2pi(2 * axis_angle - self._end_angle))
        self._sweep_dir = not self._sweep_dir
