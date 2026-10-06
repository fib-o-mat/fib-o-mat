"""Provides the :class:`Biarc` class.

A biarc is a curve of two circular arcs (or lines) which interpolates two points with given tangents. The curve is
continuously differentiable (G1), also in the joint of the two arcs.

Example::

    from fibomat.shapes import Biarc

    # S-shaped curve from (0, 0) with tangent (1, 0) to (3, 1) with tangent (1, 0)
    biarc = Biarc((0, 0), (3, 1), (1, 0), (1, 0))
    biarc.segments  # [Arc, Arc]
    biarc.to_arc_spline()

    # a line, if the points and tangents are collinear
    Biarc((0, 0), (3, 0), (1, 0), (1, 0)).segments  # [Line]

References:
    - https://www.ryanjuckett.com/biarc-interpolation/
    - https://ieeexplore.ieee.org/document/5390085
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.arc import Arc
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.line import Line
from fibomat.shapes.shape import Shape


__all__ = ['Biarc']


_REL_TOL = 1e-12
"""Relative tolerance for degenerate cases (parallel tangents, vanishing segments)."""


class Biarc(Shape, ArcSplineCompatible):
    """Biarc interpolating two points with given tangents.

    https://www.ryanjuckett.com/biarc-interpolation/
    """

    @classmethod
    def _make_biarc_seg(cls, start: Vector, end: Vector, tangent: Vector) -> t.Union[Arc, Line, None]:
        """Create an arc or line from ``start`` to ``end`` which has the tangent ``tangent`` at ``start``.

        Args:
            start (Vector): start
            end (Vector): end
            tangent (Vector): unit tangent at start (the direction in which the segment is left)

        Returns:
            Union[Arc, Line, None]: None is returned if start == end, Line is returned if (end - start) || tangent and
                                    Arc otherwise (the arc leaves `start` in direction of `tangent` and has a sweep of
                                    up to 2 pi)
        """
        # https://www.ryanjuckett.com/biarc-interpolation/
        normal = Vector(-tangent.y, tangent.x)
        start_to_end = end - start

        if start_to_end.mag <= _REL_TOL * max(start.mag, end.mag, 1e-300):
            return None

        denominator = 2 * normal.dot(start_to_end)

        if abs(denominator) <= _REL_TOL * start_to_end.mag:
            return Line(start, end)

        u = start_to_end.dot(start_to_end) / denominator

        # the center lies on the left hand side of the tangent if u > 0, hence, the arc is counterclockwise
        return Arc.from_points_center(start, end, start + u * normal, bool(u > 0))

    @classmethod
    def _make_biarc_segs(
        cls, p_1: Vector, p_2: Vector, t_1: Vector, t_2: Vector, joint: Vector
    ) -> t.List[t.Union[Arc, Line]]:
        """Create the segments of the biarc with the given joint (the point where the two arcs meet).

        Args:
            p_1 (Vector): start of biarc
            p_2 (Vector): end of biarc
            t_1 (Vector): unit tangent at start
            t_2 (Vector): unit tangent at end
            joint (Vector): joint of the arcs

        Returns:
            List[Union[Arc, Line]]: one or two segments
        """
        seg_1 = cls._make_biarc_seg(p_1, joint, t_1)

        # the second segment is built backwards from the end (leaving p_2 against the direction of t_2) and reversed
        seg_2 = cls._make_biarc_seg(p_2, joint, -t_2)
        if isinstance(seg_2, Arc):
            seg_2 = seg_2.reversed()
        elif isinstance(seg_2, Line):
            seg_2 = Line(seg_2.end, seg_2.start)

        return [seg for seg in (seg_1, seg_2) if seg is not None]

    def __init__(
        self,
        p_1: VectorLike,
        p_2: VectorLike,
        t_1: VectorLike,
        t_2: VectorLike,
        description: t.Optional[str] = None,
    ):
        """Interpolate a biarc to given points and tangents.

        For interpolation details see references.

        References:
            - https://www.ryanjuckett.com/biarc-interpolation/
            - https://ieeexplore.ieee.org/document/5390085

        Args:
            p_1 (VectorLike): start of biarc
            p_2 (VectorLike): end of biarc
            t_1 (VectorLike): tangent at start (the length is ignored)
            t_2 (VectorLike): tangent at end (the length is ignored)
            description (str, optional): optional description

        Raises:
            ValueError: Raised if p_1 == p_2 or one of the tangents is the null vector.
        """
        super().__init__(description)

        p_1 = Vector(p_1)
        p_2 = Vector(p_2)

        t_1 = Vector(t_1).normalized()
        t_2 = Vector(t_2).normalized()

        v = p_2 - p_1

        if v.mag == 0.:
            raise ValueError('p_1 == p_2. No biarc can be interpolated.')

        # d is the distance of the joint from the chord (see references)
        d_denominator = 2. * (1. - t_1.dot(t_2))

        if d_denominator <= 1e-14:
            # (nearly) equal tangents: the joint is the midpoint of the chord
            joint = (p_1 + p_2) / 2
        else:
            t = t_1 + t_2
            v_dot_t = v.dot(t)
            root = np.sqrt(v_dot_t * v_dot_t + d_denominator * v.dot(v))
            # d = (-v_dot_t + root) / d_denominator, written in a form without cancellation
            d = v.dot(v) / (v_dot_t + root) if v_dot_t >= 0. else (root - v_dot_t) / d_denominator

            joint = (p_1 + p_2 + d * (t_1 - t_2)) / 2

        segments = self._make_biarc_segs(p_1, p_2, t_1, t_2, joint)

        if not 1 <= len(segments) <= 2:
            raise RuntimeError('The biarc could not be constructed.')

        self._biarc = ArcSpline.from_segments(segments)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(start={self.start!r}, end={self.end!r}, segments={len(self.segments)})'

    @property
    def segments(self) -> t.List[t.Union[Arc, Line]]:
        """The arcs and lines of the biarc (arcs of more than a half circle are split in two arcs).

        Access:
            get
        """
        return self._biarc.segments  # type: ignore[return-value]

    def to_arc_spline(self) -> ArcSpline:
        return ArcSpline(self._biarc, description=self.description)

    @property
    def start(self) -> Vector:
        """Start point of the biarc.

        Access:
            get
        """
        return self._biarc.start

    @property
    def end(self) -> Vector:
        """End point of the biarc.

        Access:
            get
        """
        return self._biarc.end

    @property
    def is_closed(self) -> bool:
        return False

    @property
    def length(self) -> float:
        """Length of the biarc.

        Access:
            get
        """
        return self._biarc.length

    @property
    def boundary_length(self) -> float:
        """Length of the biarc. Same as :attr:`Biarc.length`.

        Access:
            get
        """
        return self.length

    @property
    def center(self) -> Vector:
        """Center of the bounding box of the biarc (same as the center of its arc spline).

        Access:
            get
        """
        return self._biarc.center

    @property
    def bounding_box(self) -> BoundingBox:
        return self._biarc.bounding_box

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._biarc._impl_translate(trans_vec)  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        self._biarc._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        self._biarc._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        self._biarc._impl_mirror(mirror_axis)  # pylint: disable=protected-access
