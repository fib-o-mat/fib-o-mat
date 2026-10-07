"""Provide smoothing (rounding of corners) of arc splines.

Example:
    >>> import math
    >>> from fibomat.shapes import Rect
    >>> from fibomat.curve_tools import smooth
    >>> square = Rect(2, 2).to_arc_spline()
    >>> rounded = smooth(square, 0.5)
    >>> len(rounded.kinks())
    0
    >>> round(rounded.area, 6) == round(4 - (4 - math.pi) * 0.25, 6)
    True
"""
from __future__ import annotations

import math
import typing as t

import numpy as np

from fibomat.shapes.arc_spline import ArcSpline


__all__ = ['smooth', 'NonSmoothableError']


class NonSmoothableError(RuntimeError):
    """Raised if a kink of a curve cannot be replaced by an arc of the requested radius.

    This happens, e.g., if the radius is too large for the adjacent segments or if the curve turns back on itself.
    """


_NEWTON_ITERATIONS = 100
_TOLERANCE = 1e-12
"""Relative tolerance."""


class _Segment:
    """A line or an arc of a curve, parametrized by arc length."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, start: np.ndarray, end: np.ndarray, bulge: float):
        """
        Args:
            start (np.ndarray): start point
            end (np.ndarray): end point
            bulge (float): bulge value (0 for lines)
        """
        self.start = start
        self.end = end
        self.bulge = bulge

        chord_vec = end - start
        chord = float(math.hypot(chord_vec[0], chord_vec[1]))
        if chord == 0.:
            raise NonSmoothableError('The curve contains a segment without length.')

        self.is_arc = bulge != 0.
        if self.is_arc:
            sweep = 4. * math.atan(bulge)
            self.direction = 1. if sweep > 0. else -1.
            self.radius = chord / (2. * abs(math.sin(sweep / 2.)))
            self.length = self.radius * abs(sweep)
            normal = np.array((-chord_vec[1], chord_vec[0])) / chord
            self.center = (start + end) / 2. + normal * (chord / 2. / math.tan(sweep / 2.))
            self.start_angle = math.atan2(start[1] - self.center[1], start[0] - self.center[0])
            self.curvature = self.direction / self.radius
        else:
            self.direction = 0.
            self.radius = math.inf
            self.length = chord
            self.unit = chord_vec / chord
            self.curvature = 0.

    def angle(self, s: float) -> float:
        """Polar angle (about the center) of the point at arc length `s` (arcs only)."""
        return self.start_angle + self.direction * s / self.radius

    def point(self, s: float) -> np.ndarray:
        """Point at arc length `s`."""
        if not self.is_arc:
            return self.start + self.unit * s
        angle = self.angle(s)
        return self.center + self.radius * np.array((math.cos(angle), math.sin(angle)))

    def tangent(self, s: float) -> np.ndarray:
        """Unit tangent at arc length `s`."""
        if not self.is_arc:
            return self.unit
        angle = self.angle(s)
        return self.direction * np.array((-math.sin(angle), math.cos(angle)))

    def trimmed_bulge(self, trim_start: float, trim_end: float) -> float:
        """Bulge value of the segment after cutting `trim_start` from its start and `trim_end` from its end."""
        if not self.is_arc:
            return 0.
        remaining = max(self.length - trim_start - trim_end, 0.)
        return math.tan(math.atan(self.bulge) * remaining / self.length)


def _left_normal(tangent: np.ndarray) -> np.ndarray:
    return np.array((-tangent[1], tangent[0]))


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


class _Fillet(t.NamedTuple):
    """Arc replacing a kink."""

    trim_incoming: float
    """Length cut from the end of the incoming segment."""
    trim_outgoing: float
    """Length cut from the start of the outgoing segment."""
    start: np.ndarray
    """Start point (on the incoming segment)."""
    end: np.ndarray
    """End point (on the outgoing segment)."""
    bulge: float
    """Bulge value of the fillet arc."""


def _make_fillet(incoming: _Segment, outgoing: _Segment, radius: float) -> _Fillet:
    """Compute the fillet of radius `radius` which is tangent to two segments meeting at a kink.

    The center of the fillet has the distance `radius` to both segments, hence, it is a point of both offset
    curves ``P(s) + side * radius * N(s)`` (N: left normal). The intersection of these offset curves is found with
    Newton's method, starting at the solution for two lines.

    Args:
        incoming (_Segment): segment ending at the kink
        outgoing (_Segment): segment starting at the kink
        radius (float): fillet radius

    Returns:
        _Fillet

    Raises:
        NonSmoothableError: Raised if there is no fillet or it does not fit on the segments.
    """
    # pylint: disable=too-many-locals
    tangent_in = incoming.tangent(incoming.length)
    tangent_out = outgoing.tangent(0.)

    turn = math.atan2(_cross(tangent_in, tangent_out), float(np.dot(tangent_in, tangent_out)))
    if abs(turn) > math.pi - 1e-6:
        raise NonSmoothableError('Cannot smooth a cusp.')

    # the center of the fillet is on the inner side of the corner
    side = 1. if turn > 0. else -1.

    guess = radius * math.tan(abs(turn) / 2.)
    s_in = incoming.length - guess
    s_out = guess

    scale = max(incoming.length, outgoing.length, radius)

    for _ in range(_NEWTON_ITERATIONS):
        residual = (
            incoming.point(s_in) + side * radius * _left_normal(incoming.tangent(s_in))
            - outgoing.point(s_out) - side * radius * _left_normal(outgoing.tangent(s_out))
        )
        if np.linalg.norm(residual) < _TOLERANCE * scale:
            break

        speed_in = 1. - side * radius * incoming.curvature
        speed_out = 1. - side * radius * outgoing.curvature
        jacobian = np.column_stack((speed_in * incoming.tangent(s_in), -speed_out * outgoing.tangent(s_out)))
        if abs(np.linalg.det(jacobian)) < 1e-14:
            raise NonSmoothableError('The fillet is not well defined (radius equals a radius of curvature).')

        step = np.linalg.solve(jacobian, -residual)
        s_in += step[0]
        s_out += step[1]
    else:
        raise NonSmoothableError('Could not find a fillet.')

    trim_in = incoming.length - s_in
    trim_out = s_out

    eps = 1e-9 * scale
    if trim_in < eps or trim_out < eps or trim_in > incoming.length + eps or trim_out > outgoing.length + eps:
        raise NonSmoothableError('The fillet does not fit on the segments, the radius is too large.')
    trim_in = min(trim_in, incoming.length)
    trim_out = min(trim_out, outgoing.length)

    start = incoming.point(incoming.length - trim_in)
    end = outgoing.point(trim_out)

    # signed angle swept by the fillet (positive: counterclockwise)
    fillet_turn = math.atan2(
        _cross(incoming.tangent(incoming.length - trim_in), outgoing.tangent(trim_out)),
        float(np.dot(incoming.tangent(incoming.length - trim_in), outgoing.tangent(trim_out)))
    )
    if fillet_turn * side <= 0.:
        raise NonSmoothableError('The fillet turns in the wrong direction, the radius is too large.')

    return _Fillet(trim_in, trim_out, start, end, math.tan(fillet_turn / 4.))


def smooth(arc_spline: ArcSpline, radius: float) -> ArcSpline:
    """Replace all kinks of an arc spline by tangent arcs of radius `radius` (fillets).

    The arcs touch both adjacent segments tangentially, hence, the result is continuously differentiable.

    Args:
        arc_spline (ArcSpline): curve to be smoothed
        radius (float): radius of the new arcs (greater than 0)

    Returns:
        ArcSpline: smoothed curve (`arc_spline` itself if it does not contain kinks)

    Raises:
        TypeError: Raised if arc_spline is not an ArcSpline.
        ValueError: Raised if radius is not positive and finite.
        NonSmoothableError: Raised if a kink cannot be smoothed, e.g. because the radius is too large for the
            adjacent segments, two fillets would overlap or the curve has a cusp.
    """
    # pylint: disable=too-many-locals
    if not isinstance(arc_spline, ArcSpline):
        raise TypeError(f'arc_spline must be an ArcSpline, got {type(arc_spline).__name__}.')

    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.:
        raise ValueError(f'radius must be positive and finite, got {radius}.')

    kinks = arc_spline.kinks()
    if not kinks:
        return arc_spline

    vertices = arc_spline.vertices
    n_vertices = len(vertices)
    n_segments = n_vertices if arc_spline.is_closed else n_vertices - 1

    segments = [
        _Segment(vertices[i, :2], vertices[(i + 1) % n_vertices, :2], float(vertices[i, 2]))
        for i in range(n_segments)
    ]

    # fillet of the kink at vertex i (between the segments i - 1 and i)
    fillets = {
        i_kink: _make_fillet(segments[(i_kink - 1) % n_segments], segments[i_kink], radius) for i_kink in kinks
    }

    trims_start = [0.] * n_segments
    trims_end = [0.] * n_segments
    for i_kink, fillet in fillets.items():
        trims_end[(i_kink - 1) % n_segments] = fillet.trim_incoming
        trims_start[i_kink] = fillet.trim_outgoing

    for segment, trim_start, trim_end in zip(segments, trims_start, trims_end):
        if trim_start + trim_end > segment.length * (1. + 1e-9):
            raise NonSmoothableError('Neighbouring fillets overlap, the radius is too large.')

    new_vertices: t.List[t.Tuple[float, float, float]] = []
    for i_segment, segment in enumerate(segments):
        remaining = segment.length - trims_start[i_segment] - trims_end[i_segment]

        # a segment which is completely used by the fillets vanishes
        if remaining > 1e-9 * segment.length:
            start = segment.point(trims_start[i_segment])
            new_vertices.append(
                (start[0], start[1], segment.trimmed_bulge(trims_start[i_segment], trims_end[i_segment]))
            )

        i_end_vertex = (i_segment + 1) % n_vertices
        if i_end_vertex in fillets:
            fillet = fillets[i_end_vertex]
            new_vertices.append((fillet.start[0], fillet.start[1], fillet.bulge))

    if not arc_spline.is_closed:
        end = vertices[-1, :2]
        if (n_vertices - 1) in fillets:
            raise NonSmoothableError('Unexpected kink at the end of an open curve.')
        new_vertices.append((end[0], end[1], 0.))

    return ArcSpline(np.array(new_vertices), arc_spline.is_closed)
