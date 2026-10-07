"""Provide rasterization routines.

Example:
    >>> from fibomat.shapes import Line
    >>> from fibomat.curve_tools import rasterize
    >>> rasterize(Line((0, 0), (1, 0)), 0.25).positions[:, 0].tolist()
    [0.0, 0.25, 0.5, 0.75, 1.0]
"""
from __future__ import annotations

import math
import typing as t

import numpy as np

from fibomat import _libfibomat
from fibomat.shapes._line_non_continuous import LineNonContinuous
from fibomat.shapes.arc_spline import ArcSpline, ArcSplineCompatible
from fibomat.shapes.parametric_curve import ParametricCurve
from fibomat.shapes.polygon import Polygon
from fibomat.shapes.polyline import Polyline
from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.shapes.shape import Shape


__all__ = ['rasterize', 'rasterize_with_const_error']


_RELATIVE_TOLERANCE = 1e-9
"""Relative tolerance (with respect to the pitch) to decide if a point lies on the end of a curve."""


def _check_pitch(pitch: float) -> float:
    """Check that the pitch is positive and finite.

    Args:
        pitch (float): pitch

    Returns:
        float: pitch

    Raises:
        ValueError: Raised if pitch is not positive or not finite.
    """
    pitch = float(pitch)
    if not math.isfinite(pitch) or pitch <= 0.:
        raise ValueError(f'pitch must be positive and finite, got {pitch}.')
    return pitch


def _segment_arrays(curve: t.Union[ArcSpline, LineNonContinuous]) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start points, end points and bulge values of all segments of a curve.

    Args:
        curve (ArcSpline, LineNonContinuous): curve

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: start points (n, 2), end points (n, 2), bulges (n,)
    """
    if isinstance(curve, LineNonContinuous):
        starts = np.array([np.asarray(line.start, dtype=float) for line in curve.segments]).reshape(-1, 2)
        ends = np.array([np.asarray(line.end, dtype=float) for line in curve.segments]).reshape(-1, 2)
        return starts, ends, np.zeros(len(starts))

    vertices = curve.vertices
    if curve.is_closed:
        return vertices[:, :2], np.roll(vertices[:, :2], -1, axis=0), vertices[:, 2]
    return vertices[:-1, :2], vertices[1:, :2], vertices[:-1, 2]


def _rasterize_segments(
    starts: np.ndarray, ends: np.ndarray, bulges: np.ndarray, pitch: float, is_closed: bool
) -> np.ndarray:
    """Points with equal arc length distance on a chain of lines and arcs.

    The chain is parametrized by the arc length of the concatenated segments (gaps between segments are ignored). The
    first point is the start of the first segment. The end of the chain is only included if its arc length is a
    multiple of the pitch. In the case of a closed chain, this point would coincide with the first point and is
    dropped.

    Args:
        starts (np.ndarray): start points of the segments, shape (n, 2)
        ends (np.ndarray): end points of the segments, shape (n, 2)
        bulges (np.ndarray): bulge values of the segments, shape (n,)
        pitch (float): arc length between the points
        is_closed (bool): True if the chain is closed

    Returns:
        np.ndarray: points with shape (m, 2)

    Raises:
        ValueError: Raised if the chain has no length.
    """
    chord_vec = ends - starts
    chord = np.hypot(chord_vec[:, 0], chord_vec[:, 1])

    keep = chord > 0.
    starts, ends, bulges, chord_vec, chord = starts[keep], ends[keep], bulges[keep], chord_vec[keep], chord[keep]
    if not len(chord):
        raise ValueError('Cannot rasterize a curve without length.')

    is_arc = bulges != 0.
    # bulge = tan(sweep / 4)
    sweep = 4. * np.arctan(bulges)
    half_sweep = sweep / 2.

    with np.errstate(divide='ignore', invalid='ignore'):
        radius = np.where(is_arc, chord / (2. * np.abs(np.sin(half_sweep))), 0.)
        # signed distance of the center from the chord midpoint (positive to the left of the chord)
        center_distance = np.where(is_arc, chord / 2. / np.tan(half_sweep), 0.)
    lengths = np.where(is_arc, radius * np.abs(sweep), chord)

    normals = np.column_stack((-chord_vec[:, 1], chord_vec[:, 0])) / chord[:, np.newaxis]
    centers = (starts + ends) / 2. + normals * center_distance[:, np.newaxis]
    start_angles = np.arctan2(starts[:, 1] - centers[:, 1], starts[:, 0] - centers[:, 0])

    cumulative = np.concatenate(([0.], np.cumsum(lengths)))
    total = cumulative[-1]

    tolerance = _RELATIVE_TOLERANCE * pitch
    n_points = int(np.floor((total + tolerance) / pitch)) + 1
    arc_lengths = np.arange(n_points) * pitch
    if is_closed and n_points > 1 and total - arc_lengths[-1] < tolerance:
        arc_lengths = arc_lengths[:-1]

    # a point between two segments belongs to the end of the first one (matters if the chain is not continuous)
    index = np.clip(np.searchsorted(cumulative, arc_lengths, side='left') - 1, 0, len(lengths) - 1)
    local = np.clip(arc_lengths - cumulative[index], 0., lengths[index])

    directions = chord_vec[index] / chord[index, np.newaxis]
    points = starts[index] + directions * local[:, np.newaxis]

    arcs = is_arc[index]
    if np.any(arcs):
        i_arc = index[arcs]
        angles = start_angles[i_arc] + np.sign(sweep[i_arc]) * local[arcs] / radius[i_arc]
        points[arcs] = centers[i_arc] + radius[i_arc, np.newaxis] * np.column_stack((np.cos(angles), np.sin(angles)))

    return points


def _with_unit_dwell_time(points: np.ndarray, is_closed: bool) -> RasterizedPoints:
    """Create RasterizedPoints with dwell time multiplicand 1.

    Args:
        points (np.ndarray): positions, shape (n, 2)
        is_closed (bool): True if the points are the points of a closed curve

    Returns:
        RasterizedPoints
    """
    dwell_points = np.ones((len(points), 3))
    dwell_points[:, :2] = points
    return RasterizedPoints(dwell_points, is_closed)


def rasterize(curve: Shape, pitch: float) -> RasterizedPoints:
    """Rasterize the outline of a shape with a given pitch uniformly.

    The distance of two consecutive points, measured along the curve, is `pitch`. The first point is the start of the
    curve. The end of an open curve is only included if its distance to the start is a multiple of the pitch. A closed
    curve does not contain its start point twice.

    The shape must be an :class:`~fibomat.shapes.arc_spline.ArcSpline`, a
    :class:`~fibomat.shapes.parametric_curve.ParametricCurve` or convertible to an ArcSpline (i.e. provide
    ``to_arc_spline()``).

    Args:
        curve (Shape): curve
        pitch (float): pitch (spacing of points)

    Returns:
        RasterizedPoints

    Raises:
        ValueError: Raised if the pitch is not positive and finite or the curve has no length.
        TypeError: Raised if the shape cannot be rasterized.
    """
    pitch = _check_pitch(pitch)

    if isinstance(curve, ParametricCurve):
        points = curve.rasterize_at(pitch)
        is_closed = curve.is_closed
        if is_closed and len(points) > 1 and np.linalg.norm(points[-1] - points[0]) < _RELATIVE_TOLERANCE * pitch:
            points = points[:-1]
        return _with_unit_dwell_time(points, is_closed)

    if isinstance(curve, LineNonContinuous):
        return _with_unit_dwell_time(_rasterize_segments(*_segment_arrays(curve), pitch, False), False)

    if not isinstance(curve, ArcSpline):
        if isinstance(curve, ArcSplineCompatible):
            curve = curve.to_arc_spline()
        else:
            raise TypeError(f'Cannot rasterize the passed object of type {type(curve).__name__}.')

    return _with_unit_dwell_time(
        _rasterize_segments(*_segment_arrays(curve), pitch, curve.is_closed), curve.is_closed
    )


def rasterize_with_const_error(curve: ArcSpline, error: float) -> t.Union[Polyline, Polygon]:
    """Convert an arc spline to a polygon or polyline.

    Args:
        curve (ArcSpline): arc spline
        error (float): maximum distance between the curve and the returned polyline/polygon (greater than 0).

    Returns:
        Polygon if arc spline is closed and polyline otherwise

    Raises:
        TypeError: Raised if curve is not an ArcSpline.
        ValueError: Raised if error is not positive and finite.
    """
    if not isinstance(curve, ArcSpline):
        raise TypeError(f'curve must be an ArcSpline, got {type(curve).__name__}.')

    error = float(error)
    if not math.isfinite(error) or error <= 0.:
        raise ValueError(f'error must be positive and finite, got {error}.')

    approx = ArcSpline(_libfibomat.convert_arcs_to_lines(curve.arc_spline_impl, error))

    if curve.is_closed:
        return Polygon(approx.vertices[:, :2])
    return Polyline(approx.vertices[:, :2])
