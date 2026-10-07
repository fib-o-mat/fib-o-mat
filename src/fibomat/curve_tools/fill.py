"""Provide routines to fill closed shapes with lines or a spiral.

Example:
    >>> from fibomat.shapes import Rect
    >>> from fibomat.curve_tools import fill_with_lines
    >>> rows = fill_with_lines(Rect(2, 2).to_arc_spline(), pitch=0.5, alpha=0., invert=False)
    >>> [row[0].start.y for row in rows]  # the scan line on the top edge of the rectangle is not part of the fill
    [0.5, 0.0, -0.5, -1.0]
"""
from __future__ import annotations

import math
import typing as t

import numpy as np

from fibomat.linalg import Vector, VectorLike
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.line import Line
from fibomat.shapes.parametric_curve import ParametricCurve

if t.TYPE_CHECKING:  # pragma: no cover
    # (the composite shapes are imported lazily, they depend on this package)
    from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline


__all__ = ['fill_with_lines', 'fill_with_spiral']


def _boundary_and_holes(
    shape: t.Union[ArcSpline, HollowArcSpline]
) -> t.Tuple[ArcSpline, t.List[ArcSpline]]:
    """Split a shape into its boundary and its holes.

    Args:
        shape (ArcSpline, HollowArcSpline): closed shape

    Returns:
        Tuple[ArcSpline, List[ArcSpline]]: boundary and holes

    Raises:
        TypeError: Raised if the shape has the wrong type.
        ValueError: Raised if an ArcSpline is not closed.
    """
    from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline  # pylint: disable=import-outside-toplevel

    if isinstance(shape, ArcSpline):
        if not shape.is_closed:
            raise ValueError('ArcSpline is not closed.')
        return shape, []
    if isinstance(shape, HollowArcSpline):
        return shape.boundary, list(shape.holes)
    raise TypeError(f'Shape must be ArcSpline or HollowArcSpline (got {type(shape).__name__}).')


def _monotone_pieces(vertices: np.ndarray) -> t.Tuple[np.ndarray, ...]:
    """Split a closed curve into pieces which are monotone in y.

    Lines are already monotone and arcs are split at their top and bottom points. The y values of the ends of the
    pieces are exactly the y values of the vertices (and ``center_y +- radius`` at the split points), so neighbouring
    pieces always agree on the y value of their common end.

    Args:
        vertices (np.ndarray): vertices of a closed curve, shape (n, 3)

    Returns:
        Tuple[np.ndarray, ...]:
            For every piece: `y_low`, `y_high`, `is_arc`, and the parameters `x_0`, `y_0`, `x_1`, `y_1` of the line
            (unused for arcs) and `center_x`, `center_y`, `radius`, `x_sign` of the arc (unused for lines). `x_sign`
            is the side of the center of the arc on which the piece lies.
    """
    # pylint: disable=too-many-locals
    starts = vertices[:, :2]
    ends = np.roll(starts, -1, axis=0)

    pieces: t.List[t.Tuple[float, ...]] = []

    for start, end, bulge in zip(starts, ends, vertices[:, 2]):
        if bulge == 0.:
            if start[1] != end[1]:
                pieces.append((
                    min(start[1], end[1]), max(start[1], end[1]), 0., start[0], start[1], end[0], end[1], 0., 0., 0.,
                    0.
                ))
            continue

        chord_vec = end - start
        chord = math.hypot(chord_vec[0], chord_vec[1])
        if chord == 0.:
            continue

        sweep = 4. * math.atan(bulge)
        direction = 1. if sweep > 0. else -1.
        radius = chord / (2. * abs(math.sin(sweep / 2.)))
        distance = chord / 2. / math.tan(sweep / 2.)
        center_x = (start[0] + end[0]) / 2. - chord_vec[1] / chord * distance
        center_y = (start[1] + end[1]) / 2. + chord_vec[0] / chord * distance
        start_angle = math.atan2(start[1] - center_y, start[0] - center_x)

        # positions of the top (pi / 2) and bottom (-pi / 2) of the circle on the arc (as swept angle)
        splits = [0.]
        for extremum_angle in (math.pi / 2., -math.pi / 2.):
            swept = (direction * (extremum_angle - start_angle)) % (2. * math.pi)
            if 0. < swept < abs(sweep):
                splits.append(swept)
        splits.sort()
        splits.append(abs(sweep))

        def y_at(swept: float, i_split: int) -> float:
            if i_split == 0:
                return start[1]
            if i_split == len(splits) - 1:
                return end[1]
            return center_y + radius * math.sin(start_angle + direction * swept)

        for i_split in range(len(splits) - 1):
            y_a = y_at(splits[i_split], i_split)
            y_b = y_at(splits[i_split + 1], i_split + 1)
            middle_angle = start_angle + direction * (splits[i_split] + splits[i_split + 1]) / 2.
            if y_a == y_b:
                continue
            pieces.append((
                min(y_a, y_b), max(y_a, y_b), 1., 0., 0., 0., 0., center_x, center_y, radius,
                1. if math.cos(middle_angle) >= 0. else -1.
            ))

    if not pieces:
        return tuple(np.empty(0) for _ in range(11))
    return tuple(np.array(pieces).T)


def _crossings(
    closed_curves: t.Sequence[np.ndarray], pitch: float
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Crossings of the curves with the lines ``y = k * pitch`` (k integer).

    A line at height y crosses a monotone piece if ``y_low <= y < y_high`` (half-open rule). With this rule, a vertex
    on the line is counted correctly (a crossing for exactly one of the two adjacent pieces if the curve crosses the
    line and for none or both of them if it touches the line) and horizontal pieces are ignored consistently. The
    number of crossings of a closed curve with any line is therefore even.

    Args:
        closed_curves (Sequence[np.ndarray]): vertices (shape (n, 3)) of closed curves
        pitch (float): distance between the lines

    Returns:
        Tuple[np.ndarray, np.ndarray]: line indices k and x coordinates of the crossings
    """
    # pylint: disable=too-many-locals
    ks: t.List[np.ndarray] = []
    xs: t.List[np.ndarray] = []

    for vertices in closed_curves:
        (y_low, y_high, is_arc, x_0, y_0, x_1, y_1, center_x, center_y, radius, x_sign) = _monotone_pieces(vertices)
        if not len(y_low):
            continue

        k_min = np.floor(y_low / pitch).astype(np.int64) - 1
        k_max = np.ceil(y_high / pitch).astype(np.int64) + 1
        counts = k_max - k_min + 1

        piece = np.repeat(np.arange(len(y_low)), counts)
        first = np.concatenate(([0], np.cumsum(counts)[:-1]))
        k = k_min[piece] + np.arange(counts.sum()) - np.repeat(first, counts)
        y = k * pitch

        inside = (y_low[piece] <= y) & (y < y_high[piece])
        piece, k, y = piece[inside], k[inside], y[inside]

        x = np.empty(len(piece))
        arc = is_arc[piece] == 1.
        line = ~arc

        i_line = piece[line]
        x[line] = x_0[i_line] + (y[line] - y_0[i_line]) * (x_1[i_line] - x_0[i_line]) / (y_1[i_line] - y_0[i_line])

        i_arc = piece[arc]
        radicand = radius[i_arc] ** 2 - (y[arc] - center_y[i_arc]) ** 2
        x[arc] = center_x[i_arc] + x_sign[i_arc] * np.sqrt(np.maximum(radicand, 0.))

        ks.append(k)
        xs.append(x)

    if not ks:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return np.concatenate(ks), np.concatenate(xs)


def fill_with_lines(
    shape: t.Union[ArcSpline, HollowArcSpline],
    pitch: float,
    alpha: float,
    invert: bool,
    seed: t.Optional[VectorLike] = None
) -> t.List[t.List[Line]]:
    """Fill a closed shape with parallel lines.

    The lines are rotated by `alpha` with respect to the x-axis and have the distance `pitch`. One of the (infinite)
    lines passes through `seed`, hence, the lines are always aligned to the same grid, independently of `invert`.
    Every line is cut at the boundary of the shape and at the holes. A line which crosses the shape several times is
    split into several parts, which are combined to a row. A line is part of the fill if its height `y` (measured
    perpendicular to the lines) satisfies ``y_min <= y < y_max``; a line which lies exactly on the upper boundary of a
    shape is therefore not part of the fill, while a line on the lower boundary is.

    Each row contains the lines on one scan line, ordered along the direction ``(cos(alpha), sin(alpha))``. Every line
    is directed along that direction. Rows without lines are omitted.

    Args:
        shape (ArcSpline, HollowArcSpline): closed shape to be filled
        pitch (float): distance between lines
        alpha (float): rotation angle of the lines with respect to the x-axis in [-pi/2, pi/2].
        invert (bool): If False, the rows are ordered from the top (with respect to the rotated frame) to the bottom
            and from the bottom to the top otherwise.
        seed (VectorLike, optional): a point on one of the lines. Defaults to the center of the shape.

    Returns:
        List[List[Line]]: rows of lines

    Raises:
        TypeError: Raised if the shape is neither an ArcSpline nor a HollowArcSpline.
        ValueError: Raised if the shape is an open ArcSpline, the pitch is not positive and finite or alpha is not in
            [-pi/2, pi/2].
        RuntimeError: Raised if the number of crossings with a line is odd (the shape is not valid, e.g. its boundary
            intersects itself).
    """
    # pylint: disable=too-many-locals
    curve, holes = _boundary_and_holes(shape)

    pitch = float(pitch)
    if not math.isfinite(pitch) or pitch <= 0.:
        raise ValueError(f'pitch must be positive and finite, got {pitch}.')

    alpha = float(alpha)
    if not -math.pi / 2. <= alpha <= math.pi / 2.:
        raise ValueError('alpha must be in [-pi/2, pi/2].')

    origin = np.asarray(Vector(seed) if seed is not None else curve.center, dtype=float)

    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    to_frame = np.array([[cos_a, sin_a], [-sin_a, cos_a]])  # rotation by -alpha

    frame_curves = []
    for part in [curve] + holes:
        vertices = part.vertices
        vertices[:, :2] = (vertices[:, :2] - origin) @ to_frame.T
        frame_curves.append(vertices)

    k, x = _crossings(frame_curves, pitch)

    order = np.lexsort((x, k))
    k, x = k[order], x[order]

    rows: t.List[t.List[Line]] = []
    if len(k):
        row_starts = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
        row_ends = np.concatenate((row_starts[1:], [len(k)]))

        tolerance = 1e-9 * pitch

        for row_start, row_end in zip(row_starts, row_ends):
            if (row_end - row_start) % 2:
                raise RuntimeError('Odd number of crossings. The boundary of the shape is not valid.')

            y = k[row_start] * pitch
            row_x = x[row_start:row_end]

            row = []
            for x_start, x_end in zip(row_x[::2], row_x[1::2]):
                if x_end - x_start <= tolerance:
                    continue
                start = origin + np.array([x_start, y]) @ to_frame
                end = origin + np.array([x_end, y]) @ to_frame
                row.append(Line(start, end))

            if row:
                rows.append(row)

    # rows are generated from the bottom to the top
    if not invert:
        rows.reverse()

    return rows


def fill_with_spiral(shape: t.Union[ArcSpline, HollowArcSpline], pitch: float) -> ParametricCurve:
    """Fill the circumscribed circle of the bounding box of a shape with an Archimedean spiral.

    The spiral starts in the center of the bounding box and winds counterclockwise outwards. In polar coordinates it
    reads ``r = pitch * phi / (2 pi)``, hence, the distance of the arms is `pitch`. It ends where it leaves the
    circumscribed circle.

    Holes are not considered, i.e. `HollowArcSpline` is filled as its boundary.

    Args:
        shape (ArcSpline, HollowArcSpline): closed shape to be filled
        pitch (float): distance between spiral arms

    Returns:
        ParametricCurve: spiral

    Raises:
        TypeError: Raised if the shape is neither an ArcSpline nor a HollowArcSpline.
        ValueError: Raised if the shape is an open ArcSpline, the pitch is not positive and finite or the bounding
            box of the shape is a point.
    """
    curve, _ = _boundary_and_holes(shape)

    pitch = float(pitch)
    if not math.isfinite(pitch) or pitch <= 0.:
        raise ValueError(f'pitch must be positive and finite, got {pitch}.')

    bbox = curve.bounding_box
    center = np.asarray(bbox.center, dtype=float)
    radius = float(np.linalg.norm(np.asarray(bbox.upper_right, dtype=float) - center))
    if radius <= 0.:
        raise ValueError('The bounding box of the shape has no extent.')

    slope = pitch / (2. * math.pi)  # r = slope * phi
    phi_max = radius / slope

    def func(phi: np.ndarray) -> np.ndarray:
        return center + slope * np.stack((phi * np.cos(phi), phi * np.sin(phi)), axis=-1)

    def d_func(phi: np.ndarray) -> np.ndarray:
        return slope * np.stack((np.cos(phi) - phi * np.sin(phi), np.sin(phi) + phi * np.cos(phi)), axis=-1)

    def d2_func(phi: np.ndarray) -> np.ndarray:
        return slope * np.stack((-2. * np.sin(phi) - phi * np.cos(phi), 2. * np.cos(phi) - phi * np.sin(phi)), axis=-1)

    def d3_func(phi: np.ndarray) -> np.ndarray:
        return slope * np.stack((-3. * np.cos(phi) + phi * np.sin(phi), -3. * np.sin(phi) - phi * np.cos(phi)), axis=-1)

    def arc_length(phi_0: float, phi_1: float) -> float:
        def primitive(phi: float) -> float:
            return slope / 2. * (phi * math.sqrt(1. + phi ** 2) + math.asinh(phi))

        return primitive(phi_1) - primitive(phi_0)

    return ParametricCurve(
        func, d_func, d2_func, (0., phi_max), length=arc_length, d3_func=d3_func,
        description='Archimedean spiral'
    )
