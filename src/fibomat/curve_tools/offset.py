"""Provide the offsetting functionality.

Example:
    >>> from fibomat.shapes import Rect
    >>> from fibomat.curve_tools import offset, deflate
    >>> square = Rect(4, 4).to_arc_spline()
    >>> offset(square, 1, 'inwards')[0].bounding_box.width
    2.0
    >>> len(deflate(square, 0.5))  # curves with side length 3, 2 and 1
    3
"""
import math
import typing as t

from fibomat import _libfibomat
from fibomat.curve_tools.modes import OffsetDirection
from fibomat.curve_tools.results import OffsetResult
from fibomat.shapes import ArcSpline


__all__ = ['offset', 'offset_with_islands', 'deflate', 'inflate', 'SAFETY']


SAFETY: int = 1000
"""Maximum number of steps of :func:`deflate` if no limit is given."""


def _check_delta(delta: float, name: str = 'delta') -> float:
    """Check that an offset distance is finite and not negative.

    Args:
        delta (float): distance
        name (str): name used in the error message

    Returns:
        float: the distance

    Raises:
        ValueError: Raised if the distance is negative or not finite.
    """
    delta = float(delta)
    if not math.isfinite(delta) or delta < 0.:
        raise ValueError(f'{name} must be finite and not negative, got {delta}.')
    return delta


def _signed_delta(curve: ArcSpline, delta: float, direction: t.Optional[t.Union[OffsetDirection, str]]) -> float:
    """Convert a distance and a direction into the signed distance of the native offset.

    The native offset moves a curve to the left (seen in the direction of the curve) for a positive distance.
    Closed curves with positive orientation (counterclockwise) are therefore moved inwards.

    Args:
        curve (ArcSpline): curve to be offset
        delta (float): not negative distance
        direction (OffsetDirection, str, optional): direction

    Returns:
        float: signed distance

    Raises:
        ValueError: Raised if the direction is missing or does not fit to the curve.
    """
    if direction is None:
        if curve.is_closed:
            raise ValueError('direction must be provided if the curve is closed.')
        return delta

    direction = OffsetDirection.parse(direction)

    if direction is OffsetDirection.LEFT:
        return delta
    if direction is OffsetDirection.RIGHT:
        return -delta

    if not curve.is_closed:
        raise ValueError(f'direction {direction.value!r} is only valid for closed curves, use "left" or "right".')

    inwards = direction is OffsetDirection.INWARDS
    return delta if inwards == bool(curve.orientation) else -delta


def offset(
    curve: ArcSpline, delta: float, direction: t.Optional[t.Union[OffsetDirection, str]] = None
) -> t.List[ArcSpline]:
    """Offset a curve by a distance.

    The distance is never negative, the side is chosen by `direction`. The result consists of several curves if the
    offset splits the curve and is empty if the curve vanishes (e.g. a circle which is deflated by more than its
    radius).

    ::

                         | left     | right    | inwards  | outwards |
        -----------------+----------+----------+----------+----------+
        open curve       | ok       | ok       | error    | error    |
        closed curve     | ok       | ok       | ok       | ok       |

    For closed curves, "left" and "right" refer to the direction of the curve, i.e. "left" means inwards if the
    orientation is positive (counterclockwise).

    Args:
        curve (ArcSpline): curve to be offset.
        delta (float): offset distance (not negative).
        direction (OffsetDirection, str, optional): offset direction. Optional for open curves (defaults to left),
            required for closed curves.

    Returns:
        List[ArcSpline]: offset curves

    Raises:
        TypeError: Raised if curve is not an ArcSpline.
        ValueError: Raised if delta is negative or not finite.
        ValueError: Raised if direction is missing for a closed curve or inwards/outwards is used for an open curve.
    """
    if not isinstance(curve, ArcSpline):
        raise TypeError(f'curve must be an ArcSpline, got {type(curve).__name__}.')

    delta = _check_delta(delta)
    signed = _signed_delta(curve, delta, direction)

    if signed == 0.:
        return [curve]

    return [ArcSpline(raw_curve) for raw_curve in _libfibomat.offset_curve(curve.arc_spline_impl, signed)]


def offset_with_islands(
    islands: t.Sequence[ArcSpline], delta: float, outer_curve: t.Optional[ArcSpline] = None
) -> OffsetResult:
    """Offset a region with islands.

    The islands grow by `delta` (islands which touch each other are merged) and the outer curve, if given, shrinks by
    `delta`. The results do not intersect each other and the offset islands and the outer curve are combined, i.e.
    an island which is grown into the outer curve is cut out of it.

    Args:
        islands (Sequence[ArcSpline]): islands. All islands must be closed curves with positive orientation.
        delta (float): offset distance (not negative).
        outer_curve (ArcSpline, optional): optional outer curve, which is offset inwards. Must be closed.

    Returns:
        OffsetResult: grown islands and shrunken outer curves.

    Raises:
        TypeError: Raised if one of the curves is not an ArcSpline.
        ValueError: Raised if delta is negative or not finite.
        ValueError: Raised if a curve is not closed or an island has negative orientation.
    """
    delta = _check_delta(delta)

    islands = list(islands)
    for island in islands:
        if not isinstance(island, ArcSpline):
            raise TypeError(f'islands must be ArcSplines, got {type(island).__name__}.')
        if not island.is_closed:
            raise ValueError('All islands must be closed.')
        if not island.orientation:
            raise ValueError('All islands must have positive orientation.')

    if outer_curve is not None:
        if not isinstance(outer_curve, ArcSpline):
            raise TypeError(f'outer_curve must be an ArcSpline, got {type(outer_curve).__name__}.')
        if not outer_curve.is_closed:
            raise ValueError('outer_curve must be closed.')

    raw_islands, raw_outer_curves = _libfibomat.offset_with_islands(
        [island.arc_spline_impl for island in islands],
        outer_curve.arc_spline_impl if outer_curve is not None else None,
        delta
    )

    return OffsetResult(
        islands=[ArcSpline(raw_curve) for raw_curve in raw_islands],
        outer_curves=[ArcSpline(raw_curve) for raw_curve in raw_outer_curves],
    )


def _number_of_steps(
    pitch: float, n_steps: t.Optional[int], distance: t.Optional[float], *, required: bool
) -> t.Optional[int]:
    """Validate the arguments of deflate and inflate and compute the maximum number of steps.

    Args:
        pitch (float): distance between two consecutive curves
        n_steps (int, optional): number of steps
        distance (float, optional): total distance
        required (bool): if True, one of n_steps and distance must be given.

    Returns:
        Optional[int]: maximum number of steps or None if unlimited.

    Raises:
        ValueError: Raised on invalid combinations or values.
    """
    pitch = float(pitch)
    if not math.isfinite(pitch) or pitch <= 0.:
        raise ValueError(f'pitch must be finite and greater than 0, got {pitch}.')

    if n_steps is not None and distance is not None:
        raise ValueError('n_steps and distance cannot be set both.')

    if n_steps is not None:
        if int(n_steps) != n_steps or n_steps < 0:
            raise ValueError(f'n_steps must be an integer not less than 0, got {n_steps}.')
        return int(n_steps)

    if distance is not None:
        distance = _check_delta(distance, 'distance')
        # tolerance: distance = 3 * pitch must give three steps although the division may give 2.9999999
        return int(math.floor(distance / pitch + 1e-9))

    if required:
        raise ValueError('Any of n_steps or distance must be set.')
    return None


def _offset_series(
    arc_spline: ArcSpline, pitch: float, direction: OffsetDirection, max_steps: t.Optional[int]
) -> t.List[ArcSpline]:
    """Offset a closed curve repeatedly.

    The k-th step offsets the original curve by k * pitch (instead of offsetting the result of the previous step) to
    avoid accumulating errors.

    Args:
        arc_spline (ArcSpline): closed curve
        pitch (float): distance between the curves
        direction (OffsetDirection): inwards or outwards
        max_steps (int, optional): maximum number of steps. If None, offset until nothing is left (inwards only).

    Returns:
        List[ArcSpline]: all offset curves, ordered by distance from the original curve.

    Raises:
        RuntimeError: Raised if more than :attr:`SAFETY` steps are performed.
    """
    if not isinstance(arc_spline, ArcSpline):
        raise TypeError(f'arc_spline must be an ArcSpline, got {type(arc_spline).__name__}.')
    if not arc_spline.is_closed:
        raise ValueError('arc_spline must be closed.')

    limit = SAFETY if max_steps is None else max_steps
    if limit > SAFETY:
        raise RuntimeError(f'More than {SAFETY} steps requested. If you need so many steps, increase SAFETY.')

    curves: t.List[ArcSpline] = []
    for step in range(1, limit + 1):
        new_curves = offset(arc_spline, step * pitch, direction)
        if not new_curves:
            return curves
        curves.extend(new_curves)

    if max_steps is None:
        raise RuntimeError(f'steps > SAFETY ({SAFETY}). If you know you need so many steps, increase SAFETY.')

    return curves


def deflate(
    arc_spline: ArcSpline, pitch: float, *, n_steps: t.Optional[int] = None, distance: t.Optional[float] = None
) -> t.List[ArcSpline]:
    """Deflate a closed curve step by step.

    The original curve is not included. The k-th step is the original curve moved inwards by ``k * pitch``. If neither
    `n_steps` nor `distance` is given, the curve is deflated until nothing is left.

    Args:
        arc_spline (ArcSpline): closed curve to be deflated.
        pitch (float): distance between two consecutive curves (greater than 0).
        n_steps (int, optional): if provided, at most `n_steps` steps are performed.
        distance (float, optional): if provided, the curve is deflated by at most `distance` in total.

    Returns:
        List[ArcSpline]: deflated curves, ordered from the outside to the inside.

    Raises:
        TypeError: Raised if arc_spline is not an ArcSpline.
        ValueError: Raised if pitch is not positive, n_steps or distance is negative, or n_steps and distance are
            provided both, or the curve is not closed.
        RuntimeError: Raised if more than :attr:`SAFETY` steps are needed.
    """
    max_steps = _number_of_steps(pitch, n_steps, distance, required=False)
    return _offset_series(arc_spline, float(pitch), OffsetDirection.INWARDS, max_steps)


def inflate(
    arc_spline: ArcSpline, pitch: float, *, n_steps: t.Optional[int] = None, distance: t.Optional[float] = None
) -> t.List[ArcSpline]:
    """Inflate a closed curve step by step.

    The original curve is not included. The k-th step is the original curve moved outwards by ``k * pitch``.

    Note, any one of `n_steps` and `distance` must be provided.

    Args:
        arc_spline (ArcSpline): closed curve to be inflated.
        pitch (float): distance between two consecutive curves (greater than 0).
        n_steps (int, optional): if provided, `n_steps` steps are performed.
        distance (float, optional): if provided, the curve is inflated by at most `distance` in total.

    Returns:
        List[ArcSpline]: inflated curves, ordered from the inside to the outside.

    Raises:
        TypeError: Raised if arc_spline is not an ArcSpline.
        ValueError: Raised if pitch is not positive, n_steps or distance is negative, none or both of n_steps and
            distance are provided, or the curve is not closed.
        RuntimeError: Raised if more than :attr:`SAFETY` steps are requested.
    """
    max_steps = _number_of_steps(pitch, n_steps, distance, required=True)
    return _offset_series(arc_spline, float(pitch), OffsetDirection.OUTWARDS, max_steps)
