"""Provide the :func:`self_intersections` and :func:`curve_intersections` function.

Example::

    from fibomat.curve_tools import curve_intersections, self_intersections
    from fibomat.shapes import Circle, Line

    circle = Circle(r=1).to_arc_spline()
    line = Line((-2, 0), (2, 0)).to_arc_spline()

    points, overlaps = curve_intersections(circle, line)
    for point in points:
        point.position, point.index_1, point.index_2

    self_intersections(circle)  # []
"""
from __future__ import annotations

import typing as t

from fibomat import _libfibomat
from fibomat.curve_tools.results import CurveIntersections, Intersection, Overlap
from fibomat.linalg import Vector
from fibomat.shapes.arc_spline import ArcSpline


__all__ = ['self_intersections', 'curve_intersections']


def _check_curve(curve: t.Any, name: str) -> None:
    if not isinstance(curve, ArcSpline):
        raise TypeError(f'{name} must be an ArcSpline, got {type(curve).__name__}.')


def self_intersections(curve: ArcSpline) -> t.List[Intersection]:
    """Self intersections of a curve. Overlapping segments are not reported.

    Args:
        curve (ArcSpline): curve

    Returns:
        List[Intersection]:
            intersections. `index_1` and `index_2` are the indices of the two segments of the curve which intersect
            (``index_1 <= index_2``).

    Raises:
        TypeError: Raised if `curve` is no ArcSpline.
    """
    _check_curve(curve, 'curve')

    return [
        Intersection(Vector(position), min(index_1, index_2), max(index_1, index_2))
        for index_1, index_2, position in _libfibomat.self_intersections(curve.arc_spline_impl)
    ]


def curve_intersections(curve_1: ArcSpline, curve_2: ArcSpline) -> CurveIntersections:
    """Intersections between two curves.

    Crossings, touching points and the end points of open curves lying on the other curve are reported as `points`.
    Parts where the curves lie on each other are reported as `overlaps` (not as points).

    Args:
        curve_1 (ArcSpline): first curve
        curve_2 (ArcSpline): second curve

    Returns:
        CurveIntersections: named tuple ``(points, overlaps)`` of :class:`Intersection` and :class:`Overlap` lists.

    Raises:
        TypeError: Raised if one of the curves is no ArcSpline.
    """
    _check_curve(curve_1, 'curve_1')
    _check_curve(curve_2, 'curve_2')

    points, overlaps = _libfibomat.curve_intersections(curve_1.arc_spline_impl, curve_2.arc_spline_impl)

    return CurveIntersections(
        points=[Intersection(Vector(position), index_1, index_2) for index_1, index_2, position in points],
        overlaps=[
            Overlap(Vector(start), Vector(end), index_1, index_2) for index_1, index_2, start, end in overlaps
        ],
    )
