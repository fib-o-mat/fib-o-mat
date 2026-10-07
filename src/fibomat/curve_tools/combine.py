"""Provide the :func:`combine_curves` function.

Example::

    from fibomat.curve_tools import combine_curves
    from fibomat.shapes import Rect

    first = Rect(width=2, height=2).to_arc_spline()
    second = Rect(width=2, height=2, center=(1, 1)).to_arc_spline()

    boundaries, holes = combine_curves(first, second, 'union')
    union = combine_curves(first, second, CombineMode.UNION).boundaries
"""
from __future__ import annotations

import typing as t

from fibomat import _libfibomat
from fibomat.curve_tools.modes import CombineMode
from fibomat.curve_tools.results import CombineResult
from fibomat.shapes.arc_spline import ArcSpline


__all__ = ['combine_curves']


def combine_curves(curve_1: ArcSpline, curve_2: ArcSpline, mode: t.Union[CombineMode, str]) -> CombineResult:
    """Combine two closed curves with a boolean operation on the regions they enclose. The result does not depend on
    the orientation of the curves.

    `mode` can be:

        - `union`: region of the first or the second curve
        - `intersect`: region of the first and the second curve
        - `exclude`: region of the first curve without the region of the second curve
        - `xor`: region of exactly one of the curves

    The result consists of the outer outlines of the resulting regions (`boundaries`) and the outlines of holes enclosed
    by these regions (`holes`). E.g. excluding a small circle in the middle of a large one gives the large circle as
    boundary and the small circle as hole.

    Args:
        curve_1 (ArcSpline): first curve (closed)
        curve_2 (ArcSpline): second curve (closed)
        mode (CombineMode, str): combining mode

    Returns:
        CombineResult: named tuple ``(boundaries, holes)`` of lists of arc splines

    Raises:
        TypeError: Raised if one of the curves is no ArcSpline.
        ValueError: Raised if one of the curves is not closed or the mode is unknown.
    """
    for name, curve in (('curve_1', curve_1), ('curve_2', curve_2)):
        if not isinstance(curve, ArcSpline):
            raise TypeError(f'{name} must be an ArcSpline, got {type(curve).__name__}.')
        if not curve.is_closed:
            raise ValueError(f'{name} must be closed.')

    mode = CombineMode.parse(mode)

    boundaries, holes = _libfibomat.combine_curves(curve_1.arc_spline_impl, curve_2.arc_spline_impl, mode.value)

    return CombineResult(
        boundaries=[ArcSpline(raw) for raw in boundaries], holes=[ArcSpline(raw) for raw in holes]
    )
