"""Result types of the functions of the package."""
from __future__ import annotations

import typing as t

from fibomat.linalg import Vector
from fibomat.shapes.arc_spline import ArcSpline


__all__ = ['Intersection', 'Overlap', 'CurveIntersections', 'CombineResult', 'OffsetResult']


class Intersection(t.NamedTuple):
    """An intersection point of two curves (or of a curve with itself)."""

    position: Vector
    """Position of the intersection."""
    index_1: int
    """Index of the segment of the first curve."""
    index_2: int
    """Index of the segment of the second curve (of the curve itself for self intersections)."""


class Overlap(t.NamedTuple):
    """A part where two curves lie on each other."""

    start: Vector
    """Start of the common part (seen along the first curve)."""
    end: Vector
    """End of the common part."""
    index_1: int
    """Index of the segment of the first curve."""
    index_2: int
    """Index of the segment of the second curve."""


class CurveIntersections(t.NamedTuple):
    """Intersections of two curves."""

    points: t.List[Intersection]
    """Intersection points (crossings and touching points)."""
    overlaps: t.List[Overlap]
    """Parts where the curves lie on each other."""


class CombineResult(t.NamedTuple):
    """Result of combining two closed curves."""

    boundaries: t.List[ArcSpline]
    """Outer outlines of the regions of the result."""
    holes: t.List[ArcSpline]
    """Outlines of the holes which are enclosed by the regions of the result."""


class OffsetResult(t.NamedTuple):
    """Result of offsetting islands and an outer curve."""

    islands: t.List[ArcSpline]
    """The (grown) islands."""
    outer_curves: t.List[ArcSpline]
    """The (shrunken) outer curve; several curves if the offset splits the outer region; empty if there was none."""
