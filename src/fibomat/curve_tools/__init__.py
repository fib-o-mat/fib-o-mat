"""Tools to work with curves (intersections, boolean operations, offsets, rasterization, filling and smoothing)."""
from fibomat.curve_tools.modes import CombineMode, OffsetDirection
from fibomat.curve_tools.results import Intersection, Overlap, CurveIntersections, CombineResult, OffsetResult
from fibomat.curve_tools.combine import combine_curves
from fibomat.curve_tools.intersections import self_intersections, curve_intersections
from fibomat.curve_tools.offset import offset, offset_with_islands, deflate, inflate
from fibomat.curve_tools.rasterize import rasterize, rasterize_with_const_error
from fibomat.curve_tools.fill import fill_with_lines, fill_with_spiral
from fibomat.curve_tools.biarc_approximation import approximate_parametric_curve
from fibomat.curve_tools.smooth import smooth, NonSmoothableError

__all__ = [
    'CombineMode', 'OffsetDirection', 'Intersection', 'Overlap', 'CurveIntersections', 'CombineResult',
    'OffsetResult', 'combine_curves', 'self_intersections', 'curve_intersections', 'offset', 'offset_with_islands',
    'rasterize', 'rasterize_with_const_error', 'approximate_parametric_curve', 'deflate', 'inflate', 'fill_with_lines',
    'smooth', 'NonSmoothableError', 'fill_with_spiral'
]
