"""Approximation of parametric curves by arc splines."""
from __future__ import annotations

import typing as t

from fibomat.curve_tools.biarc_approximation.meek_walton import approximate_vertices
from fibomat.shapes.arc_spline import ArcSpline


if t.TYPE_CHECKING:  # pragma: no cover
    from fibomat.shapes.parametric_curve import ParametricCurve


__all__ = ['approximate_parametric_curve', 'approximate_vertices']


def approximate_parametric_curve(
    param_curve: 'ParametricCurve', epsilon: float, n_samples: int = 2000, description: t.Optional[str] = None
) -> ArcSpline:
    """Approximate a ParametricCurve with an ArcSpline.

    The curve is split at its curvature extrema and inflections into spirals which are approximated by biarcs. The
    maximal distance between the curve and the arc spline is `epsilon` (see :mod:`.meek_walton`).

    Args:
        param_curve (ParametricCurve): curve to be approximated
        epsilon (float): maximal distance between original and approximated curve.
        n_samples (int): number of samples used to find curvature extrema and inflections. Increase this value if the
                         curvature of the curve oscillates strongly.
        description (str, optional): description of the arc spline

    Returns:
        ArcSpline

    Raises:
        ValueError: Raised if epsilon is not positive.
        RuntimeError: Raised if the number of arcs would exceed the limit.

    References:
        - https://www.sciencedirect.com/science/article/pii/037704279400029Z
    """
    vertices, is_closed = approximate_vertices(param_curve, epsilon, n_samples)
    return ArcSpline(vertices, is_closed, description)
