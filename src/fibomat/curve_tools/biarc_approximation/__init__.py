from typing import List, Union, TYPE_CHECKING

import numpy as np

from fibomat.shapes.line import Line
from fibomat.shapes.arc import Arc
from fibomat.shapes.biarc import Biarc


if TYPE_CHECKING:
    from fibomat.shapes.parametric_curve import ParametricCurve

from fibomat.curve_tools.biarc_approximation.monotone_intervals import intervals_by_osculating_circles, IntervalType
from fibomat.curve_tools.biarc_approximation.spiral_segment import _approximate_param_curve_greedy
from fibomat.curve_tools.biarc_approximation.utils import _make_fixed_tangent_curvature


def approximate_parametric_curve(
    param_curve: 'ParametricCurve', rasterize_pitch: float, epsilon: float
) -> List[Union[Arc, Line]]:
    """Approximate a ParametricCurve with an ArcSpline.

    Seems to work good: rasterize_pitch > 10 * epsilon

    rasterize_pitch > epsilon

    Args:
        param_curve (ParametricCurve): curve to be approximated
        rasterize_pitch (float): pitch used for rasterization.
        epsilon (float): maximal distance between original and approximated curve.

    Returns:
        ArcSpline

    References:
        - https://www.sciencedirect.com/science/article/pii/037704279400029Z
        - https://epub.jku.at/obvulihs/content/titleinfo/2474921
        - https://www.sciencedirect.com/science/article/abs/pii/S0010448508001681
        - (https://www.sciencedirect.com/science/article/pii/S0925772112000326?via%3Dihub)

    """

    if rasterize_pitch <= epsilon:
        raise ValueError('rasterize_pitch must be larger than epsilon.')

    parameter_values = param_curve.rasterize(rasterize_pitch, safety=1, add_endpoint=True)

    # For testing of FLUSH_WITHOUT_TRANSITION
    # parameter_values = np.roll(parameter_values, 10)

    monotone_curvature_intervals = intervals_by_osculating_circles(param_curve, parameter_values, epsilon)

    curve_segments = []

    for interval_type, monotone_interval in monotone_curvature_intervals:
        p_0 = param_curve.f(monotone_interval[0])
        p_1 = param_curve.f(monotone_interval[-1])

        t_0, _ = _make_fixed_tangent_curvature(param_curve, monotone_interval[0], epsilon, 'left')
        t_1, _ = _make_fixed_tangent_curvature(param_curve, monotone_interval[-1], epsilon, 'right')

        if interval_type == IntervalType.LINE:
            curve_segments.append(Line(p_0, p_1))
        elif interval_type == IntervalType.BIARC:
            # todo fix tangents
            curve_segments.append(Biarc(p_0, p_1, t_0, t_1))
        elif interval_type == IntervalType.ARC:
            if np.allclose(p_0, p_1):
                raise NotImplementedError(
                    'Cannot handle closed circles in biarc fitting currently. '
                    'Please report this as a bug if you need this.'
                )
            intermediate = param_curve.f(monotone_interval[len(monotone_interval) // 2])
            curve_segments.append(Arc.from_points(p_0, intermediate, p_1))
        else:
            new_segs = _approximate_param_curve_greedy(
                param_curve, monotone_interval, epsilon
            )
            curve_segments.extend(
                new_segs
            )

    # return parameter_values, curve_segments
    return curve_segments
