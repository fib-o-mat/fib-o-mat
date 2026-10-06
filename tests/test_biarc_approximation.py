import math
import time

import numpy as np
import pytest
import sympy
from scipy.spatial import cKDTree

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.curve_tools.biarc_approximation import approximate_parametric_curve, approximate_vertices
from fibomat.curve_tools.biarc_approximation.meek_walton import _biarc_joints_and_bulges
from fibomat.linalg import Vector
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.biarc import Biarc
from fibomat.shapes.parametric_curve import ParametricCurve

T = sympy.Symbol('t')
PI = sympy.pi

CURVES = {
    'ellipse': sympy.Curve([3 * sympy.cos(T), sympy.sin(T)], (T, 0, 2 * PI)),
    'sine': sympy.Curve([T, sympy.sin(3 * T)], (T, 0, 6)),
    'spiral': sympy.Curve([T * sympy.cos(T), T * sympy.sin(T)], (T, 0.5, 8)),
    'cubic with inflection': sympy.Curve([T, T ** 3 - T], (T, -1.5, 1.5)),
    'cusp': sympy.Curve([T ** 2, T ** 3], (T, -1, 1)),
    'line': sympy.Curve([T, 2 * T + 1], (T, 0, 3)),
    'quarter circle': sympy.Curve([sympy.cos(T), sympy.sin(T)], (T, 0, PI / 2)),
    'three quarter circle': sympy.Curve([sympy.cos(T), sympy.sin(T)], (T, 0, 3 * PI / 2)),
    'astroid': sympy.Curve([sympy.cos(T) ** 3, sympy.sin(T) ** 3], (T, 0, 2 * PI)),
    'lissajous': sympy.Curve([sympy.sin(3 * T), sympy.sin(2 * T)], (T, 0, 2 * PI)),
    'non uniform parametrization': sympy.Curve([T ** 3, sympy.sin(T ** 3)], (T, 0, 2)),
    'big and shifted': sympy.Curve([1e4 + 1e3 * sympy.cos(T), -2e4 + 5e2 * sympy.sin(T)], (T, 0.3, 5)),
}


def sample_arc_spline(vertices, closed, n=80):
    """Dense points on the arc spline."""
    points = []
    count = len(vertices)
    n_segments = count if closed else count - 1
    for i in range(n_segments):
        start, end, bulge = vertices[i, :2], vertices[(i + 1) % count, :2], vertices[i, 2]
        u = np.linspace(0, 1, n, endpoint=False)[:, None]
        if abs(bulge) < 1e-12:
            points.append(start + u * (end - start))
            continue
        theta = 4 * np.arctan(bulge)
        chord = end - start
        length = np.linalg.norm(chord)
        radius = length / (2 * np.sin(abs(theta) / 2))
        normal = np.array([-chord[1], chord[0]]) / length
        center = (start + end) / 2 + normal * radius * np.cos(theta / 2) * np.sign(bulge)
        angles = np.arctan2(*(start - center)[::-1]) + u[:, 0] * theta
        points.append(center + radius * np.c_[np.cos(angles), np.sin(angles)])
    points.append(vertices[:1, :2] if closed else vertices[-1:, :2])
    return np.concatenate(points)


def max_distances(curve, spline, n_curve=6000):
    """(max distance of the curve to the arc spline, max distance of the arc spline to the curve)."""
    dense = curve.f(np.linspace(*curve.domain, n_curve))
    to_spline = max(spline.closest_point(p)['distance'] for p in dense[::3])
    # distance to a polyline of the curve (spacing between the samples is the error of this measurement)
    spacing = np.max(np.linalg.norm(np.diff(dense, axis=0), axis=1))
    from_spline = cKDTree(dense).query(sample_arc_spline(spline.vertices, spline.is_closed))[0].max()
    return to_spline, max(from_spline - spacing / 2, 0.)


class TestAccuracy:
    @pytest.mark.parametrize('name', CURVES)
    @pytest.mark.parametrize('relative_epsilon', [1e-3, 1e-5])
    def test_distance_is_below_epsilon(self, name, relative_epsilon):
        curve = ParametricCurve.from_sympy_curve(CURVES[name])
        bbox = curve.bounding_box
        epsilon = relative_epsilon * (bbox.upper_right - bbox.lower_left).mag
        spline = curve.to_arc_spline(epsilon=epsilon)
        to_spline, from_spline = max_distances(curve, spline)
        assert to_spline <= epsilon
        assert from_spline <= epsilon

    @pytest.mark.parametrize('name', ['ellipse', 'spiral', 'sine'])
    def test_error_is_about_one_thirteenth_of_the_bound(self, name):
        # the paper: the distance of the biarc is ~ 1/13.5 of the distance of the bounding arcs
        curve = ParametricCurve.from_sympy_curve(CURVES[name])
        epsilon = 1e-4
        to_spline, _ = max_distances(curve, curve.to_arc_spline(epsilon=epsilon))
        assert 0.005 * epsilon < to_spline < 0.5 * epsilon

    @pytest.mark.parametrize('name', CURVES)
    def test_interpolates_the_end_points(self, name):
        curve = ParametricCurve.from_sympy_curve(CURVES[name])
        spline = curve.to_arc_spline(epsilon=1e-3)
        assert spline.start == pytest.approx(curve.f(curve.domain[0]), abs=1e-9)
        assert spline.end == pytest.approx(curve.f(curve.domain[1]), abs=1e-9)

    def test_number_of_arcs_grows_with_the_third_root_of_the_tolerance(self):
        curve = ParametricCurve.from_sympy_curve(CURVES['ellipse'])
        n_1 = len(curve.to_arc_spline(epsilon=1e-3))
        n_2 = len(curve.to_arc_spline(epsilon=1e-6))
        assert 7 < n_2 / n_1 < 14  # (1000 ** (1 / 3) = 10)

    def test_area_and_length_of_a_closed_curve(self):
        curve = ParametricCurve.from_sympy_curve(CURVES['ellipse'])
        spline = curve.to_arc_spline(epsilon=1e-6)
        assert spline.is_closed
        assert spline.area == pytest.approx(3 * math.pi, rel=1e-6)
        assert spline.length == pytest.approx(curve.length, rel=1e-6)


class TestTangentContinuity:
    @pytest.mark.parametrize('name', [n for n in CURVES if n not in ('cusp', 'astroid')])
    def test_smooth_curves_give_smooth_arc_splines(self, name):
        spline = ParametricCurve.from_sympy_curve(CURVES[name]).to_arc_spline(epsilon=1e-3)
        assert spline.kinks() == []

    def test_cusps_are_kinks(self):
        assert len(ParametricCurve.from_sympy_curve(CURVES['cusp']).to_arc_spline(epsilon=1e-3).kinks()) == 1
        assert len(ParametricCurve.from_sympy_curve(CURVES['astroid']).to_arc_spline(epsilon=1e-3).kinks()) == 4

    def test_closed_curves_are_smooth_at_the_closing_vertex(self):
        spline = ParametricCurve.from_sympy_curve(CURVES['ellipse']).to_arc_spline(epsilon=1e-4)
        assert spline.is_closed and 0 not in spline.kinks()

    def test_tangents_at_the_ends(self):
        curve = ParametricCurve.from_sympy_curve(CURVES['spiral'])
        spline = curve.to_arc_spline(epsilon=1e-3)
        start_tangent = Vector(curve.df(curve.domain[0])).normalized()
        end_tangent = Vector(curve.df(curve.domain[1])).normalized()
        assert spline.unit_tangents(0)[1] == pytest.approx(start_tangent, abs=1e-9)
        assert spline.unit_tangents(len(spline) - 1)[0] == pytest.approx(end_tangent, abs=1e-9)


class TestSimpleCurves:
    def test_line(self):
        spline = ParametricCurve.from_sympy_curve(CURVES['line']).to_arc_spline(epsilon=1e-6)
        assert len(spline) == 2
        assert spline.vertices[:, 2] == pytest.approx([0., 0.])
        assert spline.length == pytest.approx(3 * math.sqrt(5))

    def test_circle_arcs_need_few_vertices(self):
        # the curvature is constant: the bounding arcs coincide with the curve
        for name in ('quarter circle', 'three quarter circle'):
            spline = ParametricCurve.from_sympy_curve(CURVES[name]).to_arc_spline(epsilon=1e-9)
            assert len(spline) <= 6
        spline = ParametricCurve.from_sympy_curve(CURVES['three quarter circle']).to_arc_spline(epsilon=1e-9)
        assert spline.length == pytest.approx(1.5 * math.pi, rel=1e-9)

    def test_full_circle(self):
        curve = ParametricCurve.from_sympy_curve(sympy.Curve([sympy.cos(T), sympy.sin(T)], (T, 0, 2 * PI)))
        spline = curve.to_arc_spline(epsilon=1e-9)
        assert spline.is_closed
        assert spline.area == pytest.approx(math.pi, rel=1e-9)
        assert len(spline) <= 8

    def test_zero_curvature_inside_of_a_curve(self):
        # x^3 has an inflection point with a very flat region around it
        curve = ParametricCurve.from_sympy_curve(sympy.Curve([T, T ** 3], (T, -1, 1)))
        spline = curve.to_arc_spline(epsilon=1e-6)
        to_spline, _ = max_distances(curve, spline)
        assert to_spline <= 1e-6
        assert spline.kinks() == []

    def test_polynomial_curve_without_special_points(self):
        # a parabola has a single curvature maximum in the vertex
        curve = ParametricCurve.from_sympy_curve(sympy.Curve([T, T ** 2], (T, -2, 2)))
        spline = curve.to_arc_spline(epsilon=1e-5)
        to_spline, from_spline = max_distances(curve, spline)
        assert to_spline <= 1e-5 and from_spline <= 1e-5
        assert spline.kinks() == []

    def test_numerically_defined_curve_without_third_derivative(self):
        # curvature extrema are found with finite differences if the third derivative is not available
        curve = ParametricCurve(
            func=lambda t: np.stack([t, np.sin(3 * t)], axis=-1),
            d_func=lambda t: np.stack([np.ones_like(t), 3 * np.cos(3 * t)], axis=-1),
            d2_func=lambda t: np.stack([np.zeros_like(t), -9 * np.sin(3 * t)], axis=-1),
            domain=(0, 6),
        )
        spline = curve.to_arc_spline(epsilon=1e-4)
        to_spline, from_spline = max_distances(curve, spline)
        assert to_spline <= 1e-4 and from_spline <= 1e-4


class TestPerformance:
    def test_high_accuracy_is_fast(self):
        curve = ParametricCurve.from_sympy_curve(CURVES['ellipse'])
        start = time.perf_counter()
        spline = curve.to_arc_spline(epsilon=1e-8)
        assert len(spline) > 3000
        assert time.perf_counter() - start < 2.

    def test_many_segments_are_vectorized(self):
        curve = ParametricCurve.from_sympy_curve(CURVES['sine'])
        start = time.perf_counter()
        spline = curve.to_arc_spline(epsilon=1e-9)
        assert len(spline) > 10000
        assert time.perf_counter() - start < 5.


class TestApproximateVertices:
    def test_returns_vertices_and_closed_flag(self):
        curve = ParametricCurve.from_sympy_curve(CURVES['ellipse'])
        vertices, closed = approximate_vertices(curve, 1e-3)
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert closed is True
        assert np.all(np.abs(vertices[:, 2]) <= 1.)

    def test_approximate_parametric_curve_returns_an_arc_spline(self):
        curve = ParametricCurve.from_sympy_curve(CURVES['spiral'])
        spline = approximate_parametric_curve(curve, 1e-3, description='foo')
        assert isinstance(spline, ArcSpline) and spline.description == 'foo' and not spline.is_closed

    @pytest.mark.parametrize('epsilon', [0, -1., np.nan, np.inf])
    def test_invalid_epsilon(self, epsilon):
        curve = ParametricCurve.from_sympy_curve(CURVES['ellipse'])
        with pytest.raises(ValueError):
            approximate_vertices(curve, epsilon)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError):
            approximate_vertices(ParametricCurve.from_sympy_curve(CURVES['ellipse']), 1e-3, n_samples=3)

    def test_segment_limit(self):
        curve = ParametricCurve.from_sympy_curve(CURVES['ellipse'])
        with pytest.raises(RuntimeError, match='epsilon'):
            approximate_vertices(curve, 1e-12, max_segments=1000)

    def test_curve_without_velocity(self):
        curve = ParametricCurve(
            func=lambda t: np.zeros(t.shape + (2,)),
            d_func=lambda t: np.zeros(t.shape + (2,)),
            d2_func=lambda t: np.zeros(t.shape + (2,)),
            domain=(0, 1),
        )
        with pytest.raises(ValueError, match='velocity'):
            approximate_vertices(curve, 1e-3)

    def test_many_curvature_extrema_need_enough_samples(self):
        curve = ParametricCurve.from_sympy_curve(sympy.Curve([T, sympy.sin(40 * T) / 10], (T, 0, 6)))
        spline = curve.to_arc_spline(epsilon=1e-4)
        to_spline, from_spline = max_distances(curve, spline, n_curve=20000)
        assert to_spline <= 1e-4 and from_spline <= 1e-4


class TestBiarcConsistency:
    @pytest.mark.parametrize('seed', range(30))
    def test_vectorized_biarcs_equal_the_biarc_shape(self, seed):
        rng = np.random.default_rng(seed)
        p_a, p_b = rng.uniform(-3, 3, (2, 2))
        t_a, t_b = rng.uniform(-1, 1, (2, 2))
        t_a, t_b = t_a / np.linalg.norm(t_a), t_b / np.linalg.norm(t_b)
        joint, bulge_1, bulge_2 = _biarc_joints_and_bulges(p_a[None], p_b[None], t_a[None], t_b[None])

        biarc = Biarc(p_a, p_b, t_a, t_b)
        segments = biarc.segments
        if len(segments) == 2:
            assert joint[0] == pytest.approx(segments[0].end, abs=1e-9)
            assert bulge_1[0] == pytest.approx(segments[0].bulge, abs=1e-7)
            assert bulge_2[0] == pytest.approx(segments[1].bulge, abs=1e-7)

    def test_parallel_tangents(self):
        joint, bulge_1, bulge_2 = _biarc_joints_and_bulges(
            np.array([[0., 0.]]), np.array([[3., 1.]]), np.array([[1., 0.]]), np.array([[1., 0.]])
        )
        assert joint[0] == pytest.approx((1.5, 0.5))
        assert bulge_1[0] == pytest.approx(-bulge_2[0])
