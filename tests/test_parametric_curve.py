import copy
import math

import numpy as np
import pytest
import splipy
import splipy.curve_factory
import sympy

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, Vector, mirror, rotate, scale, translate
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.parametric_curve import ARC_SPLINE_RELATIVE_TOLERANCE, ParametricCurve
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi
T = sympy.Symbol('t')


def points(*columns):
    return np.stack(columns, axis=-1)


def circle_curve(radius=2., domain=(0, 2 * PI), **kwargs):
    return ParametricCurve(
        func=lambda t: points(radius * np.cos(t), radius * np.sin(t)),
        d_func=lambda t: points(-radius * np.sin(t), radius * np.cos(t)),
        d2_func=lambda t: points(-radius * np.cos(t), -radius * np.sin(t)),
        domain=domain,
        **kwargs,
    )


def ellipse_curve():
    return ParametricCurve.from_sympy_curve(sympy.Curve([3 * sympy.cos(T), sympy.sin(T)], (T, 0, 2 * PI)))


def sampled(curve, n=4001):
    return curve.f(np.linspace(*curve.domain, n))


class TestConstruction:
    def test_basic(self):
        curve = circle_curve(description='foo')
        assert curve.domain == (0., 2 * PI)
        assert curve.description == 'foo'
        assert isinstance(curve, Shape) and isinstance(curve, ArcSplineCompatible)
        assert repr(curve) == f'ParametricCurve(domain={(0., 2 * PI)!r})'

    @pytest.mark.parametrize('domain', [(1, 1), (2, 1), (0, np.inf), (np.nan, 1), (0, 1, 2), 5, 'ab', None])
    def test_invalid_domain(self, domain):
        with pytest.raises(ValueError):
            circle_curve(domain=domain)

    def test_functions_must_be_callable(self):
        with pytest.raises(ValueError):
            ParametricCurve(1, lambda t: t, lambda t: t, (0, 1))
        with pytest.raises(ValueError):
            circle_curve(curvature=3)
        with pytest.raises(ValueError):
            circle_curve(length='abc')

    def test_functions_must_return_the_right_shape(self):
        curve = ParametricCurve(lambda t: points(t, t)[..., :1], lambda t: points(t, t), lambda t: points(t, t), (0, 1))
        with pytest.raises(ValueError, match='func'):
            curve.f(0.5)
        # the former implementations squeezed the results; arrays of one point keep their shape
        curve = circle_curve()
        assert curve.f(np.array([0.5])).shape == (1, 2)
        assert curve.f(0.5).shape == (2,)
        assert curve.f(np.zeros((3, 4))).shape == (3, 4, 2)

    def test_bounding_box_argument_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match='bounding_box'):
            circle_curve(bounding_box=BoundingBox((0, 0), (1, 1)))


class TestEvaluation:
    def test_values(self):
        curve = circle_curve(2.)
        assert curve.f(0.) == pytest.approx((2., 0.))
        assert curve.f(np.array([0., PI / 2])) == pytest.approx(np.array([[2., 0.], [0., 2.]]))
        assert curve.df(PI / 2) == pytest.approx((-2., 0.))
        assert curve.d2f(0.) == pytest.approx((-2., 0.))

    def test_curvature_is_calculated_from_the_derivatives(self):
        curve = circle_curve(2.)
        assert curve.curvature(1.) == pytest.approx(0.5)
        assert curve.curvature(np.linspace(0, 6, 7)) == pytest.approx(np.full(7, 0.5))
        assert curve.curvature(np.array(1.)).shape == ()

    def test_curvature_sign(self):
        clockwise = ParametricCurve(
            lambda t: points(np.cos(t), -np.sin(t)), lambda t: points(-np.sin(t), -np.cos(t)),
            lambda t: points(-np.cos(t), np.sin(t)), (0, 1)
        )
        assert clockwise.curvature(0.3) == pytest.approx(-1.)

    def test_given_curvature_is_used(self):
        curve = circle_curve(2., curvature=lambda t: np.full(t.shape, 123.))
        assert curve.curvature(0.3) == 123.

    def test_d_curvature_analytic_vs_finite_differences(self):
        sympy_curve = sympy.Curve([3 * sympy.cos(T), sympy.sin(T)], (T, 0, 2 * PI))
        analytic = ParametricCurve.from_sympy_curve(sympy_curve)
        numeric = ParametricCurve(analytic._func, analytic._d_func, analytic._d2_func, analytic.domain)
        params = np.linspace(0.1, 6.2, 25)
        assert analytic.d_curvature(params) == pytest.approx(numeric.d_curvature(params), abs=1e-5)
        # analytic value of the derivative of the curvature of an ellipse
        expected = -72 * np.sin(params) * np.cos(params) / (9 * np.sin(params) ** 2 + np.cos(params) ** 2) ** 2.5
        assert analytic.d_curvature(params) == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_d_curvature_of_a_circle_is_zero(self):
        assert circle_curve().d_curvature(np.linspace(0, 6, 7)) == pytest.approx(np.zeros(7), abs=1e-6)

    def test_given_d_curvature_is_used(self):
        curve = circle_curve(d_curvature=lambda t: np.full(t.shape, 5.))
        assert curve.d_curvature(0.1) == 5.

    def test_d3f(self):
        curve = ellipse_curve()
        assert curve.d3f(0.) == pytest.approx((0., -1.))
        with pytest.raises(NotImplementedError):
            circle_curve().d3f(0.)


class TestFromSympy:
    def test_ellipse(self):
        curve = ellipse_curve()
        assert curve.domain == pytest.approx((0., 2 * PI))
        assert curve.f(np.array([0., PI / 2])) == pytest.approx(np.array([[3., 0.], [0., 1.]]))
        assert curve.curvature(0.) == pytest.approx(3.)  # a / b^2
        assert curve.curvature(PI / 2) == pytest.approx(1 / 9)  # b / a^2
        assert curve.is_closed

    def test_constant_expressions(self):
        # (sympy returns scalars for constant expressions)
        curve = ParametricCurve.from_sympy_curve(sympy.Curve([T, 2], (T, 0, 1)))
        assert curve.f(np.array([0., 1.])) == pytest.approx(np.array([[0., 2.], [1., 2.]]))
        assert curve.df(np.array([0.5])).shape == (1, 2)
        assert curve.d2f(0.5) == pytest.approx((0., 0.))

    def test_deprecated_arguments(self):
        sympy_curve = sympy.Curve([T, T ** 2], (T, 0, 1))
        with pytest.warns(DeprecationWarning, match='try_length_integration'):
            ParametricCurve.from_sympy_curve(sympy_curve, try_length_integration=True)
        with pytest.warns(DeprecationWarning):
            ParametricCurve.from_sympy_curve(sympy_curve, integration_timeout=3)

    def test_description(self):
        assert ParametricCurve.from_sympy_curve(sympy.Curve([T, T], (T, 0, 1)), description='foo').description == 'foo'

    def test_is_fast(self):
        # (no sympy.simplify: it needed seconds for complicated curves)
        import time

        expressions = [T * sympy.cos(T) / (1 + T ** 2) + sympy.exp(-T) * sympy.sin(3 * T),
                       sympy.sqrt(1 + T) * sympy.sin(T) ** 2 + sympy.log(2 + T)]
        start = time.perf_counter()
        curve = ParametricCurve.from_sympy_curve(sympy.Curve(expressions, (T, 0, 5)))
        curve.to_arc_spline(epsilon=1e-4)
        assert time.perf_counter() - start < 3.


class TestFromSplipy:
    @staticmethod
    def spline_curve():
        controlpoints = [[0, 0], [1, 2], [3, 3], [5, 0], [6, 2]]
        return splipy.curve_factory.cubic_curve(np.array(controlpoints, dtype=float))

    def test_values(self):
        spline = self.spline_curve()
        curve = ParametricCurve.from_splipy_curve(spline, description='foo')
        params = np.linspace(*curve.domain, 7)
        assert curve.domain == (float(spline.start(0)), float(spline.end(0)))
        assert curve.f(params) == pytest.approx(spline.evaluate(params)[:, :2])
        assert curve.df(params) == pytest.approx(spline.derivative(params, d=1)[:, :2])
        assert curve.f(params[3]).shape == (2,)
        assert curve.description == 'foo'
        assert curve.length == pytest.approx(float(spline.length()), rel=1e-6)

    def test_is_independent_of_the_original(self):
        spline = self.spline_curve()
        curve = ParametricCurve.from_splipy_curve(spline)
        before = curve.f(0.5)
        spline.translate([10, 10])
        assert curve.f(0.5) == pytest.approx(before)

    def test_to_arc_spline(self):
        curve = ParametricCurve.from_splipy_curve(self.spline_curve())
        spline = curve.to_arc_spline(epsilon=1e-4)
        dense = sampled(curve)
        assert max(spline.closest_point(p)['distance'] for p in dense[::5]) <= 1e-4

    def test_circle(self):
        curve = ParametricCurve.from_splipy_curve(splipy.curve_factory.circle(r=2))
        assert curve.is_closed
        # (the circle of splipy is a spline approximation)
        assert curve.length == pytest.approx(4 * PI, rel=1e-4)
        assert curve.curvature(0.3 * curve.domain[1]) == pytest.approx(0.5, rel=1e-3)


class TestLengthAndRasterization:
    def test_length(self):
        assert circle_curve(2.).length == pytest.approx(4 * PI)
        assert circle_curve(2., domain=(0, PI)).length == pytest.approx(2 * PI)
        assert ellipse_curve().length == pytest.approx(13.364893220555256)
        assert circle_curve(2.).arc_length(1., 2.) == pytest.approx(2.)

    def test_given_length_is_used_and_scaled(self):
        curve = circle_curve(2., length=lambda t_0, t_1: 2 * (t_1 - t_0))
        assert curve.length == pytest.approx(4 * PI)
        assert curve.scaled(3.).length == pytest.approx(12 * PI)

    def test_rasterize_equal_arc_length(self):
        curve = circle_curve(2.)
        params = curve.rasterize(0.1)
        assert params[0] == 0.
        assert len(params) == int(4 * PI / 0.1) + 1
        assert np.diff(params) == pytest.approx(np.full(len(params) - 1, 0.05), rel=1e-9)
        assert curve.rasterize_at(0.1).shape == (len(params), 2)

    def test_rasterize_non_uniform_parametrization(self):
        curve = ParametricCurve.from_sympy_curve(sympy.Curve([T ** 3, sympy.sin(T ** 3)], (T, 0, 2)))
        params = curve.rasterize(0.3)
        lengths = [curve.arc_length(a, b) for a, b in zip(params[:-1], params[1:])]
        assert lengths == pytest.approx(np.full(len(lengths), 0.3), rel=1e-8)
        # the last point is at most one pitch before the end
        assert 0 <= curve.arc_length(params[-1], curve.domain[1]) < 0.3 + 1e-9

    def test_rasterize_endpoint(self):
        curve = circle_curve(1., domain=(0, 1))  # length 1
        assert curve.rasterize(0.4).tolist() == pytest.approx([0., 0.4, 0.8])
        assert curve.rasterize(0.4, add_endpoint=True).tolist() == pytest.approx([0., 0.4, 0.8, 1.])
        # no duplicate if the length is a multiple of the pitch
        assert curve.rasterize(0.5, add_endpoint=True).tolist() == pytest.approx([0., 0.5, 1.])

    def test_rasterize_subdomain(self):
        curve = circle_curve(1., domain=(0, 2 * PI))
        params = curve.rasterize(0.5, domain=(1., 3.))
        assert params[0] == 1. and params[-1] <= 3.
        assert np.diff(params) == pytest.approx(np.full(len(params) - 1, 0.5), rel=1e-9)

    def test_rasterize_short_curves(self):
        params = circle_curve(1., domain=(0, 0.1)).rasterize(1.)
        assert params.tolist() == [0.]

    def test_rasterize_invalid(self):
        curve = circle_curve()
        for pitch in (0, -1., np.nan, np.inf):
            with pytest.raises(ValueError):
                curve.rasterize(pitch)
        with pytest.raises(ValueError):
            curve.rasterize(0.1, domain=(1., 0.5))
        with pytest.raises(ValueError):
            curve.rasterize(0.1, domain=(-1., 0.5))

    def test_safety_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match='safety'):
            circle_curve().rasterize(0.5, safety=1.5)

    def test_rasterize_is_fast(self):
        import time

        start = time.perf_counter()
        params = circle_curve(100.).rasterize(0.001)
        assert len(params) > 600000
        assert time.perf_counter() - start < 3.


class TestBoundingBoxAndClosed:
    def test_bounding_box_ellipse(self):
        curve = ellipse_curve()
        assert curve.bounding_box == BoundingBox((-3, -1), (3, 1))
        assert curve.center == pytest.approx((0., 0.))

    def test_bounding_box_of_arcs_and_polynomials(self):
        assert circle_curve(2., domain=(0, PI)).bounding_box == BoundingBox((-2, 0), (2, 2))
        assert circle_curve(2., domain=(0.1, 0.2)).bounding_box.lower_left == pytest.approx(
            (2 * math.cos(0.2), 2 * math.sin(0.1)))
        parabola = ParametricCurve.from_sympy_curve(sympy.Curve([T, T ** 2], (T, -2, 1)))
        assert parabola.bounding_box == BoundingBox((-2, 0), (1, 4))

    def test_bounding_box_matches_sampling(self):
        curve = ParametricCurve.from_sympy_curve(sympy.Curve([sympy.sin(3 * T), sympy.sin(2 * T) + T], (T, 0, 5)))
        dense = sampled(curve, 200001)
        bbox = curve.bounding_box
        assert (*bbox.lower_left, *bbox.upper_right) == pytest.approx(
            (*dense.min(axis=0), *dense.max(axis=0)), abs=1e-8)

    def test_bounding_box_is_cached_and_updated(self):
        curve = ellipse_curve()
        assert curve.bounding_box is curve.bounding_box
        shifted = curve.translated((1, 1))
        assert shifted.bounding_box == BoundingBox((-2, 0), (4, 2))
        assert curve.bounding_box == BoundingBox((-3, -1), (3, 1))

    def test_is_closed(self):
        assert circle_curve().is_closed is True
        assert circle_curve(domain=(0, PI)).is_closed is False
        assert isinstance(circle_curve().is_closed, bool)
        assert circle_curve().translated((1e5, 1e5)).is_closed

    def test_area_is_not_defined(self):
        with pytest.raises(NotImplementedError):
            circle_curve().area


class TestTransformations:
    @staticmethod
    def curve():
        return ParametricCurve.from_sympy_curve(sympy.Curve([T * sympy.cos(T), T * sympy.sin(T)], (T, 0.5, 6)))

    def check(self, transformed, func):
        original = self.curve()
        params = np.linspace(0.6, 5.9, 12)
        expected = np.array([func(Vector(p)) for p in original.f(params)])
        assert transformed.f(params) == pytest.approx(expected, abs=1e-9)

    def test_translate(self):
        curve = self.curve()
        res = curve.translated((1, -2))
        self.check(res, lambda p: p + (1, -2))
        assert res.df(1.) == pytest.approx(curve.df(1.))
        assert res.domain == curve.domain

    def test_rotate(self):
        curve = self.curve()
        res = curve.rotated(0.7, (2, 1))
        self.check(res, lambda p: p.rotated(0.7, (2, 1)))
        assert res.curvature(2.) == pytest.approx(curve.curvature(2.))
        assert res.length == pytest.approx(curve.length)

    def test_scale(self):
        curve = self.curve()
        res = curve.scaled(2.5, (1, 1))
        self.check(res, lambda p: (p - (1, 1)) * 2.5 + (1, 1))
        assert res.curvature(2.) == pytest.approx(curve.curvature(2.) / 2.5)
        assert res.length == pytest.approx(2.5 * curve.length)

    def test_scale_negative(self):
        curve = self.curve()
        res = curve.scaled(-2.)
        self.check(res, lambda p: p * -2.)
        assert res.curvature(2.) == pytest.approx(curve.curvature(2.) / 2.)  # (a point reflection keeps the orientation)
        assert res.length == pytest.approx(2 * curve.length)

    @pytest.mark.parametrize('axis', [(1, 0), (0, 1), (1, 1), (-2, 1)])
    def test_mirror(self, axis):
        curve = self.curve()
        res = curve.mirrored(axis)
        self.check(res, lambda p: p.mirrored(axis))
        assert res.curvature(2.) == pytest.approx(-curve.curvature(2.))  # mirroring reverses the orientation
        assert res.length == pytest.approx(curve.length)

    def test_given_curvature_functions_follow_transformations(self):
        circle = circle_curve(2., curvature=lambda t: np.full(t.shape, 0.5), d_curvature=lambda t: np.zeros(t.shape))
        assert circle.mirrored((1, 0)).curvature(1.) == pytest.approx(-0.5)
        assert circle.scaled(2.).curvature(1.) == pytest.approx(0.25)
        assert circle.scaled(2.).d_curvature(1.) == pytest.approx(0.)

    def test_transformed_chain(self):
        curve = self.curve()
        res = curve.transformed(translate((1, 1)) | rotate(0.3, (2, 2)) | scale(1.5) | mirror((1, 2)))
        self.check(res, lambda p: (((p + (1, 1)).rotated(0.3, (2, 2))) * 1.5).mirrored((1, 2)))

    def test_does_not_modify_original(self):
        curve = self.curve()
        before = curve.f(np.array([1., 2.]))
        curve.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert curve.f(np.array([1., 2.])) == pytest.approx(before)

    def test_center_follows_transformations(self):
        curve = ellipse_curve()
        assert curve.translated((1, 2)).center == pytest.approx((1., 2.))
        assert curve.rotated(PI / 2).bounding_box == BoundingBox((-1, -3), (1, 3))
        assert curve.scaled(2.).bounding_box == BoundingBox((-6, -2), (6, 2))

    def test_arc_spline_of_a_transformed_curve(self):
        curve = ellipse_curve().rotated(0.4).translated((5, 5))
        spline = curve.to_arc_spline(epsilon=1e-4)
        dense = sampled(curve)
        assert max(spline.closest_point(p)['distance'] for p in dense[::10]) <= 1e-4
        assert spline.bounding_box.center == pytest.approx(curve.center, abs=1e-3)

    def test_mirrored_curve_keeps_a_counterclockwise_orientation_of_the_spline_consistent(self):
        spline = ellipse_curve().to_arc_spline(epsilon=1e-3)
        mirrored = ellipse_curve().mirrored((1, 0)).to_arc_spline(epsilon=1e-3)
        assert spline.orientation is True and mirrored.orientation is False
        assert mirrored.area == pytest.approx(spline.area, rel=1e-6)

    def test_clone(self):
        curve = ellipse_curve().rotated(0.3)
        clone = curve.clone()
        assert clone is not curve and clone.f(1.) == pytest.approx(curve.f(1.))
        clone._impl_translate(Vector(1, 1))
        assert clone.f(1.) != pytest.approx(curve.f(1.))
        assert copy.deepcopy(curve).bounding_box == curve.bounding_box

    def test_dim_shape(self):
        dim_curve = ellipse_curve() * unit('µm')
        assert dim_curve.bounding_box.width.m_as('µm') == pytest.approx(6.)
        assert dim_curve.to('nm').bounding_box.width.m_as('nm') == pytest.approx(6000.)


class TestToArcSpline:
    def test_default_epsilon_is_relative_to_the_size(self):
        curve = ellipse_curve()
        spline = curve.to_arc_spline()
        diagonal = math.hypot(6, 2)
        dense = sampled(curve, 3001)
        assert max(spline.closest_point(p)['distance'] for p in dense[::3]) <= ARC_SPLINE_RELATIVE_TOLERANCE * diagonal
        assert ARC_SPLINE_RELATIVE_TOLERANCE == 1e-5

    def test_result(self):
        curve = ParametricCurve.from_sympy_curve(sympy.Curve([T, T ** 2], (T, 0, 1)), description='foo')
        spline = curve.to_arc_spline(epsilon=1e-4)
        assert isinstance(spline, ArcSpline) and spline.description == 'foo' and not spline.is_closed
        assert curve.to_arc_spline().description == 'foo'

    def test_rasterize_pitch_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match='rasterize_pitch'):
            ellipse_curve().to_arc_spline(rasterize_pitch=0.1, epsilon=1e-3)

    @pytest.mark.parametrize('epsilon', [0., -1., np.nan])
    def test_invalid_epsilon(self, epsilon):
        with pytest.raises(ValueError):
            ellipse_curve().to_arc_spline(epsilon=epsilon)

    def test_curve_without_extent(self):
        curve = ParametricCurve(
            lambda t: points(np.ones_like(t), np.ones_like(t)),
            lambda t: points(np.zeros_like(t), np.zeros_like(t)),
            lambda t: points(np.zeros_like(t), np.zeros_like(t)), (0, 1))
        with pytest.raises(ValueError):
            curve.to_arc_spline()
