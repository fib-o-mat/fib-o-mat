import copy
import math

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, Vector, VectorValueError, mirror, rotate, scale, translate
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.ellipse import ARC_SPLINE_TOLERANCE, Ellipse
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi


def boundary(ellipse, n=2001):
    """Points on the boundary of the ellipse."""
    params = np.linspace(0, 2 * PI, n)
    local = np.c_[ellipse.a * np.cos(params), ellipse.b * np.sin(params)]
    cos, sin = math.cos(ellipse.theta), math.sin(ellipse.theta)
    rot = np.array([[cos, -sin], [sin, cos]])
    return local @ rot.T + np.asarray(ellipse.center)


class TestConstruction:
    def test_basic(self):
        ellipse = Ellipse(3, 1, PI / 4, (1, 2), 'foo')
        assert (ellipse.a, ellipse.b) == (3., 1.)
        assert ellipse.theta == pytest.approx(PI / 4)
        assert ellipse.center == (1., 2.)
        assert ellipse.description == 'foo'
        assert isinstance(ellipse, Shape) and isinstance(ellipse, ArcSplineCompatible)
        assert Ellipse(2, 1).theta == 0.
        assert Ellipse(2, 1).center == (0., 0.)

    def test_theta_is_normalized(self):
        assert Ellipse(2, 1, -PI / 2).theta == pytest.approx(3 * PI / 2)
        assert Ellipse(2, 1, 5 * PI / 2).theta == pytest.approx(PI / 2)
        assert 0 <= Ellipse(2, 1, -1e-12).theta < 2 * PI

    @pytest.mark.parametrize('kwargs', [
        {'a': 0}, {'a': -1}, {'b': 0}, {'b': -1}, {'a': np.nan}, {'b': np.inf}, {'theta': np.nan}, {'theta': np.inf},
        {'center': (np.nan, 0)},
    ])
    def test_invalid(self, kwargs):
        args = {'a': 2., 'b': 1., 'theta': 0., 'center': (0, 0)}
        args.update(kwargs)
        with pytest.raises(ValueError):  # (the former implementation used assert)
            Ellipse(**args)

    def test_invalid_center(self):
        with pytest.raises(VectorValueError):
            Ellipse(2, 1, 0, (1, 2, 3))

    def test_repr(self):
        assert repr(Ellipse(2, 1, 0, (1, 2))) == 'Ellipse(a=2.0, b=1.0, theta=0.0, center=Vector(x=1.0, y=2.0))'


class TestProperties:
    def test_area(self):
        assert Ellipse(3, 2).area == pytest.approx(6 * PI)

    def test_boundary_length(self):
        assert Ellipse(2, 2).boundary_length == pytest.approx(4 * PI)
        assert Ellipse(3, 1).boundary_length == pytest.approx(Ellipse(1, 3).boundary_length)
        # numerical integration of the arc length
        ellipse = Ellipse(3, 1)
        params = np.linspace(0, 2 * PI, 200001)
        speed = np.sqrt((3 * np.sin(params)) ** 2 + np.cos(params) ** 2)
        assert ellipse.boundary_length == pytest.approx(np.trapz(speed, params) if hasattr(np, 'trapz') else
                                                        np.trapezoid(speed, params), rel=1e-8)

    @pytest.mark.parametrize('theta', [0., 0.3, PI / 4, PI / 2, 2., 4.])
    def test_bounding_box(self, theta):
        # the former implementation rotated the b axis by theta + pi / 4
        ellipse = Ellipse(3, 1, theta, (1, 2))
        points = boundary(ellipse)
        bbox = ellipse.bounding_box
        assert (*bbox.lower_left, *bbox.upper_right) == pytest.approx(
            (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()), abs=1e-5
        )

    def test_bounding_box_axis_aligned(self):
        assert Ellipse(3, 1).bounding_box == BoundingBox((-3, -1), (3, 1))
        assert Ellipse(3, 1, PI / 2).bounding_box == BoundingBox((-1, -3), (1, 3))

    def test_is_closed(self):
        assert Ellipse(2, 1).is_closed is True
        assert type(Ellipse(2, 1).area) is float


class TestToArcSpline:
    @pytest.mark.parametrize('a, b, theta', [(3, 1, 0.), (3, 1, 0.7), (1, 4, 2.), (10, 1, 0.3), (1.2, 1, 0.)])
    def test_accuracy(self, a, b, theta):
        ellipse = Ellipse(a, b, theta, (1, -2), 'foo')
        spline = ellipse.to_arc_spline()
        assert isinstance(spline, ArcSpline)
        assert spline.is_closed and spline.description == 'foo'
        tolerance = 10 * ARC_SPLINE_TOLERANCE * min(a, b)

        # every point of the ellipse is close to the arc spline ...
        for point in boundary(ellipse, 401):
            assert spline.closest_point(point)['distance'] <= tolerance
        # ... and the integral quantities agree
        assert spline.area == pytest.approx(ellipse.area, rel=1e-5)
        assert spline.length == pytest.approx(ellipse.boundary_length, rel=1e-5)
        assert spline.bounding_box.center == pytest.approx(ellipse.center)
        bbox, expected = spline.bounding_box, ellipse.bounding_box
        assert (*bbox.lower_left, *bbox.upper_right) == pytest.approx((*expected.lower_left, *expected.upper_right),
                                                                      abs=tolerance)

    def test_orientation_and_arcs(self):
        spline = Ellipse(3, 1).to_arc_spline()
        assert spline.orientation is True
        assert np.all(spline.vertices[:, 2] > 0)
        assert np.all(np.abs(spline.vertices[:, 2]) <= 1)

    def test_symmetric(self):
        spline = Ellipse(3, 1).to_arc_spline()
        assert spline.center == pytest.approx((0., 0.))
        assert len(spline) % 4 == 0

    def test_circle_is_converted_exactly(self):
        spline = Ellipse(2, 2, 0, (1, 1), 'foo').to_arc_spline()
        assert len(spline) == 2
        assert spline.area == pytest.approx(4 * PI)
        assert spline.description == 'foo'
        assert len(Ellipse(2, 2 * (1 + 1e-12)).to_arc_spline()) == 2

    def test_contains(self):
        spline = Ellipse(3, 1, PI / 2).to_arc_spline()
        assert spline.contains((0, 2.5)) and not spline.contains((2.5, 0))


class TestTransformations:
    @staticmethod
    def check(ellipse, transformed, func):
        """The boundary of the transformed ellipse is the transformed boundary of the original ellipse."""
        expected = np.array([func(Vector(p)) for p in boundary(ellipse, 41)])
        spline = transformed.to_arc_spline()
        for point in expected:
            assert spline.closest_point(point)['distance'] <= 1e-4

    def test_translate(self):
        ellipse = Ellipse(3, 1, 0.5, (1, 1))
        res = ellipse.translated((1, -1))
        assert res.center == (2., 0.) and res.theta == pytest.approx(0.5)
        assert (res.a, res.b) == (3., 1.)

    def test_rotate(self):
        # the former implementation assigned the rotated axis to an unused attribute and did not rotate the center
        ellipse = Ellipse(3, 1, 0.5, (1, 0))
        res = ellipse.rotated(PI / 2)
        assert res.theta == pytest.approx(0.5 + PI / 2)
        assert res.center == pytest.approx((0., 1.))
        self.check(ellipse, res, lambda p: p.rotated(PI / 2))
        self.check(ellipse, ellipse.rotated(0.7, (3, -1)), lambda p: p.rotated(0.7, (3, -1)))
        assert ellipse.rotated(0.7, 'center').center == pytest.approx((1., 0.))

    def test_scale(self):
        ellipse = Ellipse(3, 1, 0.5, (1, 2))
        res = ellipse.scaled(2.)
        assert (res.a, res.b) == (6., 2.) and res.center == (2., 4.) and res.theta == pytest.approx(0.5)
        self.check(ellipse, res, lambda p: p * 2.)

    def test_scale_negative(self):
        ellipse = Ellipse(3, 1, 0.5, (1, 2))
        res = ellipse.scaled(-2.)
        assert (res.a, res.b) == (6., 2.)  # (the former implementation produced negative half axes)
        self.check(ellipse, res, lambda p: p * -2.)

    @pytest.mark.parametrize('axis', [(1, 0), (0, 1), (1, 1), (-2, 1)])
    def test_mirror(self, axis):
        ellipse = Ellipse(3, 1, 0.5, (1, 2))
        res = ellipse.mirrored(axis)
        assert (res.a, res.b) == (3., 1.)
        self.check(ellipse, res, lambda p: p.mirrored(axis))

    def test_transformed(self):
        ellipse = Ellipse(3, 1, 0.5, (1, 2))
        res = ellipse.transformed(translate((1, 1)) | rotate(0.3, (2, 2)) | scale(1.5) | mirror((1, 2)))
        self.check(ellipse, res,
                   lambda p: (((p + (1, 1)).rotated(0.3, (2, 2))) * 1.5).mirrored((1, 2)))

    def test_does_not_modify_original(self):
        ellipse = Ellipse(3, 1, 0.5, (1, 2))
        ellipse.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert (ellipse.a, ellipse.b, ellipse.center) == (3., 1., Vector(1, 2))
        assert ellipse.theta == pytest.approx(0.5)


class TestCopyAndDimShape:
    def test_clone(self):
        ellipse = Ellipse(3, 1, 0.5, (1, 2), 'foo')
        clone = ellipse.clone()
        assert clone is not ellipse
        assert (clone.a, clone.b, clone.center, clone.description) == (3., 1., Vector(1, 2), 'foo')
        assert clone.theta == pytest.approx(0.5)
        assert copy.deepcopy(ellipse).theta == pytest.approx(0.5)

    def test_dim_shape(self):
        dim_ellipse = Ellipse(3, 1) * unit('µm')
        assert dim_ellipse.bounding_box.width.m_as('µm') == pytest.approx(6.)
        assert dim_ellipse.to('nm').shape.a == pytest.approx(3000.)
