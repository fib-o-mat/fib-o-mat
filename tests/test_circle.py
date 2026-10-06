import copy
import math
import warnings

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, Vector, VectorValueError, mirror, rotate, scale, translate
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.circle import Circle
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi


class TestConstruction:
    def test_basic(self):
        circle = Circle(2, (1, 2), 'foo')
        assert circle.r == 2.
        assert circle.center == (1., 2.)
        assert circle.description == 'foo'
        assert isinstance(circle, Shape) and isinstance(circle, ArcSplineCompatible)
        assert Circle(1).center == (0., 0.)
        assert Circle(r=1, center=Vector(1, 1)).center == (1., 1.)

    def test_numpy_radius(self):
        assert Circle(np.float32(2.)).r == 2.
        assert type(Circle(np.int64(2)).r) is float

    @pytest.mark.parametrize('r', [0, -1., np.nan, np.inf, -np.inf])
    def test_invalid_radius(self, r):
        # (nan was accepted by the former implementation)
        with pytest.raises(ValueError):
            Circle(r)

    def test_invalid_center(self):
        with pytest.raises(VectorValueError):
            Circle(1, (1, 2, 3))
        with pytest.raises(ValueError):
            Circle(1, (np.nan, 0))

    def test_repr(self):
        assert repr(Circle(2, (1, 2))) == 'Circle(r=2.0, center=Vector(x=1.0, y=2.0))'


class TestFromPoints:
    def test_unit_circle(self):
        circle = Circle.from_points((1, 0), (0, 1), (-1, 0), 'foo')
        assert circle.r == pytest.approx(1.)
        assert circle.center == pytest.approx((0., 0.))
        assert circle.description == 'foo'

    def test_orientation_does_not_matter(self):
        circle = Circle.from_points((-1, 0), (0, 1), (1, 0))
        assert circle.r == pytest.approx(1.) and circle.center == pytest.approx((0., 0.))

    @pytest.mark.parametrize('seed', range(10))
    def test_random(self, seed):
        points = np.random.default_rng(seed).uniform(-5, 5, (3, 2))
        circle = Circle.from_points(*points)
        for point in points:
            assert np.linalg.norm(point - np.asarray(circle.center)) == pytest.approx(circle.r)

    def test_invalid(self):
        with pytest.raises(ValueError, match='collinear'):
            Circle.from_points((0, 0), (1, 1), (2, 2))
        with pytest.raises(ValueError, match='different'):
            Circle.from_points((0, 0), (0, 0), (2, 2))
        with pytest.raises(ValueError, match='different'):
            Circle.from_points((0, 0), (1, 1), (1, 1))
        with pytest.raises(ValueError, match='different'):
            Circle.from_points((1, 1), (0, 0), (1, 1))


class TestProperties:
    def test_values(self):
        circle = Circle(2, (1, 1))
        assert circle.area == pytest.approx(4 * PI)
        assert circle.boundary_length == pytest.approx(4 * PI)
        assert circle.bounding_box == BoundingBox((-1, -1), (3, 3))
        assert circle.is_closed is True
        assert type(circle.area) is float

    def test_to_arc_spline(self):
        circle = Circle(2, (1, 1), 'foo')
        spline = circle.to_arc_spline()
        assert isinstance(spline, ArcSpline)
        assert spline.is_closed and spline.description == 'foo'
        assert spline.area == pytest.approx(circle.area)
        assert spline.length == pytest.approx(circle.boundary_length)
        assert spline.bounding_box == circle.bounding_box
        assert spline.contains((1, 1)) and not spline.contains((4, 1))
        assert spline.orientation is True

    def test_no_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Circle(1).to_arc_spline()


class TestTransformations:
    def test_translate(self):
        res = Circle(1, (1, 1)).translated((1, -1))
        assert res.center == (2., 0.) and res.r == 1.

    def test_rotate(self):
        res = Circle(1, (1, 0)).rotated(PI / 2)
        assert res.center == pytest.approx((0., 1.)) and res.r == 1.
        assert Circle(1, (1, 0)).rotated(PI / 2, 'center').center == (1., 0.)

    def test_scale(self):
        res = Circle(1, (1, 2)).scaled(2.)
        assert res.r == 2. and res.center == (2., 4.)
        res = Circle(1, (3, 2)).scaled(0.5, 'center')
        assert res.r == 0.5 and res.center == (3., 2.)

    def test_scale_negative(self):
        # the former implementation produced a negative radius
        res = Circle(1, (1, 2)).scaled(-2.)
        assert res.r == 2. and res.center == (-2., -4.)
        assert res.bounding_box == BoundingBox((-4, -6), (0, -2))

    def test_mirror(self):
        res = Circle(1, (1, 2)).mirrored((1, 0))
        assert res.center == pytest.approx((1., -2.)) and res.r == 1.

    def test_transformed(self):
        res = Circle(1, (1, 0)).transformed(translate((1, 1)) | rotate(PI / 2) | scale(2.) | mirror((1, 0)))
        assert res.center == pytest.approx((-2., -4.)) and res.r == 2.

    def test_does_not_modify_original(self):
        circle = Circle(1, (1, 1))
        circle.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert circle.center == (1., 1.) and circle.r == 1.


class TestCopyAndDimShape:
    def test_clone(self):
        circle = Circle(2, (1, 1), 'foo')
        clone = circle.clone()
        assert clone is not circle and clone.r == 2. and clone.center == (1., 1.) and clone.description == 'foo'
        assert copy.deepcopy(circle).center == circle.center

    def test_dim_shape(self):
        dim_circle = Circle(1, (1, 1)) * unit('µm')
        assert dim_circle.center == Vector(1, 1) * unit('µm')
        assert dim_circle.to('nm').shape.r == pytest.approx(1000.)
