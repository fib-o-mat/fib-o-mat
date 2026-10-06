import copy
import math

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, Vector, VectorValueError, mirror, rotate, scale, translate
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.line import Line
from fibomat.shapes.rect import Rect
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi


def corners_array(rect):
    return np.array(rect.corners)


class TestConstruction:
    def test_basic(self):
        rect = Rect(4, 2, PI / 6, (1, 1), 'foo')
        assert (rect.width, rect.height) == (4., 2.)
        assert rect.theta == pytest.approx(PI / 6)
        assert rect.center == pytest.approx((1., 1.))
        assert rect.description == 'foo'
        assert isinstance(rect, Shape) and isinstance(rect, ArcSplineCompatible)

    def test_default_center(self):
        # the former implementation failed without a center although it is documented as optional
        rect = Rect(4, 2)
        assert rect.center == (0., 0.)
        assert rect.theta == 0.
        assert Rect(width=4, height=2).bounding_box == BoundingBox((-2, -1), (2, 1))

    def test_numpy_arguments(self):
        rect = Rect(np.float32(4), np.int64(2), np.float64(0.5), np.array([1., 1.]))
        assert type(rect.width) is float and type(rect.height) is float

    @pytest.mark.parametrize('kwargs', [
        {'width': 0}, {'width': -1}, {'height': 0}, {'height': -1}, {'width': np.nan}, {'height': np.inf},
        {'theta': np.nan}, {'theta': np.inf}, {'center': (np.nan, 0)},
    ])
    def test_invalid(self, kwargs):
        args = {'width': 4., 'height': 2., 'theta': 0., 'center': (0, 0)}
        args.update(kwargs)
        with pytest.raises(ValueError):
            Rect(**args)

    def test_invalid_center(self):
        with pytest.raises(VectorValueError):
            Rect(1, 1, 0, (1, 2, 3))

    def test_theta_is_normalized(self):
        assert Rect(1, 1, -PI / 2).theta == pytest.approx(3 * PI / 2)
        assert Rect(1, 1, 5 * PI / 2).theta == pytest.approx(PI / 2)

    def test_repr(self):
        assert repr(Rect(4, 2, 0, (1, 2))) == 'Rect(width=4.0, height=2.0, theta=0.0, center=Vector(x=1.0, y=2.0))'


class TestFromBoundingBoxAndLine:
    def test_from_bounding_box(self):
        rect = Rect.from_bounding_box(BoundingBox((1, 2), (4, 6)))
        assert (rect.width, rect.height) == (3., 4.)
        assert rect.center == (2.5, 4.)
        assert rect.theta == 0.
        assert rect.bounding_box == BoundingBox((1, 2), (4, 6))

    def test_from_degenerate_bounding_box(self):
        with pytest.raises(ValueError):
            Rect.from_bounding_box(BoundingBox((1, 2), (1, 6)))

    def test_from_line(self):
        rect = Rect.from_line(Line((0, 0), (3, 4)), 2)
        assert rect.width == pytest.approx(5.) and rect.height == 2.
        assert rect.center == pytest.approx((1.5, 2.))
        assert rect.theta == pytest.approx(math.atan2(4, 3))

    def test_from_line_endpoints_are_midpoints_of_the_short_sides(self):
        rect = Rect.from_line(Line((0, 0), (3, 4)), 2)
        c = corners_array(rect)
        # short sides: (corner 0, corner 3) at the end of the line and (corner 1, corner 2) at its start
        assert (c[0] + c[3]) / 2 == pytest.approx((3., 4.))
        assert (c[1] + c[2]) / 2 == pytest.approx((0., 0.))

    def test_from_line_invalid(self):
        with pytest.raises(ValueError, match='length'):
            Rect.from_line(Line((1, 1), (1, 1)), 2)
        with pytest.raises(ValueError):
            Rect.from_line(Line((0, 0), (1, 1)), 0)


class TestProperties:
    def test_values(self):
        rect = Rect(4, 2, 0.7, (1, 1))
        assert rect.area == pytest.approx(8.)
        assert rect.boundary_length == pytest.approx(12.)
        assert rect.is_closed is True

    def test_corners_unrotated(self):
        corners = Rect(4, 2, 0, (1, 1)).corners
        assert [type(c) for c in corners] == [Vector] * 4
        assert [tuple(c) for c in corners] == [(3., 2.), (-1., 2.), (-1., 0.), (3., 0.)]

    def test_corners_are_counterclockwise_and_rectangular(self):
        rect = Rect(4, 2, 0.9, (1, 1))
        corners = corners_array(rect)
        x, y = corners.T
        assert 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1)) == pytest.approx(8.)  # positive area
        sides = np.roll(corners, -1, axis=0) - corners
        assert np.linalg.norm(sides, axis=1) == pytest.approx([4., 2., 4., 2.])
        for i in range(4):
            assert sides[i] @ sides[(i + 1) % 4] == pytest.approx(0., abs=1e-12)
        assert corners.mean(axis=0) == pytest.approx((1., 1.))

    @pytest.mark.parametrize('theta', [0., 0.3, PI / 4, PI / 2, 2., 4.])
    def test_bounding_box(self, theta):
        rect = Rect(4, 2, theta, (1, -1))
        assert rect.bounding_box == BoundingBox.from_points(rect.corners)

    def test_to_arc_spline(self):
        rect = Rect(4, 2, 0.5, (1, 1), 'foo')
        spline = rect.to_arc_spline()
        assert isinstance(spline, ArcSpline)
        assert spline.is_closed and spline.description == 'foo'
        assert len(spline) == 4
        assert spline.area == pytest.approx(8.)
        assert spline.length == pytest.approx(12.)
        assert spline.orientation is True
        assert spline.vertices[:, :2] == pytest.approx(corners_array(rect))
        assert spline.bounding_box == rect.bounding_box


class TestTransformations:
    @staticmethod
    def check(rect, transformed, func):
        expected = np.array([func(corner) for corner in rect.corners])
        got = corners_array(transformed)
        # same set of corners (the start corner may differ), same area and perimeter
        for point in expected:
            assert np.min(np.linalg.norm(got - point, axis=1)) < 1e-9
        assert transformed.area == pytest.approx(rect.area * abs(np.linalg.det(np.c_[func(Vector(1, 0)) - func(Vector(0, 0)), func(Vector(0, 1)) - func(Vector(0, 0))])))

    def test_translate(self):
        res = Rect(4, 2, 0.5, (1, 1)).translated((1, -1))
        assert res.center == pytest.approx((2., 0.))
        assert (res.width, res.height) == (4., 2.)
        assert res.theta == pytest.approx(0.5)

    def test_rotate(self):
        rect = Rect(4, 2, 0.5, (1, 1))
        res = rect.rotated(PI / 2)
        assert res.theta == pytest.approx(0.5 + PI / 2)
        assert res.center == pytest.approx((-1., 1.))
        self.check(rect, res, lambda p: p.rotated(PI / 2))
        self.check(rect, rect.rotated(0.7, (3, -1)), lambda p: p.rotated(0.7, (3, -1)))
        assert rect.rotated(0.7, 'center').center == pytest.approx((1., 1.))

    def test_rotate_by_default_theta(self):
        assert Rect(4, 2).rotated(PI / 3).theta == pytest.approx(PI / 3)

    def test_scale(self):
        rect = Rect(4, 2, 0.5, (1, 1))
        res = rect.scaled(2.)
        assert (res.width, res.height) == (8., 4.)
        assert res.center == pytest.approx((2., 2.))
        self.check(rect, res, lambda p: p * 2.)

    def test_scale_negative(self):
        rect = Rect(4, 2, 0.5, (1, 1))
        res = rect.scaled(-2.)
        assert (res.width, res.height) == (8., 4.)
        self.check(rect, res, lambda p: p * -2.)

    @pytest.mark.parametrize('axis', [(1, 0), (0, 1), (1, 1), (-2, 1)])
    def test_mirror(self, axis):
        # the former implementation computed the new angle from the absolute position of a corner
        rect = Rect(4, 2, 0.5, (1, 1))
        res = rect.mirrored(axis)
        assert (res.width, res.height) == (4., 2.)
        self.check(rect, res, lambda p: p.mirrored(axis))
        assert res.bounding_box == BoundingBox.from_points([c.mirrored(axis) for c in rect.corners])

    def test_mirror_axis_aligned(self):
        res = Rect(4, 2, 0, (1, 1)).mirrored((0, 1))
        assert res.center == pytest.approx((-1., 1.))
        assert res.theta in (pytest.approx(PI), pytest.approx(0.))
        assert res.bounding_box == BoundingBox((-3, 0), (1, 2))

    def test_mirrored_rect_keeps_counterclockwise_corners(self):
        spline = Rect(4, 2, 0.5, (1, 1)).mirrored((1, 0)).to_arc_spline()
        assert spline.orientation is True

    def test_transformed(self):
        rect = Rect(4, 2, 0.5, (1, 1))
        res = rect.transformed(translate((1, 1)) | rotate(0.3, (2, 2)) | scale(1.5) | mirror((1, 2)))
        self.check(rect, res, lambda p: (((p + (1, 1)).rotated(0.3, (2, 2))) * 1.5).mirrored((1, 2)))

    def test_does_not_modify_original(self):
        rect = Rect(4, 2, 0.5, (1, 1))
        rect.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert (rect.width, rect.height, rect.center) == (4., 2., Vector(1, 1))
        assert rect.theta == pytest.approx(0.5)

    def test_width_and_height_stay_exact(self):
        # the former implementation computed them from the vertices
        rect = Rect(4, 2).rotated(0.123).translated((1e3, 1e3)).mirrored((1, 2))
        assert (rect.width, rect.height) == (4., 2.)


class TestCopyAndDimShape:
    def test_clone(self):
        rect = Rect(4, 2, 0.5, (1, 1), 'foo')
        clone = rect.clone()
        assert clone is not rect
        assert (clone.width, clone.height, clone.description) == (4., 2., 'foo')
        assert clone.theta == pytest.approx(0.5) and clone.center == pytest.approx((1., 1.))
        assert copy.deepcopy(rect).center == rect.center

    def test_dim_shape(self):
        dim_rect = Rect(4, 2) * unit('µm')
        assert dim_rect.bounding_box.height.m_as('µm') == pytest.approx(2.)
        assert dim_rect.to('nm').shape.width == pytest.approx(4000.)
