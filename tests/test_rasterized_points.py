import copy
import math

import numpy as np
import pytest

from fibomat.linalg import BoundingBox, Vector, mirror, rotate, scale, translate
from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.shapes.shape import Shape
from fibomat.units import unit

POINTS = np.array([[0., 0., 1.], [2., 0., 0.5], [2., 1., 2.], [0., 1., 1.]])


class TestConstruction:
    def test_basic(self):
        points = RasterizedPoints(POINTS, True, 'foo')
        assert points.n_points == 4
        assert points.is_closed is True
        assert points.description == 'foo'
        assert isinstance(points, Shape)
        assert points.positions.tolist() == POINTS[:, :2].tolist()
        assert points.weights.tolist() == [1., 0.5, 2., 1.]
        assert points.dwell_points.tolist() == POINTS.tolist()

    def test_array_is_copied(self):
        array = POINTS.copy()
        points = RasterizedPoints(array, False)
        array[0, 0] = 100.
        assert points.positions[0, 0] == 0.
        points._impl_translate(Vector(1, 1))
        assert array[1, 0] == 2.

    def test_lists_and_integers(self):
        points = RasterizedPoints([[0, 0, 1], [1, 1, 1]], False)
        assert points.dwell_points.dtype == float

    def test_is_closed_is_converted(self):
        assert RasterizedPoints(POINTS, np.bool_(True)).is_closed is True
        assert RasterizedPoints(POINTS, 0).is_closed is False

    @pytest.mark.parametrize('array', [np.zeros(3), np.zeros((2, 2)), np.zeros((2, 4)), np.zeros((2, 3, 3))])
    def test_invalid_shapes(self, array):
        with pytest.raises(ValueError):
            RasterizedPoints(array, False)

    def test_invalid_values(self):
        with pytest.raises(ValueError, match='finite'):
            RasterizedPoints(np.array([[0., np.nan, 1.]]), False)
        with pytest.raises(ValueError):
            RasterizedPoints([['a', 'b', 'c']], False)
        with pytest.raises(ValueError):
            RasterizedPoints([[0, 0, 1], [1, 1]], False)

    def test_repr(self):
        assert repr(RasterizedPoints(POINTS, True)) == 'RasterizedPoints(n_points=4, is_closed=True)'


class TestViews:
    def test_views_are_read_only(self):
        points = RasterizedPoints(POINTS, False)
        for view in (points.positions, points.weights, points.dwell_points):
            with pytest.raises(ValueError):
                view[0] = 5.

    def test_views_do_not_make_the_object_read_only(self):
        points = RasterizedPoints(POINTS, False)
        _ = points.positions, points.weights, points.dwell_points
        points._impl_translate(Vector(1, 0))
        assert points.positions[0, 0] == 1.


class TestProperties:
    def test_bounding_box(self):
        assert RasterizedPoints(POINTS, False).bounding_box == BoundingBox((0, 0), (2, 1))

    def test_center(self):
        # (the former implementation averaged the wrong axis and failed for n != 2)
        points = RasterizedPoints(np.array([[0., 0., 1.], [4., 0., 1.], [1., 2., 1.]]), False)
        assert points.center == (2., 1.)
        assert type(points.center) is Vector
        assert RasterizedPoints(POINTS, False).center == (1., 0.5)

    def test_single_point(self):
        points = RasterizedPoints(np.array([[1., 2., 1.]]), False)
        assert points.bounding_box == BoundingBox((1, 2), (1, 2))
        assert points.center == (1., 2.)

    def test_no_points(self):
        points = RasterizedPoints(np.empty((0, 3)), False)
        assert points.n_points == 0
        with pytest.raises(ValueError):
            points.bounding_box
        with pytest.raises(ValueError):
            points.center

    def test_bounding_box_of_many_points_is_fast(self):
        import time

        points = RasterizedPoints(np.random.default_rng(0).uniform(size=(1_000_000, 3)), False)
        start = time.perf_counter()
        points.bounding_box
        assert time.perf_counter() - start < 0.5

    def test_area_and_boundary_length_are_not_defined(self):
        with pytest.raises(NotImplementedError):
            RasterizedPoints(POINTS, True).area


class TestMerged:
    def test_merged(self):
        first = RasterizedPoints(POINTS, True)
        second = RasterizedPoints(POINTS + [10, 0, 0], True)
        merged = RasterizedPoints.merged([first, second], 'foo')
        assert merged.n_points == 8
        assert merged.dwell_points[:4].tolist() == POINTS.tolist()
        assert merged.dwell_points[4:, 0].tolist() == (POINTS[:, 0] + 10).tolist()
        assert merged.is_closed is False
        assert merged.description == 'foo'

    def test_merge_nothing(self):
        merged = RasterizedPoints.merged([])
        assert merged.n_points == 0 and merged.dwell_points.shape == (0, 3)

    def test_merge_invalid(self):
        with pytest.raises(TypeError):
            RasterizedPoints.merged([RasterizedPoints(POINTS, False), POINTS])


class TestRepeats:
    def test_repeats_applied(self):
        points = RasterizedPoints(POINTS, True, 'foo')
        repeated = points.repeats_applied(3)
        assert repeated.n_points == 12
        assert repeated.dwell_points.tolist() == np.tile(POINTS, (3, 1)).tolist()
        assert repeated.is_closed is True and repeated.description == 'foo'
        assert repeated is not points

    def test_one_repeat_returns_a_copy(self):
        points = RasterizedPoints(POINTS, False)
        res = points.repeats_applied(1)
        assert res is not points
        res._impl_translate(Vector(1, 1))
        assert points.positions[0, 0] == 0.

    @pytest.mark.parametrize('repeats', [0, -1, 1.5])
    def test_invalid(self, repeats):
        with pytest.raises(ValueError):
            RasterizedPoints(POINTS, False).repeats_applied(repeats)


class TestTransformations:
    @staticmethod
    def check(points, func):
        expected = np.array([func(Vector(p)) for p in POINTS[:, :2]])
        assert points.positions == pytest.approx(expected, abs=1e-12)
        # the weights are not touched
        assert points.weights.tolist() == POINTS[:, 2].tolist()

    def test_translate(self):
        points = RasterizedPoints(POINTS, False)
        res = points.translated((1, -1))
        self.check(res, lambda p: p + (1, -1))
        assert points.positions.tolist() == POINTS[:, :2].tolist()
        assert isinstance(res, RasterizedPoints) and res is not points

    def test_rotate(self):
        # (the former implementation rotated the weights instead of the positions)
        points = RasterizedPoints(POINTS, False)
        self.check(points.rotated(math.pi / 2), lambda p: p.rotated(math.pi / 2))
        self.check(points.rotated(0.7, (3, -1)), lambda p: p.rotated(0.7, (3, -1)))
        self.check(points.rotated(1., 'center'), lambda p: p.rotated(1., (1., 0.5)))

    def test_scale(self):
        points = RasterizedPoints(POINTS, False)
        self.check(points.scaled(2.5, (1, 1)), lambda p: (p - (1, 1)) * 2.5 + (1, 1))
        self.check(points.scaled(-2.), lambda p: p * -2.)

    @pytest.mark.parametrize('axis', [(1, 0), (0, 1), (1, 1), (-2, 1), (3, 0.5)])
    def test_mirror(self, axis):
        # (the former implementation divided by the length instead of the squared length and assigned a transposed
        # array)
        self.check(RasterizedPoints(POINTS, False).mirrored(axis), lambda p: p.mirrored(axis))

    def test_transformed_chain(self):
        res = RasterizedPoints(POINTS, False).transformed(
            translate((1, 1)) | rotate(0.3, (2, 2)) | scale(1.5) | mirror((1, 2))
        )
        self.check(res, lambda p: (((p + (1, 1)).rotated(0.3, (2, 2))) * 1.5).mirrored((1, 2)))

    def test_transformations_with_other_number_of_points(self):
        # (the mirroring failed for n != 2)
        array = np.random.default_rng(1).uniform(-1, 1, (7, 3))
        points = RasterizedPoints(array, False)
        mirrored = points.mirrored((1, 0))
        assert mirrored.positions == pytest.approx(array[:, :2] * [1, -1])
        assert points.rotated(PI := math.pi).positions == pytest.approx(-array[:, :2])


class TestCopyAndDimShape:
    def test_clone(self):
        points = RasterizedPoints(POINTS, True, 'foo')
        clone = points.clone()
        assert clone is not points and clone.dwell_points.tolist() == POINTS.tolist()
        assert clone.is_closed and clone.description == 'foo'
        clone._impl_translate(Vector(1, 1))
        assert points.positions[0, 0] == 0.
        assert copy.deepcopy(points).n_points == 4

    def test_dim_shape(self):
        dim_points = RasterizedPoints(POINTS, False) * unit('µm')
        assert dim_points.bounding_box.width.m_as('µm') == pytest.approx(2.)
        assert dim_points.to('nm').shape.positions[1, 0] == pytest.approx(2000.)
