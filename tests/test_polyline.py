import copy
import math
import warnings

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, Vector, VectorValueError, mirror, rotate, scale, translate
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.polygon import Polygon
from fibomat.shapes.polyline import Polyline
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi
POINTS = [(0, 0), (2, 0), (2, 1), (0, 1)]


class TestPolylineConstruction:
    def test_basic(self):
        polyline = Polyline(POINTS, 'foo')
        assert polyline.n_points == 4
        assert polyline.points.tolist() == [[0, 0], [2, 0], [2, 1], [0, 1]]
        assert polyline.description == 'foo'
        assert not polyline.is_closed
        assert isinstance(polyline, Shape) and isinstance(polyline, ArcSplineCompatible)

    @pytest.mark.parametrize('points', [
        POINTS,
        [list(p) for p in POINTS],
        np.array(POINTS),
        tuple(POINTS),
        [Vector(*p) for p in POINTS],
        (Vector(*p) for p in POINTS),
        iter(POINTS),
    ])
    def test_point_types(self, points):
        polyline = Polyline(points)
        assert polyline.points.tolist() == [[0, 0], [2, 0], [2, 1], [0, 1]]

    @pytest.mark.parametrize('points', [
        np.zeros((3, 3)),
        np.zeros((3, 1)),
        np.zeros(4),
        [(0, 0), (1, 1, 1)],
        [(0, 0), 'ab'],
        'abcd',
        5,
        None,
        [Vector(0, 0) * unit('µm'), Vector(1, 1) * unit('µm')],
    ])
    def test_invalid_points(self, points):
        with pytest.raises(VectorValueError):
            Polyline(points)

    def test_units_are_not_dropped_silently(self):
        from fibomat.linalg import DimVector

        points = [DimVector(0 * unit('µm'), 0 * unit('µm')), DimVector(1 * unit('µm'), 1 * unit('µm'))]
        with pytest.raises(VectorValueError, match='units'):
            Polyline(points)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            Polyline([(0, 0)])
        with pytest.raises(ValueError):
            Polyline([])
        with pytest.raises(ValueError):
            Polyline(np.zeros((0, 2)))

    def test_not_finite(self):
        with pytest.raises(ValueError, match='finite'):
            Polyline([(0, 0), (np.nan, 1)])
        with pytest.raises(ValueError, match='finite'):
            Polyline([(0, 0), (np.inf, 1)])

    def test_points_array_is_not_referenced(self):
        array = np.array(POINTS, dtype=float)
        polyline = Polyline(array)
        array[0, 0] = 100.
        assert polyline.points[0, 0] == 0.
        polyline.points[1, 1] = 100.  # points is a copy
        assert polyline.points[1, 1] == 0.

    def test_repr(self):
        assert repr(Polyline([(0, 0), (1, 1)])) == 'Polyline(points=array([[0., 0.],\n       [1., 1.]]))'
        assert repr(Polyline(np.zeros((6, 2)) + np.arange(6)[:, None])) == 'Polyline(points=...)'
        assert repr(Polygon(np.zeros((6, 2)) + np.arange(6)[:, None])) == 'Polygon(points=...)'


class TestPolylineProperties:
    def test_length(self):
        polyline = Polyline(POINTS)
        assert polyline.length == pytest.approx(5.)
        assert polyline.boundary_length == pytest.approx(5.)

    def test_center_and_bounding_box(self):
        polyline = Polyline([(0, 0), (3, 0), (3, 3)])
        assert polyline.bounding_box == BoundingBox((0, 0), (3, 3))
        assert polyline.center == (1.5, 1.5)  # (center of the bounding box, not the mean of the points)

    def test_area_is_not_defined_for_open_polylines(self):
        with pytest.raises(NotImplementedError):
            Polyline(POINTS).area

    def test_to_arc_spline(self):
        polyline = Polyline(POINTS, 'foo')
        spline = polyline.to_arc_spline()
        assert isinstance(spline, ArcSpline)
        assert spline.description == 'foo' and not spline.is_closed
        assert spline.vertices[:, :2] == pytest.approx(np.array(POINTS))
        assert Polyline(POINTS).to_arc_spline().description is None

    def test_to_arc_spline_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Polyline(POINTS, 'foo').to_arc_spline()

    def test_to_arc_spline_is_a_copy(self):
        polyline = Polyline(POINTS)
        spline = polyline.to_arc_spline()
        spline._impl_translate(Vector(1, 1))
        assert polyline.points[0].tolist() == [0, 0]


class TestPolygon:
    def test_basic(self):
        polygon = Polygon(POINTS, 'foo')
        assert isinstance(polygon, Polyline)
        assert polygon.is_closed is True
        assert polygon.area == pytest.approx(2.)
        assert polygon.length == pytest.approx(6.)
        assert polygon.boundary_length == pytest.approx(6.)
        assert polygon.description == 'foo'
        assert repr(polygon).startswith('Polygon(points=')

    def test_orientation_does_not_change_the_area(self):
        assert Polygon(POINTS[::-1]).area == pytest.approx(2.)

    def test_to_arc_spline(self):
        spline = Polygon(POINTS).to_arc_spline()
        assert spline.is_closed
        assert spline.area == pytest.approx(2.)

    def test_invalid(self):
        with pytest.raises(VectorValueError):
            Polygon([(0, 0, 0), (1, 1, 1), (1, 0, 1)])
        with pytest.raises(ValueError):
            Polygon([(0, 0)])


class TestRegularNgon:
    @pytest.mark.parametrize('n', [3, 4, 5, 6, 12])
    def test_circumcircle(self, n):
        polygon = Polygon.regular_ngon(n, radius=2, center=(1, 1))
        points = polygon.points
        assert polygon.n_points == n
        assert np.linalg.norm(points - (1, 1), axis=1) == pytest.approx(2.)
        assert polygon.area == pytest.approx(0.5 * n * 4 * math.sin(2 * PI / n))
        sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        assert sides == pytest.approx(sides[0])
        if n % 2 == 0:
            # (the center is the center of the bounding box, which is the circumcenter only for even n)
            assert polygon.center == pytest.approx((1., 1.))

    @pytest.mark.parametrize('n', [3, 4, 5, 6, 12])
    def test_incircle(self, n):
        polygon = Polygon.regular_ngon(n, radius=2, circumcircle=False)
        points = polygon.points
        # distance of the sides to the center is the radius
        mid_points = (points + np.roll(points, -1, axis=0)) / 2
        assert np.linalg.norm(mid_points, axis=1) == pytest.approx(2.)

    def test_is_symmetric_about_the_x_axis(self):
        assert Polygon.regular_ngon(5).bounding_box.lower_left.y == pytest.approx(
            -Polygon.regular_ngon(5).bounding_box.upper_right.y)

    def test_defaults(self):
        polygon = Polygon.regular_ngon(6)
        assert np.linalg.norm(polygon.points, axis=1) == pytest.approx(1.)
        assert polygon.center == pytest.approx((0., 0.))

    def test_integer_valued_floats(self):
        assert Polygon.regular_ngon(4.0).n_points == 4
        assert Polygon.regular_ngon(np.int64(4)).n_points == 4

    def test_description(self):
        assert Polygon.regular_ngon(4, description='foo').description == 'foo'

    @pytest.mark.parametrize('kwargs', [
        {'n': 2}, {'n': 0}, {'n': -3}, {'n': 4.5}, {'n': 4, 'radius': 0}, {'n': 4, 'radius': -1},
        {'n': 4, 'radius': np.nan}, {'n': 4, 'radius': np.inf},
    ])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            Polygon.regular_ngon(**kwargs)

    def test_invalid_center(self):
        with pytest.raises(VectorValueError):
            Polygon.regular_ngon(4, center=(1, 2, 3))


class TestTransformations:
    @pytest.mark.parametrize('cls', [Polyline, Polygon])
    def test_translate(self, cls):
        res = cls(POINTS).translated((1, -1))
        assert res.points.tolist() == [[1, -1], [3, -1], [3, 0], [1, 0]]
        assert isinstance(res, cls)

    def test_rotate(self):
        res = Polyline([(1, 0), (2, 0)]).rotated(PI / 2)
        assert res.points == pytest.approx(np.array([[0, 1], [0, 2]]))
        res = Polygon(POINTS).rotated(PI, 'center')
        assert res.points == pytest.approx(np.array([[2, 1], [0, 1], [0, 0], [2, 0]]))
        assert res.area == pytest.approx(2.)

    def test_scale(self):
        res = Polygon(POINTS).scaled(2.)
        assert res.points.tolist() == [[0, 0], [4, 0], [4, 2], [0, 2]]
        assert res.area == pytest.approx(8.)
        res = Polygon(POINTS).scaled(0.5, 'center')
        assert res.center == pytest.approx((1., 0.5))
        assert res.area == pytest.approx(0.5)

    def test_mirror(self):
        res = Polygon(POINTS).mirrored((1, 0))
        assert res.points.tolist() == [[0, 0], [2, 0], [2, -1], [0, -1]]
        assert res.area == pytest.approx(2.)

    def test_transformed(self):
        res = Polyline([(1, 0), (2, 0)]).transformed(translate((1, 1)) | rotate(PI / 2) | scale(2.) | mirror((1, 0)))
        assert res.points == pytest.approx(np.array([[-2, -4], [-2, -6]]))

    def test_does_not_modify_original(self):
        polyline = Polyline(POINTS)
        polyline.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert polyline.points.tolist() == [[0, 0], [2, 0], [2, 1], [0, 1]]


class TestCopyAndDimShape:
    @pytest.mark.parametrize('cls', [Polyline, Polygon])
    def test_clone(self, cls):
        shape = cls(POINTS, 'foo')
        clone = shape.clone()
        assert type(clone) is cls and clone is not shape
        assert clone.points.tolist() == shape.points.tolist() and clone.description == 'foo'
        assert clone.is_closed == shape.is_closed
        clone._impl_translate(Vector(1, 1))
        assert shape.points[0].tolist() == [0, 0]
        assert copy.deepcopy(shape).points.tolist() == shape.points.tolist()

    def test_dim_shape(self):
        dim_polygon = Polygon(POINTS) * unit('µm')
        assert dim_polygon.bounding_box.width.m_as('µm') == pytest.approx(2.)
        assert dim_polygon.to('nm').shape.points[1] == pytest.approx([2000, 0])
