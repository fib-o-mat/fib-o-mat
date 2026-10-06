import copy
import math

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, Vector, VectorValueError, mirror, rotate, scale, translate
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.line import Line
from fibomat.shapes.shape import Shape
from fibomat.units import unit


class TestConstruction:
    def test_basic(self):
        line = Line((0, 0), (3, 4), description='foo')
        assert line.start == (0., 0.)
        assert line.end == (3., 4.)
        assert type(line.start) is Vector
        assert line.description == 'foo'
        assert isinstance(line, Shape) and isinstance(line, ArcSplineCompatible)

    @pytest.mark.parametrize('start, end', [
        ([0, 0], [1, 1]), (np.array([0., 0.]), np.array([1., 1.])), (Vector(0, 0), Vector(1, 1)),
    ])
    def test_vector_likes(self, start, end):
        line = Line(start, end)
        assert line.start == (0., 0.) and line.end == (1., 1.)

    def test_invalid(self):
        with pytest.raises(VectorValueError):
            Line((0, 0, 0), (1, 1))
        with pytest.raises(VectorValueError):
            Line((0, 0), 'ab')
        with pytest.raises(VectorValueError):
            Line((0, 0), Vector(1, 1) * unit('µm'))
        with pytest.raises(ValueError):
            Line((0, 0), (np.nan, 1))
        with pytest.raises(ValueError):
            Line((np.inf, 0), (1, 1))

    def test_repr(self):
        # the former repr had no closing bracket
        assert repr(Line((0, 0), (1, 2))) == 'Line(start=Vector(x=0.0, y=0.0), end=Vector(x=1.0, y=2.0))'

    def test_zero_length_line(self):
        line = Line((1, 1), (1, 1))
        assert line.length == 0.
        assert line.distance_to_point((4, 5)) == pytest.approx(5.)
        assert line.intersection(Line((0, 0), (2, 2))) is None


class TestProperties:
    def test_length(self):
        line = Line((0, 0), (3, 4))
        assert line.length == pytest.approx(5.)
        assert line.boundary_length == pytest.approx(5.)

    def test_center_and_bounding_box(self):
        line = Line((1, 5), (3, 1))
        assert line.center == (2., 3.)
        assert line.bounding_box == BoundingBox((1, 1), (3, 5))
        assert type(line.center) is Vector

    def test_is_closed(self):
        assert Line((0, 0), (1, 1)).is_closed is False

    def test_area_is_not_defined(self):
        with pytest.raises(NotImplementedError):
            Line((0, 0), (1, 1)).area

    def test_to_arc_spline(self):
        line = Line((0, 0), (3, 4), description='foo')
        spline = line.to_arc_spline()
        assert isinstance(spline, ArcSpline)
        assert not spline.is_closed
        assert spline.vertices.tolist() == [[0, 0, 0], [3, 4, 0]]
        assert spline.description == 'foo'
        assert Line((0, 0), (1, 1)).to_arc_spline().description is None

    def test_to_arc_spline_is_a_copy(self):
        line = Line((0, 0), (3, 4))
        spline = line.to_arc_spline()
        spline._impl_translate(Vector(1, 1))
        assert line.start == (0., 0.)
        line._impl_translate(Vector(2, 2))
        assert spline.start == (1., 1.)

    def test_to_arc_spline_without_deprecation_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Line((0, 0), (1, 1), description='foo').to_arc_spline()


class TestDistanceToPoint:
    def test_perpendicular_foot_on_segment(self):
        assert Line((0, 0), (4, 0)).distance_to_point((2, 3)) == pytest.approx(3.)
        assert Line((0, 0), (0, 4)).distance_to_point((-2, 1)) == pytest.approx(2.)

    def test_points_beyond_the_ends(self):
        # distance to the segment, not to the infinite line
        line = Line((0, 0), (4, 0))
        assert line.distance_to_point((7, 4)) == pytest.approx(5.)
        assert line.distance_to_point((-3, -4)) == pytest.approx(5.)
        assert line.distance_to_point((5, 0)) == pytest.approx(1.)

    def test_points_on_the_line(self):
        assert Line((0, 0), (4, 4)).distance_to_point((2, 2)) == 0.
        assert Line((0, 0), (4, 4)).distance_to_point((0, 0)) == 0.

    def test_oblique_line(self):
        assert Line((0, 0), (2, 2)).distance_to_point((2, 0)) == pytest.approx(math.sqrt(2))

    def test_vector_likes(self):
        assert Line((0, 0), (4, 0)).distance_to_point(Vector(2, 3)) == pytest.approx(3.)
        assert Line((0, 0), (4, 0)).distance_to_point([2, 3]) == pytest.approx(3.)
        assert isinstance(Line((0, 0), (4, 0)).distance_to_point((2, 3)), float)

    def test_no_output(self, capsys):
        # the former implementation printed debugging output
        Line((0, 0), (4, 0)).distance_to_point((2, 3))
        assert capsys.readouterr().out == ''

    def test_invalid_point(self):
        with pytest.raises(VectorValueError):
            Line((0, 0), (4, 0)).distance_to_point((1, 2, 3))


class TestIntersection:
    def test_crossing_segments(self):
        res = Line((0, 0), (3, 4)).intersection(Line((0, 4), (3, 0)))
        assert type(res) is Vector
        assert res == pytest.approx((1.5, 2.))

    def test_endpoint_touches(self):
        assert Line((0, 0), (1, 1)).intersection(Line((1, 1), (2, 0))) == pytest.approx((1., 1.))
        assert Line((0, 0), (2, 0)).intersection(Line((1, 0), (1, 5))) == pytest.approx((1., 0.))

    def test_no_intersection_within_the_segments(self):
        # the lines would intersect but the segments do not; the former implementation raised an IndexError
        assert Line((0, 0), (1, 0)).intersection(Line((2, -1), (2, 1))) is None
        assert Line((0, 0), (1, 1)).intersection(Line((3, 0), (0, 3))) is None  # would be at (1.5, 1.5)
        assert Line((0, 0), (1, 1)).intersection(Line((5, 0), (5, 3))) is None

    def test_parallel_lines(self):
        assert Line((0, 0), (1, 0)).intersection(Line((0, 1), (1, 1))) is None
        assert Line((0, 0), (2, 0)).intersection(Line((1, 0), (3, 0))) is None  # overlapping

    def test_symmetric(self):
        a, b = Line((0, 0), (3, 4)), Line((0, 4), (3, 0))
        assert a.intersection(b) == pytest.approx(b.intersection(a))

    def test_invalid(self):
        with pytest.raises(TypeError):
            Line((0, 0), (1, 1)).intersection((0, 0))


class TestTransformations:
    def test_translate(self):
        line = Line((0, 0), (3, 4))
        res = line.translated((1, -1))
        assert res.start == (1., -1.) and res.end == (4., 3.)
        assert isinstance(res, Line)
        assert line.start == (0., 0.)

    def test_rotate(self):
        res = Line((1, 0), (2, 0)).rotated(math.pi / 2)
        assert res.start == pytest.approx((0., 1.)) and res.end == pytest.approx((0., 2.))
        res = Line((1, 0), (3, 0)).rotated(math.pi, 'center')
        assert res.start == pytest.approx((3., 0.)) and res.end == pytest.approx((1., 0.))

    def test_scale(self):
        res = Line((1, 1), (2, 3)).scaled(2.)
        assert res.start == (2., 2.) and res.end == (4., 6.)
        assert res.length == pytest.approx(2 * Line((1, 1), (2, 3)).length)
        res = Line((1, 1), (3, 1)).scaled(2., 'center')
        assert res.start == pytest.approx((0., 1.)) and res.end == pytest.approx((4., 1.))

    def test_mirror(self):
        res = Line((1, 2), (3, 4)).mirrored((1, 0))
        assert res.start == pytest.approx((1., -2.)) and res.end == pytest.approx((3., -4.))

    def test_transformed(self):
        res = Line((1, 0), (2, 0)).transformed(translate((1, 1)) | rotate(math.pi / 2) | scale(2.) | mirror((1, 0)))
        assert res.start == pytest.approx((-2., -4.))
        assert res.end == pytest.approx((-2., -6.))

    def test_translated_to_and_pivot(self):
        line = Line((0, 0), (2, 2))
        assert line.translated_to((5, 5)).center == (5., 5.)

    def test_does_not_modify_original(self):
        line = Line((0, 0), (1, 1))
        line.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert line.start == (0., 0.) and line.end == (1., 1.)


class TestCopyAndDimShape:
    def test_clone(self):
        line = Line((0, 0), (1, 1), description='foo')
        clone = line.clone()
        assert clone is not line
        assert clone.start == line.start and clone.end == line.end
        assert clone.description == 'foo'
        clone._impl_translate(Vector(1, 1))
        assert line.start == (0., 0.)
        assert copy.deepcopy(line).end == line.end

    def test_dim_shape(self):
        dim_line = Line((0, 0), (3, 4)) * unit('µm')
        assert dim_line.center == Vector(1.5, 2) * unit('µm')
        assert dim_line.to('nm').shape.end == pytest.approx((3000., 4000.))
