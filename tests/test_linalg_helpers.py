import math

import numpy as np
import pytest

from fibomat.linalg import Vector, VectorValueError
from fibomat.linalg.helpers import GeomLine, make_perp_vector
from fibomat.units import unit


class TestMakePerpVector:
    @pytest.mark.parametrize('vec, expected', [
        ((1, 0), (0, 1)),
        ((0, 1), (-1, 0)),
        ((-1, 0), (0, -1)),
        ((1, 2), (-2, 1)),
        ((0, 0), (0, 0)),
    ])
    def test_values(self, vec, expected):
        res = make_perp_vector(vec)
        assert isinstance(res, Vector)
        assert res == expected

    def test_properties(self):
        vec = Vector(3., 4.)
        perp = make_perp_vector(vec)
        assert vec.dot(perp) == pytest.approx(0.)
        assert perp.mag == pytest.approx(vec.mag)
        assert vec.cross(perp) > 0  # rotated counterclockwise
        assert make_perp_vector(perp) == -vec

    def test_accepts_vector_likes(self):
        assert make_perp_vector([1, 0]) == (0, 1)
        assert make_perp_vector(np.array([1., 0.])) == (0, 1)
        assert make_perp_vector(Vector(1, 0)) == (0, 1)

    def test_invalid(self):
        with pytest.raises(VectorValueError):
            make_perp_vector((1, 2, 3))
        with pytest.raises(VectorValueError):
            make_perp_vector('ab')
        with pytest.raises(VectorValueError):
            make_perp_vector(Vector(1, 2) * unit('m'))


class TestGeomLineConstruction:
    def test_basic(self):
        line = GeomLine((1, 2), (3, 4))
        assert line.direction == (1., 2.)
        assert line.support == (3., 4.)
        assert isinstance(line.direction, Vector)

    def test_vector_likes(self):
        line = GeomLine(Vector(1, 2), np.array([3., 4.]))
        assert line.direction == (1., 2.)
        assert line.support == (3., 4.)

    def test_null_direction(self):
        with pytest.raises(RuntimeError):
            GeomLine((0, 0), (1, 1))

    def test_tiny_direction_is_valid(self):
        line = GeomLine((1e-12, 0), (0, 0))
        assert line.find_param((1., 0.)) == pytest.approx(1e12)
        assert line(1e12) == pytest.approx((1., 0.))

    def test_invalid_arguments(self):
        with pytest.raises(VectorValueError):
            GeomLine((1, 2, 3), (0, 0))
        with pytest.raises(VectorValueError):
            GeomLine((1, 2), 'ab')
        with pytest.raises(ValueError):
            GeomLine((np.nan, 1), (0, 0))
        with pytest.raises(ValueError):
            GeomLine((1, 1), (np.inf, 0))

    def test_does_not_keep_references(self):
        direction = np.array([1., 0.])
        line = GeomLine(direction, (0, 0))
        direction[0] = 5.
        assert line.direction == (1., 0.)

    def test_repr(self):
        assert repr(GeomLine((1, 2), (3, 4))) == 'GeomLine(direction=(1.0, 2.0), support=(3.0, 4.0))'


class TestBisectors:
    def test_make_bisector(self):
        line = GeomLine.make_bisector((0, 0), (2, 2))
        assert line.direction == (2., 2.)
        assert line.support == (1., 1.)
        assert line.find_param((0, 0)) == pytest.approx(-0.5)
        assert line.find_param((2, 2)) == pytest.approx(0.5)

    def test_make_perp_bisector(self):
        line = GeomLine.make_perp_bisector((0, 0), (2, 0))
        assert line.support == (1., 0.)
        assert line.direction.dot((2, 0)) == pytest.approx(0.)
        # all points of the perpendicular bisector have the same distance to both points
        for u in (-2., 0., 3.5):
            point = Vector(line(u))
            assert (point - (0, 0)).mag == pytest.approx((point - (2, 0)).mag)

    def test_identical_points(self):
        with pytest.raises(RuntimeError):
            GeomLine.make_bisector((1, 1), (1, 1))
        with pytest.raises(RuntimeError):
            GeomLine.make_perp_bisector((1, 1), (1, 1))


class TestCall:
    def test_scalar(self):
        line = GeomLine((1, 2), (3, 4))
        res = line(2.)
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)
        assert res == pytest.approx((5., 8.))
        assert line(0) == pytest.approx((3., 4.))
        assert line(np.float64(-1.)) == pytest.approx((2., 2.))

    def test_array_of_parameters(self):
        line = GeomLine((1, 2), (3, 4))
        res = line([0., 1., 2.])
        assert res.shape == (3, 2)
        assert res == pytest.approx(np.array([[3., 4.], [4., 6.], [5., 8.]]))
        assert line(np.array([0., 1.])).shape == (2, 2)

    def test_does_not_modify_line(self):
        line = GeomLine((1, 2), (3, 4))
        line(1.)[0] = 100.
        assert line.support == (3., 4.)


class TestParallel:
    def test_parallel(self):
        assert GeomLine((1, 0), (0, 0)).parallel_to(GeomLine((2, 0), (5, 5)))
        assert GeomLine((1, 1), (0, 0)).parallel_to(GeomLine((-3, -3), (1, 0)))  # antiparallel
        assert GeomLine((1, 0), (0, 0)).parallel_to(GeomLine((1, 0), (0, 0)))  # identical
        assert GeomLine((1, 0), (0, 0)).parallel_to(GeomLine((-1, 0), (0, 0)))

    def test_not_parallel(self):
        assert not GeomLine((1, 0), (0, 0)).parallel_to(GeomLine((0, 1), (0, 0)))
        assert not GeomLine((1, 0), (0, 0)).parallel_to(GeomLine((1, 1), (0, 0)))

    def test_small_angles_are_not_parallel(self):
        # the former criterion (cos close to 1) treated angles up to ~0.26 degree as parallel
        assert not GeomLine((1, 0), (0, 0)).parallel_to(GeomLine((1, 1e-3), (0, 0)))
        assert not GeomLine((1, 0), (0, 0)).parallel_to(GeomLine((-1, 1e-3), (0, 0)))
        assert GeomLine((1, 0), (0, 0)).parallel_to(GeomLine((1, 1e-3), (0, 0)), tol=1e-2)

    def test_length_of_direction_is_irrelevant(self):
        assert GeomLine((1e-6, 0), (0, 0)).parallel_to(GeomLine((1e6, 0), (0, 0)))
        assert not GeomLine((1e-6, 0), (0, 0)).parallel_to(GeomLine((0, 1e6), (0, 0)))

    def test_symmetric(self):
        a, b = GeomLine((1, 0.5), (0, 0)), GeomLine((2, 1), (3, 3))
        assert a.parallel_to(b) and b.parallel_to(a)

    def test_invalid(self):
        with pytest.raises(TypeError):
            GeomLine((1, 0), (0, 0)).parallel_to((1, 0))


class TestIntersection:
    def test_perpendicular_lines(self):
        line_1 = GeomLine((1, 0), (0, 0))
        line_2 = GeomLine((0, 1), (2, -1))
        assert line_1.intersect_at_param(line_2) == pytest.approx(2.)
        assert line_2.intersect_at_param(line_1) == pytest.approx(1.)
        assert line_1.intersect_at(line_2) == pytest.approx((2., 0.))
        assert line_2.intersect_at(line_1) == pytest.approx((2., 0.))
        assert isinstance(line_1.intersect_at_param(line_2), float)

    def test_general_lines(self):
        line_1 = GeomLine((1, 1), (0, 0))
        line_2 = GeomLine((-1, 1), (4, 0))
        assert line_1.intersect_at(line_2) == pytest.approx((2., 2.))
        point = line_1.intersect_at(line_2)
        assert line_1.find_param(point) == pytest.approx(2.)
        assert line_2.find_param(point) == pytest.approx(2.)

    def test_parameter_scales_with_direction(self):
        line = GeomLine((2, 0), (0, 0))
        other = GeomLine((0, 1), (3, -1))
        assert line.intersect_at_param(other) == pytest.approx(1.5)
        assert line.intersect_at(other) == pytest.approx((3., 0.))

    def test_intersection_outside_of_support(self):
        line_1 = GeomLine((1, 0), (0, 0))
        line_2 = GeomLine((0, 1), (-5, 0))
        assert line_1.intersect_at_param(line_2) == pytest.approx(-5.)

    def test_parallel_lines(self):
        with pytest.raises(ValueError):
            GeomLine((1, 0), (0, 0)).intersect_at_param(GeomLine((1, 0), (0, 1)))
        with pytest.raises(ValueError):
            GeomLine((1, 0), (0, 0)).intersect_at(GeomLine((-1, 0), (0, 1)))
        with pytest.raises(ValueError):
            GeomLine((1, 0), (0, 0)).intersect_at(GeomLine((1, 0), (0, 0)))  # identical lines

    def test_bisector_intersection(self):
        # circumcenter of the triangle (0, 0), (4, 0), (0, 4) is the intersection of two perpendicular bisectors
        line_1 = GeomLine.make_perp_bisector((0, 0), (4, 0))
        line_2 = GeomLine.make_perp_bisector((0, 0), (0, 4))
        assert line_1.intersect_at(line_2) == pytest.approx((2., 2.))

    def test_invalid(self):
        with pytest.raises(TypeError):
            GeomLine((1, 0), (0, 0)).intersect_at_param((1, 0))
        with pytest.raises(TypeError):
            GeomLine((1, 0), (0, 0)).intersect_at((1, 0))


class TestFindParam:
    def test_horizontal_vertical_and_general(self):
        assert GeomLine((2, 0), (1, 1)).find_param((5, 1)) == pytest.approx(2.)
        assert GeomLine((0, 2), (1, 1)).find_param((1, -3)) == pytest.approx(-2.)
        assert GeomLine((1, 2), (1, 1)).find_param((3, 5)) == pytest.approx(2.)

    def test_support_and_roundtrip(self):
        line = GeomLine((0.3, -0.7), (1., 2.))
        assert line.find_param((1., 2.)) == pytest.approx(0.)
        for u in (-3., 0.25, 10.):
            assert line.find_param(line(u)) == pytest.approx(u)

    def test_returns_float(self):
        assert isinstance(GeomLine((1, 0), (0, 0)).find_param((1, 0)), float)

    def test_point_not_on_line(self):
        with pytest.raises(ValueError):
            GeomLine((1, 0), (0, 0)).find_param((1, 1))
        with pytest.raises(ValueError):
            GeomLine((0, 1), (0, 0)).find_param((1, 1))
        with pytest.raises(ValueError):
            GeomLine((1, 1), (0, 0)).find_param((1, 2))

    def test_nearly_axis_parallel_line(self):
        # the former implementation treated direction components below 1e-8 as zero
        line = GeomLine((1e-9, 1.), (0., 0.))
        assert line.find_param((1e-9, 1.)) == pytest.approx(1.)
        assert line.find_param((2e-9, 2.)) == pytest.approx(2.)

    def test_no_output(self, capsys):
        # the former implementation printed debugging output
        with pytest.raises(ValueError):
            GeomLine((1, 1), (0, 0)).find_param((1, 2))
        GeomLine((1, 1), (0, 0)).find_param((1, 1))
        assert capsys.readouterr().out == ''

    def test_point_on_line_within_tolerance(self):
        assert GeomLine((1, 1), (0, 0)).find_param((1., 1. + 1e-12)) == pytest.approx(1.)

    def test_invalid_point(self):
        with pytest.raises(VectorValueError):
            GeomLine((1, 0), (0, 0)).find_param((1, 2, 3))
