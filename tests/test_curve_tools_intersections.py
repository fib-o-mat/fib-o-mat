"""Tests of `fibomat.curve_tools.intersections` and `fibomat.curve_tools.combine`."""
import numpy as np
import pytest

from fibomat.curve_tools import (
    CombineMode, CombineResult, CurveIntersections, Intersection, Overlap, combine_curves, curve_intersections,
    self_intersections
)
from fibomat.linalg import Vector
from fibomat.shapes import ArcSpline, Line


def line(x0, y0, x1, y1):
    return ArcSpline([(x0, y0, 0), (x1, y1, 0)], False)


def circle(cx=0., cy=0., r=1.):
    return ArcSpline([(cx + r, cy, 1), (cx - r, cy, 1)], True)


def square(x0, y0, size):
    return ArcSpline([(x0, y0, 0), (x0 + size, y0, 0), (x0 + size, y0 + size, 0), (x0, y0 + size, 0)], True)


def positions(result):
    return sorted((round(p.position.x, 9), round(p.position.y, 9)) for p in result.points)


class TestCurveIntersections:
    def test_result_types(self):
        result = curve_intersections(line(0, 0, 1, 1), line(0, 1, 1, 0))
        assert isinstance(result, CurveIntersections)
        points, overlaps = result  # NamedTuple unpacking
        assert overlaps == []
        assert isinstance(points[0], Intersection)
        assert isinstance(points[0].position, Vector)
        assert points[0].position == Vector(0.5, 0.5)
        assert (points[0].index_1, points[0].index_2) == (0, 0)

    def test_crossing(self):
        assert positions(curve_intersections(line(0, 0, 2, 2), line(0, 2, 2, 0))) == [(1., 1.)]

    def test_no_intersection(self):
        result = curve_intersections(line(0, 0, 1, 0), line(5, 5, 6, 6))
        assert result.points == []
        assert result.overlaps == []

    def test_parallel_lines(self):
        result = curve_intersections(line(0, 0, 1, 0), line(0, 1, 1, 1))
        assert result.points == [] and result.overlaps == []

    @pytest.mark.parametrize('first, second', [
        (line(0, 0, 1, 0), line(1, 0, 1, 1)),  # end - start
        (line(0, 0, 1, 0), line(0, 0, 0, 1)),  # start - start
        (line(0, 0, 1, 0), line(0, 1, 0, 0)),  # start - end
        (line(0, 0, 1, 0), line(1, 1, 1, 0)),  # end - end
    ])
    def test_touching_end_points(self, first, second):
        result = curve_intersections(first, second)
        assert len(result.points) == 1 and not result.overlaps

    def test_end_point_touches_middle_of_line(self):
        assert positions(curve_intersections(line(0, 0, 2, 0), line(1, 0, 1, 1))) == [(1., 0.)]
        assert positions(curve_intersections(line(1, 0, 1, 1), line(0, 0, 2, 0))) == [(1., 0.)]

    def test_overlapping_lines(self):
        result = curve_intersections(line(0, 0, 2, 0), line(1, 0, 3, 0))
        assert result.points == []
        assert len(result.overlaps) == 1
        overlap = result.overlaps[0]
        assert isinstance(overlap, Overlap)
        assert {tuple(overlap.start), tuple(overlap.end)} == {(1., 0.), (2., 0.)}

    def test_identical_lines(self):
        result = curve_intersections(line(0, 0, 1, 1), line(0, 0, 1, 1))
        assert result.points == []
        assert len(result.overlaps) == 1

    def test_identical_circles(self):
        result = curve_intersections(circle(), circle())
        assert result.points == []
        assert len(result.overlaps) == 2

    def test_two_circles(self):
        result = curve_intersections(circle(), circle(1., 0., 1.))
        assert positions(result) == [(0.5, -0.866025404), (0.5, 0.866025404)]
        # each intersection is on a different half circle
        assert sorted(p.index_1 for p in result.points) == [0, 1]

    def test_circle_line_tangent(self):
        assert positions(curve_intersections(circle(), line(-2, 1, 2, 1))) == [(0., 1.)]

    def test_circle_line_two_points(self):
        assert positions(curve_intersections(circle(), line(-2, 0.5, 2, 0.5))) == [
            (-0.866025404, 0.5), (0.866025404, 0.5)
        ]

    def test_crossing_at_vertex_is_reported_once(self):
        poly = ArcSpline([(0, 0, 0), (1, 1, 0), (2, 0, 0)], False)
        assert positions(curve_intersections(poly, line(1, 0, 1, 2))) == [(1., 1.)]

    def test_indices_refer_to_segments(self):
        poly = ArcSpline([(0, 0, 0), (2, 0, 0), (2, 2, 0)], False)
        result = curve_intersections(poly, line(1.5, -1, 1.5, 1))
        assert [(p.index_1, p.index_2) for p in result.points] == [(0, 0)]
        result = curve_intersections(poly, line(1, 1, 3, 1))
        assert [(p.index_1, p.index_2) for p in result.points] == [(1, 0)]

    def test_symmetry(self):
        first, second = circle(), line(-2, 0.5, 2, 0.5)
        assert positions(curve_intersections(first, second)) == positions(curve_intersections(second, first))

    def test_type_errors(self):
        with pytest.raises(TypeError):
            curve_intersections(Line((0, 0), (1, 1)), line(0, 0, 1, 1))
        with pytest.raises(TypeError):
            curve_intersections(line(0, 0, 1, 1), 'foo')


class TestSelfIntersections:
    def test_simple_curves_have_none(self):
        assert self_intersections(square(0, 0, 1)) == []
        assert self_intersections(circle()) == []
        assert self_intersections(line(0, 0, 1, 1)) == []

    def test_bow_tie(self):
        bow_tie = ArcSpline([(0, 0, 0), (2, 2, 0), (2, 0, 0), (0, 2, 0)], True)
        result = self_intersections(bow_tie)
        assert len(result) == 1
        assert isinstance(result[0], Intersection)
        assert result[0].position == Vector(1, 1)
        assert (result[0].index_1, result[0].index_2) == (0, 2)

    def test_indices_are_sorted(self):
        curve = ArcSpline([(0, 0, 0), (2, 2, 0), (2, 0, 0), (0, 2, 0)], False)
        for hit in self_intersections(curve):
            assert hit.index_1 <= hit.index_2

    def test_type_error(self):
        with pytest.raises(TypeError):
            self_intersections(Line((0, 0), (1, 1)))


class TestCombineCurves:
    @staticmethod
    def areas(curves):
        return sorted(round(abs(curve.area), 9) for curve in curves)

    @pytest.mark.parametrize('mode, boundaries', [
        (CombineMode.UNION, [7.]),
        (CombineMode.INTERSECT, [1.]),
        (CombineMode.EXCLUDE, [3.]),
        (CombineMode.XOR, [3., 3.]),
    ])
    def test_overlapping_squares(self, mode, boundaries):
        result = combine_curves(square(0, 0, 2), square(1, 1, 2), mode)
        assert isinstance(result, CombineResult)
        assert self.areas(result.boundaries) == boundaries
        assert result.holes == []

    def test_string_modes(self):
        for mode in CombineMode:
            assert self.areas(combine_curves(square(0, 0, 2), square(1, 1, 2), mode.value).boundaries) == \
                self.areas(combine_curves(square(0, 0, 2), square(1, 1, 2), mode).boundaries)

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match='union'):
            combine_curves(square(0, 0, 2), square(1, 1, 2), 'foo')

    def test_hole_inside(self):
        result = combine_curves(square(0, 0, 2), square(0.5, 0.5, 1), 'exclude')
        assert self.areas(result.boundaries) == [4.]
        assert self.areas(result.holes) == [1.]

    def test_union_of_contained_curve(self):
        result = combine_curves(square(0, 0, 2), square(0.5, 0.5, 1), 'union')
        assert self.areas(result.boundaries) == [4.] and result.holes == []

    def test_intersection_of_contained_curve(self):
        result = combine_curves(square(0, 0, 2), square(0.5, 0.5, 1), 'intersect')
        assert self.areas(result.boundaries) == [1.]

    def test_disjoint_curves(self):
        first, second = square(0, 0, 1), square(5, 5, 1)
        assert self.areas(combine_curves(first, second, 'union').boundaries) == [1., 1.]
        assert combine_curves(first, second, 'intersect').boundaries == []
        assert self.areas(combine_curves(first, second, 'exclude').boundaries) == [1.]

    def test_orientation_does_not_matter(self):
        first, second = square(0, 0, 2), square(1, 1, 2)
        for a in (first, first.reversed()):
            for b in (second, second.reversed()):
                assert self.areas(combine_curves(a, b, 'union').boundaries) == [7.]
                assert self.areas(combine_curves(a, b, 'intersect').boundaries) == [1.]

    def test_circles(self):
        result = combine_curves(circle(), circle(1., 0., 1.), 'intersect')
        assert len(result.boundaries) == 1
        # lens area of two unit circles with distance 1
        expected = 2 * np.arccos(0.5) - 0.5 * np.sqrt(3)
        assert abs(result.boundaries[0].area) == pytest.approx(expected, rel=1e-9)

    def test_results_are_closed_arc_splines(self):
        result = combine_curves(square(0, 0, 2), square(1, 1, 2), 'union')
        assert all(isinstance(curve, ArcSpline) and curve.is_closed for curve in result.boundaries)

    def test_open_curve_raises(self):
        with pytest.raises(ValueError):
            combine_curves(line(0, 0, 1, 1), square(0, 0, 1), 'union')

    def test_type_error(self):
        with pytest.raises(TypeError):
            combine_curves(Line((0, 0), (1, 1)), square(0, 0, 1), 'union')
