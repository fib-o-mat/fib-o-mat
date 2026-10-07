"""Tests of `fibomat.curve_tools.offset`."""
import math
import sys

import pytest

from fibomat.curve_tools import OffsetDirection, OffsetResult, deflate, inflate, offset, offset_with_islands
from fibomat.shapes import ArcSpline, Line


def square(size=4., x0=0., y0=0.):
    return ArcSpline([(x0, y0, 0), (x0 + size, y0, 0), (x0 + size, y0 + size, 0), (x0, y0 + size, 0)], True)


def circle(r=1.):
    return ArcSpline([(r, 0, 1), (-r, 0, 1)], True)


def open_line():
    return ArcSpline([(0, 0, 0), (4, 0, 0)], False)


def width(curve):
    return curve.bounding_box.width


class TestOffset:
    @pytest.mark.parametrize('curve', [square(), square().reversed()])
    def test_inwards_independent_of_orientation(self, curve):
        result = offset(curve, 1., 'inwards')
        assert len(result) == 1
        assert width(result[0]) == pytest.approx(2.)

    @pytest.mark.parametrize('curve', [square(), square().reversed()])
    def test_outwards_independent_of_orientation(self, curve):
        result = offset(curve, 1., OffsetDirection.OUTWARDS)
        assert len(result) == 1
        # corners are rounded with arcs, the extent grows by the distance on every side
        assert width(result[0]) == pytest.approx(6.)

    def test_outwards_area_of_circle(self):
        result = offset(circle(), 0.5, 'outwards')
        assert abs(result[0].area) == pytest.approx(math.pi * 1.5 ** 2)

    def test_inwards_area_of_circle(self):
        result = offset(circle(), 0.5, 'inwards')
        assert abs(result[0].area) == pytest.approx(math.pi * 0.5 ** 2)

    def test_left_and_right_of_closed_curve(self):
        ccw = square()
        assert ccw.orientation
        assert width(offset(ccw, 1., 'left')[0]) == pytest.approx(2.)
        assert width(offset(ccw, 1., 'right')[0]) == pytest.approx(6.)

    def test_vanishing_curve(self):
        assert offset(circle(), 2., 'inwards') == []

    def test_zero_distance_returns_curve(self):
        curve = square()
        assert offset(curve, 0., 'inwards') == [curve]

    def test_split(self):
        # dumbbell: two squares connected by a thin neck
        dumbbell = ArcSpline([
            (0, 0, 0), (4, 0, 0), (4, 1.8, 0), (6, 1.8, 0), (6, 0, 0), (10, 0, 0), (10, 4, 0), (6, 4, 0), (6, 2.2, 0),
            (4, 2.2, 0), (4, 4, 0), (0, 4, 0)
        ], True)
        assert len(offset(dumbbell, 0.5, 'inwards')) == 2

    def test_open_curve(self):
        left = offset(open_line(), 1.)
        assert len(left) == 1 and left[0].start.y == pytest.approx(1.)
        right = offset(open_line(), 1., 'right')
        assert right[0].start.y == pytest.approx(-1.)
        assert offset(open_line(), 1., 'left')[0].start.y == pytest.approx(1.)

    def test_open_curve_rejects_inwards_and_outwards(self):
        for direction in ('inwards', OffsetDirection.OUTWARDS):
            with pytest.raises(ValueError):
                offset(open_line(), 1., direction)

    def test_closed_curve_needs_direction(self):
        with pytest.raises(ValueError):
            offset(square(), 1.)

    @pytest.mark.parametrize('delta', [-1., math.inf, math.nan])
    def test_invalid_delta(self, delta):
        with pytest.raises(ValueError):
            offset(square(), delta, 'inwards')

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match='inwards'):
            offset(square(), 1., 'sideways')

    def test_type_error(self):
        with pytest.raises(TypeError):
            offset(Line((0, 0), (1, 0)), 1.)


class TestOffsetWithIslands:
    def test_grow_island(self):
        result = offset_with_islands([square(2.)], 0.5)
        assert isinstance(result, OffsetResult)
        assert len(result.islands) == 1
        assert width(result.islands[0]) == pytest.approx(3.)
        assert result.outer_curves == []

    def test_islands_merge(self):
        result = offset_with_islands([square(1.), square(1., 1.5, 0.)], 0.5)
        assert len(result.islands) == 1

    def test_outer_curve_shrinks(self):
        result = offset_with_islands([square(1., 1.5, 1.5)], 0.5, square(4.))
        assert len(result.outer_curves) == 1
        assert width(result.outer_curves[0]) == pytest.approx(3.)
        assert width(result.islands[0]) == pytest.approx(2.)

    def test_negative_orientation_island(self):
        with pytest.raises(ValueError):
            offset_with_islands([square().reversed()], 1.)

    def test_open_island(self):
        with pytest.raises(ValueError):
            offset_with_islands([open_line()], 1.)

    def test_open_outer_curve(self):
        with pytest.raises(ValueError):
            offset_with_islands([square()], 1., open_line())

    def test_negative_delta(self):
        with pytest.raises(ValueError):
            offset_with_islands([square()], -1.)

    def test_type_errors(self):
        with pytest.raises(TypeError):
            offset_with_islands([Line((0, 0), (1, 1))], 1.)
        with pytest.raises(TypeError):
            offset_with_islands([square()], 1., 'outer')


class TestDeflateInflate:
    def test_deflate_completely(self):
        curves = deflate(square(4.), 0.5)
        assert [round(width(c), 6) for c in curves] == [3., 2., 1.]

    def test_deflate_excludes_original(self):
        assert all(width(c) < 4. for c in deflate(square(4.), 0.5))

    def test_deflate_n_steps(self):
        assert [round(width(c), 6) for c in deflate(square(4.), 0.5, n_steps=2)] == [3., 2.]

    def test_deflate_zero_steps(self):
        assert deflate(square(4.), 0.5, n_steps=0) == []

    def test_deflate_distance(self):
        # the distance 1.5 is exactly three pitches (floating point division is not exact)
        assert len(deflate(square(4.), 0.1 * 5, distance=1.5)) == 3
        assert len(deflate(square(4.), 0.3, distance=0.9)) == 3
        assert len(deflate(square(4.), 0.5, distance=0.9)) == 1

    def test_deflate_does_not_accumulate_errors(self):
        curves = deflate(circle(1.), 0.1)
        assert len(curves) == 9
        for i, curve in enumerate(curves, start=1):
            assert abs(curve.area) == pytest.approx(math.pi * (1 - 0.1 * i) ** 2, rel=1e-9)

    def test_deflate_negative_orientation(self):
        assert len(deflate(square(4.).reversed(), 0.5)) == 3

    def test_inflate_n_steps(self):
        curves = inflate(circle(1.), 0.25, n_steps=3)
        assert [round(abs(c.area) / math.pi, 6) for c in curves] == [1.25 ** 2, 1.5 ** 2, 1.75 ** 2]

    def test_inflate_distance(self):
        assert len(inflate(circle(1.), 0.25, distance=1.)) == 4

    def test_inflate_needs_limit(self):
        with pytest.raises(ValueError):
            inflate(circle(), 0.25)

    @pytest.mark.parametrize('function', [deflate, inflate])
    def test_invalid_arguments(self, function):
        with pytest.raises(ValueError):
            function(circle(), 0.25, n_steps=1, distance=1.)
        with pytest.raises(ValueError):
            function(circle(), 0.)
        with pytest.raises(ValueError):
            function(circle(), -1., n_steps=1)
        with pytest.raises(ValueError):
            function(circle(), 0.25, n_steps=-1)
        with pytest.raises(ValueError):
            function(circle(), 0.25, n_steps=1.5)
        with pytest.raises(ValueError):
            function(circle(), 0.25, distance=-1.)

    @pytest.mark.parametrize('function', [deflate, inflate])
    def test_open_curve(self, function):
        with pytest.raises(ValueError):
            function(open_line(), 0.5, n_steps=1)

    @pytest.mark.parametrize('function', [deflate, inflate])
    def test_type_error(self, function):
        with pytest.raises(TypeError):
            function(Line((0, 0), (1, 1)), 0.5, n_steps=1)

    def test_safety(self, monkeypatch):
        # (the function `offset` shadows the module in the package namespace)
        offset_module = sys.modules['fibomat.curve_tools.offset']
        monkeypatch.setattr(offset_module, 'SAFETY', 3)
        with pytest.raises(RuntimeError):
            deflate(circle(1.), 0.01)
        with pytest.raises(RuntimeError):
            inflate(circle(1.), 0.01, n_steps=4)
