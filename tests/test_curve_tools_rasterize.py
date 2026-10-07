"""Tests of `fibomat.curve_tools.rasterize`."""
import math

import numpy as np
import pytest

from fibomat.curve_tools import rasterize, rasterize_with_const_error
from fibomat.shapes import ArcSpline, Circle, Line, ParametricCurve, Polygon, Polyline, Rect, Spot
from fibomat.shapes._line_non_continuous import LineNonContinuous
from fibomat.shapes.rasterizedpoints import RasterizedPoints


def circle(r=1.):
    return ArcSpline([(r, 0, 1), (-r, 0, 1)], True)


def square(size=2.):
    return ArcSpline([(0, 0, 0), (size, 0, 0), (size, size, 0), (0, size, 0)], True)


def spacing(points, closed=False):
    positions = points.positions
    if closed:
        positions = np.vstack((positions, positions[:1]))
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


class TestRasterize:
    def test_returns_rasterized_points(self):
        result = rasterize(Line((0, 0), (1, 0)), 0.25)
        assert isinstance(result, RasterizedPoints)
        assert not result.is_closed
        assert np.all(result.weights == 1.)

    def test_line(self):
        result = rasterize(Line((0, 0), (1, 0)), 0.25)
        assert result.positions[:, 0].tolist() == pytest.approx([0., 0.25, 0.5, 0.75, 1.])
        assert np.all(result.positions[:, 1] == 0.)

    def test_line_end_not_a_multiple_of_pitch(self):
        result = rasterize(Line((0, 0), (1, 1)), 0.5)
        assert result.n_points == 3
        assert np.allclose(result.positions[-1], (0.5 * math.sqrt(2), 0.5 * math.sqrt(2)))

    def test_start_is_first_point(self):
        result = rasterize(Line((1, 2), (3, 4)), 0.3)
        assert np.allclose(result.positions[0], (1., 2.))

    def test_pitch_along_polyline(self):
        curve = ArcSpline([(0, 0, 0), (1, 0, 0), (1, 1, 0), (3, 1, 0)], False)
        result = rasterize(curve, 0.3)
        # the pitch is measured along the curve, points on a corner are closer (in air line distance)
        assert result.n_points == int(curve.length / 0.3) + 1
        assert np.allclose(result.positions[0], (0., 0.))
        # arc length 0.3 * 4 = 1.2 is on the second segment
        assert np.allclose(result.positions[4], (1., 0.2))

    def test_arc_spacing(self):
        arc = ArcSpline([(1, 0, math.tan(math.pi / 4)), (-1, 0, 0)], False)  # half circle
        result = rasterize(arc, 0.1)
        assert result.n_points == int(math.pi / 0.1) + 1
        radius = np.linalg.norm(result.positions, axis=1)
        assert np.allclose(radius, 1.)
        # chord = 2 sin(0.05)
        assert np.allclose(spacing(result), 2 * math.sin(0.05))

    def test_clockwise_arc(self):
        arc = ArcSpline([(1, 0, -1), (-1, 0, 0)], False)  # lower half circle (clockwise)
        result = rasterize(arc, 0.1)
        assert np.all(result.positions[1:, 1] < 0.)
        assert np.allclose(np.linalg.norm(result.positions, axis=1), 1.)

    def test_closed_curve_has_no_duplicate_point(self):
        result = rasterize(square(2.), 0.5)
        assert result.is_closed
        assert result.n_points == 16
        assert len(np.unique(np.round(result.positions, 9), axis=0)) == 16

    def test_closed_curve_spacing_including_closing_segment(self):
        result = rasterize(circle(1.), 2 * math.pi / 100)
        assert result.n_points == 100
        assert np.allclose(spacing(result, closed=True), 2 * math.sin(math.pi / 100))

    def test_closed_curve_not_a_multiple(self):
        result = rasterize(square(2.), 0.3)
        assert result.n_points == int(8 / 0.3) + 1

    def test_points_lie_on_curve(self):
        curve = ArcSpline([(0, 0, 0.3), (2, 0, -0.5), (3, 2, 0), (0, 3, 0.2)], True)
        result = rasterize(curve, 0.07)
        for position in result.positions[::7]:
            assert curve.closest_point(position)['distance'] < 1e-9

    def test_arc_length_parametrization_of_mixed_curve(self):
        curve = ArcSpline([(0, 0, 0.3), (2, 0, -0.5), (3, 2, 0), (0, 3, 0.2)], True)
        pitch = 0.05
        fine = rasterize(curve, 1e-4).positions
        coarse = rasterize(curve, pitch)
        # every coarse point corresponds to the fine point with the index pitch / 1e-4
        step = int(round(pitch / 1e-4))
        assert np.allclose(coarse.positions[:len(fine[::step])], fine[::step], atol=1e-6)

    def test_degenerate_segments_are_ignored(self):
        curve = ArcSpline([(0, 0, 0), (0, 0, 0), (1, 0, 0)], False)
        assert rasterize(curve, 0.5).n_points == 3

    def test_no_length(self):
        with pytest.raises(ValueError):
            rasterize(ArcSpline([(0, 0, 0), (0, 0, 0)], False), 0.5)

    def test_arc_spline_compatible(self):
        result = rasterize(Circle(1.), 0.1)
        assert result.is_closed
        assert result.n_points == int(2 * math.pi / 0.1) + 1

        assert rasterize(Rect(2, 2), 1.).n_points == 8

    def test_line_non_continuous(self):
        lines = LineNonContinuous([Line((0, 0), (1, 0)), Line((5, 5), (5, 6))])
        result = rasterize(lines, 0.5)
        # the gap is ignored; the pitch continues on the second line
        assert np.allclose(result.positions, [[0, 0], [0.5, 0], [1, 0], [5, 5.5], [5, 6]])
        assert not result.is_closed

    def test_parametric_curve(self):
        curve = ParametricCurve(
            lambda u: np.stack((np.cos(u), np.sin(u)), axis=-1),
            lambda u: np.stack((-np.sin(u), np.cos(u)), axis=-1),
            lambda u: np.stack((-np.cos(u), -np.sin(u)), axis=-1),
            (0., math.pi),
        )
        result = rasterize(curve, 0.1)
        assert result.n_points == int(math.pi / 0.1) + 1
        assert np.allclose(np.linalg.norm(result.positions, axis=1), 1.)
        assert np.allclose(spacing(result), 2 * math.sin(0.05), atol=1e-6)

    @pytest.mark.parametrize('pitch', [0., -1., math.inf, math.nan])
    def test_invalid_pitch(self, pitch):
        with pytest.raises(ValueError):
            rasterize(Line((0, 0), (1, 0)), pitch)

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            rasterize(Spot((0, 0)), 0.1)
        with pytest.raises(TypeError):
            rasterize('foo', 0.1)


class TestRasterizeWithConstError:
    def test_closed_returns_polygon(self):
        result = rasterize_with_const_error(circle(1.), 0.01)
        assert isinstance(result, Polygon)

    def test_open_returns_polyline(self):
        result = rasterize_with_const_error(ArcSpline([(1, 0, 1), (-1, 0, 0)], False), 0.01)
        assert isinstance(result, Polyline)

    def test_error_bound(self):
        error = 0.01
        result = rasterize_with_const_error(circle(1.), error)
        points = np.asarray(result.points)
        assert np.allclose(np.linalg.norm(points, axis=1), 1.)  # polygon vertices are on the circle
        # maximum distance of the edge midpoints to the circle
        mid = (points + np.roll(points, -1, axis=0)) / 2
        assert np.max(1 - np.linalg.norm(mid, axis=1)) <= error * (1 + 1e-9)

    def test_lines_are_kept(self):
        result = rasterize_with_const_error(square(2.), 0.1)
        assert len(np.asarray(result.points)) == 4

    def test_invalid_error(self):
        for error in (0., -1., math.inf, math.nan):
            with pytest.raises(ValueError):
                rasterize_with_const_error(circle(), error)

    def test_type_error(self):
        with pytest.raises(TypeError):
            rasterize_with_const_error(Circle(1.), 0.1)
