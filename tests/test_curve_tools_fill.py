"""Tests of `fibomat.curve_tools.fill`."""
import math

import numpy as np
import pytest

from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline
from fibomat.curve_tools import fill_with_lines, fill_with_spiral
from fibomat.linalg import rotate
from fibomat.shapes import ArcSpline, Line, ParametricCurve


def rect(x0, y0, x1, y1):
    return ArcSpline([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], True)


def circle(r=1., cx=0., cy=0.):
    return ArcSpline([(cx + r, cy, 1), (cx - r, cy, 1)], True)


def total_length(rows):
    return sum(line.length for row in rows for line in row)


def flat(rows):
    return [line for row in rows for line in row]


class TestFillWithLines:
    def test_rectangle(self):
        rows = fill_with_lines(rect(-1, -1, 1, 1), 0.5, 0., False)
        assert all(len(row) == 1 for row in rows)
        # y in [y_min, y_max): the line on the top is not part of the fill
        assert [row[0].start.y for row in rows] == pytest.approx([0.5, 0., -0.5, -1.])
        for row in rows:
            assert row[0].start.x == pytest.approx(-1.)
            assert row[0].end.x == pytest.approx(1.)
        assert isinstance(rows[0][0], Line)

    def test_invert_reverses_rows(self):
        rows = fill_with_lines(rect(-1, -1, 1, 1), 0.5, 0., False)
        inverted = fill_with_lines(rect(-1, -1, 1, 1), 0.5, 0., True)
        assert [row[0].start.y for row in inverted] == pytest.approx([-1., -0.5, 0., 0.5])
        # same lines, the grid does not depend on invert
        for row, inverted_row in zip(rows, reversed(inverted)):
            assert row[0].start == inverted_row[0].start and row[0].end == inverted_row[0].end

    @pytest.mark.parametrize('invert', [False, True])
    def test_seed_aligns_grid(self, invert):
        rows = fill_with_lines(rect(0, 0, 3, 3), 0.7, 0., invert, seed=(10., 0.25))
        ys = np.array([row[0].start.y for row in rows])
        assert np.allclose((ys - 0.25 + 0.35) % 0.7, 0.35)
        assert ys.min() >= 0. and ys.max() < 3.
        assert sorted(ys.tolist()) == pytest.approx([0.25, 0.95, 1.65, 2.35])

    def test_seed_grid_with_rotation(self):
        alpha = 0.4
        seed = np.array((1., 2.))
        rows = fill_with_lines(circle(3.), 0.5, alpha, False, seed=seed)
        normal = np.array((-math.sin(alpha), math.cos(alpha)))
        for row in rows:
            for line in row:
                height = np.dot(np.asarray(line.start) - seed, normal)
                assert height / 0.5 == pytest.approx(round(height / 0.5), abs=1e-9)
                direction = np.asarray(line.end - line.start)
                assert direction / np.linalg.norm(direction) == pytest.approx((math.cos(alpha), math.sin(alpha)))

    def test_circle_area(self):
        pitch = 0.01
        rows = fill_with_lines(circle(1.), pitch, 0., False)
        assert total_length(rows) * pitch == pytest.approx(math.pi, rel=1e-3)
        for line in flat(rows):
            assert np.linalg.norm(line.start) == pytest.approx(1.)
            assert np.linalg.norm(line.end) == pytest.approx(1.)

    @pytest.mark.parametrize('alpha', [-math.pi / 2, -0.7, 0., 0.3, math.pi / 4, math.pi / 2])
    def test_rotated_shape_area(self, alpha):
        shape = ArcSpline([(0, 0, 0.2), (4, 0, 0), (4, 1, -0.3), (2, 3, 0), (0, 1, 0)], True)
        pitch = 0.005
        rows = fill_with_lines(shape, pitch, alpha, False)
        assert total_length(rows) * pitch == pytest.approx(abs(shape.area), rel=2e-3)

    def test_lines_inside_the_shape(self):
        shape = ArcSpline([(0, 0, 0.2), (4, 0, 0), (4, 1, -0.3), (2, 3, 0), (0, 1, 0)], True)
        for line in flat(fill_with_lines(shape, 0.1, 0.3, False)):
            assert shape.contains(line.start + (line.end - line.start) * 0.5)

    def test_non_convex_shape_has_several_lines_in_a_row(self):
        u_shape = ArcSpline([(0, 0, 0), (3, 0, 0), (3, 3, 0), (2, 3, 0), (2, 1, 0), (1, 1, 0), (1, 3, 0), (0, 3, 0)], True)
        rows = fill_with_lines(u_shape, 0.5, 0., True)
        lengths = {round(row[0].start.y, 6): len(row) for row in rows}
        assert lengths[0.] == 1 and lengths[0.5] == 1
        assert lengths[1.] == 2 and lengths[2.5] == 2
        assert 3. not in lengths
        for row in rows:
            xs = [x for line in row for x in (line.start.x, line.end.x)]
            assert xs == sorted(xs)

    def test_lines_touching_a_vertex_are_not_split(self):
        # a scan line through the tip of a triangle
        triangle = ArcSpline([(0, 0, 0), (2, 0, 0), (1, 2, 0)], True)
        rows = fill_with_lines(triangle, 1., 0., False, seed=(0, 0))
        assert [round(row[0].start.y, 9) for row in rows] == [1., 0.]
        assert rows[0][0].length == pytest.approx(1.)
        assert rows[1][0].length == pytest.approx(2.)

    def test_scan_line_through_the_extremum_of_a_circle(self):
        rows = fill_with_lines(circle(1.), 0.5, 0., False, seed=(0, 0))
        ys = [row[0].start.y for row in rows]
        # bottom of the circle (y = -1) is a degenerate line which is dropped, the top is excluded
        assert ys == pytest.approx([0.5, 0., -0.5])
        assert rows[1][0].length == pytest.approx(2.)

    def test_hollow_arc_spline(self):
        shape = HollowArcSpline(rect(0, 0, 4, 4), [rect(1, 1, 3, 3)])
        pitch = 0.01
        rows = fill_with_lines(shape, pitch, 0., False)
        assert total_length(rows) * pitch == pytest.approx(16 - 4, rel=2e-3)
        # lines through the hole are split in two parts
        middle = [row for row in rows if 1. < row[0].start.y < 3.]
        assert all(len(row) == 2 for row in middle)
        for row in middle:
            assert row[0].end.x == pytest.approx(1.) and row[1].start.x == pytest.approx(3.)

    def test_hollow_arc_spline_rotated(self):
        base = HollowArcSpline(rect(-2, -2, 2, 2), [circle(1.)])
        pitch = 0.01
        rows = fill_with_lines(base, pitch, 0.5, True)
        assert total_length(rows) * pitch == pytest.approx(16 - math.pi, rel=2e-3)

    def test_translated_shape(self):
        rows = fill_with_lines(rect(100, 200, 102, 202), 0.5, 0., False, seed=(100, 200))
        assert [row[0].start.y for row in rows] == pytest.approx([201.5, 201., 200.5, 200.])

    def test_shape_smaller_than_pitch(self):
        rows = fill_with_lines(rect(0.1, 0.1, 0.2, 0.2), 1., 0., False, seed=(0, 0))
        assert rows == []

    def test_rotation_by_transformed_shape_equals_alpha(self):
        shape = rect(-2, -1, 2, 1)
        rows_alpha = fill_with_lines(shape, 0.25, 0.6, False, seed=(0, 0))
        rows_rotated = fill_with_lines(shape.transformed(rotate(-0.6)), 0.25, 0., False, seed=(0, 0))
        assert total_length(rows_alpha) == pytest.approx(total_length(rows_rotated))
        assert len(rows_alpha) == len(rows_rotated)

    def test_errors(self):
        with pytest.raises(ValueError):
            fill_with_lines(ArcSpline([(0, 0, 0), (1, 0, 0), (1, 1, 0)], False), 0.5, 0., False)
        with pytest.raises(TypeError):
            fill_with_lines(Line((0, 0), (1, 1)), 0.5, 0., False)
        for pitch in (0., -1., math.inf, math.nan):
            with pytest.raises(ValueError):
                fill_with_lines(rect(0, 0, 1, 1), pitch, 0., False)
        for alpha in (-2., 2.):
            with pytest.raises(ValueError):
                fill_with_lines(rect(0, 0, 1, 1), 0.5, alpha, False)


class TestFillWithSpiral:
    def test_returns_parametric_curve(self):
        spiral = fill_with_spiral(rect(0, 0, 4, 2), 0.5)
        assert isinstance(spiral, ParametricCurve)

    def test_starts_in_the_center(self):
        spiral = fill_with_spiral(rect(0, 0, 4, 2), 0.5)
        assert np.allclose(spiral.f(spiral.domain[0]), (2., 1.))

    def test_ends_on_circumscribed_circle(self):
        spiral = fill_with_spiral(rect(0, 0, 4, 2), 0.5)
        end = spiral.f(spiral.domain[1])
        assert np.linalg.norm(end - (2., 1.)) == pytest.approx(math.hypot(2., 1.))

    def test_pitch_between_arms(self):
        pitch = 0.5
        spiral = fill_with_spiral(circle(3.), pitch)
        phi = np.linspace(2 * math.pi, spiral.domain[1], 50)
        radius_now = np.linalg.norm(spiral.f(phi) - (0., 0.), axis=1)
        radius_next = np.linalg.norm(spiral.f(phi + 2 * math.pi) - (0., 0.), axis=1)
        assert np.allclose(radius_next - radius_now, pitch)

    def test_length(self):
        spiral = fill_with_spiral(circle(3.), 0.5)
        # Archimedean spiral of the pitch p from 0 to r: length ~ pi r^2 / p (for r >> p); the radius is the half
        # diagonal of the bounding box
        assert spiral.length == pytest.approx(math.pi * (3 * math.sqrt(2)) ** 2 / 0.5, rel=0.01)

    def test_rasterized_spacing(self):
        spiral = fill_with_spiral(circle(3.), 0.5)
        points = spiral.rasterize_at(0.05)
        # (the chords are shorter than the arc length close to the center, where the curvature is large)
        assert np.allclose(np.linalg.norm(np.diff(points, axis=0), axis=1)[200:], 0.05, atol=1e-3)

    def test_hollow_arc_spline(self):
        shape = HollowArcSpline(rect(0, 0, 4, 4), [rect(1, 1, 3, 3)])
        assert np.allclose(fill_with_spiral(shape, 0.5).f(0.), (2., 2.))

    def test_errors(self):
        with pytest.raises(ValueError):
            fill_with_spiral(ArcSpline([(0, 0, 0), (1, 0, 0)], False), 0.5)
        with pytest.raises(TypeError):
            fill_with_spiral(Line((0, 0), (1, 1)), 0.5)
        for pitch in (0., -1., math.inf):
            with pytest.raises(ValueError):
                fill_with_spiral(rect(0, 0, 1, 1), pitch)
