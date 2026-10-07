"""Tests of `fibomat.raster_styles.two_d.LineByLine`."""
import math

import numpy as np
import pytest

from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline
from fibomat.composite_shapes.ring import Ring
from fibomat.mill import Mill
from fibomat.raster_styles import RasterStyle, ScanSequence
from fibomat.raster_styles.one_d import Curve
from fibomat.raster_styles.two_d import LineByLine
from fibomat.shapes import ArcSpline, Circle, Line, Rect, Spot
from fibomat.units import unit


def line_style(pitch=0.5):
    return Curve(pitch * unit('µm'), ScanSequence.CONSECUTIVE)


def style(sequence=ScanSequence.CONSECUTIVE, pitch=0.5, alpha=0., invert=False, line_pitch=None):
    return LineByLine(pitch * unit('µm'), sequence, alpha, invert, line_style(line_pitch or pitch))


def rasterize(line_by_line, shape, repeats=1, dwell=1., length=unit('µm'), time=unit('ms')):
    return line_by_line.rasterize(shape * unit('µm'), Mill(dwell * unit('ms'), repeats), length, time)


class TestConstruction:
    def test_properties(self):
        line_by_line = LineByLine(0.5 * unit('µm'), 'serpentine', 0.3, True, line_style())
        assert isinstance(line_by_line, RasterStyle)
        assert line_by_line.dimension == 2
        assert line_by_line.line_pitch == 0.5 * unit('µm')
        assert line_by_line.scan_sequence is ScanSequence.SERPENTINE
        assert line_by_line.alpha == 0.3
        assert line_by_line.invert is True
        assert isinstance(line_by_line.line_style, Curve)
        assert 'LineByLine(' in repr(line_by_line)

    def test_invalid_line_pitch(self):
        with pytest.raises(TypeError):
            LineByLine(0.5, ScanSequence.CONSECUTIVE, 0., False, line_style())
        with pytest.raises(ValueError):
            LineByLine(0. * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, line_style())
        with pytest.raises(ValueError, match='length'):
            LineByLine(1. * unit('s'), ScanSequence.CONSECUTIVE, 0., False, line_style())

    @pytest.mark.parametrize('alpha', [-2., 2., math.nan])
    def test_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match='alpha'):
            style(alpha=alpha)

    def test_invalid_scan_sequence(self):
        with pytest.raises(ValueError):
            style(sequence='foo')

    def test_invalid_line_style(self):
        from fibomat.raster_styles.zero_d import SingleSpot
        with pytest.raises(ValueError, match='dimension'):
            LineByLine(0.5 * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, SingleSpot())
        with pytest.raises(ValueError):
            LineByLine(0.5 * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, 'foo')


class TestRasterize:
    def test_rect(self):
        pattern = rasterize(style(), Rect(1, 1))
        # lines at y = 0 and y = -0.5, from the top to the bottom (the upper edge is not part of the fill)
        assert np.allclose(pattern.positions, [
            [-0.5, 0], [0, 0], [0.5, 0], [-0.5, -0.5], [0, -0.5], [0.5, -0.5]
        ])
        assert np.allclose(pattern.dwell_times, 1.)

    def test_invert(self):
        pattern = rasterize(style(invert=True), Rect(1, 1))
        assert np.allclose(pattern.positions[:, 1], [-0.5] * 3 + [0] * 3)

    def test_serpentine(self):
        pattern = rasterize(style(ScanSequence.SERPENTINE), Rect(1, 1))
        assert np.allclose(pattern.positions[:, 0], [-0.5, 0, 0.5, 0.5, 0, -0.5])

    def test_repeats_and_units(self):
        pattern = rasterize(style(), Rect(1, 1), repeats=2, dwell=3., length=unit('nm'), time=unit('µs'))
        assert len(pattern.dwell_points) == 12
        assert np.allclose(pattern.dwell_times, 3000.)
        assert np.max(np.abs(pattern.positions)) == pytest.approx(500.)

    def test_crossection_repeats_every_line(self):
        pattern = rasterize(style(ScanSequence.CROSSECTION), Rect(1, 1), repeats=2)
        assert len(pattern.dwell_points) == 12
        assert np.allclose(pattern.positions[:6, 1], 0.)

    def test_rotated_lines(self):
        # vertical lines, a pitch which keeps the lines away from the edges (where rounding errors decide)
        pattern = rasterize(style(alpha=math.pi / 2, pitch=0.4), Rect(1, 1))
        assert np.allclose(np.unique(pattern.positions[:, 0].round(9)), [-0.4, 0., 0.4])
        assert np.all(np.abs(pattern.positions[:, 1]) <= 0.5 + 1e-9)

    def test_filled_length_is_the_area_divided_by_the_pitch(self):
        line_by_line = LineByLine(0.01 * unit('µm'), ScanSequence.CONSECUTIVE, 0.3, False, line_style(0.01))
        pattern = rasterize(line_by_line, Circle(1.))
        # points on the lines have a distance of 0.01: number of points ~ area / pitch^2
        assert len(pattern.dwell_points) == pytest.approx(math.pi / 0.01 ** 2, rel=0.01)

    def test_hollow_arc_spline_has_a_hole(self):
        hollow = HollowArcSpline(
            ArcSpline([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)], True),
            [ArcSpline([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], True)]
        )
        pattern = rasterize(style(pitch=0.25), hollow)
        radius = np.max(np.abs(pattern.positions), axis=1)
        assert np.all(radius > 1. - 1e-9)

    def test_ring(self):
        pattern = rasterize(style(pitch=0.1), Ring(2., 1.))
        assert len(pattern.dwell_points) > 0
        distance = np.linalg.norm(pattern.positions, axis=1)
        assert np.all(distance > 1. - 1e-6) and np.all(distance < 2. + 1e-6)

    def test_shape_smaller_than_pitch_gets_one_line(self):
        # (a line always passes through the center of the shape)
        pattern = rasterize(style(pitch=100.), Rect(0.1, 0.1).translated((0.2, 0.2)))
        assert pattern.dwell_points.shape == (1, 3)
        assert np.allclose(pattern.positions, [[0.15, 0.2]])

    def test_open_shape(self):
        with pytest.raises(ValueError):
            rasterize(style(), Line((0, 0), (1, 1)))

    def test_unsupported_shape(self):
        with pytest.raises(TypeError):
            rasterize(style(), Spot((0, 0)))
