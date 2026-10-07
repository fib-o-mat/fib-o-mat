"""Tests of `fibomat.raster_styles.one_d.Curve`."""
import math

import numpy as np
import pytest

from fibomat.mill import Mill
from fibomat.raster_styles import RasterStyle, ScanSequence
from fibomat.raster_styles.one_d import Curve
from fibomat.shapes import ArcSpline, Circle, Line, Spot
from fibomat.units import unit


def rasterize(style, shape, repeats=1, dwell=1., length=unit('µm'), time=unit('ms')):
    return style.rasterize(shape * unit('µm'), Mill(dwell * unit('ms'), repeats), length, time)


def curve(pitch=0.25, sequence=ScanSequence.CONSECUTIVE):
    return Curve(pitch * unit('µm'), sequence)


class TestConstruction:
    def test_properties(self):
        style = curve(0.5, ScanSequence.BACKSTITCH)
        assert isinstance(style, RasterStyle)
        assert style.dimension == 1
        assert style.pitch == 0.5 * unit('µm')
        assert style.scan_sequence is ScanSequence.BACKSTITCH
        assert 'pitch=' in repr(style)

    def test_string_sequence(self):
        assert Curve(1. * unit('µm'), 'back_and_forth').scan_sequence is ScanSequence.BACK_AND_FORTH

    @pytest.mark.parametrize('pitch', [0., -1.])
    def test_invalid_pitch_value(self, pitch):
        with pytest.raises(ValueError):
            Curve(pitch * unit('µm'), ScanSequence.CONSECUTIVE)

    def test_pitch_dimension(self):
        with pytest.raises(ValueError, match='length'):
            Curve(1. * unit('s'), ScanSequence.CONSECUTIVE)

    def test_pitch_type(self):
        with pytest.raises(TypeError):
            Curve(1., ScanSequence.CONSECUTIVE)

    @pytest.mark.parametrize('sequence', [ScanSequence.SERPENTINE, ScanSequence.CROSSECTION, 'foo'])
    def test_invalid_sequence(self, sequence):
        with pytest.raises(ValueError):
            Curve(1. * unit('µm'), sequence)

    def test_clone(self):
        style = curve()
        clone = style.clone()
        assert clone is not style and clone.pitch == style.pitch


class TestRasterize:
    def test_line(self):
        pattern = rasterize(curve(0.25), Line((0, 0), (1, 0)))
        assert np.allclose(pattern.positions[:, 0], [0, 0.25, 0.5, 0.75, 1.])
        assert np.allclose(pattern.dwell_times, 1.)

    def test_units(self):
        pattern = rasterize(curve(0.25), Line((0, 0), (1, 0)), dwell=2., length=unit('nm'), time=unit('µs'))
        assert np.allclose(pattern.positions[:, 0], [0, 250, 500, 750, 1000])
        assert np.allclose(pattern.dwell_times, 2000.)

    def test_pitch_in_other_unit_than_shape(self):
        pattern = Curve(250. * unit('nm'), ScanSequence.CONSECUTIVE).rasterize(
            Line((0, 0), (1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1), unit('µm'), unit('ms')
        )
        assert len(pattern.dwell_points) == 5

    def test_shape_unit(self):
        pattern = curve(0.25).rasterize(
            Line((0, 0), (1000, 0)) * unit('nm'), Mill(1. * unit('ms'), 1), unit('µm'), unit('ms')
        )
        assert np.allclose(pattern.positions[:, 0], [0, 0.25, 0.5, 0.75, 1.])

    def test_circle_has_no_duplicated_start_point(self):
        radius = 1.
        pattern = rasterize(curve(2 * math.pi * radius / 100), Circle(radius))
        assert len(pattern.dwell_points) == 100
        assert np.allclose(np.linalg.norm(pattern.positions, axis=1), radius)

    def test_arc_spline(self):
        spline = ArcSpline([(0, 0, 0), (1, 0, 0), (1, 1, 0)], False)
        assert len(rasterize(curve(0.5), spline).dwell_points) == 5

    def test_repeats_consecutive(self):
        once = rasterize(curve(), Line((0, 0), (1, 0))).dwell_points
        pattern = rasterize(curve(), Line((0, 0), (1, 0)), repeats=3)
        assert np.allclose(pattern.dwell_points, np.tile(once, (3, 1)))

    def test_backstitch_swaps_pairs(self):
        pattern = rasterize(curve(0.25, ScanSequence.BACKSTITCH), Line((0, 0), (1, 0)))
        # five points: 1, 0, 3, 2, 4
        assert np.allclose(pattern.positions[:, 0], [0.25, 0, 0.75, 0.5, 1.])

    def test_backstitch_even_number_of_points(self):
        pattern = rasterize(curve(0.25, ScanSequence.BACKSTITCH), Line((0, 0), (0.75, 0)))
        assert np.allclose(pattern.positions[:, 0], [0.25, 0, 0.75, 0.5])

    def test_back_and_forth(self):
        pattern = rasterize(curve(0.5, ScanSequence.BACK_AND_FORTH), Line((0, 0), (1, 0)), repeats=3)
        assert np.allclose(pattern.positions[:, 0], [0, 0.5, 1, 1, 0.5, 0, 0, 0.5, 1])

    def test_unsupported_shape(self):
        with pytest.raises(TypeError):
            rasterize(curve(), Spot((0, 0)))

    def test_wrong_mill(self):
        with pytest.raises(TypeError, match='Mill'):
            curve().rasterize(Line((0, 0), (1, 0)) * unit('µm'), 'foo', unit('µm'), unit('ms'))
