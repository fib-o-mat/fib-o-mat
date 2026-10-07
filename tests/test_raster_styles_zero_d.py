"""Tests of `fibomat.raster_styles.zero_d`."""
import numpy as np
import pytest

from fibomat.mill import Mill
from fibomat.raster_styles import RasterStyle
from fibomat.raster_styles.zero_d import PreRasterized, SingleSpot
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes import Line, RasterizedPoints, Spot
from fibomat.units import unit


class TestSingleSpot:
    def rasterize(self, shape, mill=None, **units):
        return SingleSpot().rasterize(
            shape * unit('µm'), mill or Mill(2. * unit('ms'), 3),
            units.get('length', unit('nm')), units.get('time', unit('µs'))
        )

    def test_is_raster_style(self):
        style = SingleSpot()
        assert isinstance(style, RasterStyle)
        assert style.dimension == 0
        assert repr(style) == 'SingleSpot()'

    def test_rasterize(self):
        pattern = self.rasterize(Spot((1., 2.)))
        assert isinstance(pattern, RasterizedPattern)
        assert pattern.dwell_points.shape == (3, 3)
        assert np.allclose(pattern.dwell_points, [[1000., 2000., 2000.]] * 3)

    def test_units_are_kept(self):
        pattern = self.rasterize(Spot((1., 2.)), length=unit('µm'), time=unit('ms'))
        assert np.allclose(pattern.dwell_points[0], [1., 2., 2.])

    def test_wrong_shape(self):
        with pytest.raises(TypeError, match='Spot'):
            self.rasterize(Line((0, 0), (1, 1)))

    def test_wrong_mill(self):
        with pytest.raises(TypeError, match='Mill'):
            self.rasterize(Spot((0, 0)), mill='foo')


class TestPreRasterized:
    points = RasterizedPoints(np.array([[0., 0., 1.], [1., 2., 2.]]), False)

    def rasterize(self, shape, mill=None, length=unit('µm'), time=unit('ms')):
        return PreRasterized().rasterize(shape * unit('µm'), mill or Mill(3. * unit('ms'), 1), length, time)

    def test_is_raster_style(self):
        assert PreRasterized().dimension == 0
        assert repr(PreRasterized()) == 'PreRasterized()'

    def test_weights_are_multiplied_with_the_dwell_time(self):
        pattern = self.rasterize(self.points)
        assert np.allclose(pattern.dwell_points, [[0, 0, 3], [1, 2, 6]])

    def test_units(self):
        pattern = self.rasterize(self.points, length=unit('nm'), time=unit('µs'))
        assert np.allclose(pattern.dwell_points, [[0, 0, 3000], [1000, 2000, 6000]])

    def test_repeats_repeat_the_sequence(self):
        pattern = self.rasterize(self.points, Mill(1. * unit('ms'), 2))
        assert np.allclose(pattern.dwell_points, [[0, 0, 1], [1, 2, 2], [0, 0, 1], [1, 2, 2]])

    def test_input_is_not_modified(self):
        self.rasterize(self.points, length=unit('nm'))
        assert np.allclose(self.points.dwell_points, [[0, 0, 1], [1, 2, 2]])

    def test_wrong_shape(self):
        with pytest.raises(TypeError, match='RasterizedPoints'):
            self.rasterize(Spot((0, 0)))
