"""Smoke tests of `fibomat.raster_styles.two_d.Spiral` (it is only adapted to the new units, see TODO.md)."""
import numpy as np

from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence
from fibomat.raster_styles.two_d import Spiral
from fibomat.shapes import Circle
from fibomat.units import unit


def rasterize(direction):
    style = Spiral(0.05 * unit('µm'), 0.2 * unit('µm'), ScanSequence.CONSECUTIVE, direction)
    return style.rasterize(Circle(1.) * unit('µm'), Mill(2. * unit('ms'), 1), unit('nm'), unit('µs'))


def test_constant_dwell_time_and_units():
    pattern = rasterize('outwards')
    assert len(pattern.dwell_points) > 100
    assert np.allclose(pattern.dwell_times, 2000.)
    # positions are in nm and inside of the circle (radius 1 µm)
    assert np.all(np.linalg.norm(pattern.positions, axis=1) <= 1000. + 1e-6)


def test_directions():
    outwards = rasterize('outwards').dwell_points
    inwards = rasterize('inwards').dwell_points
    out_in = rasterize('out-in').dwell_points
    assert np.allclose(inwards, outwards[::-1])
    assert np.allclose(out_in, np.concatenate([outwards, outwards[::-1]]))
