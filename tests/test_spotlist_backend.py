"""Tests of `fibomat.default_backends.spotlist_backend`."""
import warnings

import numpy as np
import pytest

from fibomat.composite_shapes import Ring
from fibomat.default_backends import SpotListBackend
from fibomat.default_backends.spotlist_backend import _default_save_impl
from fibomat.layout import Layout
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d, two_d, zero_d
from fibomat.shapes import Circle, Line, Rect, Spot
from fibomat.units import unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def curve_style(pitch=0.25):
    return one_d.Curve(pitch * unit('µm'), ScanSequence.CONSECUTIVE)


def layout_with_line(site_center=(10, 0), dwell=1., repeats=1):
    layout = Layout(description='design')
    site = layout.create_site(um(*site_center))
    site.create_pattern(Line((0, 0), (1, 0)) * unit('µm'), Mill(dwell * unit('ms'), repeats), curve_style())
    return layout


class TestConstruction:
    def test_defaults(self):
        backend = SpotListBackend()
        assert backend.description is None
        with pytest.raises(ValueError, match='no dwell points'):
            backend.dwell_points()

    @pytest.mark.parametrize('length_unit, time_unit', [(unit('nm'), unit('ms')), ('nm', 'ms')])
    def test_unit_arguments(self, length_unit, time_unit):
        pattern = layout_with_line().export(SpotListBackend, length_unit=length_unit, time_unit=time_unit).dwell_points()
        assert pattern[1, 0] - pattern[0, 0] == pytest.approx(250.)
        assert pattern[0, 2] == pytest.approx(1.)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match='length'):
            SpotListBackend(length_unit=unit('s'))
        with pytest.raises(ValueError, match='time'):
            SpotListBackend(time_unit=unit('µm'))
        with pytest.raises(TypeError):
            SpotListBackend(length_unit=3)
        with pytest.raises(TypeError, match='dimensioned'):
            SpotListBackend(base_dwell_time=0.1)
        with pytest.raises(ValueError, match='positive time'):
            SpotListBackend(base_dwell_time=0. * unit('µs'))
        with pytest.raises(ValueError):
            SpotListBackend(base_dwell_time=1. * unit('µm'))
        with pytest.raises(TypeError, match='callable'):
            SpotListBackend(save_impl='foo')
        with pytest.raises(TypeError):
            SpotListBackend(unknown=1)


class TestDwellPoints:
    def test_positions_are_absolute(self):
        points = layout_with_line((10, 5)).export(SpotListBackend).dwell_points()
        assert points[:, 0].tolist() == pytest.approx([10., 10.25, 10.5, 10.75, 11.])
        assert np.allclose(points[:, 1], 5.)

    def test_units_and_dwell_times(self):
        points = layout_with_line(dwell=2.).export(SpotListBackend, length_unit=unit('nm'), time_unit=unit('ns')).dwell_points()
        assert points[0, 0] == pytest.approx(10000.)
        assert np.allclose(points[:, 2], 2e6)

    def test_base_dwell_time(self):
        backend = layout_with_line(dwell=1.).export(SpotListBackend, base_dwell_time=100 * unit('µs'))
        assert np.all(backend.dwell_points()[:, 2] == 10)  # exactly integers

    def test_base_dwell_time_rounds_noise(self):
        # 3 * 0.1 µs / 0.1 µs is not exactly 3 in floating point arithmetic
        backend = layout_with_line(dwell=0.3e-3).export(SpotListBackend, base_dwell_time=0.1 * unit('µs'))
        assert np.all(backend.dwell_points()[:, 2] == 3)

    def test_non_integer_multiple_warns(self):
        backend = layout_with_line(dwell=1.).export(SpotListBackend, base_dwell_time=300 * unit('µs'))
        with pytest.warns(UserWarning, match='integer multiples'):
            backend.dwell_points()

    def test_several_sites_and_patterns(self):
        layout = layout_with_line((0, 0))
        site = layout.create_site(um(0, 100))
        site.create_pattern(Spot((0, 0)) * unit('µm'), Mill(1. * unit('ms'), 3), zero_d.SingleSpot())
        points = layout.export(SpotListBackend).dwell_points()
        assert len(points) == 5 + 3
        assert np.allclose(points[-3:, :2], [[0, 100]] * 3)

    @pytest.mark.parametrize('shape', [
        Rect(2, 2), Circle(1.), Ring(2., 1.),
    ])
    def test_areas(self, shape):
        layout = Layout()
        site = layout.create_site(um(0, 0))
        site.create_pattern(
            shape * unit('µm'), Mill(1. * unit('ms'), 1),
            two_d.LineByLine(0.5 * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, curve_style(0.5))
        )
        assert len(layout.export(SpotListBackend).dwell_points()) > 4

    def test_unsupported_shape_style_combination(self):
        layout = Layout()
        layout.create_site(um(0, 0), um(5, 5)).create_pattern(
            Line((0, 0), (1, 1)) * unit('µm'), Mill(1. * unit('ms'), 1), zero_d.SingleSpot()
        )
        with pytest.raises(TypeError):
            layout.export(SpotListBackend)


class TestSave:
    def test_default_file(self, tmp_path):
        backend = layout_with_line((10, 0)).export(SpotListBackend, length_unit=unit('nm'), time_unit=unit('µs'))
        filename = tmp_path / 'points.txt'
        backend.save(filename)
        text = filename.read_text(encoding='utf-8')

        assert '[Info]' in text and '[Points]' in text
        assert 'length_unit = nm' in text and 'time_unit = µs' in text
        assert 'number_of_points = 5' in text and 'description = design' in text
        points = np.loadtxt(text.split('[Points]\n')[1].splitlines())
        assert points.shape == (5, 3)
        assert points[0].tolist() == [10000., 0., 1000.]

    def test_dwell_times_are_rounded_not_truncated(self, tmp_path):
        layout = layout_with_line(dwell=0.3)  # 300 µs, converted from ms: 0.3 * 1000 = 300.00000000000006 or 299.99...
        backend = layout.export(SpotListBackend, time_unit=unit('µs'))
        filename = tmp_path / 'points.txt'
        backend.save(filename)
        points = np.loadtxt(filename.read_text(encoding='utf-8').split('[Points]\n')[1].splitlines())
        assert np.all(points[:, 2] == 300)

    def test_non_integer_dwell_time_warns(self, tmp_path):
        backend = layout_with_line(dwell=0.0005).export(SpotListBackend, time_unit=unit('µs'))  # 0.5 µs
        with pytest.warns(UserWarning, match='rounded'):
            backend.save(tmp_path / 'points.txt')

    def test_custom_save_function(self, tmp_path):
        calls = {}

        def save_impl(filename, dwell_points, parameters):
            calls.update(filename=filename, dwell_points=dwell_points, parameters=parameters)

        backend = layout_with_line((10, 0)).export(
            SpotListBackend, save_impl=save_impl, base_dwell_time=100 * unit('µs'), length_unit=unit('µm')
        )
        backend.save(tmp_path / 'x')
        parameters = calls['parameters']
        assert calls['filename'] == tmp_path / 'x'
        assert calls['dwell_points'].shape == (5, 3)
        assert set(parameters) == {
            'length_unit', 'time_unit', 'fov', 'base_dwell_time', 'total_dwell_time', 'number_of_points', 'description',
            'time_stamp'
        }
        assert parameters['number_of_points'] == 5 and parameters['description'] == 'design'
        assert parameters['base_dwell_time'] == 100 * unit('µs')
        assert parameters['total_dwell_time'] == 50

    def test_fov_is_the_bounding_box_of_the_points(self):
        parameters = {}
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(
            Rect(4, 2).translated((5, 5)) * unit('µm'), Mill(1. * unit('ms'), 1),
            two_d.LineByLine(0.5 * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, curve_style(0.5))
        )
        layout.export(SpotListBackend, save_impl=lambda f, d, p: parameters.update(p)).save('unused')
        fov = parameters['fov']
        assert fov.width == pytest.approx(4.) and fov.height == pytest.approx(1.5)

    def test_fov_of_a_line_is_the_fov_of_the_site(self):
        parameters = {}
        layout_with_line((10, 0)).export(SpotListBackend, save_impl=lambda f, d, p: parameters.update(p)).save('unused')
        fov = parameters['fov']
        # the points are on a line (no extent in y): the field of view of the site (2 µm * 1.1) is used
        assert fov.width == pytest.approx(2.2) and fov.height == pytest.approx(2.2)
        assert tuple(fov.center) == pytest.approx((10., 0.))

    def test_several_sites_warn(self, tmp_path):
        layout = layout_with_line()
        layout.add_site(layout.sites[0].translated(um(100, 0)))
        backend = layout.export(SpotListBackend)
        with pytest.warns(UserWarning, match='merged'):
            backend.save(tmp_path / 'points.txt')

    def test_nothing_to_save(self, tmp_path):
        layout = Layout()
        layout.create_site(um(0, 0), um(1, 1))
        with pytest.raises(ValueError, match='no dwell points'):
            layout.export(SpotListBackend).save(tmp_path / 'points.txt')

    def test_default_header_with_base_dwell_time(self, tmp_path):
        backend = layout_with_line().export(SpotListBackend, base_dwell_time=100 * unit('µs'))
        filename = tmp_path / 'points.txt'
        backend.save(filename)
        assert 'base_dwell_time = 100 µs' in filename.read_text(encoding='utf-8')
