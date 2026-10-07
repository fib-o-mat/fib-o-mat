"""Tests of `fibomat.default_backends.patterning_duration_calculator`."""
import pytest

from fibomat.default_backends import PatterningDurationCalculator
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


def area_style(line_pitch=0.5, point_pitch=0.25):
    return two_d.LineByLine(
        line_pitch * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, curve_style(point_pitch)
    )


def seconds(duration):
    return duration.m_as('s')


def calculate(patterns, current=1. * unit('nA'), **site_kwargs):
    layout = Layout()
    site = layout.create_site(um(0, 0), um(100, 100), **site_kwargs)
    for dim_shape, mill, style in patterns:
        site.create_pattern(dim_shape, mill, style)
    return layout.export(PatterningDurationCalculator, current=current)


class TestConstruction:
    def test_current(self):
        assert PatterningDurationCalculator(2. * unit('nA')).current == 2. * unit('nA')

    def test_invalid_current(self):
        with pytest.raises(TypeError):
            PatterningDurationCalculator(1.)
        with pytest.raises(ValueError, match='current'):
            PatterningDurationCalculator(1. * unit('s'))
        with pytest.raises(ValueError):
            PatterningDurationCalculator(-1. * unit('nA'))
        with pytest.raises(TypeError):
            PatterningDurationCalculator()


class TestEstimate:
    def test_spot(self):
        calculator = calculate([(Spot((0, 0)) * unit('µm'), Mill(2. * unit('s'), 3), zero_d.SingleSpot())])
        assert seconds(calculator.total_duration) == pytest.approx(6.)

    def test_curve(self):
        # 10 µm / 0.25 µm = 40 points, 2 repeats, 1 ms
        calculator = calculate([(Line((0, 0), (10, 0)) * unit('µm'), Mill(1. * unit('ms'), 2), curve_style())])
        assert seconds(calculator.total_duration) == pytest.approx(0.08)

    def test_closed_curve_includes_the_closing_segment(self):
        calculator = calculate([(Rect(2, 2) * unit('µm'), Mill(1. * unit('ms'), 1), curve_style(0.25))])
        assert seconds(calculator.total_duration) == pytest.approx(32 * 1e-3)

    def test_area(self):
        # 100 µm^2 / (0.5 µm * 0.25 µm) = 800 points
        calculator = calculate([(Rect(10, 10) * unit('µm'), Mill(1. * unit('ms'), 1), area_style())])
        assert seconds(calculator.total_duration) == pytest.approx(0.8)

    def test_area_of_a_circle(self):
        calculator = calculate([(Circle(1.) * unit('µm'), Mill(1. * unit('ms'), 1), area_style(0.1, 0.1))])
        assert seconds(calculator.total_duration) == pytest.approx(3.14159 / 0.01 * 1e-3, rel=1e-4)

    def test_units_of_shape_and_pitch_are_converted(self):
        calculator = calculate([(
            Line((0, 0), (10000, 0)) * unit('nm'), Mill(1. * unit('ms'), 1),
            one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
        )])
        assert seconds(calculator.total_duration) == pytest.approx(0.04)

    def test_estimate_matches_the_rasterization(self):
        from fibomat.default_backends import SpotListBackend
        layout = Layout()
        layout.create_site(um(0, 0), um(100, 100)).create_pattern(
            Rect(10, 10) * unit('µm'), Mill(1. * unit('ms'), 2), area_style(0.5, 0.5)
        )
        estimated = seconds(layout.export(PatterningDurationCalculator, current=1. * unit('nA')).total_duration)
        exposed = layout.export(SpotListBackend, time_unit=unit('s')).dwell_points()[:, 2].sum()
        assert estimated == pytest.approx(exposed, rel=0.1)

    def test_several_sites_and_patterns(self):
        layout = Layout()
        layout.create_site(um(0, 0), um(100, 100), description='a').create_pattern(
            Spot((0, 0)) * unit('µm'), Mill(1. * unit('s'), 1), zero_d.SingleSpot()
        )
        site = layout.create_site(um(0, 0), um(100, 100))
        site.create_pattern(Spot((0, 0)) * unit('µm'), Mill(2. * unit('s'), 1), zero_d.SingleSpot())
        site.create_pattern(Spot((1, 0)) * unit('µm'), Mill(3. * unit('s'), 1), zero_d.SingleSpot())
        calculator = layout.export(PatterningDurationCalculator, current=1. * unit('nA'))

        assert [site['name'] for site in calculator.durations] == ['Site 0 a', 'Site 1']
        assert [len(site['patterns']) for site in calculator.durations] == [1, 2]
        assert [p['name'] for p in calculator.durations[1]['patterns']] == ['0 Spot', '1 Spot']
        assert seconds(calculator.total_duration) == pytest.approx(6.)

    def test_repeats_below_one_warn_is_not_possible_for_a_mill(self):
        # (a Mill needs at least one repeat)
        with pytest.raises(ValueError):
            Mill(1. * unit('s'), 0)

    def test_unsupported_raster_style(self):
        layout = Layout()
        layout.create_site(um(0, 0), um(10, 10)).create_pattern(
            Spot((0, 0)) * unit('µm'), Mill(1. * unit('s'), 1), zero_d.PreRasterized()
        )
        with pytest.raises(NotImplementedError, match='cannot be estimated'):
            layout.export(PatterningDurationCalculator, current=1. * unit('nA'))

    def test_mill_without_repeats(self):
        from fibomat.mill import SpecialMill
        layout = Layout()
        layout.create_site(um(0, 0), um(10, 10)).create_pattern(
            Spot((0, 0)) * unit('µm'), SpecialMill(dwell_time=1. * unit('s')), zero_d.SingleSpot()
        )
        with pytest.raises(TypeError, match='repeats'):
            layout.export(PatterningDurationCalculator, current=1. * unit('nA'))

    def test_empty_layout(self):
        calculator = Layout().export(PatterningDurationCalculator, current=1. * unit('nA'))
        assert seconds(calculator.total_duration) == 0.


class TestOutput:
    def test_table(self):
        calculator = calculate([(Spot((0, 0)) * unit('µm'), Mill(90. * unit('s'), 1), zero_d.SingleSpot())])
        table = calculator.table()
        assert 'Site 0' in table and '1.50 min' in table
        assert 'Total duration: 1.50 min @ 1000.00 pA' in table
        assert '(1.000)' in table

    def test_time_formats(self):
        for dwell, expected in ((5., '5.00 s'), (7200., '2.00 h')):
            table = calculate([(Spot((0, 0)) * unit('µm'), Mill(dwell * unit('s'), 1), zero_d.SingleSpot())]).table()
            assert f'Total duration: {expected}' in table

    def test_print(self, capsys):
        calculate([(Spot((0, 0)) * unit('µm'), Mill(1. * unit('s'), 1), zero_d.SingleSpot())]).print()
        assert 'Total duration' in capsys.readouterr().out

    def test_empty_table(self):
        table = Layout().export(PatterningDurationCalculator, current=1. * unit('nA')).table()
        assert 'Total duration: 0.00 s' in table
