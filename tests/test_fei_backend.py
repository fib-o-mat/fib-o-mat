"""Tests of the FEI stream file backend (`fibomat.default_backends.fei`).

The files are compared with the output of the legacy implementation (the code before the rework, only adapted to the
current vector class), so that the file content did not change.
"""
import math
import pathlib
import re
import warnings

import numpy as np
import pytest

from fibomat.default_backends import SpotListBackend
from fibomat.default_backends.fei import FEIStreamFile, stream_file_impl
from fibomat.layout import Layout
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d, two_d, zero_d
from fibomat.shapes import Circle, Line, Polygon, Rect, Spot
from fibomat.units import scale_factor, unit


def legacy_stream_file_impl(n_rep=1, margin=0.9, dac16bit=True, hfw_rounding=True, time_low_res=True):
    """The implementation before the rework (the unit handling and the vector arithmetic are adapted)."""
    def streamfile_type(dac16bit=dac16bit, time_low_res=time_low_res):
        if dac16bit:
            x_res = 65536
            y_res = x_res
            if time_low_res == 'low':
                time_unit = 100
                header = 's16\n'
            else:
                time_unit = 25
                header = 's16,25ns\n'
        else:
            x_res = 4095
            y_res = 3816
            header = 's\n'
            time_unit = 100
        return x_res, y_res, time_unit, header

    def round_hfw(hfw, hfw_rounding=hfw_rounding):
        if hfw_rounding:
            oom = math.floor(math.log(hfw, 10))
            hfw = np.round(np.ceil(hfw*4*10**(-oom)) / (4*10**(-oom)))
        return hfw

    def _custom_save_impl(filename, dwell_points, parameters, n_rep=n_rep, margin=margin):
        x_res, y_res, time_unit, header = streamfile_type()
        fov = parameters["fov"]
        center = fov.center
        width, height = fov.width, fov.height
        xy_aspect_ratio = x_res/y_res
        if height != 0 and width/height > xy_aspect_ratio:
            hfw = width / margin
        else:
            hfw = height * xy_aspect_ratio / margin
        if hfw > 0:
            hfw = round_hfw(hfw)
        if hfw == 0:
            hfw = 1
        shift = np.asarray(center) - (hfw/2, hfw/xy_aspect_ratio/2)
        dwell_points[:, 0:2] = (dwell_points[:, 0:2]-shift)*x_res/hfw
        dwell_points[:, 2] *= (scale_factor(unit('ns'), parameters["base_dwell_time"])*parameters["base_dwell_time"].magnitude)/time_unit
        dwell_points = dwell_points.round()

        stripped = filename.split('.', 1)[0]
        filename = stripped + f'(HFW={hfw:0.0f}{parameters["length_unit"]:~P})' + '.str'
        with open(filename, 'w') as fp:
            fp.writelines([
                header,
                f'{n_rep:d}\n',
                f'{parameters["number_of_points"]:d}\n',
            ])
            np.savetxt(fp, dwell_points[:, [2, 0, 1]], "%d %d %d")

    return _custom_save_impl


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def curve(pitch=0.5):
    return one_d.Curve(pitch * unit('µm'), ScanSequence.CONSECUTIVE)


def area(line_pitch=0.5, pitch=0.5):
    return two_d.LineByLine(line_pitch * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, curve(pitch))


def layouts():
    result = {}

    layout = Layout()
    layout.create_site(um(3, 1)).create_pattern(
        Rect(4, 2).translated((1, 1)) * unit('µm'), Mill(0.3 * unit('µs'), 2), area()
    )
    result['rect'] = layout

    layout = Layout()
    site = layout.create_site(um(0, 0), um(40, 40))
    site.create_pattern(Circle(7.3, center=(2, -3)) * unit('µm'), Mill(1.5 * unit('µs'), 1), area(0.4, 0.2))
    site.create_pattern(Line((0, 0), (12, 5)) * unit('µm'), Mill(0.1 * unit('µs'), 3), curve(0.1))
    result['circle and line'] = layout

    layout = Layout()
    layout.create_site(um(10, -10), um(100, 100)).create_pattern(
        Polygon([(0, 0), (30, 0), (10, 20)]) * unit('µm'), Mill(0.7 * unit('µs'), 1), area(1., 0.5)
    )
    layout.create_site(um(100, 100), um(100, 100)).create_pattern(
        Rect(3, 3) * unit('µm'), Mill(0.2 * unit('µs'), 1), area(0.3, 0.3)
    )
    result['two sites'] = layout

    layout = Layout()
    layout.create_site(um(1, 1), um(10, 10)).create_pattern(
        Line((0, 0), (5, 0)) * unit('µm'), Mill(0.5 * unit('µs'), 1), curve(0.25)
    )
    result['horizontal line (no height)'] = layout

    layout = Layout()
    layout.create_site(um(5, 5), um(4, 4)).create_pattern(
        Spot((0, 0)) * unit('µm'), Mill(1. * unit('µs'), 2), zero_d.SingleSpot()
    )
    result['single spot'] = layout

    layout = Layout()
    layout.create_site(um(0, 0), um(0.5, 0.5)).create_pattern(
        Rect(0.2, 0.1) * unit('µm'), Mill(0.4 * unit('µs'), 1), area(0.05, 0.05)
    )
    result['small pattern'] = layout

    return result


OPTIONS = [
    {},
    {'n_rep': 3},
    {'margin': 0.5},
    {'dac16bit': False},
    {'dac16bit': False, 'time_low_res': 'low'},
    {'time_low_res': 'low'},
    {'hfw_rounding': False},
    {'n_rep': 2, 'margin': 0.75, 'hfw_rounding': False, 'time_low_res': 'low'},
]


def created_file(directory):
    files = list(pathlib.Path(directory).glob('*.str'))
    assert len(files) == 1
    return files[0]


@pytest.mark.parametrize('name', sorted(layouts()))
@pytest.mark.parametrize('options', OPTIONS, ids=str)
def test_files_are_identical_to_the_legacy_implementation(tmp_path, name, options):
    legacy_dir, new_dir = tmp_path / 'legacy', tmp_path / 'new'
    legacy_dir.mkdir()
    new_dir.mkdir()

    layout = layouts()[name]

    layout.export(
        SpotListBackend, base_dwell_time=0.1 * unit('µs'), length_unit=unit('µm'), time_unit=unit('µs'),
        save_impl=legacy_stream_file_impl(**options)
    ).save(str(legacy_dir / 'out.txt'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        layout.export(FEIStreamFile, **options).save(new_dir / 'out.txt')

    legacy, new = created_file(legacy_dir), created_file(new_dir)
    assert legacy.name == new.name
    assert legacy.read_text(encoding='utf-8') == new.read_text(encoding='utf-8')


class TestStreamFile:
    @staticmethod
    def export(**options):
        layout = layouts()['rect']
        return layout.export(FEIStreamFile, **options)

    def test_file_name_contains_the_hfw(self, tmp_path):
        self.export().save(tmp_path / 'pattern.txt')
        assert re.fullmatch(r'pattern\(HFW=\d+µm\)\.str', created_file(tmp_path).name)

    def test_directories_with_dots(self, tmp_path):
        directory = tmp_path / 'my.directory'
        directory.mkdir()
        self.export().save(directory / 'pattern.v2.txt')
        assert re.fullmatch(r'pattern\(HFW=\d+µm\)\.str', created_file(directory).name)

    def test_header(self, tmp_path):
        self.export(n_rep=3).save(tmp_path / 'pattern.txt')
        lines = created_file(tmp_path).read_text(encoding='utf-8').splitlines()
        assert lines[0] == 's16,25ns' and lines[1] == '3'
        assert int(lines[2]) == len(lines) - 3

    @pytest.mark.parametrize('options, header', [
        ({'time_low_res': 'low'}, 's16'), ({'dac16bit': False}, 's'), ({}, 's16,25ns'),
    ])
    def test_header_types(self, tmp_path, options, header):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.export(**options).save(tmp_path / 'pattern.txt')
        assert created_file(tmp_path).read_text(encoding='utf-8').splitlines()[0] == header

    def test_points_are_integers_in_the_range_of_the_converter(self, tmp_path):
        self.export().save(tmp_path / 'pattern.txt')
        points = np.loadtxt(created_file(tmp_path), skiprows=3, dtype=int)
        assert points.shape[1] == 3
        assert points[:, 1:].min() >= 0 and points[:, 1:].max() <= 65536
        # dwell time 0.3 µs in units of 25 ns
        assert np.all(points[:, 0] == 12)

    def test_twelve_bit_converter_with_high_resolution_warns(self, tmp_path):
        with pytest.warns(UserWarning, match='12 bit'):
            self.export(dac16bit=False, time_low_res=False).save(tmp_path / 'pattern.txt')

    def test_hfw_of_zero_is_replaced_with_one(self, tmp_path):
        layout = Layout()
        layout.create_site(um(5, 5), um(0.001, 0.001)).create_pattern(
            Spot((0, 0)) * unit('µm'), Mill(1. * unit('µs'), 1), zero_d.SingleSpot()
        )
        with pytest.warns(UserWarning, match='horizontal field width'):
            layout.export(FEIStreamFile).save(tmp_path / 'pattern.txt')
        assert created_file(tmp_path).name == 'pattern(HFW=1µm).str'

    def test_description(self):
        assert self.export(description='design').description == 'design'

    @pytest.mark.parametrize('options', [{'n_rep': 0}, {'n_rep': 1.5}, {'margin': 0.05}, {'margin': 1.5}])
    def test_invalid_options(self, options):
        with pytest.raises(ValueError):
            FEIStreamFile(**options)
        with pytest.raises(ValueError):
            stream_file_impl(**options)

    def test_spot_list_without_base_dwell_time(self, tmp_path):
        layout = layouts()['rect']
        backend = layout.export(SpotListBackend, length_unit=unit('µm'), save_impl=stream_file_impl())
        with pytest.raises(ValueError, match='base dwell time'):
            backend.save(str(tmp_path / 'pattern.txt'))
