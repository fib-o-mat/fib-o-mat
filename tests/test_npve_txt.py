"""Tests of the NPVE deflection list backend."""
import numpy as np
import pytest

from fibomat.default_backends.npve import NPVETxt
from fibomat.layout import Layout
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d
from fibomat.shapes import Line
from fibomat.units import unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def layout(dwell=0.3):
    result = Layout()
    result.create_site(um(10, 5)).create_pattern(
        Line((0, 0), (2, 0)) * unit('µm'), Mill(dwell * unit('µs'), 1),
        one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE)
    )
    return result


def test_file_content(tmp_path):
    with pytest.warns(UserWarning, match='0.1 us'):
        layout().export(NPVETxt).save(tmp_path / 'points.txt')

    content = (tmp_path / 'points.txt').read_bytes().decode('utf-8')
    expected_points = '10.00000 5.00000 3\r\n11.00000 5.00000 3\r\n12.00000 5.00000 3\r\n'
    assert content == (
        'NPVE DEFLECTION LIST\r\nUNITS=MICRONS\r\nDWELL=0.1\r\nFOV=4.4\r\nSTART\r\n' + expected_points + 'END\r\n'
    )


def test_line_endings_are_crlf_on_every_platform(tmp_path):
    with pytest.warns(UserWarning):
        layout().export(NPVETxt).save(tmp_path / 'points.txt')
    data = (tmp_path / 'points.txt').read_bytes()
    assert data.count(b'\r\n') == data.count(b'\n') and b'\r\r' not in data


def test_dwell_times_are_multiples_of_the_base_dwell_time(tmp_path):
    # 0.3 µs / 0.1 µs is not exactly 3 in floating point arithmetic: the value must not be truncated to 2
    with pytest.warns(UserWarning):
        layout(0.3).export(NPVETxt).save(tmp_path / 'points.txt')
    points = np.loadtxt((tmp_path / 'points.txt').read_text().split('START')[1].split('END')[0].splitlines())
    assert points[:, 2].tolist() == [3, 3, 3]


def test_description():
    assert NPVETxt(description='x').description == 'x'
