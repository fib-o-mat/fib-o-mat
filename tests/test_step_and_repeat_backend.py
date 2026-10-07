"""Tests of the NPVE step and repeat backend.

The golden master tests compare the generated XML with files which were created with the backend before its rework
(`tests/sar_corpus.py`): the file format must not change at all.
"""
import pathlib
import warnings

import pytest
import xmltodict

from fibomat.composite_shapes import Ring
from fibomat.default_backends.npve import (
    LineByLineOutlined, NPVEMill, OutlineAlignement, OutlineNodeStyle, OutlineScanStyle, StepAndRepeatBackend
)
from fibomat.layout import Layout
from fibomat.raster_styles import ScanSequence, one_d, two_d, zero_d
from fibomat.shapes import Circle, Ellipse, Rect, Spot, Line, ArcSpline
from fibomat.units import unit

import sar_corpus
from sar_corpus import curve, dose_mill, lines, outlined, repeats_mill, single_site, um


GOLDEN = pathlib.Path(__file__).parent / 'data' / 'sar_golden'


@pytest.mark.parametrize('name', sorted(sar_corpus.cases()))
def test_file_format_did_not_change(name):
    layout, kwargs = sar_corpus.cases()[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        generated = layout.export(StepAndRepeatBackend, **kwargs)._to_str()  # pylint: disable=protected-access
    assert generated == (GOLDEN / f'{name}.xml').read_text(encoding='utf-8')


def test_all_golden_files_are_used():
    assert {path.stem for path in GOLDEN.glob('*.xml')} == set(sar_corpus.cases())


def parse(layout, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        text = layout.export(StepAndRepeatBackend, **kwargs)._to_str()  # pylint: disable=protected-access
    return xmltodict.parse(text)['StepAndRepeatPatterning']


def shapes_of(document, site=0):
    sites = document['Sites']['Site']
    sites = sites if isinstance(sites, list) else [sites]
    shapes = sites[site]['Shapes']['FIBShape']
    return shapes if isinstance(shapes, list) else [shapes]


class TestStructure:
    def test_shape_classes_and_ids(self):
        document = parse(single_site([
            (Spot((0, 0)) * unit('µm'), repeats_mill(), zero_d.SingleSpot()),
            (Line((0, 0), (1, 1)) * unit('µm'), repeats_mill(), curve()),
            (Rect(1, 1) * unit('µm'), repeats_mill(), lines()),
            (Circle(1.) * unit('µm'), repeats_mill(), lines()),
            (Ellipse(2, 1) * unit('µm'), repeats_mill(), lines()),
        ]))
        shapes = shapes_of(document)
        assert [shape['Class'] for shape in shapes] == ['TFIBSpot', 'TLine', 'TPolygon', 'TEllipse', 'TEllipse']
        assert [shape['DisplayID'] for shape in shapes] == ['10000', '10001', '10002', '10003', '10004']

    def test_mill_values(self):
        document = parse(single_site([
            (Rect(1, 1) * unit('µm'), repeats_mill(3, 2.5), lines(ScanSequence.SERPENTINE, 0.3, True, 0.04, 0.02)),
        ]))
        mill = shapes_of(document)[0]['Mill']
        assert mill['NumFrames'] == '3' and mill['TargetMode'] == '3'
        assert mill['DwellTime'] == '2.5'
        assert mill['TargetDu'] == '0.02' and mill['TargetDv'] == '0.04'
        assert mill['RasterStyle'] == '1'
        assert float(mill['Angle']) == pytest.approx(0.3 - 3.141592653589793 / 2 + 3.141592653589793)

    def test_dose_mode(self):
        document = parse(single_site([(Rect(1, 1) * unit('µm'), dose_mill('nC/µm**2', 5.), lines())]))
        mill = shapes_of(document)[0]['Mill']
        assert mill['TargetMode'] == '0' and mill['TargetDose'] == '5.0' and mill['NumFrames'] == '0'

    def test_sites_are_relative(self):
        layout = Layout()
        for center in ((0, 0), (10, 5), (4, 4)):
            layout.create_site(um(*center), um(4, 4)).create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
        sites = parse(layout)['Sites']['Site']
        assert [(site['X'], site['Y']) for site in sites] == [('0.0', '0.0'), ('10.0', '5.0'), ('-6.0', '-1.0')]
        assert [site['FOV'] for site in sites] == ['4.0'] * 3

    def test_lengths_are_converted_to_microns(self):
        document = parse(single_site([(Circle(1500., center=(1000., 0.)) * unit('nm'), repeats_mill(), lines())]))
        center = shapes_of(document)[0]['RotationCenter']
        assert float(center['X']) == pytest.approx(1.) and float(center['Y']) == pytest.approx(0.)

    def test_ring(self):
        document = parse(single_site([(Ring(3., 1., center=(1, 1)) * unit('µm'), repeats_mill(), lines())]))
        ring = shapes_of(document)[0]
        assert ring['Class'] == 'TRing' and ring['ShapeName'] == 'Ring'
        assert [(float(node['X']), float(node['Y']), node['NodeType']) for node in ring['Nodes']['Node']] == [
            (-2., 4., '0'), (4., 4., '1'), (4., -2., '1'), (-2., -2., '129')
        ]
        assert ring['RotationCenter']['X'] == '1.0'
        assert ring['Angle'] == '0.0'
        assert ring['Outline'] == {
            'Outlined': 'true', 'Thickness': '1.0', 'NodeStyle': '0', 'StrokeStyle': '0', 'PenAlignment': '1',
            'OutlineOffset': '0.0', 'Direction': '1'
        }

    def test_outline_of_a_line_by_line_outlined_style(self):
        style = outlined(OutlineAlignement.OUTSET, OutlineScanStyle.ALTERNATING, OutlineNodeStyle.ROUND, 0.07)
        outline = shapes_of(parse(single_site([(Rect(1, 1) * unit('µm'), repeats_mill(), style)])))[0]['Outline']
        assert outline == {
            'Outlined': 'true', 'Thickness': '0.07', 'NodeStyle': '2', 'StrokeStyle': '0', 'PenAlignment': '2',
            'OutlineOffset': '0.0', 'Direction': '2'
        }

    def test_hollow_arc_spline_with_a_custom_approximation_error(self):
        layout = single_site([(sar_corpus.hollow() * unit('µm'), repeats_mill(), lines())])
        coarse = parse(layout, approximation_error=0.1 * unit('µm'))
        fine = parse(layout, approximation_error=0.001 * unit('µm'))
        assert len(shapes_of(coarse)[0]['Nodes']['Node']) < len(shapes_of(fine)[0]['Nodes']['Node'])

    def test_shapes_are_scaled_by_the_site_unit_only_once(self):
        a = parse(single_site([(Rect(1, 1) * unit('µm'), repeats_mill(), lines())]))
        b = parse(single_site([(Rect(1000, 1000) * unit('nm'), repeats_mill(), lines())]))
        assert shapes_of(a)[0]['Nodes'] == shapes_of(b)[0]['Nodes']


class TestBackend:
    def test_save(self, tmp_path):
        layout, _ = sar_corpus.cases()['curves']
        backend = layout.export(StepAndRepeatBackend)
        backend.save(tmp_path / 'file.xml')
        assert (tmp_path / 'file.xml').read_text(encoding='iso-8859-1') == backend._to_str()  # pylint: disable=protected-access
        assert (tmp_path / 'file.xml').read_bytes().startswith(b'<?xml version="1.0" encoding="iso-8859-1"?>')

    def test_print(self, capsys):
        layout, _ = sar_corpus.cases()['curves']
        backend = layout.export(StepAndRepeatBackend)
        backend.print()
        assert capsys.readouterr().out.strip() == backend._to_str().strip()  # pylint: disable=protected-access

    def test_description(self):
        assert StepAndRepeatBackend(description='x').description == 'x'

    def test_no_sites(self):
        with pytest.raises(RuntimeError, match='any sites'):
            Layout().export(StepAndRepeatBackend)._to_str()  # pylint: disable=protected-access

    def test_only_empty_sites(self):
        layout = Layout()
        layout.create_site(um(0, 0), um(1, 1))
        with pytest.raises(RuntimeError, match='any sites'):
            layout.export(StepAndRepeatBackend)._to_str()  # pylint: disable=protected-access

    def test_empty_sites_can_be_forbidden(self):
        layout = Layout()
        layout.create_site(um(0, 0), um(1, 1))
        with pytest.raises(ValueError, match='may not be empty'):
            layout.export(StepAndRepeatBackend, skip_empty_sites=False)

    def test_shared_patterns_need_empty_following_sites(self):
        layout = Layout()
        for _ in range(2):
            layout.create_site(um(0, 0), um(4, 4)).create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
        with pytest.raises(ValueError, match='Only first site'):
            layout.export(StepAndRepeatBackend, share_patterns=True)

    def test_shared_patterns(self):
        layout, kwargs = sar_corpus.cases()['shared_patterns']
        document = parse(layout, **kwargs)
        assert document['Options']['ShareShapes'] == 'true'
        assert len(document['SharedShapes']['FIBShape']) == 2
        assert all(site['Shapes'] is None for site in document['Sites']['Site'])

    def test_spot_dose_warning_only_once(self):
        layout = single_site([
            (Spot((0, 0)) * unit('µm'), repeats_mill(), zero_d.SingleSpot()),
            (Spot((1, 0)) * unit('µm'), repeats_mill(), zero_d.SingleSpot()),
        ])
        with pytest.warns(UserWarning, match='spot dose') as record:
            layout.export(StepAndRepeatBackend)
        assert len([warning for warning in record if 'spot dose' in str(warning.message)]) == 1

    def test_rotated_ellipse(self):
        layout = single_site([(Ellipse(2, 1, 0.5) * unit('µm'), repeats_mill(), lines())])
        with pytest.raises(NotImplementedError, match='Rotated ellipses'):
            layout.export(StepAndRepeatBackend)

    def test_unsupported_shape(self):
        from fibomat.backend import ShapeNotSupportedError
        layout = single_site([(ArcSpline([(0, 0, 0.3), (1, 0, 0)], False) * unit('µm'), repeats_mill(), curve())])
        with pytest.raises(ShapeNotSupportedError):
            layout.export(StepAndRepeatBackend)

    def test_mill_must_be_an_npve_mill(self):
        from fibomat.mill import Mill
        layout = single_site([(Rect(1, 1) * unit('µm'), Mill(1. * unit('µs'), 1), lines())])
        with pytest.raises(TypeError, match='NPVEMill'):
            layout.export(StepAndRepeatBackend)

    def test_unsupported_scan_sequences(self):
        with pytest.raises(NotImplementedError, match='not supported by NPVE'):
            single_site([(Rect(1, 1) * unit('µm'), repeats_mill(), lines(ScanSequence.BACKSTITCH))]).export(
                StepAndRepeatBackend
            )
        with pytest.raises(ValueError, match='CONSECUTIVE'):
            single_site([(Line((0, 0), (1, 1)) * unit('µm'), repeats_mill(), one_d.Curve(
                0.1 * unit('µm'), ScanSequence.BACKSTITCH))]).export(StepAndRepeatBackend)

    def test_unsupported_raster_style(self):
        with pytest.raises(TypeError, match='Unsupported raster style'):
            single_site([(Rect(1, 1) * unit('µm'), repeats_mill(), zero_d.PreRasterized())]).export(
                StepAndRepeatBackend
            )

    @pytest.mark.parametrize('error', [0. * unit('µm'), 1. * unit('s')])
    def test_invalid_approximation_error(self, error):
        with pytest.raises(ValueError):
            StepAndRepeatBackend(approximation_error=error)
        with pytest.raises(TypeError):
            StepAndRepeatBackend(approximation_error=0.01)

    def test_bitmap_patterns(self, monkeypatch):
        # the bitmap backend is not reworked yet: a replacement which only provides the image is used
        import sys
        import types

        import PIL.Image

        from fibomat.backend import BackendBase

        class FakeBitmapBackend(BackendBase):
            def rect(self, ptn):
                pass

            def image(self):
                return PIL.Image.new('RGBA', (4, 4))

        module = types.ModuleType('fibomat.default_backends.bitmap_backend')
        module.BitmapBackend = FakeBitmapBackend
        monkeypatch.setitem(sys.modules, 'fibomat.default_backends.bitmap_backend', module)

        layout = Layout()
        layout.create_site(um(0, 0), um(4, 4)).create_pattern(
            Rect(2, 1).translated((1, 1)) * unit('µm'), repeats_mill(), lines(), use_bitmap=True
        )
        shape = shapes_of(parse(layout))[0]
        assert shape['Class'] == 'TRectangle' and shape['HasBitmap'] == 'true'
        assert [(float(node['X']), float(node['Y']), node['NodeType']) for node in shape['Nodes']['Node']] == [
            (0., 0.5, '0'), (2., 0.5, '1'), (2., 1.5, '1'), (0., 1.5, '129')
        ]

    def test_shape_texture(self):
        import PIL.Image

        from fibomat.default_backends.npve.step_and_repeat.common_models import ShapeTexture, encode_image
        texture = ShapeTexture(PIL.Image.new('RGBA', (2, 2)), (1. * unit('µm'), 1. * unit('µm')), lines(line_pitch=0.04))
        assert (texture.du, texture.dv) == (0.01, 0.04)
        assert texture.encoded_image['filename'] == 'bitmap.png'
        assert encode_image(b'Man') == 'JM5k'  # base64 'TWFu' in the alphabet of NPVE
        with pytest.raises(ValueError, match='LineByLine'):
            ShapeTexture(PIL.Image.new('RGBA', (2, 2)), (1. * unit('µm'), 1. * unit('µm')), curve())


class TestNPVEMill:
    def test_repeats(self):
        mill = NPVEMill(dwell_time=2. * unit('µs'), repeats=3, scan_direction=0.5, custom='x')
        assert (mill.repeats, mill.dose, mill.scan_direction) == (3, None, 0.5)
        assert mill.dwell_time == 2. * unit('µs')
        assert mill['custom'] == 'x'

    def test_dose(self):
        mill = NPVEMill(dwell_time=1. * unit('µs'), dose=2. * unit('nC/µm'))
        assert mill.repeats is None and mill.dose == 2. * unit('nC/µm')
        with pytest.raises(KeyError):
            mill['repeats']

    @pytest.mark.parametrize('dose', ['ions', 'nC/µm', 'nC/µm**2', 'pA*µs'])
    def test_dose_dimensions(self, dose):
        NPVEMill(dwell_time=1. * unit('µs'), dose=1. * unit(dose))

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match='not both'):
            NPVEMill(dwell_time=1. * unit('µs'))
        with pytest.raises(ValueError, match='not both'):
            NPVEMill(dwell_time=1. * unit('µs'), repeats=1, dose=1. * unit('ions'))
        with pytest.raises(TypeError, match='dimensioned'):
            NPVEMill(dwell_time=1., repeats=1)
        with pytest.raises(ValueError, match='positive time'):
            NPVEMill(dwell_time=1. * unit('µm'), repeats=1)
        with pytest.raises(ValueError):
            NPVEMill(dwell_time=0. * unit('µs'), repeats=1)
        for repeats in (0, -1):
            with pytest.raises(ValueError, match='at least 1'):
                NPVEMill(dwell_time=1. * unit('µs'), repeats=repeats)
        for repeats in (1.5, 'a', True):
            with pytest.raises(TypeError):
                NPVEMill(dwell_time=1. * unit('µs'), repeats=repeats)
        with pytest.raises(TypeError):
            NPVEMill(dwell_time=1. * unit('µs'), dose=1.)
        with pytest.raises(ValueError, match='dose'):
            NPVEMill(dwell_time=1. * unit('µs'), dose=1. * unit('µm'))
        with pytest.raises(ValueError):
            NPVEMill(dwell_time=1. * unit('µs'), dose=-1. * unit('ions'))

    def test_positional_arguments_are_not_allowed(self):
        with pytest.raises(TypeError):
            NPVEMill(1. * unit('µs'), 1)


class TestOutlineStyle:
    def test_properties(self):
        style = outlined(OutlineAlignement.CENTER, OutlineScanStyle.OUTSIDE_IN, OutlineNodeStyle.ROUND, 0.1)
        assert style.outline_offset == 0.1 * unit('µm')
        assert style.outline_alignement is OutlineAlignement.CENTER
        assert style.outline_scan_style is OutlineScanStyle.OUTSIDE_IN
        assert style.outline_node_style is OutlineNodeStyle.ROUND
        assert isinstance(style, two_d.LineByLine) and style.dimension == 2
        assert 'outline_offset' in repr(style)

    def test_enum_codes_are_the_npve_codes(self):
        assert [member.value for member in OutlineAlignement] == [1, 2, 0]
        assert [member.value for member in OutlineScanStyle] == [0, 1, 2]
        assert [member.value for member in OutlineNodeStyle] == [0, 2]

    def test_invalid_arguments(self):
        base = dict(
            line_pitch=0.02 * unit('µm'), scan_sequence=ScanSequence.CONSECUTIVE, alpha=0., invert=False,
            line_style=curve(), outline_offset=0.05 * unit('µm'), outline_alignement=OutlineAlignement.INSET,
            outline_scan_style=OutlineScanStyle.INSIDE_OUT, outline_node_style=OutlineNodeStyle.MITERED
        )
        LineByLineOutlined(**base)
        with pytest.raises(TypeError, match='outline_offset'):
            LineByLineOutlined(**{**base, 'outline_offset': 0.05})
        with pytest.raises(ValueError):
            LineByLineOutlined(**{**base, 'outline_offset': 0. * unit('µm')})
        for key in ('outline_alignement', 'outline_scan_style', 'outline_node_style'):
            with pytest.raises(TypeError, match=key):
                LineByLineOutlined(**{**base, key: 1})
        with pytest.raises(ValueError, match='alpha'):
            LineByLineOutlined(**{**base, 'alpha': 4.})
