"""Tests of `fibomat.layout.pattern`."""
import math

import pytest

from fibomat.arrangements import DimGroup
from fibomat.layout import Pattern
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d
from fibomat.shapes import Line, Rect
from fibomat.units import unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def make_pattern(shape=None, **kwargs):
    return Pattern(
        (shape or Rect(2, 2)) * unit('µm'), Mill(1. * unit('ms'), 2),
        one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE), **kwargs
    )


class TestConstruction:
    def test_properties(self):
        mill = Mill(1. * unit('ms'), 2)
        style = one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
        dim_shape = Rect(2, 2) * unit('µm')
        pattern = Pattern(dim_shape, mill, style, description='rect', _color='red', custom=1)
        assert pattern.dim_shape is dim_shape
        assert pattern.mill is mill
        assert pattern.raster_style is style
        assert pattern.description == 'rect'
        assert pattern.kwargs == {'_color': 'red', 'custom': 1}

    def test_repr(self):
        assert repr(make_pattern()).startswith('Pattern(dim_shape=')

    def test_mill_may_be_none(self):
        pattern = Pattern(Rect(1, 1) * unit('µm'), None, one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE))
        assert pattern.mill is None

    def test_arrangement_as_shape(self):
        group = DimGroup([Rect(1, 1) * unit('µm'), (Rect(1, 1) * unit('µm')).translated(um(5, 0))])
        pattern = Pattern(group, Mill(1. * unit('ms'), 1), one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE))
        assert pattern.bounding_box.width.m_as('µm') == pytest.approx(6.)

    @pytest.mark.parametrize('dim_shape', [Rect(1, 1), 'foo', None])
    def test_invalid_shape(self, dim_shape):
        with pytest.raises(TypeError, match='dim_shape'):
            Pattern(dim_shape, Mill(1. * unit('ms'), 1), one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE))

    def test_invalid_mill(self):
        with pytest.raises(TypeError, match='mill'):
            Pattern(Rect(1, 1) * unit('µm'), 'foo', one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE))

    def test_invalid_raster_style(self):
        with pytest.raises(TypeError, match='raster_style'):
            Pattern(Rect(1, 1) * unit('µm'), Mill(1. * unit('ms'), 1), 'foo')


class TestGeometryAndTransformations:
    def test_center_and_bounding_box(self):
        pattern = make_pattern(Rect(2, 4).translated((1, 1)))
        assert pattern.center.x.m_as('µm') == pytest.approx(1.)
        assert pattern.bounding_box.height.m_as('µm') == pytest.approx(4.)

    def test_translated_returns_new_pattern(self):
        pattern = make_pattern()
        moved = pattern.translated(um(3, 0))
        assert moved is not pattern
        assert moved.center.x.m_as('µm') == pytest.approx(3.)
        assert pattern.center.x.m_as('µm') == pytest.approx(0.)

    def test_translation_with_other_unit(self):
        moved = make_pattern().translated(DimVector(1000 * unit('nm'), 0 * unit('nm')))
        assert moved.center.x.m_as('µm') == pytest.approx(1.)

    def test_rotated_scaled_mirrored(self):
        pattern = make_pattern(Line((0, 0), (2, 0)))
        rotated = pattern.rotated(math.pi / 2)
        assert rotated.bounding_box.height.m_as('µm') == pytest.approx(2.)
        assert pattern.bounding_box.height.m_as('µm') == pytest.approx(0.)
        assert pattern.scaled(3.).bounding_box.width.m_as('µm') == pytest.approx(6.)
        mirrored = pattern.mirrored(um(0, 1))
        assert mirrored.bounding_box.lower_left.x.m_as('µm') == pytest.approx(-2.)

    def test_clone_shares_mill_and_style_but_copies_the_shape(self):
        pattern = make_pattern(_color='red')
        clone = pattern.clone()
        assert clone.mill is pattern.mill
        assert clone.raster_style is pattern.raster_style
        assert clone.dim_shape is not pattern.dim_shape
        assert clone.kwargs == pattern.kwargs and clone.kwargs is not pattern.kwargs

    def test_transformed_keeps_description_and_kwargs(self):
        moved = make_pattern(description='d', _color='red').translated(um(1, 1))
        assert moved.description == 'd'
        assert moved.kwargs == {'_color': 'red'}

    def test_with_changed_description(self):
        pattern = make_pattern(description='a')
        assert pattern.with_changed_description('b').description == 'b'
        assert pattern.description == 'a'
