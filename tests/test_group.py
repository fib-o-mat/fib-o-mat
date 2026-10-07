"""Tests of `fibomat.arrangements.groups` (`Group` and `DimGroup`) and `ArrangementBase`."""
import math

import numpy as np
import pytest

from fibomat.arrangements import ArrangementBase, DimGroup, Group
from fibomat.layout import Pattern, Site
from fibomat.linalg import DimVector, Vector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d
from fibomat.shapes import Circle, DimShape, Line, Rect
from fibomat.units import DimensionError, unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def rects(n=3, size=1., distance=4.):
    return [Rect(size, size).translated((distance * i, 0)) for i in range(n)]


def dim_rects(n=3, size=1., distance=4.):
    return [Rect(size, size).translated((distance * i, 0)) * unit('µm') for i in range(n)]


class TestGroup:
    def test_properties(self):
        elements = rects()
        group = Group(elements, description='rects')
        assert isinstance(group, ArrangementBase)
        assert group.elements == tuple(elements)
        assert len(group) == 3 and list(group) == elements and group[1] is elements[1]
        assert group[1:] == tuple(elements[1:])
        assert group.description == 'rects'
        assert repr(group) == 'Group(n_elements=3)'

    def test_elements_can_be_a_generator(self):
        assert len(Group(rect for rect in rects())) == 3

    def test_elements_are_immutable(self):
        group = Group(rects())
        assert isinstance(group.elements, tuple)

    def test_center_and_bounding_box(self):
        group = Group(rects())
        assert group.center == Vector(4, 0)
        bbox = group.bounding_box
        assert bbox.lower_left == Vector(-0.5, -0.5) and bbox.upper_right == Vector(8.5, 0.5)

    def test_no_elements(self):
        with pytest.raises(ValueError, match='at least one'):
            Group([])

    @pytest.mark.parametrize('element', ['foo', 1, None])
    def test_elements_must_be_transformable(self, element):
        with pytest.raises(TypeError, match='transformable'):
            Group([Rect(1, 1), element])

    def test_elements_must_not_have_units(self):
        with pytest.raises(TypeError, match='DimGroup'):
            Group([Rect(1, 1), Rect(1, 1) * unit('µm')])

    def test_groups_of_groups(self):
        group = Group([Group(rects(2)), Rect(1, 1).translated((20, 0))])
        assert group.bounding_box.upper_right == Vector(20.5, 0.5)


class TestGroupTransformations:
    def test_translated(self):
        group = Group(rects())
        moved = group.translated((1, 2))
        assert moved is not group
        assert moved.center == Vector(5, 2)
        assert group.center == Vector(4, 0)  # the original is not changed
        assert group.elements[0].center == Vector(0, 0)

    def test_center_and_bounding_box_are_not_stale_after_transformations(self):
        group = Group(rects())
        # (the center and the bounding box used to be cached and copied by the transformations)
        assert group.center == Vector(4, 0) and group.bounding_box.upper_right == Vector(8.5, 0.5)
        moved = group.translated((10, 10))
        assert moved.center == Vector(14, 10)
        assert moved.bounding_box.lower_left == Vector(9.5, 9.5)
        scaled = group.scaled(2.)
        assert scaled.center == Vector(8, 0)
        assert scaled.bounding_box.upper_right == Vector(17, 1)

    def test_rotated(self):
        rotated = Group([Line((1, 0), (2, 0))]).rotated(math.pi / 2)
        assert rotated.elements[0].start.close_to((0, 1))
        assert rotated.bounding_box.upper_right.close_to((0, 2))

    def test_rotated_about_the_center(self):
        group = Group([Line((0, 0), (2, 0)), Line((0, 2), (2, 2))])
        rotated = group.rotated(math.pi, origin='center')
        assert rotated.center.close_to(group.center)
        assert rotated.elements[0].start.close_to((2, 2))

    def test_scaled_mirrored(self):
        group = Group(rects(2))
        assert group.scaled(3., origin='center').bounding_box.width == pytest.approx(3 * 5)
        assert group.mirrored((0, 1)).bounding_box.lower_left.close_to((-4.5, -0.5))

    def test_transformed(self):
        from fibomat.linalg import rotate, translate
        group = Group([Line((1, 0), (2, 0))]).transformed(translate((1, 1)) | rotate(math.pi))
        assert group.bounding_box.upper_right.close_to((-2, -1))

    def test_pivot(self):
        group = Group(rects(2))
        group.pivot = lambda obj: Vector(100, 100)
        assert group.translated_to((0, 0)).center.close_to((2 - 100, 0 - 100))

    def test_clone_and_description(self):
        group = Group(rects(), description='a')
        clone = group.with_changed_description('b')
        assert clone.description == 'b' and group.description == 'a'
        assert clone.elements[0] is not group.elements[0]

    def test_nested_groups_are_transformed(self):
        group = Group([Group(rects(2)), Rect(1, 1)])
        assert group.translated((5, 0)).elements[0].elements[0].center == Vector(5, 0)


class TestArrangementElements:
    def test_elements(self):
        elements = rects()
        assert list(Group(elements).arrangement_elements()) == elements

    def test_nested_arrangements_are_resolved(self):
        a, b, c, d = rects(4)
        group = Group([a, Group([b, Group([c, Group([d])])])])
        assert list(group.arrangement_elements()) == [a, b, c, d]  # (more than one level)

    def test_dim_group(self):
        elements = dim_rects(2)
        group = DimGroup([DimGroup([elements[0]]), elements[1]])
        assert list(group.arrangement_elements()) == elements

    def test_the_base_class_is_abstract(self):
        with pytest.raises(TypeError):
            ArrangementBase()


class TestDimGroup:
    def test_properties(self):
        elements = dim_rects()
        group = DimGroup(elements, description='rects')
        assert isinstance(group, ArrangementBase)
        assert group.elements == tuple(elements) and len(group) == 3
        assert group.description == 'rects'
        assert repr(group) == 'DimGroup(n_elements=3)'

    def test_center_and_bounding_box(self):
        group = DimGroup(dim_rects())
        assert group.center.vector_as(unit('µm')).close_to((4, 0))
        assert group.bounding_box.width.m_as('µm') == pytest.approx(9.)

    def test_mixed_units(self):
        group = DimGroup([Rect(1, 1) * unit('µm'), Rect(1000, 1000).translated((4000, 0)) * unit('nm')])
        assert group.bounding_box.width.m_as('µm') == pytest.approx(5.)
        assert group.center.vector_as(unit('µm')).close_to((2, 0))

    def test_translated_with_other_unit(self):
        moved = DimGroup(dim_rects(2)).translated(DimVector(1000 * unit('nm'), 0 * unit('nm')))
        assert moved.bounding_box.lower_left.vector_as(unit('µm')).close_to((0.5, -0.5))

    def test_transformations(self):
        group = DimGroup(dim_rects(2))
        assert group.rotated(math.pi / 2).bounding_box.height.m_as('µm') == pytest.approx(5.)
        assert group.scaled(2.).bounding_box.width.m_as('µm') == pytest.approx(2 * 5.)
        assert group.mirrored(um(0, 1)).bounding_box.lower_left.vector_as(unit('µm')).close_to((-4.5, -0.5))
        # not stale after translation
        assert group.center.vector_as(unit('µm')).close_to((2, 0))
        assert group.translated(um(10, 0)).center.vector_as(unit('µm')).close_to((12, 0))

    def test_elements_must_have_units(self):
        with pytest.raises(TypeError, match='need units'):
            DimGroup([Rect(1, 1) * unit('µm'), Rect(1, 1)])
        with pytest.raises(TypeError):
            DimGroup(['foo'])
        with pytest.raises(ValueError):
            DimGroup([])

    def test_group_of_patterns(self):
        pattern = Pattern(
            Rect(1, 1) * unit('µm'), Mill(1. * unit('ms'), 1), one_d.Curve(0.1 * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        group = DimGroup([pattern, pattern.translated(um(5, 0))])
        assert group.translated(um(0, 5)).elements[1].center.vector_as(unit('µm')).close_to((5, 5))
        assert list(group.arrangement_elements())[0] is pattern

    def test_group_of_sites(self):
        sites = [Site(um(0, 0), um(1, 1)), Site(um(10, 0), um(1, 1))]
        group = DimGroup(sites)
        assert group.translated(um(1, 1)).elements[1].center.vector_as(unit('µm')).close_to((11, 1))


class TestMultiplicationWithUnits:
    def test_group_times_unit(self):
        group = Group(rects(2), description='rects')
        dim_group = group * unit('µm')
        assert isinstance(dim_group, DimGroup)
        assert dim_group.description == 'rects'
        assert all(isinstance(element, DimShape) for element in dim_group.elements)
        assert dim_group.bounding_box.width.m_as('µm') == pytest.approx(5.)

    def test_unit_times_group(self):
        assert isinstance(unit('nm') * Group(rects(2)), DimGroup)

    def test_nested_groups(self):
        dim_group = Group([Group(rects(2)), Circle(1.)]) * unit('µm')
        assert isinstance(dim_group.elements[0], DimGroup)

    def test_unit_conversion(self):
        in_nm = Group(rects(2)) * unit('nm')
        assert in_nm.bounding_box.width.m_as('µm') == pytest.approx(5e-3)

    def test_other_operands(self):
        group = Group(rects(2))
        with pytest.raises(TypeError):
            group * 2
        with pytest.raises(TypeError):
            group * 'µm'

    def test_unit_of_another_dimension(self):
        with pytest.raises(DimensionError):
            Group(rects(2)) * unit('s')
