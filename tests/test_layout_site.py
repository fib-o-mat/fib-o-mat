"""Tests of `fibomat.layout.site`."""
import math

import pytest

from fibomat.arrangements import DimGroup
from fibomat.layout import Pattern, Site
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d
from fibomat.shapes import Line, Rect, Spot
from fibomat.units import unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def style():
    return one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)


def mill():
    return Mill(1. * unit('ms'), 1)


def add_shape(site, shape, shape_unit='µm'):
    return site.create_pattern(shape * unit(shape_unit), mill(), style())


def fov(site):
    return (site.fov.x.m_as('µm'), site.fov.y.m_as('µm'))


class TestAutomaticFov:
    def test_centered_pattern(self):
        site = Site(um(0, 0))
        add_shape(site, Rect(4, 2))
        # half size 2 (largest distance of the pattern from the center of the site), square, scale 1.1
        assert fov(site) == pytest.approx((1.1 * 2 * 2, 1.1 * 2 * 2))

    @pytest.mark.parametrize('shape, half_size', [
        (Line((4, 0), (6, 0)), 6.),  # not centered at the site: the fov must cover the origin, too
        (Line((-6, 1), (-4, 1)), 6.),
        (Line((0, 3), (1, 5)), 5.),
        (Rect(2, 2).translated((10, -10)), 11.),
    ])
    def test_pattern_not_centered_at_the_site(self, shape, half_size):
        site = Site(um(100, 100))
        add_shape(site, shape)
        assert fov(site) == pytest.approx((1.1 * 2 * half_size,) * 2)
        # all corners of the pattern are inside of the fov
        bbox, fov_box = site.bounding_box_abs, site.fov_bounding_box
        assert fov_box.contains(bbox)

    def test_fov_scale(self):
        site = Site(um(0, 0), fov_scale=2.)
        add_shape(site, Line((0, 0), (3, 0)))
        assert fov(site) == pytest.approx((12., 12.))
        assert site.fov_scale == 2.

    def test_default_fov_scale(self):
        assert Site(um(0, 0)).fov_scale == 1.1

    def test_fov_scale_of_one_is_the_minimal_fov(self):
        site = Site(um(0, 0), fov_scale=1)
        add_shape(site, Line((0, 0), (3, 0)))
        assert fov(site) == pytest.approx((6., 6.))

    def test_several_patterns_and_units(self):
        site = Site(um(0, 0))
        add_shape(site, Line((0, 0), (2, 0)))
        add_shape(site, Line((0, 0), (0, -5000)), 'nm')  # 5 µm
        assert fov(site) == pytest.approx((1.1 * 2 * 5,) * 2)

    def test_fov_follows_new_patterns(self):
        site = Site(um(0, 0), fov_scale=1.)
        add_shape(site, Line((0, 0), (1, 0)))
        assert fov(site) == pytest.approx((2., 2.))
        add_shape(site, Line((0, 0), (0, 4)))
        assert fov(site) == pytest.approx((8., 8.))

    def test_result_has_the_unit_of_the_patterns(self):
        site = Site(um(0, 0), fov_scale=1.)
        add_shape(site, Line((0, 0), (500, 0)), 'nm')
        assert site.fov.x.m_as('nm') == pytest.approx(1000.)

    def test_empty_site(self):
        with pytest.raises(ValueError, match='empty'):
            Site(um(0, 0)).fov

    def test_pattern_without_extent(self):
        site = Site(um(5, 5))
        add_shape(site, Spot((0, 0)))
        with pytest.raises(ValueError, match='extent'):
            site.fov

    def test_spot_not_in_the_center_has_an_fov(self):
        site = Site(um(0, 0), fov_scale=1.)
        add_shape(site, Spot((2, -3)))
        assert fov(site) == pytest.approx((6., 6.))


class TestExplicitFov:
    def test_explicit_fov_is_used(self):
        site = Site(um(0, 0), um(10, 6))
        add_shape(site, Rect(100, 100))
        assert fov(site) == (10., 6.)
        assert site.has_explicit_fov

    def test_explicit_fov_of_an_empty_site(self):
        assert fov(Site(um(0, 0), um(10, 6))) == (10., 6.)

    def test_fov_scale_is_ignored(self):
        assert fov(Site(um(0, 0), um(10, 6), fov_scale=3.)) == (10., 6.)

    def test_not_explicit(self):
        assert not Site(um(0, 0)).has_explicit_fov

    @pytest.mark.parametrize('dim_fov', [(0, 1), (1, 0), (-1, 1)])
    def test_invalid_fov(self, dim_fov):
        with pytest.raises(ValueError, match='positive'):
            Site(um(0, 0), um(*dim_fov))

    @pytest.mark.parametrize('scale', [0.9, 0., -1., math.inf, math.nan])
    def test_invalid_fov_scale(self, scale):
        with pytest.raises(ValueError, match='fov_scale'):
            Site(um(0, 0), fov_scale=scale)

    def test_square_fov(self):
        assert Site(um(0, 0), um(10, 6)).square_fov.y.m_as('µm') == 10.
        assert Site(um(0, 0), um(4, 6)).square_fov.x.m_as('µm') == 6.

    def test_fov_bounding_box(self):
        box = Site(um(10, 20), um(4, 2)).fov_bounding_box
        assert box.lower_left.vector_as(unit('µm')).close_to((8, 19))
        assert box.upper_right.vector_as(unit('µm')).close_to((12, 21))


class TestPatterns:
    def test_empty(self):
        site = Site(um(0, 0))
        assert site.empty
        with pytest.raises(RuntimeError, match='empty'):
            site.bounding_box
        with pytest.raises(RuntimeError, match='empty'):
            site.bounding_box_abs

    def test_create_and_add(self):
        site = Site(um(0, 0), description='site')
        created = add_shape(site, Rect(1, 1))
        site += Pattern(Rect(1, 1) * unit('µm'), mill(), style())
        site.add_pattern(Pattern(Rect(1, 1) * unit('µm'), mill(), style()))
        assert site.description == 'site'
        assert len(site.patterns) == 3 and site.patterns[0] is created
        assert not site.empty

    def test_add_arrangement(self):
        site = Site(um(0, 0))
        pattern = Pattern(Rect(1, 1) * unit('µm'), mill(), style())
        site += DimGroup([pattern, pattern.translated(um(3, 0))])
        assert len(site.patterns) == 2

    def test_add_wrong_type(self):
        site = Site(um(0, 0))
        with pytest.raises(TypeError):
            site.add_pattern(Rect(1, 1) * unit('µm'))
        with pytest.raises(TypeError):
            site.add_pattern(DimGroup([Rect(1, 1) * unit('µm')]))
        assert site.empty

    def test_patterns_is_a_copy(self):
        site = Site(um(0, 0))
        add_shape(site, Rect(1, 1))
        site.patterns.clear()
        assert len(site.patterns) == 1

    def test_bounding_boxes(self):
        site = Site(um(10, 0))
        add_shape(site, Rect(2, 2).translated((1, 0)))
        relative, absolute = site.bounding_box, site.bounding_box_abs
        assert relative.lower_left.vector_as(unit('µm')).close_to((0, -1))
        assert absolute.lower_left.vector_as(unit('µm')).close_to((10, -1))
        assert absolute.upper_right.vector_as(unit('µm')).close_to((12, 1))

    def test_bounding_box_abs_does_not_change_the_site(self):
        site = Site(um(10, 0))
        add_shape(site, Rect(2, 2))
        first = site.bounding_box_abs
        second = site.bounding_box_abs
        assert first.close_to(second)
        assert site.bounding_box.lower_left.vector_as(unit('µm')).close_to((-1, -1))

    def test_patterns_absolute(self):
        site = Site(um(10, 5))
        add_shape(site, Rect(2, 2))
        absolute = site.patterns_absolute
        assert absolute[0].center.vector_as(unit('µm')).close_to((10, 5))
        assert site.patterns[0].center.vector_as(unit('µm')).close_to((0, 0))

    def test_repr(self):
        assert 'n_patterns=0' in repr(Site(um(0, 0)))


class TestTransformations:
    @staticmethod
    def site(**kwargs):
        site = Site(um(10, 0), **kwargs)
        add_shape(site, Line((1, 0), (3, 0)))
        return site

    def test_translated_moves_only_the_center(self):
        moved = self.site(dim_fov=um(8, 4)).translated(um(5, 5))
        assert moved.center.vector_as(unit('µm')).close_to((15, 5))
        assert moved.bounding_box.upper_right.vector_as(unit('µm')).close_to((3, 0))
        assert fov(moved) == (8., 4.)

    def test_translated_returns_a_new_site(self):
        site = self.site()
        site.translated(um(5, 5))
        assert site.center.vector_as(unit('µm')).close_to((10, 0))

    def test_rotated_quarter_turn_swaps_the_explicit_fov(self):
        rotated = self.site(dim_fov=um(8, 4)).rotated(math.pi / 2)
        assert fov(rotated) == pytest.approx((4., 8.))
        assert rotated.center.vector_as(unit('µm')).close_to((0, 10))
        assert rotated.bounding_box.upper_right.vector_as(unit('µm')).close_to((0, 3))
        assert rotated.theta == pytest.approx(math.pi / 2)

    def test_rotated_half_turn_keeps_the_fov(self):
        rotated = self.site(dim_fov=um(8, 4)).rotated(math.pi)
        assert fov(rotated) == pytest.approx((8., 4.))

    @pytest.mark.parametrize('theta', [-math.pi / 2, 3 * math.pi / 2, math.pi / 2 + 1e-12, 5 * math.pi / 2])
    def test_rotation_by_multiples_of_a_quarter_turn(self, theta):
        rotated = self.site(dim_fov=um(8, 4)).rotated(theta)
        assert fov(rotated) == pytest.approx((4., 8.))

    def test_rotated_with_automatic_fov(self):
        rotated = self.site(fov_scale=1.).rotated(math.pi / 2)
        assert fov(rotated) == pytest.approx((6., 6.))

    def test_rotation_by_other_angles_is_not_allowed(self):
        with pytest.raises(ValueError, match='pi/2'):
            self.site().rotated(0.3)

    def test_rotated_about_the_center(self):
        rotated = self.site().rotated(math.pi, origin='center')
        assert rotated.center.vector_as(unit('µm')).close_to((10, 0))
        assert rotated.bounding_box.lower_left.vector_as(unit('µm')).close_to((-3, 0))

    def test_scaled(self):
        scaled = self.site(dim_fov=um(8, 4)).scaled(2.)
        assert scaled.center.vector_as(unit('µm')).close_to((20, 0))
        assert fov(scaled) == pytest.approx((16., 8.))
        assert scaled.bounding_box.upper_right.vector_as(unit('µm')).close_to((6, 0))

    def test_scaled_with_automatic_fov(self):
        assert fov(self.site(fov_scale=1.).scaled(2.)) == pytest.approx((12., 12.))

    def test_negative_scale_keeps_the_fov_positive(self):
        scaled = self.site(dim_fov=um(8, 4)).scaled(-2.)
        assert fov(scaled) == pytest.approx((16., 8.))
        assert scaled.center.vector_as(unit('µm')).close_to((-20, 0))

    def test_mirror_on_the_x_axis(self):
        mirrored = self.site(dim_fov=um(8, 4)).mirrored(um(1, 0))
        assert fov(mirrored) == (8., 4.)
        assert mirrored.center.vector_as(unit('µm')).close_to((10, 0))

    def test_mirror_on_the_y_axis_keeps_the_fov(self):
        mirrored = self.site(dim_fov=um(8, 4)).mirrored(um(0, 1))
        assert fov(mirrored) == (8., 4.)
        assert mirrored.center.vector_as(unit('µm')).close_to((-10, 0))
        assert mirrored.bounding_box.lower_left.vector_as(unit('µm')).close_to((-3, 0))

    @pytest.mark.parametrize('axis', [(1, 1), (-1, 1), (3, 3)])
    def test_mirror_on_a_diagonal_swaps_the_fov(self, axis):
        mirrored = self.site(dim_fov=um(8, 4)).mirrored(um(*axis))
        assert fov(mirrored) == pytest.approx((4., 8.))

    def test_mirror_on_other_axes_is_not_allowed(self):
        with pytest.raises(ValueError, match='diagonals'):
            self.site().mirrored(um(1, 2))

    def test_mirror_with_automatic_fov(self):
        assert fov(self.site(fov_scale=1.).mirrored(um(1, 1))) == pytest.approx((6., 6.))

    def test_transformations_keep_fov_scale_and_description(self):
        site = Site(um(0, 0), fov_scale=1.5, description='s')
        add_shape(site, Line((0, 0), (1, 0)))
        moved = site.translated(um(1, 1))
        assert moved.fov_scale == 1.5 and moved.description == 's'
