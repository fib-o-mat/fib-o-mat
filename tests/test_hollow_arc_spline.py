import copy
import math

import numpy as np
import pytest

from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline
from fibomat.linalg import BoundingBox, Vector, mirror, rotate, scale, translate
from fibomat.shapes import Circle, Rect, ArcSpline, Line
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi


def boundary(size=10.):
    return Rect(width=size, height=size).to_arc_spline()


def hole(radius=1., center=(0, 0)):
    return Circle(r=radius, center=center).to_arc_spline()


def square_hole(size=2., center=(0, 0)):
    return Rect(width=size, height=size, center=center).to_arc_spline()


class TestConstruction:
    def test_boundary_only(self):
        hollow = HollowArcSpline(boundary(), description='foo')
        assert hollow.holes == []
        assert hollow.description == 'foo'
        assert isinstance(hollow, Shape)
        assert hollow.is_closed is True
        assert hollow.area == pytest.approx(100.)

    def test_with_hole(self):
        # (holes=None crashed the former implementation)
        hollow = HollowArcSpline(boundary(), [hole(2.)])
        assert len(hollow.holes) == 1
        assert hollow.area == pytest.approx(100. - 4 * PI)
        assert hollow.boundary_length == pytest.approx(40. + 4 * PI)
        assert hollow.bounding_box == BoundingBox((-5, -5), (5, 5))
        assert hollow.center == (0., 0.)

    @pytest.mark.parametrize('holes', [None, [], (), iter(())])
    def test_no_holes(self, holes):
        assert HollowArcSpline(boundary(), holes).holes == []

    def test_holes_may_be_given_as_generator(self):
        hollow = HollowArcSpline(boundary(), (hole(1., (x, 0)) for x in (-3, 3)))
        assert len(hollow.holes) == 2

    def test_orientation_of_the_hole_does_not_change_the_area(self):
        # (the former implementation added or subtracted the area of a hole depending on its orientation)
        for reverse in (False, True):
            curve = hole(2.)
            curve = curve.reversed() if reverse else curve
            assert HollowArcSpline(boundary(), [curve]).area == pytest.approx(100. - 4 * PI)

    def test_input_is_copied(self):
        outer, inner = boundary(), hole(2.)
        hollow = HollowArcSpline(outer, [inner], disable_checks=True)
        outer._impl_translate(Vector(100, 100))
        inner._impl_translate(Vector(100, 100))
        assert hollow.boundary.center == (0., 0.)
        assert hollow.holes[0].center == (0., 0.)

    def test_holes_property_returns_a_new_list(self):
        hollow = HollowArcSpline(boundary(), [hole(2.)])
        hollow.holes.append(hole(1.))
        assert len(hollow.holes) == 1

    def test_invalid_types(self):
        with pytest.raises(TypeError):
            HollowArcSpline(Circle(r=1), [])
        with pytest.raises(TypeError):
            HollowArcSpline(boundary(), [Circle(r=1)])
        with pytest.raises(TypeError):
            HollowArcSpline(boundary(), [hole(), 'foo'])

    def test_open_curves(self):
        line = ArcSpline([(0, 0, 0), (1, 0, 0)], False)
        with pytest.raises(ValueError, match='boundary'):
            HollowArcSpline(line)
        with pytest.raises(ValueError, match='holes'):
            HollowArcSpline(boundary(), [line])

    def test_repr(self):
        assert repr(HollowArcSpline(boundary(), [hole()])).endswith('n_holes=1)')

    def test_from_points(self):
        hollow = HollowArcSpline.from_points(
            [(0, 0), (10, 0), (10, 10), (0, 10)], [[(4, 4), (6, 4), (6, 6), (4, 6)]], description='foo'
        )
        assert hollow.area == pytest.approx(96.)
        assert hollow.description == 'foo'
        assert HollowArcSpline.from_points(np.array([(0, 0), (1, 0), (1, 1)])).area == pytest.approx(0.5)
        with pytest.raises(ValueError):
            HollowArcSpline.from_points([(0, 0, 0), (1, 0, 0), (1, 1, 0)])


class TestHoleHandling:
    def test_disjoint_holes(self):
        hollow = HollowArcSpline(boundary(), [hole(1., (-3, 0)), hole(1., (3, 0))])
        assert len(hollow.holes) == 2
        assert hollow.area == pytest.approx(100. - 2 * PI)

    def test_overlapping_holes_are_merged(self):
        hollow = HollowArcSpline(boundary(), [square_hole(2., (-0.5, 0)), square_hole(2., (0.5, 0))])
        assert len(hollow.holes) == 1
        assert hollow.holes[0].area == pytest.approx(6.)
        assert hollow.area == pytest.approx(94.)

    def test_chain_of_overlapping_holes(self):
        holes = [square_hole(2., (x, 0)) for x in (-2, -1, 0, 1, 2)]
        hollow = HollowArcSpline(boundary(), holes)
        assert len(hollow.holes) == 1
        assert hollow.holes[0].area == pytest.approx(12.)

    def test_hole_touching_the_boundary_is_cut_out(self):
        hollow = HollowArcSpline(boundary(), [square_hole(2., (5, 0))])
        assert hollow.holes == []
        assert hollow.area == pytest.approx(98.)
        assert hollow.boundary.area == pytest.approx(98.)
        assert not hollow.contains((4.5, 0.))
        assert hollow.contains((3., 0.))

    def test_hole_outside_of_the_boundary(self):
        # (the former implementation silently kept such holes)
        with pytest.raises(ValueError, match='outside'):
            HollowArcSpline(boundary(), [hole(1., (20, 0))])

    def test_boundary_inside_of_a_hole(self):
        with pytest.raises(ValueError, match='inside of a hole'):
            HollowArcSpline(hole(1.), [boundary()])

    def test_hole_which_separates_the_shape(self):
        with pytest.raises(ValueError, match='simply connected'):
            HollowArcSpline(boundary(), [Rect(width=2, height=20).to_arc_spline()])

    def test_holes_which_enclose_an_island(self):
        # a C shaped hole closed by a second hole encloses an island
        c_shape = ArcSpline([(-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0), (-3, 2, 0), (2, 2, 0), (2, -2, 0), (-3, -2, 0)], True)
        closing = Rect(width=1, height=5, center=(-3, 0)).to_arc_spline()
        with pytest.raises(ValueError, match='simply connected'):
            HollowArcSpline(boundary(), [c_shape, closing])

    def test_disable_checks(self):
        # the holes are used as they are
        overlapping = [square_hole(2., (-0.5, 0)), square_hole(2., (0.5, 0))]
        assert len(HollowArcSpline(boundary(), overlapping, disable_checks=True).holes) == 2
        assert len(HollowArcSpline(boundary(), overlapping).holes) == 1


class TestContains:
    def test_contains(self):
        hollow = HollowArcSpline(boundary(), [hole(2.)])
        assert hollow.contains((3., 3.))
        assert hollow.contains(Vector(-4., 0.))
        assert not hollow.contains((0., 0.))
        assert not hollow.contains((1.5, 0.))
        assert not hollow.contains((6., 0.))

    def test_invalid_position(self):
        with pytest.raises(ValueError):
            HollowArcSpline(boundary()).contains((1, 2, 3))


class TestTransformations:
    def hollow(self):
        return HollowArcSpline(boundary(), [hole(1., (-3, 0)), hole(1., (3, 1))])

    def test_translate(self):
        hollow = self.hollow()
        res = hollow.translated((1, -1))
        assert isinstance(res, HollowArcSpline)
        assert res.boundary.center == (1., -1.)
        assert res.holes[0].center == (-2., -1.)
        assert res.area == pytest.approx(hollow.area)
        assert res.contains((1., -1.))
        assert hollow.boundary.center == (0., 0.)

    def test_rotate_scale_mirror(self):
        hollow = self.hollow()
        assert hollow.rotated(0.7, (2, 1)).area == pytest.approx(hollow.area)
        assert hollow.mirrored((1, 2)).area == pytest.approx(hollow.area)
        assert hollow.scaled(2.).area == pytest.approx(4 * hollow.area)
        assert hollow.scaled(-2.).area == pytest.approx(4 * hollow.area)
        rotated = hollow.rotated(PI / 2)
        assert rotated.holes[0].center == pytest.approx((0., -3.))
        assert rotated.contains((0., 0.)) and not rotated.contains((0., -3.))

    def test_transformed_chain(self):
        res = self.hollow().transformed(translate((1, 1)) | rotate(0.3) | scale(1.5) | mirror((1, 2)))
        assert res.area == pytest.approx(2.25 * self.hollow().area)

    def test_clone_and_copy(self):
        hollow = HollowArcSpline(boundary(), [hole(1.)], 'foo')
        for clone in (hollow.clone(), hollow.to_hollow_arc_spline(), copy.deepcopy(hollow)):
            assert clone is not hollow
            assert clone.description == 'foo' and clone.area == pytest.approx(hollow.area)
            assert clone.boundary is not hollow.boundary
            clone._impl_translate(Vector(5, 5))
            assert hollow.boundary.center == (0., 0.)

    def test_dim_shape(self):
        dim_hollow = self.hollow() * unit('µm')
        assert dim_hollow.bounding_box.width.m_as('µm') == pytest.approx(10.)
        assert dim_hollow.to('nm').shape.boundary.length == pytest.approx(40000.)
