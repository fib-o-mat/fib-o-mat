import copy
import math

import numpy as np
import pytest

from fibomat.linalg import BoundingBox, DimBoundingBox, DimVector, Vector, VectorValueError
from fibomat.units import unit, DimFloat

um = unit('µm')
nm = unit('nm')


def dbox(ll, ur, u=um):
    return DimBoundingBox(Vector(*ll) * u, Vector(*ur) * u)


class TestConstruction:
    def test_basic(self):
        box = BoundingBox((1, 2), (3, 5))
        assert box.lower_left == (1., 2.)
        assert box.upper_right == (3., 5.)
        assert type(box.lower_left) is Vector

    def test_vector_likes(self):
        assert BoundingBox([1, 2], np.array([3., 4.])) == BoundingBox(Vector(1, 2), Vector(3, 4))

    def test_degenerate_box(self):
        box = BoundingBox((1, 1), (1, 1))
        assert box.width == 0. and box.height == 0. and box.area == 0.

    def test_invalid(self):
        with pytest.raises(ValueError):
            BoundingBox((1, 2), (0, 5))
        with pytest.raises(ValueError):
            BoundingBox((1, 2), (3, 1))
        with pytest.raises(ValueError):
            BoundingBox((np.nan, 0), (1, 1))
        with pytest.raises(ValueError):
            BoundingBox((0, 0), (np.inf, 1))
        with pytest.raises(VectorValueError):
            BoundingBox((1, 2, 3), (4, 5))
        with pytest.raises(VectorValueError):
            BoundingBox((0, 0), (1 * um, 1 * um))

    def test_dim_box(self):
        box = dbox((1, 2), (3, 5))
        assert isinstance(box, BoundingBox)
        assert type(box.lower_left) is DimVector
        assert DimBoundingBox(Vector(1, 2) * um, Vector(3000, 5000) * nm).upper_right == Vector(3, 5) * um

    def test_dim_box_invalid(self):
        with pytest.raises(VectorValueError):
            DimBoundingBox((0, 0), (1, 1))
        with pytest.raises(ValueError):
            DimBoundingBox(Vector(1, 1) * um, Vector(500, 500) * nm)

    def test_unhashable(self):
        with pytest.raises(TypeError):
            hash(BoundingBox((0, 0), (1, 1)))

    def test_mul_unit(self):
        box = BoundingBox((1, 2), (3, 5)) * nm
        assert isinstance(box, DimBoundingBox)
        assert box == dbox((1, 2), (3, 5), nm)
        assert nm * BoundingBox((1, 2), (3, 5)) == box
        with pytest.raises(TypeError):
            dbox((0, 0), (1, 1)) * nm
        with pytest.raises(TypeError):
            BoundingBox((0, 0), (1, 1)) * 2
        with pytest.raises(TypeError):
            BoundingBox((0, 0), (1, 1)) * unit('s')


class TestFromPoints:
    def test_from_points(self):
        box = BoundingBox.from_points([(0, 1), (3, -1), (2, 5)])
        assert box == BoundingBox((0, -1), (3, 5))
        assert type(box) is BoundingBox

    def test_from_array_and_generator(self):
        assert BoundingBox.from_points(np.array([[0., 1.], [3., -1.]])) == BoundingBox((0, -1), (3, 1))
        assert BoundingBox.from_points(p for p in [(0, 1), (3, -1)]) == BoundingBox((0, -1), (3, 1))

    def test_single_point(self):
        assert BoundingBox.from_points([(1, 2)]) == BoundingBox((1, 2), (1, 2))

    def test_invalid(self):
        with pytest.raises(ValueError):
            BoundingBox.from_points([])
        with pytest.raises(ValueError):
            BoundingBox.from_points(np.zeros((0, 2)))
        with pytest.raises(VectorValueError):
            BoundingBox.from_points([(1, 2, 3)])
        with pytest.raises(VectorValueError):
            BoundingBox.from_points(np.zeros((2, 3)))
        with pytest.raises(VectorValueError):
            BoundingBox.from_points([Vector(1, 2) * um])

    def test_dim_from_points(self):
        box = DimBoundingBox.from_points([Vector(0, 1) * um, Vector(3000, -1000) * nm, Vector(2, 5) * um])
        assert isinstance(box, DimBoundingBox)
        assert box == dbox((0, -1), (3, 5))
        assert box.lower_left.unit == um.__rmul__(1.).units  # unit of the first point

    def test_dim_from_points_invalid(self):
        with pytest.raises(ValueError):
            DimBoundingBox.from_points([])
        with pytest.raises(VectorValueError):
            DimBoundingBox.from_points([(1, 2)])


class TestProperties:
    def test_values(self):
        box = BoundingBox((1, 2), (4, 4))
        assert box.width == 3.
        assert box.height == 2.
        assert box.area == 6.
        assert box.center == (2.5, 3.)

    def test_dim_values(self):
        box = dbox((1, 2), (4, 4))
        assert isinstance(box.width, DimFloat)
        assert box.width.m_as('µm') == pytest.approx(3.)
        assert box.height.m_as('nm') == pytest.approx(2000.)
        assert box.area.m_as('µm**2') == pytest.approx(6.)
        assert box.center == Vector(2.5, 3.) * um

    def test_corners_counterclockwise(self):
        box = BoundingBox((0, 0), (2, 1))
        assert list(box.corners) == [(0., 0.), (2., 0.), (2., 1.), (0., 1.)]
        # area of the polygon is positive for counterclockwise order
        x, y = np.array(box.corners).T
        assert 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1)) == pytest.approx(2.)

    def test_dim_corners(self):
        corners = dbox((0, 0), (2, 1)).corners
        assert [type(c) for c in corners] == [DimVector] * 4
        assert corners[1] == Vector(2., 0.) * um

    def test_repr(self):
        assert repr(BoundingBox((0, 0), (1, 1))) == (
            'BoundingBox(lower_left=Vector(x=0.0, y=0.0), upper_right=Vector(x=1.0, y=1.0))'
        )
        assert repr(dbox((0, 0), (1, 1))).startswith('DimBoundingBox(lower_left=DimVector(x=0.0 µm')


class TestComparison:
    def test_eq(self):
        assert BoundingBox((0, 0), (1, 1)) == BoundingBox((0, 0), (1, 1))
        assert BoundingBox((0, 0), (1, 1)) != BoundingBox((0, 0), (1, 2))
        assert BoundingBox((0, 0), (1, 1)) == BoundingBox((0, 0), (1, 1 + 1e-10))
        assert dbox((0, 0), (1, 1)) == dbox((0, 0), (1000, 1000), nm)

    def test_eq_with_other_types(self):
        assert BoundingBox((0, 0), (1, 1)) != 1
        assert BoundingBox((0, 0), (1, 1)) != ((0, 0), (1, 1))
        assert not BoundingBox((0, 0), (1, 1)) == None  # noqa: E711
        assert BoundingBox((0, 0), (1, 1)) != dbox((0, 0), (1, 1))
        assert dbox((0, 0), (1, 1)) != BoundingBox((0, 0), (1, 1))

    def test_close_to(self):
        assert BoundingBox((0, 0), (1, 1)).close_to(BoundingBox((0, 0), (1, 1)))
        assert not BoundingBox((0, 0), (1, 1)).close_to(BoundingBox((0, 0), (2, 1)))
        assert not BoundingBox((0, 0), (1, 1)).close_to('foo')

    def test_clone(self):
        box = BoundingBox((0, 0), (1, 1))
        clone = box.clone()
        assert clone == box and clone is not box
        dim = dbox((0, 0), (1, 1))
        assert dim.clone() == dim and isinstance(dim.clone(), DimBoundingBox)
        assert copy.deepcopy(dim) == dim


class TestScaled:
    @pytest.mark.parametrize('scale', [0.25, 0.5, 1., 1.5, 2., 3.])
    def test_scaled(self, scale):
        for box in (BoundingBox((.5, 1.), (1.5, 3.)), dbox((.5, 1.), (1.5, 3.))):
            scaled = box.scaled(scale)
            assert scaled.center.close_to(box.center)
            assert float(scaled.width / box.width) == pytest.approx(scale)
            assert float(scaled.height / box.height) == pytest.approx(scale)
            assert type(scaled) is type(box)

    def test_scaled_one_is_identity(self):
        box = BoundingBox((.5, 1.), (1.5, 3.))
        assert box.scaled(1.) == box

    def test_scaled_invalid(self):
        for bad in (0., -1., np.nan, np.inf):
            with pytest.raises(ValueError):
                BoundingBox((0, 0), (1, 1)).scaled(bad)


class TestOverlapContains:
    def test_overlaps_with(self):
        box = BoundingBox((0, 0), (2, 2))
        assert box.overlaps_with(BoundingBox((1, 1), (3, 3)))
        assert box.overlaps_with(BoundingBox((0.5, 0.5), (1, 1)))
        assert box.overlaps_with(BoundingBox((2, 2), (3, 3)))  # touching
        assert not box.overlaps_with(BoundingBox((2.1, 0), (3, 1)))
        assert not box.overlaps_with(BoundingBox((0, 2.1), (1, 3)))
        assert not box.overlaps_with(BoundingBox((-2, -2), (-0.1, -0.1)))
        assert box.overlaps_with(box)

    def test_overlaps_with_symmetric(self):
        a, b = BoundingBox((0, 0), (2, 2)), BoundingBox((1, -5), (1.5, 5))
        assert a.overlaps_with(b) and b.overlaps_with(a)

    def test_overlaps_dim(self):
        assert dbox((0, 0), (2, 2)).overlaps_with(dbox((1000, 1000), (3000, 3000), nm))
        assert not dbox((0, 0), (2, 2)).overlaps_with(dbox((2100, 0), (3000, 1000), nm))

    def test_overlaps_invalid(self):
        with pytest.raises(TypeError):
            BoundingBox((0, 0), (1, 1)).overlaps_with(dbox((0, 0), (1, 1)))
        with pytest.raises(TypeError):
            dbox((0, 0), (1, 1)).overlaps_with(BoundingBox((0, 0), (1, 1)))
        with pytest.raises(TypeError):
            BoundingBox((0, 0), (1, 1)).overlaps_with((0, 0))

    def test_contains_box(self):
        box = BoundingBox((0, 0), (4, 4))
        assert box.contains(BoundingBox((1, 1), (2, 2)))
        assert box.contains(box)
        assert BoundingBox((1, 1), (2, 2)) in box
        assert not box.contains(BoundingBox((1, 1), (5, 2)))
        assert BoundingBox((1, 1), (2, 2)) in box

    def test_contains_point(self):
        box = BoundingBox((0, 0), (4, 4))
        assert (1, 1) in box
        assert box.contains((0, 0)) and box.contains(Vector(4, 4))
        assert not box.contains((4.1, 1))
        assert not box.contains((1, -0.1))

    def test_contains_invalid(self):
        with pytest.raises(TypeError):
            BoundingBox((0, 0), (4, 4)).contains('foo')
        with pytest.raises(TypeError):
            BoundingBox((0, 0), (4, 4)).contains((1, 2, 3))
        with pytest.raises(TypeError):
            BoundingBox((0, 0), (4, 4)).contains(dbox((0, 0), (1, 1)))
        with pytest.raises(TypeError):
            BoundingBox((0, 0), (4, 4)).contains(Vector(1, 1) * um)
        with pytest.raises(TypeError):
            dbox((0, 0), (4, 4)).contains(BoundingBox((0, 0), (1, 1)))
        with pytest.raises(TypeError):
            dbox((0, 0), (4, 4)).contains((1, 1))

    def test_contains_dim(self):
        box = dbox((0, 0), (4, 4))
        assert box.contains(dbox((1000, 1000), (2000, 2000), nm))
        assert Vector(1, 1) * um in box
        assert Vector(3999, 1000) * nm in box
        assert Vector(4001, 1000) * nm not in box


class TestExtended:
    def test_extended_with_box(self):
        box = BoundingBox((0, 0), (1, 1))
        assert box.extended(BoundingBox((2, -1), (3, 0.5))) == BoundingBox((0, -1), (3, 1))
        assert box.extended(BoundingBox((0.2, 0.2), (0.5, 0.5))) == box
        assert box.extended(box) == box

    def test_extended_with_point(self):
        box = BoundingBox((0, 0), (1, 1))
        assert box.extended((2, 3)) == BoundingBox((0, 0), (2, 3))
        assert box.extended(Vector(-1, 0.5)) == BoundingBox((-1, 0), (1, 1))
        assert box.extended([-1, -1]) == BoundingBox((-1, -1), (1, 1))
        assert box.extended(np.array([0.5, 0.5])) == box

    def test_extended_with_points(self):
        box = BoundingBox((0, 0), (1, 1))
        assert box.extended([(2, 3), (-1, 0.5), (0, -4)]) == BoundingBox((-1, -4), (2, 3))
        assert box.extended([]) == box

    def test_extended_with_generator(self):
        # a generator can be consumed only once
        box = BoundingBox((0, 0), (1, 1))
        assert box.extended(p for p in [(2, 3), (-1, 0.5)]) == BoundingBox((-1, 0), (2, 3))
        assert box.extended(p for p in [(2, 3), (-1, 0.5), (5, 5)]) == BoundingBox((-1, 0), (5, 5))
        assert box.extended(p for p in [(2, 3), (-1, 0.5)]) == box.extended([(2, 3), (-1, 0.5)])

    def test_extended_with_two_points_is_not_a_vector(self):
        assert BoundingBox((0, 0), (1, 1)).extended([(5, 5), (6, 6)]) == BoundingBox((0, 0), (6, 6))

    def test_extended_does_not_modify(self):
        box = BoundingBox((0, 0), (1, 1))
        box.extended((5, 5))
        assert box == BoundingBox((0, 0), (1, 1))

    def test_extended_invalid(self):
        box = BoundingBox((0, 0), (1, 1))
        for bad in (1, 'foo', [1, 2, 3], [(1, 2), (1, 2, 3)], [(1, 2), 'a']):
            with pytest.raises(TypeError):
                box.extended(bad)
        with pytest.raises(TypeError):
            box.extended(dbox((0, 0), (1, 1)))
        with pytest.raises(TypeError):
            box.extended(Vector(1, 1) * um)

    def test_extended_dim(self):
        box = dbox((0, 0), (1, 1))
        res = box.extended(dbox((2000, -1000), (3000, 500), nm))
        assert isinstance(res, DimBoundingBox)
        assert res == dbox((0, -1), (3, 1))
        assert res.lower_left.unit == box.lower_left.unit
        assert box.extended(Vector(5000, 5000) * nm) == dbox((0, 0), (5, 5))
        assert box.extended([Vector(5, 5) * um, Vector(-1000, 0) * nm]) == dbox((-1, 0), (5, 5))
        with pytest.raises(TypeError):
            box.extended((1, 1))
        with pytest.raises(TypeError):
            box.extended(BoundingBox((0, 0), (1, 1)))
