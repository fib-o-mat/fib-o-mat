import copy
import math

import numpy as np
import pytest

from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline
from fibomat.composite_shapes.ring import Ring
from fibomat.linalg import BoundingBox, Vector, VectorValueError, mirror, rotate, scale, translate
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi


class TestConstruction:
    def test_basic(self):
        ring = Ring(2, 0.5, (1, 1), 'foo')
        assert (ring.r_outer, ring.thickness, ring.r_inner) == (2., 0.5, 1.5)
        assert ring.center == (1., 1.)
        assert ring.description == 'foo'
        assert isinstance(ring, Shape) and ring.is_closed is True
        assert Ring(2, 0.5).center == (0., 0.)

    def test_from_radii(self):
        ring = Ring.from_radii(3, 2, center=(1, 2), description='foo')
        assert ring.r_outer == 3. and ring.r_inner == pytest.approx(2.) and ring.thickness == pytest.approx(1.)
        assert ring.center == (1., 2.) and ring.description == 'foo'
        assert Ring.from_radii(3, 0).r_inner == 0.

    @pytest.mark.parametrize('r_inner', [-1., 3., 4., np.nan, np.inf])
    def test_from_radii_invalid(self, r_inner):
        with pytest.raises(ValueError):
            Ring.from_radii(3, r_inner)

    @pytest.mark.parametrize('r_outer, thickness', [
        (0, 1), (-1, 1), (1, 0), (1, -1), (1, 2), (np.nan, 1), (1, np.nan), (np.inf, 1), (1, np.inf),
    ])
    def test_invalid(self, r_outer, thickness):
        with pytest.raises(ValueError):
            Ring(r_outer, thickness)

    def test_invalid_center(self):
        with pytest.raises(VectorValueError):
            Ring(2, 1, (1, 2, 3))
        with pytest.raises(ValueError):
            Ring(2, 1, (np.nan, 0))

    def test_disk(self):
        ring = Ring(2, 2)
        assert ring.r_inner == 0.
        assert ring.area == pytest.approx(4 * PI)

    def test_no_arc_spline_contract(self):
        # a ring cannot be an arc spline (the former `to_arc_spline` returned a HollowArcSpline)
        assert not isinstance(Ring(2, 1), ArcSplineCompatible)
        assert not hasattr(Ring(2, 1), 'to_arc_spline')

    def test_repr(self):
        assert repr(Ring(2, 0.5, (1, 1))) == 'Ring(r_outer=2.0, thickness=0.5, center=Vector(x=1.0, y=1.0))'


class TestProperties:
    def test_values(self):
        ring = Ring(2, 0.5, (1, 1))
        assert ring.area == pytest.approx(PI * (4 - 2.25))
        assert ring.boundary_length == pytest.approx(2 * PI * 3.5)
        assert ring.bounding_box == BoundingBox((-1, -1), (3, 3))

    def test_contains(self):
        ring = Ring(2, 1, (1, 1))
        assert ring.contains((2.5, 1.)) and ring.contains((1., 2.9)) and ring.contains((3., 1.)) and ring.contains((2., 1.))
        assert not ring.contains((1., 1.)) and not ring.contains((1.5, 1.)) and not ring.contains((4., 1.))
        assert Ring(2, 2).contains((0., 0.))

    def test_to_hollow_arc_spline(self):
        ring = Ring(2, 0.5, (1, 1), 'foo')
        hollow = ring.to_hollow_arc_spline()
        assert isinstance(hollow, HollowArcSpline)
        assert hollow.description == 'foo'
        assert len(hollow.holes) == 1
        assert hollow.area == pytest.approx(ring.area)
        assert hollow.boundary_length == pytest.approx(ring.boundary_length)
        assert hollow.bounding_box == ring.bounding_box
        assert hollow.contains((2.8, 1.)) and not hollow.contains((1., 1.))

    def test_disk_to_hollow_arc_spline(self):
        hollow = Ring(2, 2).to_hollow_arc_spline()
        assert hollow.holes == [] and hollow.area == pytest.approx(4 * PI)


class TestTransformations:
    def test_translate(self):
        res = Ring(2, 0.5, (1, 1)).translated((1, -1))
        assert res.center == (2., 0.) and (res.r_outer, res.thickness) == (2., 0.5)
        assert isinstance(res, Ring)

    def test_rotate(self):
        res = Ring(2, 0.5, (1, 0)).rotated(PI / 2)
        assert res.center == pytest.approx((0., 1.)) and res.r_outer == 2.
        assert Ring(2, 0.5, (1, 0)).rotated(1., 'center').center == (1., 0.)

    def test_scale(self):
        res = Ring(2, 0.5, (1, 2)).scaled(2.)
        assert (res.r_outer, res.thickness, res.center) == (4., 1., Vector(2, 4))
        # (negative factors produced negative radii)
        res = Ring(2, 0.5, (1, 2)).scaled(-2.)
        assert (res.r_outer, res.thickness, res.r_inner) == (4., 1., 3.)
        assert res.center == (-2., -4.)

    def test_mirror(self):
        res = Ring(2, 0.5, (1, 2)).mirrored((1, 0))
        assert res.center == pytest.approx((1., -2.))

    def test_transformed_chain(self):
        res = Ring(2, 0.5, (1, 0)).transformed(translate((1, 1)) | rotate(PI / 2) | scale(2.) | mirror((1, 0)))
        assert res.center == pytest.approx((-2., -4.)) and res.r_outer == 4.

    def test_does_not_modify_original(self):
        ring = Ring(2, 0.5, (1, 1))
        ring.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert (ring.r_outer, ring.thickness, ring.center) == (2., 0.5, Vector(1, 1))


class TestCopyAndDimShape:
    def test_clone(self):
        ring = Ring(2, 0.5, (1, 1), 'foo')
        clone = ring.clone()
        assert clone is not ring and clone.description == 'foo' and clone.center == (1., 1.)
        assert copy.deepcopy(ring).r_outer == 2.

    def test_dim_shape(self):
        dim_ring = Ring(2, 0.5) * unit('µm')
        assert dim_ring.bounding_box.width.m_as('µm') == pytest.approx(4.)
        assert dim_ring.to('nm').shape.r_outer == pytest.approx(2000.)
