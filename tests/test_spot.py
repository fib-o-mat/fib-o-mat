import copy
import math

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, DimTransformable, Transformable, Vector, VectorValueError, mirror, rotate, scale, translate
from fibomat.shapes.shape import Shape
from fibomat.shapes.spot import Spot
from fibomat.units import unit


class TestConstruction:
    def test_default_position(self):
        assert Spot().position == (0., 0.)

    @pytest.mark.parametrize('position', [(1, 2), [1, 2], np.array([1., 2.]), Vector(1, 2)])
    def test_position(self, position):
        spot = Spot(position)
        assert spot.position == (1., 2.)
        assert type(spot.position) is Vector

    def test_keyword_arguments(self):
        spot = Spot(position=(1, 2), description='foo')
        assert spot.position == (1., 2.)
        assert spot.description == 'foo'
        assert Spot().description is None

    def test_description_is_keyword_only(self):
        with pytest.raises(TypeError):
            Spot((1, 2), 'foo')

    def test_invalid_position(self):
        with pytest.raises(VectorValueError):
            Spot((1, 2, 3))
        with pytest.raises(VectorValueError):
            Spot('ab')
        with pytest.raises(VectorValueError):
            Spot(1)
        with pytest.raises(VectorValueError):
            Spot(Vector(1, 2) * unit('µm'))

    def test_non_finite_position(self):
        for bad in ((np.nan, 0), (0, np.inf), (-np.inf, 0)):
            with pytest.raises(ValueError):
                Spot(bad)

    def test_hierarchy(self):
        assert issubclass(Spot, Shape)
        assert isinstance(Spot(), Transformable)
        assert not isinstance(Spot(), DimTransformable)

    def test_repr(self):
        assert repr(Spot((1, 2))) == 'Spot(position=Vector(x=1.0, y=2.0))'


class TestProperties:
    def test_position_center_and_bounding_box(self):
        spot = Spot((1, 2))
        assert spot.center == (1., 2.)
        assert spot.center == spot.position
        bbox = spot.bounding_box
        assert bbox == BoundingBox((1, 2), (1, 2))
        assert bbox.width == 0. and bbox.height == 0.

    def test_is_closed_is_not_defined(self):
        with pytest.raises(NotImplementedError, match='0-dim'):
            Spot().is_closed

    def test_area_and_boundary_length_are_not_defined(self):
        with pytest.raises(NotImplementedError):
            Spot().area
        with pytest.raises(NotImplementedError):
            Spot().boundary_length


class TestTransformations:
    def test_translated(self):
        spot = Spot((1, 2))
        res = spot.translated((1, -1))
        assert isinstance(res, Spot)
        assert res.position == (2., 1.)
        assert spot.position == (1., 2.)

    def test_translated_to(self):
        assert Spot((1, 2)).translated_to((5, 5)).position == (5., 5.)

    def test_rotated(self):
        assert Spot((1, 0)).rotated(math.pi / 2).position == (0., 1.)
        assert Spot((2, 0)).rotated(math.pi / 2, (1, 0)).position == (1., 1.)
        assert Spot((2, 1)).rotated(math.pi, 'center').position == (2., 1.)

    def test_scaled(self):
        assert Spot((1, 2)).scaled(2.).position == (2., 4.)
        assert Spot((3, 2)).scaled(2., (1, 2)).position == (5., 2.)
        assert Spot((3, 2)).scaled(2., 'center').position == (3., 2.)

    def test_mirrored(self):
        assert Spot((1, 2)).mirrored((1, 0)).position == (1., -2.)
        assert Spot((1, 2)).mirrored((1, 1)).position == (2., 1.)

    def test_transformed(self):
        res = Spot((1, 0)).transformed(translate((1, 1)) | rotate(math.pi / 2) | scale(2.) | mirror((1, 0)))
        assert res.position == pytest.approx((-2., -4.))

    def test_transformations_do_not_modify_original(self):
        spot = Spot((1, 2))
        spot.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert spot.position == (1., 2.)

    def test_position_is_not_changed_in_place(self):
        # the position vector handed out earlier must not change when the spot is transformed
        spot = Spot((1, 2))
        position = spot.position
        spot._impl_translate(Vector(1, 1))
        assert position == (1., 2.)

    def test_pivot(self):
        spot = Spot((1, 2))
        assert spot.pivot == (1., 2.)
        spot.pivot = lambda obj: obj.center + (1, 0)
        assert spot.translated_to((0, 0)).position == (-1., 0.)


class TestCopyAndDimShape:
    def test_clone(self):
        spot = Spot((1, 2), description='foo')
        clone = spot.clone()
        assert clone is not spot
        assert clone.position == spot.position
        assert clone.description == 'foo'
        assert copy.deepcopy(spot).position == spot.position

    def test_dim_shape(self):
        from fibomat.shapes.dim_shape import DimShape
        from fibomat.units import ureg

        dim_spot = Spot((1, 2)) * unit('µm')
        assert isinstance(dim_spot, DimShape)
        assert dim_spot.unit == ureg.Unit('µm')
        assert dim_spot.center == Vector(1, 2) * unit('µm')
        assert dim_spot.to('nm').shape.position == pytest.approx((1000., 2000.))
        assert (unit('nm') * Spot()).unit == ureg.Unit('nm')
