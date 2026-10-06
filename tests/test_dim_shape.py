import copy
import math
import pickle

import pytest

from fibomat.linalg import DimBoundingBox, DimTransformable, DimVector, Vector
from fibomat.units import DimensionError, ureg, unit

from helpers_shapes import Box, DimShape, Dot, Shape

um = unit('µm')
nm = unit('nm')


def dv(x, y, u=um):
    return Vector(x, y) * u


class TestConstruction:
    def test_basic(self):
        shape = Dot((1, 2))
        dim_shape = DimShape(shape, ureg.Unit('µm'), description='foo')
        assert dim_shape.shape is shape
        assert dim_shape.unit == ureg.Unit('µm')
        assert dim_shape.description == 'foo'
        assert isinstance(dim_shape, DimTransformable)

    @pytest.mark.parametrize('unit_, expected', [
        ('µm', 'µm'), (ureg.Unit('µm'), 'µm'), (um, 'µm'), (unit('um'), 'µm'), ('nm', 'nm'), (unit('mm'), 'mm'),
        ('m', 'm'),
    ])
    def test_unit_types(self, unit_, expected):
        assert DimShape(Dot(), unit_).unit == ureg.Unit(expected)

    def test_invalid_unit(self):
        with pytest.raises(TypeError):
            DimShape(Dot(), 5)
        with pytest.raises(TypeError):
            DimShape(Dot(), None)
        with pytest.raises(TypeError):
            DimShape(Dot(), 1. * um)
        with pytest.raises(DimensionError):
            DimShape(Dot(), 's')
        with pytest.raises(DimensionError):
            DimShape(Dot(), unit('s'))
        with pytest.raises(DimensionError):
            DimShape(Dot(), ureg.Unit('kg'))

    def test_invalid_shape(self):
        with pytest.raises(TypeError):
            DimShape((1, 2), 'µm')
        with pytest.raises(TypeError):
            DimShape(None, 'µm')
        with pytest.raises(TypeError):
            DimShape(Dot() * um, 'µm')

    def test_created_by_multiplication(self):
        shape = Dot()
        assert isinstance(shape * um, DimShape)
        assert (shape * nm).unit == ureg.Unit('nm')
        assert (nm * shape).shape is shape

    def test_repr(self):
        assert repr(DimShape(Dot((1, 2)), 'µm')) == "DimShape(shape=Dot(Vector(x=1.0, y=2.0)), unit=<Unit('micrometer')>)"


class TestProperties:
    def test_center(self):
        center = DimShape(Dot((1, 2)), 'nm').center
        assert type(center) is DimVector
        assert center == dv(1, 2, nm)
        assert center.unit == ureg.Unit('nm')

    def test_bounding_box(self):
        bbox = DimShape(Box((1, 2), (3, 5)), 'nm').bounding_box
        assert type(bbox) is DimBoundingBox
        assert bbox == DimBoundingBox(dv(1, 2, nm), dv(3, 5, nm))
        assert bbox.width.m_as('nm') == pytest.approx(2.)
        assert bbox.height.m_as('µm') == pytest.approx(0.003)

    def test_pivot_defaults_to_shape_pivot(self):
        shape = Dot((1, 2))
        dim_shape = DimShape(shape, 'µm')
        assert dim_shape.pivot == dv(1, 2)

        shape.pivot = lambda obj: (5, 5)
        assert dim_shape.pivot == dv(5, 5)
        assert type(dim_shape.pivot) is DimVector

    def test_own_pivot(self):
        dim_shape = DimShape(Dot((1, 2)), 'µm')
        dim_shape.pivot = lambda obj: obj.center + dv(1, 0)
        assert dim_shape.pivot == dv(2, 2)
        dim_shape.pivot = None
        assert dim_shape.pivot == dv(1, 2)
        with pytest.raises(TypeError):
            dim_shape.pivot = (1, 2)

    def test_translated_to_uses_pivot(self):
        shape = Dot((1, 2))
        shape.pivot = lambda obj: obj.center + Vector(1, 1)
        res = DimShape(shape, 'µm').translated_to(dv(5, 5))
        assert res.shape.position == (4., 4.)

    def test_area_of_shape_is_unitless(self):
        with pytest.raises(NotImplementedError):
            DimShape(Dot(), 'µm').shape.area


class TestTransformations:
    def test_translate_same_unit(self):
        dim_shape = DimShape(Dot((1, 2)), 'µm')
        res = dim_shape.translated(dv(1, 1))
        assert res.shape.position == (2., 3.)
        assert res.center == dv(2, 3)

    def test_translate_other_unit(self):
        res = DimShape(Dot((1, 2)), 'µm').translated(dv(1000, 500, nm))
        assert res.shape.position == pytest.approx((2., 2.5))
        res = DimShape(Dot((1, 2)), 'nm').translated(dv(1, 1, um))
        assert res.shape.position == pytest.approx((1001., 1002.))

    def test_translate_requires_units(self):
        with pytest.raises(ValueError):
            DimShape(Dot(), 'µm').translated((1, 1))

    def test_translated_does_not_modify(self):
        shape = Dot((1, 2))
        dim_shape = DimShape(shape, 'µm')
        res = dim_shape.translated(dv(1, 1))
        assert shape.position == (1., 2.)
        assert dim_shape.center == dv(1, 2)
        assert res.shape is not shape
        assert res.unit == dim_shape.unit

    def test_translated_to(self):
        res = DimShape(Dot((1, 2)), 'nm').translated_to(dv(1, 1, um))
        assert res.center == dv(1, 1, um)
        assert res.shape.position == pytest.approx((1000., 1000.))

    def test_rotate(self):
        res = DimShape(Dot((1, 0)), 'µm').rotated(math.pi / 2)
        assert res.shape.position == pytest.approx((0., 1.))
        res = DimShape(Dot((2, 0)), 'µm').rotated(math.pi / 2, dv(1000, 0, nm))
        assert res.shape.position == pytest.approx((1., 1.))
        res = DimShape(Dot((2, 0)), 'µm').rotated(math.pi, 'center')
        assert res.shape.position == pytest.approx((2., 0.))

    def test_scale(self):
        res = DimShape(Dot((1, 2)), 'µm').scaled(2.)
        assert res.shape.position == (2., 4.)
        assert res.unit == ureg.Unit('µm')
        res = DimShape(Dot((3, 2)), 'µm').scaled(2., dv(1, 2))
        assert res.shape.position == pytest.approx((5., 2.))

    def test_mirror(self):
        res = DimShape(Dot((1, 2)), 'µm').mirrored(dv(1, 0, nm))
        assert res.shape.position == pytest.approx((1., -2.))
        with pytest.raises(ValueError):
            DimShape(Dot(), 'µm').mirrored((1, 0))
        with pytest.raises(ValueError):
            DimShape(Dot(), 'µm').mirrored(DimVector())

    def test_transformed(self):
        from fibomat.linalg import translate, rotate, scale, mirror

        res = DimShape(Dot((1, 0)), 'µm').transformed(
            translate(dv(1, 1)) | rotate(math.pi / 2) | scale(2.) | mirror(dv(1, 0))
        )
        assert res.shape.position == pytest.approx((-2., -4.))


class TestTo:
    def test_to_scales_shape(self):
        shape = Box((1, 2), (3, 5))
        dim_shape = DimShape(shape, 'µm')
        res = dim_shape.to('nm')
        assert isinstance(res, DimShape)
        assert res.unit == ureg.Unit('nm')
        assert res.shape.ll == pytest.approx((1000., 2000.))
        assert res.shape.ur == pytest.approx((3000., 5000.))
        # physical size is unchanged
        assert res.bounding_box == dim_shape.bounding_box
        assert res.center == dv(2, 3.5)

    def test_original_is_not_modified(self):
        shape = Box((1, 2), (3, 5))
        dim_shape = DimShape(shape, 'µm')
        res = dim_shape.to('nm')
        assert res is not dim_shape
        assert res.shape is not shape
        assert dim_shape.unit == ureg.Unit('µm')
        assert dim_shape.shape is shape
        assert shape.ll == (1., 2.)
        assert shape.ur == (3., 5.)

    def test_larger_unit(self):
        res = DimShape(Dot((1000, 2000)), 'nm').to(um)
        assert res.shape.position == pytest.approx((1., 2.))

    def test_same_unit_returns_a_copy(self):
        dim_shape = DimShape(Dot((1, 2)), 'µm')
        res = dim_shape.to(ureg.Unit('µm'))
        assert res is not dim_shape
        assert res.shape is not dim_shape.shape
        assert res.shape.position == (1., 2.)

    def test_unit_types(self):
        dim_shape = DimShape(Dot((1, 2)), 'µm')
        assert dim_shape.to(nm).unit == ureg.Unit('nm')
        res = dim_shape.to('mm')
        assert res.unit == ureg.Unit('mm')
        assert res.shape.position == pytest.approx((1e-3, 2e-3))
        assert dim_shape.to(ureg.Unit('nm')).unit == ureg.Unit('nm')

    def test_keeps_description_and_pivot(self):
        dim_shape = DimShape(Dot((1, 2)), 'µm', description='foo')
        dim_shape.pivot = lambda obj: obj.center + dv(1, 0)
        res = dim_shape.to('nm')
        assert res.description == 'foo'
        assert res.pivot == dv(2, 2)

    def test_invalid_unit(self):
        dim_shape = DimShape(Dot((1, 2)), 'µm')
        with pytest.raises(DimensionError):
            dim_shape.to('s')
        with pytest.raises(TypeError):
            dim_shape.to(5)
        assert dim_shape.shape.position == (1., 2.)

    def test_no_set_unit(self):
        assert not hasattr(DimShape, 'set_unit')

    def test_roundtrip(self):
        dim_shape = DimShape(Box((1, 2), (3, 5)), 'µm')
        res = dim_shape.to('nm').to('µm')
        assert res.shape.ll == pytest.approx((1., 2.))
        assert res.shape.ur == pytest.approx((3., 5.))


class TestCopy:
    def test_clone(self):
        dim_shape = DimShape(Dot((1, 2)), 'µm', description='foo')
        clone = dim_shape.clone()
        assert clone is not dim_shape and clone.shape is not dim_shape.shape
        assert clone.unit == dim_shape.unit
        assert clone.unit is not None and clone.description == 'foo'
        assert clone.center == dim_shape.center
        assert copy.deepcopy(dim_shape).center == dim_shape.center

    def test_with_changed_description(self):
        assert DimShape(Dot(), 'µm').with_changed_description('bar').description == 'bar'

    def test_pickle(self):
        dim_shape = DimShape(Dot((1, 2)), 'nm', description='foo')
        restored = pickle.loads(pickle.dumps(dim_shape))
        assert restored.unit == ureg.Unit('nm')
        assert restored.unit._REGISTRY is ureg
        assert restored.center == dim_shape.center
        assert restored.description == 'foo'
        # the restored object is fully functional
        assert restored.translated(dv(1, 1, nm)).shape.position == (2., 3.)
