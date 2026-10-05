import copy
import math

import numpy as np
import pytest

from fibomat.linalg import (
    BoundingBox, DimBoundingBox, DimTransformable, DimVector, Transformable, Vector, mirror, rotate, scale, translate
)
from fibomat.linalg.transformables.transformation_builder import (
    _TransformationBuilder, _TranslationBuilder, _RotationBuilder, _ScaleBuilder, _MirrorBuilder
)
from fibomat.units import unit


class Point(Transformable):
    """Transformable point which records the calls of the _impl_* methods."""

    def __init__(self, x=0., y=0., description=None):
        super().__init__(description)
        self.pos = Vector(x, y)
        self.calls = []

    @property
    def center(self):
        return self.pos

    @property
    def bounding_box(self):
        return BoundingBox(self.pos, self.pos)

    def _impl_translate(self, trans_vec):
        assert type(trans_vec) is Vector
        self.calls.append(('translate', trans_vec))
        self.pos = self.pos + trans_vec

    def _impl_rotate(self, theta):
        self.calls.append(('rotate', theta))
        self.pos = self.pos.rotated(theta)

    def _impl_scale(self, fac):
        self.calls.append(('scale', fac))
        self.pos = self.pos * fac

    def _impl_mirror(self, mirror_axis):
        assert type(mirror_axis) is Vector
        self.calls.append(('mirror', mirror_axis))
        self.pos = self.pos.mirrored(mirror_axis)


class DimPoint(DimTransformable):
    def __init__(self, x=0., y=0., description=None):
        super().__init__(description)
        self.pos = Vector(x, y) * unit('µm')

    @property
    def center(self):
        return self.pos

    @property
    def bounding_box(self):
        return DimBoundingBox(self.pos, self.pos)

    def _impl_translate(self, trans_vec):
        assert type(trans_vec) is DimVector
        self.pos = self.pos + trans_vec

    def _impl_rotate(self, theta):
        self.pos = self.pos.rotated(theta)

    def _impl_scale(self, fac):
        self.pos = self.pos * fac

    def _impl_mirror(self, mirror_axis):
        assert type(mirror_axis) is DimVector
        self.pos = self.pos.mirrored(mirror_axis)


class TestAbstract:
    def test_cannot_instantiate_incomplete(self):
        with pytest.raises(TypeError):
            Transformable()
        with pytest.raises(TypeError):
            DimTransformable()

        class Incomplete(Transformable):
            center = Vector()

        with pytest.raises(TypeError):
            Incomplete()

    def test_hierarchy(self):
        assert issubclass(DimTransformable, Transformable)
        assert Point._VectorClass is Vector
        assert DimPoint._VectorClass is DimVector


class TestDescription:
    def test_description(self):
        assert Point().description is None
        assert Point(description='foo').description == 'foo'
        assert Point(description='foo').with_changed_description('bar').description == 'bar'


class TestPivot:
    def test_default_is_center(self):
        assert Point(1., 2.).pivot == (1., 2.)

    def test_set_pivot(self):
        p = Point(1., 2.)
        p.pivot = lambda obj: (obj.center.x + 1, 0)
        assert p.pivot == (2., 0.)
        assert type(p.pivot) is Vector

    def test_reset_pivot(self):
        p = Point(1., 2.)
        p.pivot = lambda obj: (5, 5)
        p.pivot = None
        assert p.pivot == (1., 2.)

    def test_invalid_pivot(self):
        with pytest.raises(TypeError):
            Point().pivot = (1, 2)

    def test_pivot_is_kept_in_clone(self):
        p = Point(1., 2.)
        p.pivot = lambda obj: obj.center + (1, 0)
        assert p.translated((1, 1)).pivot == (3., 3.)


class TestTranslate:
    def test_translated(self):
        p = Point(1., 2.)
        res = p.translated((1, 1))
        assert res.pos == (2., 3.)
        assert p.pos == (1., 2.)  # original not modified
        assert res is not p

    def test_translated_converts_argument(self):
        res = Point().translated([1, 2])
        assert res.calls == [('translate', Vector(1, 2))]
        assert Point().translated(np.array([1., 2.])).pos == (1., 2.)

    def test_invalid_vector(self):
        with pytest.raises(ValueError):
            Point().translated((1, 2, 3))
        with pytest.raises(ValueError):
            Point().translated(1)

    def test_translated_to(self):
        p = Point(1., 2.)
        assert p.translated_to((5, 5)).pos == (5., 5.)

    def test_translated_to_pivot(self):
        p = Point(1., 2.)
        p.pivot = lambda obj: obj.center + (1, 1)
        res = p.translated_to((5, 5))
        assert res.pos == (4., 4.)
        assert res.pivot == (5., 5.)


class TestRotateScale:
    @pytest.mark.parametrize('method, name, arg', [('rotated', 'rotate', 2.), ('scaled', 'scale', 2.)])
    def test_without_origin(self, method, name, arg):
        p = Point(1., 2.)
        res = getattr(p, method)(arg)
        assert res.calls == [(name, arg)]
        assert p.calls == []

    @pytest.mark.parametrize('method, name', [('rotated', 'rotate'), ('scaled', 'scale')])
    def test_with_origin_vector(self, method, name):
        res = getattr(Point(1., 2.), method)(2., (3, 4))
        assert res.calls == [('translate', Vector(-3, -4)), (name, 2.), ('translate', Vector(3, 4))]

    @pytest.mark.parametrize('method, name', [('rotated', 'rotate'), ('scaled', 'scale')])
    def test_with_center_and_pivot(self, method, name):
        p = Point(7., 8.)
        p.pivot = lambda obj: (1, 1)
        res = getattr(p, method)(2., 'center')
        assert res.calls[0] == ('translate', Vector(-7, -8))
        assert res.calls[2] == ('translate', Vector(7, 8))
        res = getattr(p, method)(2., 'pivot')
        assert res.calls[0] == ('translate', Vector(-1, -1))
        assert res.calls[2] == ('translate', Vector(1, 1))

    @pytest.mark.parametrize('method', ['rotated', 'scaled'])
    def test_unknown_origin(self, method):
        with pytest.raises(ValueError):
            getattr(Point(), method)(2., 'foo')
        with pytest.raises(ValueError):
            getattr(Point(), method)(2., '')

    def test_origin_zero_vector_is_origin(self):
        res = Point(1., 0.).rotated(math.pi / 2, (0, 0))
        assert res.pos == (0., 1.)
        assert len(res.calls) == 3

    def test_rotated_values(self):
        assert Point(1., 0.).rotated(math.pi / 2).pos == (0., 1.)
        assert Point(2., 1.).rotated(math.pi, 'center').pos == (2., 1.)
        assert Point(2., 0.).rotated(math.pi / 2, (1, 0)).pos == (1., 1.)

    def test_scaled_values(self):
        assert Point(1., 2.).scaled(2.).pos == (2., 4.)
        assert Point(1., 2.).scaled(2., (1, 2)).pos == (1., 2.)
        assert Point(3., 2.).scaled(2., (1, 2)).pos == (5., 2.)
        assert Point(1., 2.).scaled(-1.).pos == (-1., -2.)

    def test_always_returns_a_clone(self):
        p = Point(1., 2.)
        for res in (p.rotated(0.), p.scaled(1.), p.rotated(0., 'center'), p.scaled(1., 'center')):
            assert res is not p
            assert res.pos == p.pos

    def test_small_rotation_and_scale_are_applied(self):
        # the former implementation skipped rotations by angles < 1e-8 and scale factors close to 1
        assert Point().rotated(1e-9).calls == [('rotate', 1e-9)]
        assert Point().scaled(1. + 1e-7).calls == [('scale', 1. + 1e-7)]

    def test_numpy_numbers(self):
        assert Point(1., 0.).rotated(np.float64(math.pi / 2)).pos == (0., 1.)
        assert Point(1., 1.).scaled(np.int64(2)).pos == (2., 2.)

    def test_invalid_numbers(self):
        for bad in (np.nan, np.inf):
            with pytest.raises(ValueError):
                Point().rotated(bad)
            with pytest.raises(ValueError):
                Point().scaled(bad)
        with pytest.raises(TypeError):
            Point().rotated('a')
        with pytest.raises(ValueError):
            Point().scaled(0.)


class TestMirror:
    def test_mirrored(self):
        p = Point(1., 2.)
        res = p.mirrored((1, 0))
        assert res.pos == (1., -2.)
        assert res.calls == [('mirror', Vector(1, 0))]
        assert p.pos == (1., 2.)

    def test_null_axis(self):
        with pytest.raises(ValueError):
            Point().mirrored((0, 0))
        with pytest.raises(ValueError):
            Point().transformed(mirror((0, 0)))

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            Point().mirrored((1, 2, 3))


class TestTransformed:
    def test_chain(self):
        p = Point(1., 0.)
        res = p.transformed(translate((1, 1)) | rotate(math.pi / 2) | scale(2.) | mirror((1, 0)))
        assert [c[0] for c in res.calls] == ['translate', 'rotate', 'scale', 'mirror']
        # (1, 0) + (1, 1) = (2, 1) -> rotated (-1, 2) -> scaled (-2, 4) -> mirrored (-2, -4)
        assert res.pos == (-2., -4.)
        assert p.pos == (1., 0.)

    def test_single_trafo(self):
        assert Point(1., 0.).transformed(translate((1, 1))).pos == (2., 1.)

    def test_origins(self):
        res = Point(2., 0.).transformed(rotate(math.pi / 2, (1, 0)) | scale(2., 'center'))
        assert res.pos == (1., 1.)

    def test_converts_arguments(self):
        res = Point().transformed(translate([1, 2]) | mirror([1, 0]))
        assert all(type(c[1]) is Vector for c in res.calls)

    def test_invalid(self):
        with pytest.raises(TypeError):
            Point().transformed(1)
        with pytest.raises(TypeError):
            Point().transformed(_TransformationBuilder())

    def test_builder_types(self):
        assert isinstance(translate((1, 1)), _TranslationBuilder)
        assert isinstance(rotate(1.), _RotationBuilder)
        assert isinstance(scale(2.), _ScaleBuilder)
        assert isinstance(mirror((1, 0)), _MirrorBuilder)

    def test_builder_validation(self):
        with pytest.raises(ValueError):
            rotate(np.nan)
        with pytest.raises(TypeError):
            rotate('a')
        with pytest.raises(ValueError):
            scale(0.)
        with pytest.raises(ValueError):
            scale(np.inf)

    def test_or_does_not_modify_operands(self):
        t_1, t_2, t_3 = translate((1, 0)), rotate(1.), scale(2.)
        chain = t_1 | t_2
        chain_2 = chain | t_3
        assert t_1.transformations == [t_1]
        assert t_2.transformations == [t_2]
        assert chain.transformations == [t_1, t_2]
        assert chain_2.transformations == [t_1, t_2, t_3]
        assert (t_3 | chain).transformations == [t_3, t_1, t_2]

    def test_or_invalid(self):
        with pytest.raises(TypeError):
            translate((1, 1)) | 1

    def test_transformations_is_a_copy(self):
        t_1 = translate((1, 0))
        t_1.transformations.append(rotate(1.))
        assert len(t_1.transformations) == 1

    def test_reusable_builders(self):
        base = translate((1, 0))
        first = Point().transformed(base | rotate(math.pi / 2))
        second = Point().transformed(base)
        assert first.pos == (0., 1.)
        assert second.pos == (1., 0.)


class TestDimTransformable:
    def test_translated(self):
        res = DimPoint(1., 2.).translated(Vector(1., 1.) * unit('µm'))
        assert res.pos == Vector(2., 3.) * unit('µm')
        assert res.pos.unit == DimPoint().pos.unit

    def test_translated_other_unit(self):
        res = DimPoint(1., 2.).translated(Vector(1000., 0.) * unit('nm'))
        assert res.pos == Vector(2., 2.) * unit('µm')

    def test_translated_requires_units(self):
        with pytest.raises(ValueError):
            DimPoint().translated((1., 1.))

    def test_translated_to(self):
        res = DimPoint(1., 2.).translated_to(Vector(5., 5.) * unit('µm'))
        assert res.pos == Vector(5., 5.) * unit('µm')

    def test_rotated_with_origin(self):
        res = DimPoint(2., 0.).rotated(math.pi / 2, Vector(1., 0.) * unit('µm'))
        assert res.pos == Vector(1., 1.) * unit('µm')
        res = DimPoint(2., 0.).rotated(math.pi / 2, Vector(1000., 0.) * unit('nm'))
        assert res.pos == Vector(1., 1.) * unit('µm')
        assert DimPoint(2., 0.).rotated(math.pi, 'center').pos == Vector(2., 0.) * unit('µm')

    def test_origin_without_unit(self):
        with pytest.raises(ValueError):
            DimPoint().rotated(1., (1., 1.))

    def test_scaled_mirrored(self):
        assert DimPoint(1., 2.).scaled(2.).pos == Vector(2., 4.) * unit('µm')
        assert DimPoint(1., 2.).mirrored(Vector(1., 0.) * unit('m')).pos == Vector(1., -2.) * unit('µm')
        with pytest.raises(ValueError):
            DimPoint().mirrored(DimVector())

    def test_transformed(self):
        res = DimPoint(1., 0.).transformed(
            translate(Vector(1., 1.) * unit('µm')) | rotate(math.pi / 2) | mirror(Vector(1., 0.) * unit('µm'))
        )
        assert res.pos == Vector(-1., -2.) * unit('µm')

    def test_clone(self):
        p = DimPoint(1., 2.)
        c = p.clone()
        assert c is not p and c.pos == p.pos
        assert copy.deepcopy(p).pos == p.pos
