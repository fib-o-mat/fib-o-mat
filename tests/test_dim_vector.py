import math
import pickle
import warnings

import numpy as np
import pint
import pytest

from fibomat.linalg.vectors import (
    Vector, DimVector, VectorValueError, angle_between, signed_angle_between
)
from fibomat.units import unit, DimFloat, DimensionError, ureg


um = unit('µm')
nm = unit('nm')
m = unit('m')


def dv(x, y, u=um):
    return DimVector(x * u, y * u)


class TestDimVectorConstruction:
    def test_components(self):
        v = DimVector(1. * um, 2. * um)
        assert isinstance(v, Vector)
        assert v.x.m_as('µm') == pytest.approx(1.)
        assert v.y.m_as('µm') == pytest.approx(2.)

    def test_keywords(self):
        v = DimVector(x=1. * um, y=2. * um)
        assert v.y.m_as('µm') == pytest.approx(2.)

    def test_iterable(self):
        assert DimVector([1. * um, 2. * um]) == dv(1, 2)
        assert DimVector((1. * um, 2. * um)) == dv(1, 2)
        assert DimVector(dv(1, 2)) == dv(1, 2)

    def test_mixed_units(self):
        v = DimVector(1. * um, 500. * nm)
        assert v.unit == ureg.Unit('µm')
        assert v.y.m_as('µm') == pytest.approx(0.5)

    def test_mixed_units_do_not_modify_arguments(self):
        x, y = 1. * um, 500. * nm
        DimVector(x, y)
        assert y.units == ureg.Unit('nm')
        assert y.magnitude == 500.

    def test_polar(self):
        v = DimVector(r=2. * um, phi=math.pi / 2)
        assert v.x.m_as('µm') == pytest.approx(0., abs=1e-12)
        assert v.y.m_as('µm') == pytest.approx(2.)

    def test_zero_vector(self):
        v = DimVector()
        assert v.x.m_as('m') == 0. and v.y.m_as('m') == 0.

    def test_integer_magnitudes(self):
        v = DimVector(1 * um, 2 * um)
        assert v.x.m_as('µm') == 1.

    @pytest.mark.parametrize('args, kwargs', [
        ((1., 2.), {}),
        (([1., 2.],), {}),
        ((1. * um,), {}),
        ((1. * um, 2.), {}),
        ((1. * um, 2. * unit('s')), {}),
        ((1. * unit('s'), 2. * unit('s')), {}),
        ((1. * unit('kg'), 2. * unit('kg')), {}),
        (([1. * um, 2. * um, 3. * um],), {}),
        (([1. * um],), {}),
        ((Vector(1., 2.),), {}),
        ((), {'x': 1. * um}),
        ((), {'r': 1. * um}),
        ((), {'r': 1., 'phi': 0.}),
        ((), {'r': 1. * um, 'phi': 'a'}),
        ((1. * um, 2. * um), {'r': 1. * um, 'phi': 0.}),
    ])
    def test_invalid(self, args, kwargs):
        with pytest.raises(VectorValueError):
            DimVector(*args, **kwargs)

    def test_from_vector(self):
        v = DimVector.from_vector(Vector(1., 2.), 'nm')
        assert v.y.m_as('nm') == pytest.approx(2.)
        assert DimVector.from_vector(Vector(1., 2.), um).unit == ureg.Unit('µm')
        assert DimVector.from_vector(Vector(1., 2.), ureg.Unit('mm')).unit == ureg.Unit('mm')

    def test_from_vector_invalid(self):
        with pytest.raises(DimensionError):
            DimVector.from_vector(Vector(1., 2.), 's')
        with pytest.raises(DimensionError):
            DimVector.from_vector(Vector(1., 2.), unit('s'))
        with pytest.raises(VectorValueError):
            DimVector.from_vector(dv(1, 2), 'nm')

    def test_mul_unit(self):
        v = Vector(1., 2.) * nm
        assert isinstance(v, DimVector)
        assert v == DimVector(1. * nm, 2. * nm)
        with pytest.raises(DimensionError):
            Vector(1., 2.) * unit('s')
        with pytest.raises(TypeError):
            dv(1, 2) * nm

    def test_pint_quantity_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match='pint.Quantity'):
            v = DimVector(1. * ureg.micrometer, 2. * ureg.micrometer)
        assert v == dv(1, 2)

        with pytest.warns(DeprecationWarning):
            v = DimVector([1. * ureg.nm, 2. * ureg.nm])
        assert v == dv(0.001, 0.002)

        with pytest.warns(DeprecationWarning):
            DimVector(r=1. * ureg.nm, phi=0.)

    def test_pint_quantity_wrong_dimension(self):
        with pytest.raises(VectorValueError):
            DimVector(1. * ureg.second, 2. * ureg.second)

    def test_dim_float_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            DimVector(1. * um, 2. * um) + dv(1, 1)


class TestDimVectorAccess:
    def test_components_are_dim_floats(self):
        v = dv(1, 2)
        assert isinstance(v.x, DimFloat)
        assert isinstance(v.y, DimFloat)
        assert isinstance(v.mag, DimFloat)
        assert isinstance(v.r, DimFloat)
        assert [type(c) for c in v] == [DimFloat, DimFloat]
        assert v[0] == 1. * um
        assert v[-1] == 2. * um
        assert v[:] == [1. * um, 2. * um]
        assert len(v) == 2

    def test_magnitude(self):
        v = dv(3, 4)
        assert v.mag.m_as('µm') == pytest.approx(5.)
        assert v.magnitude.m_as('nm') == pytest.approx(5000.)
        assert v.phi == pytest.approx(math.atan2(4, 3))

    def test_vector_and_unit(self):
        v = DimVector(1. * um, 2. * um)
        assert v.vector == Vector(1., 2.)
        assert v.unit == ureg.Unit('µm')
        assert isinstance(v.vector, Vector) and not isinstance(v.vector, DimVector)

    def test_vector_as(self):
        v = dv(1, 2)
        assert v.vector_as('nm') == Vector(1000., 2000.)
        assert v.vector_as(nm) == Vector(1000., 2000.)
        assert v.vector_as(ureg.Unit('mm')) == Vector(1e-3, 2e-3)
        assert not isinstance(v.vector_as('nm'), DimVector)
        with pytest.raises(DimensionError):
            v.vector_as('s')

    def test_to(self):
        v = dv(1, 2).to('nm')
        assert v.unit == ureg.Unit('nm')
        assert v.vector == Vector(1000., 2000.)
        assert v == dv(1, 2)

    def test_array_drops_unit(self):
        arr = np.asarray(dv(1, 2, nm))
        assert arr.shape == (2,)
        assert list(arr) == [1., 2.]

    def test_repr(self):
        assert repr(dv(1, 2)) == 'DimVector(x=1.0 µm, y=2.0 µm)'

    def test_angle_about_x_axis(self):
        assert dv(0, 1).angle_about_x_axis == pytest.approx(math.pi / 2)
        with pytest.raises(ValueError):
            DimVector().angle_about_x_axis

    def test_nanometer_scale_in_meters_is_not_null(self):
        v = DimVector(5e-9 * m, 0. * m)
        assert v.angle_about_x_axis == 0.
        assert v != DimVector(0. * m, 0. * m)

    def test_unhashable(self):
        with pytest.raises(TypeError):
            hash(dv(1, 2))

    def test_pickle(self):
        v = dv(1, 2)
        restored = pickle.loads(pickle.dumps(v))
        assert restored == v
        assert restored.unit == v.unit


class TestDimVectorComparison:
    def test_eq(self):
        assert dv(1, 2) == dv(1, 2)
        assert dv(1, 2) == DimVector(1000. * nm, 2000. * nm)
        assert dv(1, 2) == (1. * um, 2. * um)
        assert dv(1, 2) != dv(1, 3)

    def test_eq_with_other_types(self):
        assert dv(1, 2) != Vector(1., 2.)
        assert dv(1, 2) != (1., 2.)
        assert dv(1, 2) != 1.
        assert dv(1, 2) != (1. * unit('s'), 2. * unit('s'))
        assert Vector(1., 2.) != dv(1, 2)

    def test_close_to_tolerance_is_physical(self):
        # near zero, picometer differences are equal, nanometer differences are not (independent of the internal unit)
        assert DimVector(1e-13 * m, 0. * m) == DimVector(0. * m, 0. * m)
        assert DimVector(0.1 * nm, 0. * nm) != DimVector(0. * um, 0. * um)
        assert DimVector(1. * um, 0. * um) != DimVector(1. * um + 1. * nm, 0. * um)
        assert DimVector(0. * nm, 0. * nm) != DimVector(1. * nm, 0. * nm)
        assert DimVector(0. * m, 0. * m) != DimVector(1. * nm, 0. * m)

    def test_close_to_invalid(self):
        with pytest.raises(VectorValueError):
            dv(1, 2).close_to((1., 2.))


class TestDimVectorGeometry:
    def test_normalized(self):
        v = dv(3, 4).normalized()
        assert isinstance(v, DimVector)
        assert v.mag.m_as('µm') == pytest.approx(1.)
        assert v.unit == ureg.Unit('µm')
        assert v.x.m_as('µm') == pytest.approx(0.6)

    def test_normalized_null_vector(self):
        with pytest.raises(ValueError):
            DimVector().normalized()

    def test_normalized_to(self):
        v = dv(3, 4).normalized_to(10. * nm)
        assert v.mag.m_as('nm') == pytest.approx(10.)
        assert v.x.m_as('nm') == pytest.approx(6.)
        assert v.unit == ureg.Unit('µm')

    def test_normalized_to_invalid(self):
        with pytest.raises(VectorValueError):
            dv(3, 4).normalized_to(10.)
        with pytest.raises(VectorValueError):
            dv(3, 4).normalized_to(10. * unit('s'))

    def test_rotated(self):
        v = dv(1, 0).rotated(math.pi / 2)
        assert isinstance(v, DimVector)
        assert v == dv(0, 1)
        assert dv(1, 0).rotated(math.pi / 2, origin=DimVector(1000. * nm, 1000. * nm)) == dv(2, 1)
        assert dv(1, 0).rotated(math.pi / 2, origin=dv(1, 1, m)) != dv(2, 1)
        with pytest.raises(VectorValueError):
            dv(1, 0).rotated(1., origin=(1., 1.))

    def test_mirrored(self):
        assert dv(1, 2).mirrored(DimVector(1. * nm, 0. * nm)) == dv(1, -2)
        assert dv(1, 2).mirrored(dv(1, 1)) == dv(2, 1)
        assert dv(1, 2).mirrored(DimVector(1. * m, 1. * m)) == dv(2, 1)
        with pytest.raises(VectorValueError):
            dv(1, 2).mirrored((1., 0.))
        with pytest.raises(ValueError):
            dv(1, 2).mirrored(DimVector())

    def test_projected(self):
        assert dv(2, 0).projected(dv(1, 1)) == dv(1, 0)
        assert dv(2, 0).projected(DimVector(1000. * nm, 1000. * nm)) == dv(1, 0)
        with pytest.raises(ValueError):
            DimVector().projected(dv(1, 1))

    def test_dot_cross_have_area_units(self):
        d = dv(1, 2).dot(dv(3, 4))
        assert isinstance(d, DimFloat)
        assert d.m_as('µm**2') == pytest.approx(11.)
        assert d.m_as('nm**2') == pytest.approx(11e6)

        c = dv(1, 0).cross(DimVector(0. * nm, 1000. * nm))
        assert c.m_as('µm**2') == pytest.approx(1.)

        with pytest.raises(VectorValueError):
            dv(1, 2).dot((1., 2.))


class TestDimVectorArithmetic:
    def test_add_sub(self):
        assert dv(1, 2) + dv(3, 4) == dv(4, 6)
        assert dv(1, 2) + DimVector(1000. * nm, 1000. * nm) == dv(2, 3)
        assert dv(1, 2) - dv(3, 5) == dv(-2, -3)
        assert (dv(1, 2) + DimVector(1000. * nm, 1000. * nm)).unit == ureg.Unit('µm')
        assert (DimVector(1000. * nm, 1000. * nm) + dv(1, 2)).unit == ureg.Unit('nm')
        assert (1. * um, 2. * um) + dv(1, 1) == dv(2, 3)
        assert (3. * um, 5. * um) - dv(1, 2) == dv(2, 3)

    def test_sum(self):
        assert sum([dv(1, 2), dv(3, 4), dv(5, 6)]) == dv(9, 12)

    def test_incompatible_operands(self):
        with pytest.raises(VectorValueError):
            dv(1, 2) + Vector(1., 2.)
        with pytest.raises(VectorValueError):
            dv(1, 2) + (1., 2.)
        with pytest.raises(VectorValueError):
            Vector(1., 2.) + dv(1, 2)
        with pytest.raises(VectorValueError):
            dv(1, 2) - DimVector(1. * unit('s'), 2. * unit('s'))

    def test_mul_div(self):
        assert dv(1, 2) * 2 == dv(2, 4)
        assert 2 * dv(1, 2) == dv(2, 4)
        assert np.float64(2.) * dv(1, 2) == dv(2, 4)
        assert isinstance(np.float64(2.) * dv(1, 2), DimVector)
        assert dv(2, 4) / 2 == dv(1, 2)
        assert -dv(1, -2) == dv(-1, 2)
        assert (2 * dv(1, 2)).unit == ureg.Unit('µm')

    def test_invalid_mul(self):
        with pytest.raises(TypeError):
            dv(1, 2) * (1. * um)
        with pytest.raises(TypeError):
            dv(1, 2) * dv(1, 2)
        with pytest.raises(TypeError):
            dv(1, 2) / (1. * um)
        with pytest.raises(ZeroDivisionError):
            dv(1, 2) / 0

    def test_operations_do_not_modify(self):
        a = dv(1, 2)
        b = DimVector(500. * nm, 500. * nm)
        _ = a + b, a - b, b + a, 2 * a, a.rotated(1.), a.normalized()
        assert a == dv(1, 2)
        assert b.unit == ureg.Unit('nm') and b == dv(0.5, 0.5)


class TestDimVectorAngles:
    def test_angle_between(self):
        assert angle_between(dv(1, 0), dv(0, 1)) == pytest.approx(math.pi / 2)
        assert angle_between(dv(1, 0), DimVector(0. * nm, 1. * nm)) == pytest.approx(math.pi / 2)

    def test_angle_between_converts_units(self):
        # (1 um, 0) and (1000 nm, 1 um) -> 45 degree. Comparing the raw numbers would give a wrong angle.
        assert angle_between(dv(1, 0), DimVector(1000. * nm, 1. * um)) == pytest.approx(math.pi / 4)
        assert signed_angle_between(dv(1, 0), DimVector(1000. * nm, 1. * um)) == pytest.approx(math.pi / 4)
        assert signed_angle_between(DimVector(1000. * nm, 0. * nm), DimVector(1. * um, -1. * um)) == pytest.approx(
            -math.pi / 4
        )

    def test_mixing_vector_and_dim_vector(self):
        with pytest.raises(VectorValueError):
            angle_between(dv(1, 0), (0., 1.))
        with pytest.raises(VectorValueError):
            signed_angle_between((1., 0.), dv(0, 1))

    def test_null_vector(self):
        with pytest.raises(ValueError):
            angle_between(dv(1, 0), DimVector())
