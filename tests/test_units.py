import itertools
import typing as t
import warnings

import numpy as np
import pytest

from fibomat.units import (
    ureg, Q_, U_, unit, UnitType, QuantityType, scale_factor, scale_to, has_length_dim, has_time_dim,
    Dimension, DimBase, DimFloat, DimensionError, LengthDimension, TimeDimension, DimensionlessDimension,
    SpotDoseDimension, LineDoseDimension, AreaDoseDimension, UNIT_DIMENSIONS, check_units
)


def test_ureg():
    assert ureg
    assert UnitType
    assert QuantityType


def test_u_is_unit():
    assert U_ is unit


def test_q_deprecated():
    with pytest.warns(DeprecationWarning):
        q = Q_('1 cm')
    assert q == 1 * ureg.cm


def test_ions_and_doses():
    assert (1 * ureg('ions')) == (1 * ureg('elementary_charge'))
    # 1 coulomb expressed in ions is identical to 1 coulomb / elementary_charge
    assert (1 * ureg.coulomb).to('ions').magnitude == pytest.approx((1 * ureg.coulomb / ureg.elementary_charge).m_as(''))

    # ions are a charge, i.e. [current] * [time]
    assert SpotDoseDimension.matches(ureg('ions'))
    assert LineDoseDimension.matches(ureg('ions / m'))
    assert AreaDoseDimension.matches(ureg('ions / m**2'))
    assert not AreaDoseDimension.matches(ureg('ions / m'))
    assert not SpotDoseDimension.matches(ureg('ions * s'))

    dose = 12 * unit('ions / nm**2', AreaDoseDimension)
    assert dose.to('coulomb / µm**2').magnitude == pytest.approx(12 * 1.602176634e-19 * 1e6)


class TestDimension:
    def test_matches(self):
        assert LengthDimension.matches(ureg('µm'))
        assert LengthDimension.matches(1 * ureg.m)
        assert not LengthDimension.matches(ureg('s'))
        assert DimensionlessDimension.matches(ureg('m / m'))

    def test_subclass_needs_pint_dimension(self):
        with pytest.raises(TypeError):
            class Broken(Dimension):
                pass

    def test_unit_table(self):
        for symbol, dim in UNIT_DIMENSIONS.items():
            assert dim.matches(ureg(symbol)), symbol


class TestUnitTag:
    def test_inferred_dimension(self):
        length = 4. * unit('m')
        assert isinstance(length, DimFloat)
        assert length.m_as('cm') == pytest.approx(400)
        assert (unit('µs') * 2.).m_as('s') == pytest.approx(2e-6)

    def test_explicit_dimension(self):
        assert (4. * unit('m', LengthDimension)) == 4. * unit('m')
        with pytest.raises(DimensionError):
            4. * unit('kg', LengthDimension)

    def test_unknown_unit_is_unchecked(self):
        assert isinstance(4. * unit('kg'), DimFloat)

    def test_dimension_mismatch_via_table(self):
        with pytest.raises(DimensionError):
            DimFloat.of(1., 's', LengthDimension)


class TestDimFloat:
    def test_requires_registry_quantity(self):
        with pytest.raises(TypeError):
            DimFloat(1.)

    def test_arithmetic(self):
        a, b = 2. * unit('m'), 50. * unit('cm')
        assert (a + b).m_as('m') == pytest.approx(2.5)
        assert (a - b).m_as('m') == pytest.approx(1.5)
        assert (-a).m_as('m') == pytest.approx(-2)
        assert (a * 2).m_as('m') == pytest.approx(4)
        assert (2 * a).m_as('m') == pytest.approx(4)
        assert (a / 2).m_as('m') == pytest.approx(1)
        assert (a / (4. * unit('s'))).m_as('m/s') == pytest.approx(0.5)
        assert float(a) == 2.

    def test_comparison(self):
        a, b = 2. * unit('m'), 50. * unit('cm')
        assert b < a and b <= a and a > b and a >= b
        assert a == 200. * unit('cm')
        assert a != b
        assert a != 2

    def test_access(self):
        a = 2. * unit('m')
        assert a.magnitude == 2.
        assert a.units == ureg.m
        assert a.to('cm').magnitude == pytest.approx(200)
        assert a.quantity == 2 * ureg.m


class TestCheckUnits:
    def test_arguments_and_return(self):
        @check_units
        def double(x: DimFloat[LengthDimension]) -> DimFloat[LengthDimension]:
            return 2 * x

        assert double(1. * unit('m')).m_as('m') == pytest.approx(2)

        with pytest.raises(DimensionError):
            double(1. * unit('s'))
        with pytest.raises(DimensionError):
            double(1.)

    def test_return_checked(self):
        @check_units
        def wrong(x: DimFloat[LengthDimension]) -> DimFloat[TimeDimension]:
            return x

        with pytest.raises(DimensionError):
            wrong(1. * unit('m'))

    def test_kwargs_and_defaults(self):
        @check_units
        def f(x: DimFloat[LengthDimension], y: DimFloat[TimeDimension] = 1. * unit('s')):
            return x, y

        f(x=1. * unit('m'))
        with pytest.raises(DimensionError):
            f(1. * unit('m'), y=1. * unit('m'))

    def test_custom_dim_type(self):
        D = t.TypeVar('D', bound=Dimension)

        class Pair(DimBase[D]):
            def __init__(self, a, b):
                self.a, self.b = a, b

            def _dim_quantities(self):
                yield 'a', self.a.quantity
                yield 'b', self.b.quantity

        @check_units
        def f(p: Pair[LengthDimension]):
            pass

        f(Pair(1. * unit('m'), 2. * unit('m')))
        with pytest.raises(DimensionError, match="'p.b'"):
            f(Pair(1. * unit('m'), 2. * unit('s')))

    def test_dim_base_requires_implementation(self):
        with pytest.raises(NotImplementedError):
            DimBase()._dim_quantities()


class TestHelpers:
    def test_scaling(self):
        for s in itertools.combinations_with_replacement([1., 2., 3.], 2):
            assert np.isclose(scale_factor(s[0] * ureg('cm'), s[1] * ureg('m')), 100)
            assert np.isclose(scale_factor(s[0] * ureg.Unit('cm'), s[1] * ureg('m')), 100)
            assert np.isclose(scale_factor(unit('cm'), s[1] * unit('m')), 100)
            assert np.isclose(scale_factor(ureg.Unit('cm'), ureg.Unit('m')), 100)

            assert np.isclose(scale_to(unit('m'), s[0] * ureg('cm')), s[0] / 100)
            assert np.isclose(scale_to(s[1] * ureg('m'), s[0] * ureg('cm')), s[0] / 100)
            assert np.isclose(scale_to(unit('m'), s[0] * unit('cm')), s[0] / 100)

    def test_has_dim(self):
        assert has_length_dim(ureg.Unit('µm'))
        assert has_length_dim(unit('µm'))
        assert has_length_dim(1. * unit('µm'))
        assert has_length_dim(1. * ureg('µm'))
        assert not has_length_dim(unit('s'))

        assert has_time_dim(ureg.Unit('µs'))
        assert has_time_dim(unit('µs'))
        assert not has_time_dim(unit('m'))


class TestOperandRefusal:
    """Operands which are no numbers are refused (TypeError) instead of being wrapped into a quantity."""

    class Dummy:
        pass

    def test_unit_tag_and_dim_float_refuse_non_numbers(self):
        for operand in ('a', None, [1., 2.], self.Dummy(), (1., 2.)):
            with pytest.raises(TypeError):
                operand * unit('m')
            with pytest.raises(TypeError):
                unit('m') * operand
            with pytest.raises(TypeError):
                operand * (1. * unit('m'))
            with pytest.raises(TypeError):
                (1. * unit('m')) * operand
            with pytest.raises(TypeError):
                (1. * unit('m')) / operand

    def test_dim_float_arithmetic_with_non_dim_float(self):
        a = 1. * unit('m')
        for op in (lambda: a + 1., lambda: a - 1., lambda: 1. + a, lambda: a + unit('m')):
            with pytest.raises(TypeError):
                op()
        for op in (lambda: a < 1., lambda: a <= 1., lambda: a > 1., lambda: a >= 1.):
            with pytest.raises(TypeError):
                op()

    def test_numpy_scalars(self):
        for scalar in (np.float64(2.), np.float32(2.), np.int64(2)):
            res = scalar * unit('m')
            assert isinstance(res, DimFloat)
            assert res.m_as('m') == pytest.approx(2.)
            res = scalar * (1. * unit('m'))
            assert isinstance(res, DimFloat)
            assert (unit('m') * scalar).m_as('m') == pytest.approx(2.)
            assert ((1. * unit('m')) * scalar).m_as('m') == pytest.approx(2.)

    def test_numpy_array_is_refused(self):
        with pytest.raises(TypeError):
            np.array([1., 2.]) * unit('m')

    def test_unit_times_vector_creates_dim_vector(self):
        from fibomat.linalg.vectors import Vector, DimVector

        for res in (Vector(1., 2.) * unit('um'), unit('um') * Vector(1., 2.)):
            assert isinstance(res, DimVector)
            assert res.y.m_as('um') == pytest.approx(2.)

        with pytest.raises(TypeError):
            unit('um') * DimVector(1. * unit('um'), 2. * unit('um'))
        with pytest.raises(TypeError):
            (1. * unit('um')) * Vector(1., 2.)
        with pytest.raises(TypeError):
            Vector(1., 2.) * (1. * unit('um'))
