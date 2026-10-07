"""Tests of `fibomat.mill`."""
import copy
import pickle

import numpy as np
import pytest

from fibomat.mill import DDDMill, Mill, MillBase, SILMill, SpecialMill
from fibomat.units import unit


class TestMillBase:
    def test_getitem(self):
        mill = MillBase(a=1, b=2)
        assert mill['a'] == 1
        assert mill['b'] == 2

    def test_missing_item(self):
        with pytest.raises(KeyError, match='SpecialMill'):
            MillBase(a=1)['c']

    def test_none_is_missing(self):
        with pytest.raises(KeyError):
            MillBase(a=None)['a']

    def test_immutable(self):
        mill = MillBase(a=1)
        with pytest.raises(TypeError):
            mill['a'] = 2

    def test_kwargs_are_copied(self):
        settings = {'a': 1}
        mill = MillBase(**settings)
        settings['a'] = 5
        assert mill['a'] == 1

    def test_repr(self):
        assert repr(MillBase(a=1, b='x')) == "MillBase(a=1, b='x')"
        assert repr(SpecialMill(a=1)) == 'SpecialMill(a=1)'


class TestMill:
    def test_properties(self):
        mill = Mill(2. * unit('ms'), 3)
        assert mill.dwell_time == 2. * unit('ms')
        assert mill.dwell_time.m_as('µs') == pytest.approx(2000.)
        assert mill.repeats == 3

    def test_items(self):
        mill = Mill(1. * unit('s'), 2)
        assert mill['dwell_time'] == 1. * unit('s')
        assert mill['repeats'] == 2

    def test_dwell_time_is_constant_not_a_function(self):
        assert not callable(Mill(1. * unit('s'), 1).dwell_time)

    def test_is_not_a_ddd_mill(self):
        assert isinstance(Mill(1. * unit('s'), 1), MillBase)
        assert not isinstance(Mill(1. * unit('s'), 1), DDDMill)

    def test_numpy_integer_repeats(self):
        assert Mill(1. * unit('s'), np.int64(4)).repeats == 4
        assert type(Mill(1. * unit('s'), np.int64(4)).repeats) is int

    def test_repr(self):
        assert repr(Mill(1. * unit('s'), 2)).startswith('Mill(dwell_time=DimFloat(')

    @pytest.mark.parametrize('dwell_time', ['foo', 1., 1])
    def test_dwell_time_type(self, dwell_time):
        with pytest.raises(TypeError, match='unit'):
            Mill(dwell_time, 2)

    def test_plain_pint_quantity_is_rejected(self):
        from fibomat.units import ureg
        with pytest.raises(TypeError):
            Mill(1. * ureg.second, 2)

    def test_dwell_time_dimension(self):
        with pytest.raises(ValueError, match='time'):
            Mill(1. * unit('µm'), 2)

    @pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan')])
    def test_dwell_time_value(self, value):
        with pytest.raises(ValueError):
            Mill(value * unit('s'), 2)

    @pytest.mark.parametrize('repeats', ['foo', 1.5, 2., None, True])
    def test_repeats_type(self, repeats):
        with pytest.raises(TypeError):
            Mill(1. * unit('s'), repeats)

    @pytest.mark.parametrize('repeats', [0, -1])
    def test_repeats_value(self, repeats):
        with pytest.raises(ValueError):
            Mill(1. * unit('s'), repeats)

    def test_copy_and_pickle(self):
        mill = Mill(1. * unit('s'), 2)
        for clone in (copy.deepcopy(mill), pickle.loads(pickle.dumps(mill))):
            assert clone.dwell_time == mill.dwell_time
            assert clone.repeats == 2


class TestSpecialMill:
    def test_settings(self):
        mill = SpecialMill(a=1, b=2)
        assert mill['a'] == 1 and mill['b'] == 2
        assert isinstance(mill, MillBase)

    def test_missing_setting(self):
        with pytest.raises(KeyError):
            SpecialMill(a=1)['b']


def test_ddd_and_sil_mill_are_exported():
    # not reworked yet (see TODO.md); only the exports are checked
    assert issubclass(DDDMill, MillBase) and issubclass(SILMill, DDDMill)
