"""Dimensioned scalars (physical units) for the package, backed by `pint`.

Physical values are wrapped in :class:`DimFloat`, which carries a :class:`Dimension` tag (e.g.
:class:`LengthDimension`) as a generic parameter. This gives two layers of safety:

1. **Static**: ``DimFloat[LengthDimension]`` and ``DimFloat[TimeDimension]`` are different types for mypy/pyright.
2. **Runtime**: :func:`check_units` validates the actual pint dimensionality of annotated arguments and return values.

The most convenient way to create values is :func:`unit` (also available as :data:`U_`)::

    from fibomat.units import unit, U_, LengthDimension, AreaDoseDimension, DimFloat, check_units

    length = 4. * unit('m')  # DimFloat[LengthDimension], inferred statically and at runtime
    same_length = 4. * U_('m')  # U_ is the same object as unit

    length_in_um = length.to('µm')
    raw = length.m_as('µm')  # magnitude as float

    # unit strings unknown to UNIT_DIMENSIONS need an explicit dimension for static typing
    dose = 12. * unit('ions / nm**2', AreaDoseDimension)

    @check_units
    def double(distance: DimFloat[LengthDimension]) -> DimFloat[LengthDimension]:
        return 2 * distance

    double(4. * unit('s'))  # raises DimensionError

Custom units and dimensions are added to the registry :data:`ureg` (see :mod:`fibomat.units.registry`). ``ions`` is
defined as ``elementary_charge`` (only charges of +1 are of interest), so ``1 ions`` is identical to one elementary
charge and ``(1 * ureg.coulomb).to('ions')`` to ``1 coulomb / elementary_charge``. The doses

* ``[spotdose]``: ``[current] * [time]``,
* ``[linedose]``: ``[current] * [time] / [length]`` and
* ``[areadose]``: ``[current] * [time] / [length] ** 2``

are available as derived dimensions (see :class:`SpotDoseDimension`, :class:`LineDoseDimension` and
:class:`AreaDoseDimension`).

The submodules are:

* ``registry``: the pint registry :data:`ureg` and custom definitions,
* ``typing_aliases``: aliases of pint types for annotations,
* ``dimensions``: :class:`Dimension` tags and :class:`DimensionError`,
* ``unit_table``: :data:`UNIT_DIMENSIONS`,
* ``dim_base``, ``dim_float``: :class:`DimBase` and :class:`DimFloat`,
* ``unit_tag``: :func:`unit` and :data:`U_`,
* ``legacy``: deprecated :func:`Q_`,
* ``helpers``: :func:`scale_factor`, :func:`scale_to`, :func:`has_length_dim`, :func:`has_time_dim`,
* ``checking``: :func:`check_units`.

.. note:: The block of ``unit()`` overloads in ``unit_tag.py`` is generated from :data:`UNIT_DIMENSIONS`. After
    editing the table in ``unit_table.py``, run ``python scripts/generate_unit_overloads.py``.

.. warning:: ``Q_`` is deprecated. Use :func:`unit` / :data:`U_` instead.
"""
from __future__ import annotations

from fibomat.units.registry import ureg
from fibomat.units.typing_aliases import UnitType, QuantityType, LengthUnit, TimeUnit, LengthQuantity, TimeQuantity
from fibomat.units.dimensions import DimensionError, Dimension, LengthDimension, TimeDimension, DimensionlessDimension, SpotDoseDimension, LineDoseDimension, AreaDoseDimension
from fibomat.units.unit_table import UNIT_DIMENSIONS
from fibomat.units.dim_base import DimBase
from fibomat.units.dim_float import DimFloat
from fibomat.units.unit_tag import unit, U_
from fibomat.units.legacy import Q_
from fibomat.units.helpers import has_length_dim, has_time_dim, scale_factor, scale_to
from fibomat.units.checking import check_units


__all__ = [
    'ureg',
    'UnitType',
    'QuantityType',
    'LengthUnit',
    'TimeUnit',
    'LengthQuantity',
    'TimeQuantity',
    'DimensionError',
    'Dimension',
    'LengthDimension',
    'TimeDimension',
    'DimensionlessDimension',
    'SpotDoseDimension',
    'LineDoseDimension',
    'AreaDoseDimension',
    'UNIT_DIMENSIONS',
    'DimBase',
    'DimFloat',
    'unit',
    'U_',
    'Q_',
    'has_length_dim',
    'has_time_dim',
    'scale_factor',
    'scale_to',
    'check_units',
]
