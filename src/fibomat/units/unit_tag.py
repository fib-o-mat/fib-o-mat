"""The :func:`unit` constructor (alias :data:`U_`) creating dimensioned values, e.g. ``4. * unit('m')``."""
from __future__ import annotations

import numbers
import typing as t

from fibomat.units.dim_float import DimFloat
from fibomat.units.dimensions import D, Dimension, LengthDimension, TimeDimension
from fibomat.units.unit_table import UNIT_DIMENSIONS


__all__ = ['unit', 'U_']


class _UnitTag(t.Generic[D]):
    """Returned by :func:`unit`. Multiplying a number with it builds a ``DimFloat[D]``, e.g. ``4. * unit('m')``.

    `D` is resolved by the overloads of :func:`unit` from the literal unit string. (``4. * ureg('m')`` would give a
    plain `pint.Quantity`, not a :class:`DimFloat`.)
    """

    __slots__ = ('_symbol', '_dimension')

    def __init__(self, symbol: str, dimension: t.Optional[t.Type[Dimension]]) -> None:
        self._symbol = symbol
        self._dimension = dimension

    # numpy scalars on the left side of the operator defer to __rmul__
    __array_ufunc__ = None

    def __rmul__(self, value: float) -> DimFloat[D]:
        if not isinstance(value, numbers.Real):
            # e.g. vectors: let python try the reflected operation of the other operand
            return NotImplemented
        return DimFloat.of(value, self._symbol, self._dimension)

    __mul__ = __rmul__  # unit('m') * 4. also works

    def __repr__(self) -> str:
        return f'unit({self._symbol!r})'


# Overloads are grouped by dimension. They have to be literal source text since type checkers never execute this file.
# >>> BEGIN GENERATED unit() OVERLOADS (see generate_unit_overloads.py) >>>
@t.overload
def unit(symbol: t.Literal['cm', 'ft', 'in', 'km', 'm', 'mi', 'mm', 'nm', 'um', 'yd', 'µm'], dimension: None = None) -> _UnitTag[LengthDimension]: ...
@t.overload
def unit(symbol: t.Literal['day', 'h', 'min', 'ms', 'ns', 's', 'us', 'µs'], dimension: None = None) -> _UnitTag[TimeDimension]: ...
# <<< END GENERATED unit() OVERLOADS <<<


# Fixed overloads (not generated): an explicit `dimension` always wins; a unit string outside UNIT_DIMENSIONS without
# explicit dimension is not validated and statically untagged.
@t.overload
def unit(symbol: str, dimension: t.Type[D]) -> _UnitTag[D]: ...
@t.overload
def unit(symbol: str, dimension: None = None) -> _UnitTag[t.Any]: ...


def unit(symbol: str, dimension: t.Optional[t.Type[Dimension]] = None) -> _UnitTag[t.Any]:
    """Create a unit tag. ``4. * unit('m')`` is a ``DimFloat[LengthDimension]``.

    The dimension is inferred from :data:`UNIT_DIMENSIONS` (statically via overloads and at runtime). An explicit
    `dimension` overrides the lookup and is required for statically typed values of unknown units::

        4. * unit('N', ForceDimension)  # unit unknown to the table
        4. * unit('m', LengthDimension)  # redundant, same as 4. * unit('m')
        4. * unit('kg', LengthDimension)  # raises DimensionError, 'kg' is not a length

    Args:
        symbol (str): unit string understood by pint.
        dimension (Type[Dimension], optional): expected dimension.

    Returns:
        _UnitTag: object which creates a :class:`DimFloat` if multiplied with a number.
    """
    if dimension is None:
        dimension = UNIT_DIMENSIONS.get(symbol)
    return _UnitTag(symbol, dimension)


U_ = unit
"""Alias of :func:`unit` (formerly the pint unit constructor)."""
