"""Dimension-tagged scalar."""
from __future__ import annotations

import numbers
import typing as t

import pint  # type: ignore

from fibomat.units.dim_base import DimBase
from fibomat.units.dimensions import D, Dimension, DimensionError
from fibomat.units.registry import ureg
from fibomat.units.unit_table import UNIT_DIMENSIONS


__all__ = ['DimFloat']


class DimFloat(DimBase[D]):
    """A scalar physical quantity, tagged at type level with a :class:`Dimension`.

    At runtime this wraps a `pint.Quantity`; the generic parameter `D` is only used for static type checking. Use
    :func:`check_units` or :meth:`of` with `dimension` to enforce it at runtime.
    """

    __slots__ = ('_q',)

    # numpy scalars on the left side of an operator defer to the reflected operators of this class
    __array_ufunc__ = None

    def __init__(self, quantity: pint.Quantity) -> None:
        """
        Args:
            quantity (pint.Quantity): quantity created with :data:`ureg`.

        Raises:
            TypeError: `quantity` is not a quantity of this module's registry.
        """
        if not isinstance(quantity, ureg.Quantity):
            raise TypeError(f"expected a pint.Quantity from this library's `ureg`, got {type(quantity)!r}")
        self._q = quantity

    @classmethod
    def of(cls, value: float, unit: str, dimension: t.Optional[t.Type[Dimension]] = None) -> DimFloat[D]:
        """Build a tagged value, e.g. ``DimFloat[LengthDimension].of(3., 'm')``.

        If `dimension` is omitted, it is looked up in :data:`UNIT_DIMENSIONS`. If the unit is unknown there, no
        validation takes place.

        Args:
            value (float): magnitude
            unit (str): unit string
            dimension (Type[Dimension], optional): expected dimension

        Returns:
            DimFloat

        Raises:
            DimensionError: the unit does not have the (given or inferred) dimension.
        """
        quantity = value * ureg(unit)
        if dimension is None:
            dimension = UNIT_DIMENSIONS.get(unit)
        if dimension is not None and not dimension.matches(quantity):
            raise DimensionError(
                f'unit {unit!r} has dimensionality {quantity.dimensionality}, '
                f'expected {dimension.pint_dimension!r} ({dimension.__name__})'
            )
        return cls(quantity)

    @property
    def quantity(self) -> pint.Quantity:
        """The wrapped pint quantity."""
        return self._q

    @property
    def magnitude(self) -> float:
        """Magnitude in the current unit."""
        return self._q.magnitude

    @property
    def units(self) -> pint.Unit:
        """The pint unit."""
        return self._q.units

    def to(self, unit: t.Union[str, pint.Unit]) -> DimFloat[D]:
        """Convert to another unit of the same dimension."""
        return type(self)(self._q.to(unit))

    def m_as(self, unit: t.Union[str, pint.Unit]) -> float:
        """Return the magnitude in `unit`."""
        return self._q.m_as(unit)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._q!r})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DimFloat):
            return bool(self._q == other._q)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __lt__(self, other: DimFloat[D]) -> bool:
        if not isinstance(other, DimFloat):
            return NotImplemented
        return bool(self._q < other._q)

    def __le__(self, other: DimFloat[D]) -> bool:
        if not isinstance(other, DimFloat):
            return NotImplemented
        return bool(self._q <= other._q)

    def __gt__(self, other: DimFloat[D]) -> bool:
        if not isinstance(other, DimFloat):
            return NotImplemented
        return bool(self._q > other._q)

    def __ge__(self, other: DimFloat[D]) -> bool:
        if not isinstance(other, DimFloat):
            return NotImplemented
        return bool(self._q >= other._q)

    def __add__(self, other: DimFloat[D]) -> DimFloat[D]:
        if not isinstance(other, DimFloat):
            return NotImplemented
        return DimFloat(self._q + other._q)

    def __sub__(self, other: DimFloat[D]) -> DimFloat[D]:
        if not isinstance(other, DimFloat):
            return NotImplemented
        return DimFloat(self._q - other._q)

    def __neg__(self) -> DimFloat[D]:
        return DimFloat(-self._q)

    # Operands which are neither DimFloat nor a number (e.g. vectors) are refused with NotImplemented, so that python
    # tries the reflected operation of the other operand and raises a TypeError otherwise.
    @t.overload
    def __mul__(self, other: float) -> DimFloat[D]: ...
    @t.overload
    def __mul__(self, other: DimFloat[t.Any]) -> DimFloat[t.Any]: ...
    def __mul__(self, other: t.Union[DimFloat[t.Any], float]) -> DimFloat[t.Any]:
        if isinstance(other, DimFloat):
            return DimFloat(self._q * other._q)
        if isinstance(other, numbers.Real):
            return DimFloat(self._q * other)
        return NotImplemented

    __rmul__ = __mul__

    @t.overload
    def __truediv__(self, other: float) -> DimFloat[D]: ...
    @t.overload
    def __truediv__(self, other: DimFloat[t.Any]) -> DimFloat[t.Any]: ...
    def __truediv__(self, other: t.Union[DimFloat[t.Any], float]) -> DimFloat[t.Any]:
        if isinstance(other, DimFloat):
            return DimFloat(self._q / other._q)
        if isinstance(other, numbers.Real):
            return DimFloat(self._q / other)
        return NotImplemented

    def __float__(self) -> float:
        return float(self._q.magnitude)

    def _dim_quantities(self) -> t.Iterator[t.Tuple[str, pint.Quantity]]:
        yield '', self._q
