"""Helper functions to inspect and scale units and quantities."""
from __future__ import annotations

import typing as t

import pint  # type: ignore

from fibomat.units.dim_float import DimFloat
from fibomat.units.dimensions import LengthDimension, TimeDimension
from fibomat.units.registry import ureg
from fibomat.units.unit_tag import _UnitTag


__all__ = ['has_length_dim', 'has_time_dim', 'scale_factor', 'scale_to']


_UnitLike = t.Union[pint.Unit, pint.Quantity, DimFloat[t.Any], _UnitTag[t.Any]]


def _as_unit(obj: _UnitLike) -> pint.Unit:
    """Return the pint unit of a unit, quantity, DimFloat or unit tag (magnitudes are ignored)."""
    if isinstance(obj, _UnitTag):
        return ureg.Unit(obj._symbol)  # pylint: disable=protected-access
    if isinstance(obj, pint.Unit):
        return obj
    return obj.units


def has_length_dim(quant_or_unit: _UnitLike) -> bool:
    """Check if `quant_or_unit` has dimension `[length]`.

    Args:
        quant_or_unit (UnitType, QuantityType, DimFloat): unit or quantity to be checked.

    Returns:
        bool
    """
    return LengthDimension.matches(_as_unit(quant_or_unit))


def has_time_dim(quant_or_unit: _UnitLike) -> bool:
    """Check if `quant_or_unit` has dimension `[time]`.

    Args:
        quant_or_unit (UnitType, QuantityType, DimFloat): unit or quantity to be checked.

    Returns:
        bool
    """
    return TimeDimension.matches(_as_unit(quant_or_unit))


def scale_factor(base_unit: _UnitLike, other_unit: _UnitLike) -> float:
    """Calculate the scaling factor needed to convert `other_unit` to `base_unit`.

    .. warning:: If `base_unit` or `other_unit` are quantities, the magnitude is ignored.

    Args:
        base_unit (UnitType, QuantityType, DimFloat): base unit
        other_unit (UnitType, QuantityType, DimFloat): other unit

    Returns:
        float: scaling factor
    """
    return (1. * _as_unit(other_unit)).to(_as_unit(base_unit)).magnitude


def scale_to(base_unit: _UnitLike, quantity: t.Union[pint.Quantity, DimFloat[t.Any]]) -> float:
    """Scale `quantity` to `base_unit` and return the result as `float`.

    Args:
        base_unit (UnitType, QuantityType, DimFloat): base unit
        quantity (QuantityType, DimFloat): quantity to be scaled

    Returns:
        float: scaled quantity
    """
    if isinstance(quantity, DimFloat):
        quantity = quantity.quantity
    return quantity.to(_as_unit(base_unit)).magnitude
