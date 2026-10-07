"""Helpers to convert units which are given by the user (shared by the default backends)."""
from __future__ import annotations

import typing as t

import pint  # type: ignore

from fibomat.units import has_length_dim, has_time_dim, ureg


__all__ = ['to_length_unit', 'to_time_unit']


def _to_pint_unit(value: t.Any) -> pint.Unit:
    """Convert a unit string, a pint unit or a unit tag (``unit('µm')``) to a pint unit.

    Raises:
        TypeError: Raised if `value` is no unit.
    """
    if isinstance(value, pint.Unit):
        return value
    if isinstance(value, str):
        return ureg.Unit(value)
    try:
        return (1. * value).units  # unit tags
    except (TypeError, AttributeError):
        raise TypeError(f'Expected a unit (e.g. unit("µm")), got {value!r}.') from None


def to_length_unit(value: t.Any) -> pint.Unit:
    """Convert a length unit (a unit string, a pint unit or ``unit('µm')``) to a pint unit.

    Args:
        value: unit

    Returns:
        pint.Unit

    Raises:
        TypeError: Raised if `value` is no unit.
        ValueError: Raised if the unit is no unit of length.
    """
    pint_unit = _to_pint_unit(value)
    if not has_length_dim(pint_unit):
        raise ValueError(f'The unit must have dimension [length], got {pint_unit}.')
    return pint_unit


def to_time_unit(value: t.Any) -> pint.Unit:
    """Convert a time unit (a unit string, a pint unit or ``unit('µs')``) to a pint unit.

    Args:
        value: unit

    Returns:
        pint.Unit

    Raises:
        TypeError: Raised if `value` is no unit.
        ValueError: Raised if the unit is no unit of time.
    """
    pint_unit = _to_pint_unit(value)
    if not has_time_dim(pint_unit):
        raise ValueError(f'The unit must have dimension [time], got {pint_unit}.')
    return pint_unit
