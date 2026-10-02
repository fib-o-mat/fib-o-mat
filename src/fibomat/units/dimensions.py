"""Dimension tags and :class:`DimensionError`.

Tags are used as generic parameters of the `Dim...` types (e.g. ``DimFloat[LengthDimension]``) and are never
instantiated.
"""
from __future__ import annotations

import typing as t

import pint  # type: ignore

from fibomat.units.registry import ureg


__all__ = [
    'DimensionError',
    'Dimension',
    'LengthDimension',
    'TimeDimension',
    'DimensionlessDimension',
    'SpotDoseDimension',
    'LineDoseDimension',
    'AreaDoseDimension',
    'D',
]


class DimensionError(TypeError):
    """Raised if the actual pint dimensionality of a value does not match its annotation or explicit dimension."""


class Dimension:
    """Base class for dimension tags.

    Subclasses are pure markers: they are used as generic parameters and are never instantiated. Each subclass names
    the pint dimensionality it stands for in the class variable `pint_dimension`::

        class ForceDimension(Dimension):
            pint_dimension = '[length] * [mass] / [time] ** 2'
    """

    pint_dimension: t.ClassVar[str]
    """pint dimensionality string, e.g. ``'[length]'`` or ``'[length] / [time]'``."""

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        if 'pint_dimension' not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a 'pint_dimension' class variable")

    @classmethod
    def dimensionality(cls) -> t.Any:
        """Return the pint dimensionality (a ``UnitsContainer``) of this dimension."""
        return ureg.get_dimensionality(cls.pint_dimension)

    @classmethod
    def matches(cls, quantity_or_unit: t.Any) -> bool:
        """Check if a pint quantity or unit has this dimension.

        Args:
            quantity_or_unit (pint.Quantity, pint.Unit): object to be checked.

        Returns:
            bool
        """
        return bool(quantity_or_unit.dimensionality == cls.dimensionality())


class LengthDimension(Dimension):
    """Dimension ``[length]``."""
    pint_dimension = '[length]'


class TimeDimension(Dimension):
    """Dimension ``[time]``."""
    pint_dimension = '[time]'


class DimensionlessDimension(Dimension):
    """Dimensionless quantities."""
    pint_dimension = ''


class SpotDoseDimension(Dimension):
    """Dose per spot, ``[current] * [time]`` (``[spotdose]``)."""
    pint_dimension = '[spotdose]'


class LineDoseDimension(Dimension):
    """Dose per length, ``[current] * [time] / [length]`` (``[linedose]``)."""
    pint_dimension = '[linedose]'


class AreaDoseDimension(Dimension):
    """Dose per area, ``[current] * [time] / [length] ** 2`` (``[areadose]``)."""
    pint_dimension = '[areadose]'


D = t.TypeVar('D', bound=Dimension)
"""Type variable for dimension tags."""
