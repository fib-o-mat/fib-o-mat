"""Provide the :class:`Mill` class.

Example:
    >>> from fibomat.units import unit
    >>> mill = Mill(2. * unit('ms'), repeats=3)
    >>> mill.repeats
    3
    >>> mill.dwell_time.m_as('µs')
    2000.0
"""
from __future__ import annotations

import math
import operator
import typing as t

from fibomat.mill.mill_base import MillBase
from fibomat.units import DimFloat, TimeDimension, has_time_dim


__all__ = ['Mill']


class Mill(MillBase):
    """Constant dwell time per spot and number of repeats of a pattern. It is used for all shapes.

    The settings are available as properties and as items (``mill['dwell_time']``, ``mill['repeats']``).
    """

    def __init__(self, dwell_time: DimFloat[TimeDimension], repeats: int):
        """
        Args:
            dwell_time (DimFloat[TimeDimension]): dwell time per spot (greater than 0), e.g. ``2. * unit('ms')``.
            repeats (int): number of repeats (at least 1)

        Raises:
            TypeError: Raised if dwell_time is not a dimensioned value (e.g. a float or a plain pint quantity) or
                repeats is not an integer.
            ValueError: Raised if dwell_time has not the dimension [time] or is not positive and finite, or if repeats
                is less than 1.
        """
        if not isinstance(dwell_time, DimFloat):
            raise TypeError(
                f'dwell_time must be a dimensioned value like 2. * unit("ms"), got {type(dwell_time).__name__}.'
            )

        if not has_time_dim(dwell_time):
            raise ValueError('dwell_time must have dimension [time].')

        if not math.isfinite(dwell_time.magnitude) or dwell_time.magnitude <= 0.:
            raise ValueError(f'dwell_time must be positive and finite, got {dwell_time!r}.')

        if isinstance(repeats, bool):
            raise TypeError('repeats must be an int.')
        try:
            repeats = operator.index(repeats)
        except TypeError:
            raise TypeError(f'repeats must be an int, got {type(repeats).__name__}.') from None

        if repeats < 1:
            raise ValueError('repeats must be at least 1.')

        super().__init__(dwell_time=dwell_time, repeats=repeats)

    @property
    def dwell_time(self) -> DimFloat[TimeDimension]:
        """Dwell time per spot.

        Access:
            get
        """
        return self['dwell_time']

    @property
    def repeats(self) -> int:
        """Number of repeats.

        Access:
            get
        """
        return self['repeats']
