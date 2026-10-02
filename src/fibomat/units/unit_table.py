"""Table of unit strings known to :func:`fibomat.units.unit`."""
from __future__ import annotations

import typing as t

from fibomat.units.dimensions import Dimension, LengthDimension, TimeDimension


__all__ = ['UNIT_DIMENSIONS']


UNIT_DIMENSIONS: t.Dict[str, t.Type[Dimension]] = {}
"""Unit strings known to :func:`unit`, mapped to their dimension.

This is the single source of truth for the generated ``@overload`` block of :func:`unit` (static typing) and for the
dimension inference/validation at runtime if no explicit dimension is given.
"""
for _symbol in ('m', 'km', 'cm', 'mm', 'um', 'µm', 'nm', 'mi', 'yd', 'ft', 'in'):
    UNIT_DIMENSIONS[_symbol] = LengthDimension
for _symbol in ('s', 'ms', 'us', 'µs', 'ns', 'min', 'h', 'day'):
    UNIT_DIMENSIONS[_symbol] = TimeDimension
del _symbol
