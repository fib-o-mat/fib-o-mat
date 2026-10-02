"""Aliases of pint types used for type annotations."""
from __future__ import annotations

import pint  # type: ignore


__all__ = ['UnitType', 'QuantityType', 'LengthUnit', 'TimeUnit', 'LengthQuantity', 'TimeQuantity']


UnitType = pint.Unit
"""Generic pint unit type used for typing annotation."""

QuantityType = pint.Quantity
"""pint quantity type used for typing annotation."""

LengthUnit = pint.Unit
"""pint unit with dimension ``[length]`` (annotation only, not enforced)."""

TimeUnit = pint.Unit
"""pint unit with dimension ``[time]`` (annotation only, not enforced)."""

LengthQuantity = QuantityType
"""pint quantity with dimension ``[length]`` (annotation only, not enforced)."""

TimeQuantity = QuantityType
"""pint quantity with dimension ``[time]`` (annotation only, not enforced)."""
