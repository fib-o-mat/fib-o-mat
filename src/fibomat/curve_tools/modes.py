"""Enums of the options of the functions of the package.

The functions accept the enum members or their string values (``'union'`` or ``CombineMode.UNION``).
"""
from __future__ import annotations

import enum
import typing as t


__all__ = ['CombineMode', 'OffsetDirection']


E = t.TypeVar('E', bound=enum.Enum)


class _StrEnum(str, enum.Enum):
    """Enum with string values which can be created from a member or a value."""

    @classmethod
    def parse(cls: t.Type[E], value: t.Union[E, str]) -> E:
        """Return the member for `value` (a member or its string value).

        Args:
            value (Enum, str): member or value

        Returns:
            Enum member

        Raises:
            ValueError: Raised if `value` is unknown.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            options = ', '.join(repr(member.value) for member in cls)  # type: ignore[attr-defined]
            raise ValueError(f'Unknown {cls.__name__} {value!r}, expected one of {options}.') from None


class CombineMode(_StrEnum):
    """How two closed curves are combined (boolean operations on the enclosed regions)."""

    UNION = 'union'
    """Region of the first or the second curve."""
    INTERSECT = 'intersect'
    """Region of the first and the second curve."""
    EXCLUDE = 'exclude'
    """Region of the first curve without the region of the second curve."""
    XOR = 'xor'
    """Region of exactly one of the curves."""


class OffsetDirection(_StrEnum):
    """Direction of an offset."""

    INWARDS = 'inwards'
    """To the inside of a closed curve."""
    OUTWARDS = 'outwards'
    """To the outside of a closed curve."""
    LEFT = 'left'
    """To the left of an open curve (seen in the direction of the curve)."""
    RIGHT = 'right'
    """To the right of an open curve (seen in the direction of the curve)."""
