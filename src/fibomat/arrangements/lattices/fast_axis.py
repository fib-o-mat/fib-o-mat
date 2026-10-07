"""Provide the :class:`FastAxis` enum."""
from __future__ import annotations

import enum
import typing as t


__all__ = ['FastAxis']


class FastAxis(str, enum.Enum):
    """The lattice direction along which the points follow each other directly (the "fast axis" of the order of the
    lattice points): with ``U``, the points are ordered row by row (the index along `u` changes fastest), with ``V``
    column by column.

    For lattices with the lattice vectors along the x and y axis, ``'x'`` and ``'y'`` can be used instead of ``'u'`` and
    ``'v'``.
    """

    U = 'u'
    V = 'v'

    @classmethod
    def parse(cls, value: t.Union[FastAxis, str]) -> FastAxis:
        """Return the member for `value` (a member or one of the strings ``'u'``, ``'v'``, ``'x'``, ``'y'``).

        Args:
            value (FastAxis, str): member or string

        Returns:
            FastAxis

        Raises:
            ValueError: Raised if `value` is unknown.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.lower()
            if key in ('u', 'x'):
                return cls.U
            if key in ('v', 'y'):
                return cls.V
        raise ValueError(f"Unknown fast axis {value!r}, expected one of 'u', 'v', 'x', 'y'.")
