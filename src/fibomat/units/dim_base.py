"""Base class of all `Dim...` types."""
from __future__ import annotations

import typing as t

import pint  # type: ignore

from fibomat.units.dimensions import D


__all__ = ['DimBase']


class DimBase(t.Generic[D]):
    """Base class for every `Dim...` type: a Python value tagged at type level with a :class:`Dimension`.

    :func:`check_units` recognizes any subclass structurally, so adding a new `Dim...` type (3-vector, matrix, ...)
    only requires implementing :meth:`_dim_quantities`: yield a ``(label, pint.Quantity)`` pair for every embedded
    quantity which must share dimension `D`. A scalar wrapper yields one entry with an empty label, a container one
    labeled entry per component. The label is used in error messages like ``'displacement.y': expected ...``::

        class DimTriple(DimBase[D]):
            def __init__(self, a: DimFloat[D], b: DimFloat[D], c: DimFloat[D]) -> None:
                self._a, self._b, self._c = a, b, c

            def _dim_quantities(self) -> t.Iterator[t.Tuple[str, pint.Quantity]]:
                yield 'a', self._a.quantity
                yield 'b', self._b.quantity
                yield 'c', self._c.quantity
    """

    __slots__ = ()

    def _dim_quantities(self) -> t.Iterator[t.Tuple[str, pint.Quantity]]:
        raise NotImplementedError(f'{type(self).__name__} must implement _dim_quantities() (see DimBase docstring)')
