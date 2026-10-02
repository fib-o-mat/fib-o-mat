"""Deprecated parts of the units API."""
from __future__ import annotations

import typing as t
import warnings

import pint  # type: ignore

from fibomat.units.registry import ureg


__all__ = ['Q_']


def Q_(*args: t.Any, **kwargs: t.Any) -> pint.Quantity:
    """Create a plain pint quantity.

    .. deprecated:: 0.6.0
        Use :func:`unit` (or :data:`U_`) instead, e.g. ``12. * unit('µm')``.
    """
    warnings.warn("Q_ is deprecated, use unit(...) / U_(...) instead.", DeprecationWarning, stacklevel=2)
    return ureg.Quantity(*args, **kwargs)
