"""Helpers shared by the raster styles."""
from __future__ import annotations

import math
import typing as t

import numpy as np

from fibomat.mill import Mill
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.units import DimFloat, LengthUnit, TimeUnit, has_length_dim, scale_factor, scale_to


__all__ = ['check_length', 'dwell_time_in', 'apply_repeats', 'empty_pattern', 'position_scale']


def check_length(value: DimFloat[t.Any], name: str) -> DimFloat[t.Any]:
    """Check that `value` is a positive, finite length.

    Args:
        value (DimFloat): value to be checked, e.g. ``0.5 * unit('µm')``
        name (str): name of the argument used in error messages

    Returns:
        DimFloat: `value`

    Raises:
        TypeError: Raised if `value` is not a dimensioned value (e.g. a float).
        ValueError: Raised if `value` has not the dimension [length] or is not positive and finite.
    """
    if not isinstance(value, DimFloat):
        raise TypeError(f'{name} must be a dimensioned value like 0.5 * unit("µm"), got {type(value).__name__}.')

    if not has_length_dim(value):
        raise ValueError(f'{name} must have dimension [length].')

    if not math.isfinite(value.magnitude) or value.magnitude <= 0.:
        raise ValueError(f'{name} must be positive and finite, got {value!r}.')

    return value


def dwell_time_in(mill: Mill, time_unit: TimeUnit) -> float:
    """Dwell time of a mill as a float in `time_unit`.

    Args:
        mill (Mill): mill
        time_unit (TimeUnit): unit of the result

    Returns:
        float: dwell time

    Raises:
        TypeError: Raised if `mill` is not a :class:`~fibomat.mill.Mill` (mills with a position dependent dwell time are
            not supported by this raster style yet).
    """
    if not isinstance(mill, Mill):
        raise TypeError(f'This raster style needs a Mill with a constant dwell time, got {type(mill).__name__}.')
    return scale_to(time_unit, mill.dwell_time)


def position_scale(from_unit: LengthUnit, to_unit: LengthUnit) -> float:
    """Factor which converts lengths given in `from_unit` to `to_unit`."""
    return scale_factor(to_unit, from_unit)


def apply_repeats(dwell_points: np.ndarray, repeats: int) -> np.ndarray:
    """Repeat the dwell points `repeats` times (one pass after the other).

    Args:
        dwell_points (np.ndarray): dwell points, shape (n, 3)
        repeats (int): number of repeats

    Returns:
        np.ndarray: dwell points, shape (n * repeats, 3)
    """
    return np.tile(dwell_points, (repeats, 1))


def empty_pattern(length_unit: LengthUnit, time_unit: TimeUnit) -> RasterizedPattern:
    """A pattern without dwell points."""
    return RasterizedPattern(np.empty((0, 3)), length_unit, time_unit)
