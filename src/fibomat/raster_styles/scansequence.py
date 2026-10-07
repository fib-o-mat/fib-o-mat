"""Provide the :class:`ScanSequence` enum and the helpers which apply a scan sequence to the lines of a filling.

Example:
    >>> from fibomat.raster_styles.scansequence import ScanSequence, _make_scan_index_sequence
    >>> indices, directions = _make_scan_index_sequence(3, 2, ScanSequence.SERPENTINE)
    >>> indices.tolist()
    [0, 1, 2, 0, 1, 2]
    >>> directions.tolist()  # True: line is exposed from its start to its end
    [True, False, True, True, False, True]
"""
from __future__ import annotations

import enum
import typing as t

import numpy as np

from fibomat.mill import Mill
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.raster_styles._helpers import empty_pattern
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.shapes import DimShape, Shape
from fibomat.units import LengthUnit, TimeUnit


__all__ = ['ScanSequence']


@enum.unique
class ScanSequence(enum.Enum):
    """Order in which the lines of a filling (or the points of a curve) are exposed.

    ``CONSECUTIVE``: all lines in order, the whole sequence is repeated.

    ``BACKSTITCH``: like ``CONSECUTIVE`` but consecutive pairs of lines are swapped (1, 0, 3, 2, ...).

    ``BACK_AND_FORTH``: (curves only) the curve is exposed forwards and backwards alternately.

    ``SERPENTINE``: all lines in order, every second line is exposed in reverse direction.

    ``DOUBLE_SERPENTINE``: like ``SERPENTINE``, but every second repeat runs through the lines in reverse order. The
    directions alternate over the whole sequence, hence, the beam never jumps between two exposures of a line.

    ``DOUBLE_SERPENTINE_SAME_PATH``: every second repeat runs through the lines in reverse order, all lines are
    exposed in the same direction.

    ``CROSSECTION``: every line is repeated before the next line is started.
    """
    CONSECUTIVE = 'consecutive'
    BACKSTITCH = 'backstitch'
    BACK_AND_FORTH = 'back_and_forth'
    SERPENTINE = 'serpentine'
    DOUBLE_SERPENTINE = 'double_serpentine'
    DOUBLE_SERPENTINE_SAME_PATH = 'double_serpentine_same_path'
    CROSSECTION = 'crossection'

    @classmethod
    def parse(cls, value: t.Union[ScanSequence, str]) -> ScanSequence:
        """Return the member for `value` (a member or its string value, e.g. ``'serpentine'``).

        Args:
            value (ScanSequence, str): member or value

        Returns:
            ScanSequence

        Raises:
            ValueError: Raised if `value` is unknown.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            options = ', '.join(repr(member.value) for member in cls)
            raise ValueError(f'Unknown scan sequence {value!r}, expected one of {options}.') from None


def _swap_pairs(n_lines: int) -> np.ndarray:
    """Indices 0 .. n_lines - 1 with swapped consecutive pairs (1, 0, 3, 2, ...). A last single index stays."""
    order = np.arange(n_lines)
    n_paired = n_lines // 2 * 2
    order[:n_paired:2], order[1:n_paired:2] = order[1:n_paired:2].copy(), order[:n_paired:2].copy()
    return order


def _make_scan_index_sequence(
    n_lines: int, repeats: int, scan_sequence: ScanSequence
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Create the order and the directions in which lines are exposed.

    Args:
        n_lines (int): number of lines
        repeats (int): number of repeats
        scan_sequence (ScanSequence): scan sequence

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            indices of the lines in the order of exposure and, for each of them, if the line is exposed from start to
            end (True) or reversed (False). For ``CROSSECTION`` every line appears once (the repeats must be applied
            by the line raster style).

    Raises:
        ValueError: Raised if the scan sequence is not supported for lines (e.g. ``BACK_AND_FORTH``).
    """
    consecutive = np.arange(n_lines)
    alternating = consecutive % 2 == 0  # True, False, True, ...

    if scan_sequence == ScanSequence.CONSECUTIVE:
        indices = np.tile(consecutive, repeats)
        directions = np.ones(len(indices), dtype=bool)
    elif scan_sequence == ScanSequence.BACKSTITCH:
        indices = np.tile(_swap_pairs(n_lines), repeats)
        directions = np.ones(len(indices), dtype=bool)
    elif scan_sequence == ScanSequence.CROSSECTION:
        indices = consecutive
        directions = np.ones(n_lines, dtype=bool)
    elif scan_sequence == ScanSequence.SERPENTINE:
        indices = np.tile(consecutive, repeats)
        directions = np.tile(alternating, repeats)
    elif scan_sequence in (ScanSequence.DOUBLE_SERPENTINE, ScanSequence.DOUBLE_SERPENTINE_SAME_PATH):
        indices = np.concatenate([consecutive if i % 2 == 0 else consecutive[::-1] for i in range(repeats)])
        if scan_sequence == ScanSequence.DOUBLE_SERPENTINE:
            # The directions alternate over the whole sequence (also for an odd number of lines): the line at the end
            # of a pass is exposed again at the start of the next pass, now from the end where the beam is.
            directions = np.arange(len(indices)) % 2 == 0
        else:
            directions = np.ones(len(indices), dtype=bool)
    else:
        raise ValueError(f'Scan sequence {scan_sequence} not supported.')

    return indices, directions


def _apply_scan_sequence(
    filling_rows: t.Sequence[Shape],
    filling_rows_length_unit: LengthUnit,
    scan_sequence: ScanSequence,
    line_style: RasterStyle,
    mill: Mill,
    out_length_unit: LengthUnit,
    out_time_unit: TimeUnit
) -> RasterizedPattern:
    """Rasterize the rows of a filling with a line style and order them according to the scan sequence.

    Args:
        filling_rows (Sequence[Shape]): rows of the filling, e.g. :class:`LineNonContinuous` shapes
        filling_rows_length_unit (LengthUnit): length unit of the rows
        scan_sequence (ScanSequence): scan sequence
        line_style (RasterStyle): raster style which is applied to each row (dimension 1)
        mill (Mill): mill. Its repeats are applied by the scan sequence (or, for ``CROSSECTION``, by the line style)
        out_length_unit (LengthUnit): length unit of the returned pattern
        out_time_unit (TimeUnit): time unit of the returned pattern

    Returns:
        RasterizedPattern: pattern with all rows (empty if there are no rows)
    """
    scan_sequence = ScanSequence.parse(scan_sequence)

    # the line style must not repeat the lines if the scan sequence repeats them
    line_mill = mill if scan_sequence == ScanSequence.CROSSECTION else Mill(mill.dwell_time, 1)

    rasterized_rows = [
        np.array(
            line_style.rasterize(
                DimShape(row, filling_rows_length_unit),
                mill=line_mill,
                out_length_unit=out_length_unit,
                out_time_unit=out_time_unit
            ).dwell_points
        )
        for row in filling_rows
    ]

    if not rasterized_rows:
        return empty_pattern(out_length_unit, out_time_unit)

    indices, directions = _make_scan_index_sequence(len(rasterized_rows), mill.repeats, scan_sequence)

    scan_lines = [
        rasterized_rows[index] if forward else rasterized_rows[index][::-1]
        for index, forward in zip(indices, directions)
    ]

    return RasterizedPattern(np.concatenate(scan_lines), out_length_unit, out_time_unit)
