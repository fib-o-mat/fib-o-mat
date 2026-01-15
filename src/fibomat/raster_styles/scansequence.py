from typing import Tuple, List
import enum

import numpy as np

from fibomat.shapes import Shape
from fibomat.units import LengthUnit, TimeUnit
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.mill import Mill, DDDMill
from fibomat.rasterizedpattern import RasterizedPattern


@enum.unique
class ScanSequence(enum.Enum):
    CONSECUTIVE = 'consecutive'
    BACKSTITCH = 'backstitch'
    BACK_AND_FORTH = 'back_and_forth'
    SERPENTINE = 'serpentine'
    DOUBLE_SERPENTINE = 'double_serpentine'
    DOUBLE_SERPENTINE_SAME_PATH = 'double_serpentine_same_path'
    CROSSECTION = 'crossection'


def _make_scan_index_sequence(
    n_lines: int, repeats: int, scan_sequence: ScanSequence
) -> Tuple[np.ndarray, Tuple[bool, np.ndarray]]:
    consecutive = np.arange(n_lines)

    if scan_sequence != ScanSequence.CROSSECTION:
        index_sequence = np.empty(n_lines * repeats, dtype=consecutive.dtype)
    else:
        index_sequence = consecutive
    # line_directions = np.empty(n_lines * repeats, dtype=bool)

    if scan_sequence == ScanSequence.CONSECUTIVE:
        for i in range(repeats):
            index_sequence[i*n_lines:(i + 1) * n_lines] = consecutive
        line_directions = True
    elif scan_sequence == ScanSequence.BACKSTITCH:
        for i in range(0, n_lines - 1, 2):
            index_sequence[i], index_sequence[i + 1] = (
                consecutive[i + 1], consecutive[i]
            )
        if n_lines % 2 == 1:
            index_sequence[n_lines-1] = n_lines-1
        for i in range(1, repeats):
            index_sequence[i*n_lines:(i + 1) * n_lines] = index_sequence[0:n_lines]
        line_directions = True
    elif scan_sequence == ScanSequence.CROSSECTION:
        # for i in range(n_lines):
        #     index_sequence[i*repeats:(i + 1) * repeats] = i
        line_directions = True
    elif scan_sequence == ScanSequence.SERPENTINE:
        line_directions = np.empty(n_lines * repeats, dtype=bool)
        for i in range(repeats):
            index_sequence[i * n_lines:(i + 1) * n_lines] = consecutive
            line_directions[i * n_lines:(i + 1) * n_lines:2] = True
            line_directions[i * n_lines + 1:(i + 1) * n_lines:2] = False
    elif scan_sequence == ScanSequence.DOUBLE_SERPENTINE:
        line_directions = np.empty(n_lines * repeats, dtype=bool)
        direction = True
        for i in range(repeats):
            if direction:
                index_sequence[i*n_lines:(i + 1) * n_lines] = consecutive
            else:
                index_sequence[i*n_lines:(i + 1) * n_lines] = consecutive[::-1]
            line_directions[i * n_lines:(i + 1) * n_lines:2] = True
            line_directions[i * n_lines + 1:(i + 1) * n_lines:2] = False
            direction = not direction
    elif scan_sequence == ScanSequence.DOUBLE_SERPENTINE_SAME_PATH:
        direction = True
        for i in range(repeats):
            if direction:
                index_sequence[i*n_lines:(i + 1) * n_lines] = consecutive
            else:
                index_sequence[i*n_lines:(i + 1) * n_lines] = consecutive[::-1]
            direction = not direction

        line_directions = True
    else:
        raise ValueError('Scan sequence not supported.')

    if isinstance(line_directions, bool):
        line_directions = np.full(n_lines * repeats, line_directions, dtype=bool)

    return index_sequence, line_directions


def _apply_scan_sequence(
    filling_rows: List[Shape],
    filling_rows_length_unit: LengthUnit,
    scan_sequence: ScanSequence,
    line_style: RasterStyle,
    mill: Mill,
    out_length_unit: LengthUnit,
    out_time_unit: TimeUnit
):
    if scan_sequence == ScanSequence.CROSSECTION:
        line_mill = mill
    else:
        line_mill = DDDMill(mill.dwell_time, repeats=1)
        #line_mill = Mill(mill.dwell_time, repeats=1)

    rasterized_lines = []

    for row in filling_rows:
        rasterized_lines.append(
            line_style.rasterize(
                row * filling_rows_length_unit,
                mill=line_mill,
                out_length_unit=out_length_unit,
                out_time_unit=out_time_unit
            ).dwell_points
        )

    scan_lines = []

    for index, direction in zip(*_make_scan_index_sequence(len(rasterized_lines), mill.repeats, scan_sequence)):
        if not direction:
            scan_lines.append(rasterized_lines[index][::-1])
        else:
            scan_lines.append(rasterized_lines[index])

    return RasterizedPattern(np.concatenate(scan_lines), out_length_unit, out_time_unit)
