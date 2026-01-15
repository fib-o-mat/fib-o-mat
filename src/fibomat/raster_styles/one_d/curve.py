from typing import Tuple, Sequence

import numpy as np

from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.units import LengthUnit, TimeUnit, scale_to, has_length_dim, QuantityType, scale_factor
from fibomat.shapes import Shape, DimShape
from fibomat.mill import Mill
from fibomat.curve_tools import rasterize
from fibomat.raster_styles.scansequence import ScanSequence


class Curve(RasterStyle):
    def __init__(self, pitch: QuantityType, scan_sequence: ScanSequence):
        self._pitch = pitch

        if not has_length_dim(self._pitch):
            raise ValueError('Pitch must have [length] as dimension')

        if self._pitch.m <= 0:
            raise ValueError('pitch must be greater than 0.')

        if scan_sequence not in [ScanSequence.CONSECUTIVE, ScanSequence.BACKSTITCH, ScanSequence.BACK_AND_FORTH]:
            raise ValueError(
                'scan_sequence must be "ScanSequence.CONSECUTIVE", " ScanSequence.BACKSTITCH" or '
                '"ScanSequence.BACK_AND_FORTH".'
            )

        self._scan_sequence = scan_sequence

    def __repr__(self):
        return '{}(pitch={!r})'.format(self.__class__.__name__, self._pitch)

    @property
    def dimension(self) -> int:
        return 1

    @property
    def pitch(self) -> QuantityType:
        return self._pitch

    @property
    def scan_sequence(self) -> ScanSequence:
        return self._scan_sequence

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:
        # dim_shape = DimObj.create(dim_shape)

        pitch_scaled = scale_to(dim_shape.unit, self._pitch)

        points = rasterize(dim_shape.shape, pitch_scaled)

        points._dwell_points[:, :2] *= scale_factor(out_length_unit, dim_shape.unit)
        points._dwell_points[:, 2] = [scale_to(out_time_unit,mill.dwell_time(p[0:2])) for p in points._dwell_points]
        #points._dwell_points[:, 2] = scale_to(out_time_unit, mill.dwell_time)

        if self._scan_sequence in [ScanSequence.BACKSTITCH, ScanSequence.CONSECUTIVE]:
            if self._scan_sequence == ScanSequence.BACKSTITCH:
                # dwell points are indexed by 0, 1, 2, 3, 4, 5, ...
                # with backstiztch, points should be ordered like 1, 0, 3, 2, 5, 4, ...
                # hence, consecutive pairs of points must be swapped
                for i in range(0, points.n_points - 1, 2):
                    points._dwell_points[i], points._dwell_points[i + 1] = (
                        np.array(points._dwell_points[i + 1]), np.array(points._dwell_points[i])
                    )

            return RasterizedPattern(points.repeats_applied(mill.repeats).dwell_points, out_length_unit, out_time_unit)
        elif self._scan_sequence == ScanSequence.BACK_AND_FORTH:
            back_and_force = []
            reversed_points = points.dwell_points[::-1]
            # print(points)
            # print(reversed_points)
            for i in range(mill.repeats):
                if i % 2 == 0:
                    back_and_force.append(points.dwell_points)
                else:
                    back_and_force.append(reversed_points)
            return RasterizedPattern(np.concatenate(back_and_force), out_length_unit, out_time_unit)
        else:
            raise ValueError('Scan sequence not supported.')


