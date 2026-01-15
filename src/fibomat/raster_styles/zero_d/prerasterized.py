from typing import Tuple, Sequence, Union

import numpy as np

from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.units import LengthUnit, TimeUnit, scale_to, scale_factor
from fibomat.shapes import Shape, DimShape
from fibomat.mill import Mill


class PreRasterized(RasterStyle):
    def __init__(self):
        pass

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    @property
    def dimension(self) -> int:
        return 0

    @staticmethod
    def _prepare_and_scale(
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:
        # dim_shape = DimObj.create(dim_shape)

        # if isinstance(dim_shape, RasterizedPattern):
        #     rasterized_pattern = dim_shape
        #
        #     if rasterized_pattern.length_unit != out_length_unit:
        #         rasterized_pattern._dwell_points[:, :2] *= scale_factor(out_length_unit, rasterized_pattern.length_unit)
        #
        #     if rasterized_pattern.time_unit != out_time_unit:
        #         rasterized_pattern._dwell_points[:, 2] *= scale_factor(out_time_unit, rasterized_pattern.time_unit)
        #
        #     return rasterized_pattern
        if isinstance(dim_shape.shape, RasterizedPoints):
            points = np.array(dim_shape.shape.dwell_points)

            points[:, :2] *= scale_factor(out_length_unit, dim_shape.unit)
            points[:, 2] *= [scale_to(out_time_unit,mill.dwell_time(p[0:2])) for p in points._dwell_points] #scale_to(out_time_unit, mill.dwell_time)

            if mill.repeats != 1:
                return RasterizedPattern(np.concatenate([points]*mill.repeats), out_length_unit, out_time_unit)
            else:
                return RasterizedPattern(points, out_length_unit, out_time_unit)
        else:
            raise TypeError('Shape must be of type RasterizedPoints or RasterizedPattern for PreRasterized style')


    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:
        return self._prepare_and_scale(dim_shape, mill, out_length_unit, out_time_unit)
