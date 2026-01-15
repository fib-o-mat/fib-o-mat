from typing import Tuple, Sequence

import numpy as np

from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.units import LengthUnit, TimeUnit, scale_to, scale_factor
from fibomat.shapes import Spot, DimShape
from fibomat.mill import Mill


class SingleSpot(RasterStyle):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return '{}()'.format(self.__class__.__name__)

    @property
    def dimension(self) -> int:
        return 0

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:

        # dim_shape = DimShape(dim_shape)

        if not isinstance(dim_shape.shape, Spot):
            raise RuntimeError('Only `shapes.Spot`s can have `SpotStyle` as raster style.')

        spot = dim_shape.shape
        dwell_point = [
            *(spot.position * scale_factor(out_length_unit, dim_shape.unit)),
            scale_to(out_time_unit, mill.dwell_time(spot.position))
        ]
        print(dwell_point)

        # return RasterizedPoints(np.array([dwell_point]*mill.repeats), False)
        return RasterizedPattern(np.array([dwell_point]*mill.repeats), out_length_unit, out_time_unit)
