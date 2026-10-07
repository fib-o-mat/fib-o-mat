"""Provide the :class:`PreRasterized` raster style.

Example:
    >>> import numpy as np
    >>> from fibomat.raster_styles.zero_d import PreRasterized
    >>> from fibomat.mill import Mill
    >>> from fibomat.shapes import RasterizedPoints
    >>> from fibomat.units import unit
    >>> points = RasterizedPoints(np.array([[0., 0., 1.], [1., 0., 2.]]), False)
    >>> pattern = PreRasterized().rasterize(points * unit('µm'), Mill(1. * unit('ms'), 1), unit('µm'), unit('ms'))
    >>> pattern.dwell_points.tolist()  # the weights are multiplied with the dwell time
    [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]]
"""
from __future__ import annotations

import numpy as np

from fibomat.mill import Mill
from fibomat.raster_styles._helpers import apply_repeats, dwell_time_in, position_scale
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes import DimShape
from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.units import LengthUnit, TimeUnit


__all__ = ['PreRasterized']


class PreRasterized(RasterStyle):
    """Raster style for :class:`~fibomat.shapes.rasterizedpoints.RasterizedPoints`.

    The positions are only converted to the output unit. The weight of each point is multiplied with the dwell time of
    the mill. The whole sequence of points is repeated `repeats` times.
    """

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

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
        """Convert pre-rasterized points to a pattern.

        Args:
            dim_shape (DimShape): points with length unit
            mill (Mill): mill
            out_length_unit (LengthUnit): length unit of the returned pattern
            out_time_unit (TimeUnit): time unit of the returned pattern

        Returns:
            RasterizedPattern

        Raises:
            TypeError: Raised if the shape is no RasterizedPoints or the mill is no :class:`~fibomat.mill.Mill`.
        """
        if not isinstance(dim_shape.shape, RasterizedPoints):
            raise TypeError('Shape must be of type RasterizedPoints for the PreRasterized style.')

        points = np.array(dim_shape.shape.dwell_points)
        points[:, :2] *= position_scale(dim_shape.unit, out_length_unit)
        points[:, 2] *= dwell_time_in(mill, out_time_unit)

        return RasterizedPattern(apply_repeats(points, mill.repeats), out_length_unit, out_time_unit)
