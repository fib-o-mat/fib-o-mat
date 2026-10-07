"""Provide the :class:`SingleSpot` raster style.

Example:
    >>> from fibomat.raster_styles.zero_d import SingleSpot
    >>> from fibomat.mill import Mill
    >>> from fibomat.shapes import Spot
    >>> from fibomat.units import unit
    >>> pattern = SingleSpot().rasterize(Spot((1, 2)) * unit('µm'), Mill(2. * unit('ms'), 2), unit('nm'), unit('µs'))
    >>> pattern.dwell_points.round(6).tolist()
    [[1000.0, 2000.0, 2000.0], [1000.0, 2000.0, 2000.0]]
"""
from __future__ import annotations

import numpy as np

from fibomat.mill import Mill
from fibomat.raster_styles._helpers import apply_repeats, dwell_time_in, position_scale
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes import DimShape, Spot
from fibomat.units import LengthUnit, TimeUnit


__all__ = ['SingleSpot']


class SingleSpot(RasterStyle):
    """Raster style for :class:`~fibomat.shapes.spot.Spot` shapes. The spot is exposed `repeats` times."""

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
        """Rasterize a spot.

        Args:
            dim_shape (DimShape): spot with length unit
            mill (Mill): mill
            out_length_unit (LengthUnit): length unit of the returned pattern
            out_time_unit (TimeUnit): time unit of the returned pattern

        Returns:
            RasterizedPattern: `mill.repeats` times the same dwell point

        Raises:
            TypeError: Raised if the shape is no spot or the mill is no :class:`~fibomat.mill.Mill`.
        """
        if not isinstance(dim_shape.shape, Spot):
            raise TypeError('Only `shapes.Spot`s can have `SingleSpot` as raster style.')

        scale = position_scale(dim_shape.unit, out_length_unit)
        x, y = np.asarray(dim_shape.shape.position, dtype=float) * scale

        dwell_point = np.array([[x, y, dwell_time_in(mill, out_time_unit)]])

        return RasterizedPattern(apply_repeats(dwell_point, mill.repeats), out_length_unit, out_time_unit)
