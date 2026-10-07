"""Provides the :class:`StubRasterStyle` class."""
from __future__ import annotations

import typing as t

from fibomat.mill import Mill
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes import DimShape
from fibomat.units import LengthUnit, TimeUnit


__all__ = ['StubRasterStyle']


class StubRasterStyle(RasterStyle):
    """Raster style of the annotations which are only plotted. It cannot rasterize."""

    def __init__(self, dimension: int):
        """
        Args:
            dimension (int): dimension of the shapes the style stands for (1: curve, 2: filled)
        """
        self._dimension = dimension

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._dimension})'

    @property
    def dimension(self) -> int:
        return self._dimension

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: t.Optional[Mill],
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit,
    ) -> RasterizedPattern:
        raise NotImplementedError('Annotations cannot be rasterized.')
