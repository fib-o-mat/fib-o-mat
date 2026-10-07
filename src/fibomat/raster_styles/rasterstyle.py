"""Provide the :class:`RasterStyle` class."""
from __future__ import annotations

import abc
import copy

from fibomat.mill import Mill
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes.dim_shape import DimShape
from fibomat.units import LengthUnit, TimeUnit


__all__ = ['RasterStyle']


class RasterStyle(abc.ABC):
    """Base class to define a raster style.

    All raster styles have the required parameters as attributes (e.g. pitch, ...).
    Further, all raster styles must implement a rasterize method which applies the raster style to a shape.
    """

    def __repr__(self) -> str:
        return self.__class__.__name__

    def clone(self) -> RasterStyle:
        """Deep copy the raster style.

        Returns:
            RasterStyle
        """
        return copy.deepcopy(self)

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the shapes this raster style can be applied to.

        0 for spots and pre-rasterized points, 1 for curves and 2 for areas.

        Access:
            get
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:
        """Rasterize the given shape.

        The number of repeats and the dwell time defined in the mill object must be applied!
        The points and dwell times in the returned RasterizedPattern must be scaled according to out_length_unit
        and out_time_unit.

        Args:
            dim_shape (DimShape): shape with length unit to be rasterized.
            mill (Mill): mill
            out_length_unit (LengthUnit): length unit of the returned RasterizedPattern
            out_time_unit (TimeUnit): time unit of the returned RasterizedPattern

        Returns:
            RasterizedPattern
        """
        raise NotImplementedError
