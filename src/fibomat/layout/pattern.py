"""
Provides the :class:`Pattern` class.
"""
# pylint: disable=protected-access
from __future__ import annotations
from typing import Generic, TypeVar, Optional, Dict

from fibomat.mill import MillBase
from fibomat.linalg import DimTransformable, DimVector, DimVectorLike, DimBoundingBox
from fibomat.raster_styles import RasterStyle
from fibomat.shapes import DimShape


DimTransformableT = TypeVar('DimTransformableT', bound=DimTransformable)


class Pattern(DimTransformable, Generic[DimTransformableT]):
    """
    Class is used to collect a shape with a length unit,  mill settings and optional settings.

    Clones of a pattern (e.g. the copies created by transformations or lattices) copy the shape but share the mill and
    the raster style, which are treated as immutable configurations.
    """

    _shared_attributes = ('_mill', '_raster_style')

    def __init__(
        self,
        dim_shape: DimShape,
        mill: MillBase,
        raster_style: RasterStyle,
        *,
        description: Optional[str] = None,
        **kwargs
    ):
        """

        Args:
            dim_shape (Tuple[ShapeType, units.LengthUnit]):
                tuple of a shape type and its length unit. ShapeType can be any transformable, e.g. an arrangements.Group or
                shapes.Line, ...
            mill (Mill): mill object
            **kwargs: additional args
        """
        super().__init__(description=description)

        self._dim_shape = dim_shape
        self._mill = mill
        self._raster_style = raster_style
        self._kwargs = kwargs

    @property
    def dim_shape(self) -> DimShape:
        """Dimensioned shape of the pattern.

        Access:
            get

        Returns:
            DimShape
        """
        return self._dim_shape

    @property
    def mill(self) -> MillBase:
        """Mill of the pattern.

        Access:
            get

        Returns:
            MillBase
        """
        return self._mill

    @property
    def raster_style(self) -> RasterStyle:
        """Raster style of the pattern.

        Access:
            get

        Returns:
            RasterStyle
        """
        return self._raster_style

    @property
    def kwargs(self) -> Dict:
        """Additional passed arguments.

        Access:
            get

        Returns:
            Dict
        """
        return self._kwargs

    @property
    def bounding_box(self) -> DimBoundingBox:
        return self.dim_shape.bounding_box

    @property
    def center(self) -> DimVector:
        return self._dim_shape.center

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        self._dim_shape._impl_translate(DimVector(trans_vec))

    def _impl_rotate(self, theta: float) -> None:
        self._dim_shape._impl_rotate(theta)

    def _impl_scale(self, fac: float) -> None:
        self._dim_shape._impl_scale(fac)

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        self._dim_shape._impl_mirror(DimVector(mirror_axis))
