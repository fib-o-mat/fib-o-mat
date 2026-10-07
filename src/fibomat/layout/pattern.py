"""Provides the :class:`Pattern` class.

Example:
    >>> from fibomat.layout import Pattern
    >>> from fibomat.mill import Mill
    >>> from fibomat.raster_styles import ScanSequence, one_d
    >>> from fibomat.shapes import Line
    >>> from fibomat.units import unit
    >>> pattern = Pattern(
    ...     Line((0, 0), (1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1),
    ...     one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
    ... )
    >>> pattern.translated((1 * unit('µm'), 0 * unit('µm'))).center.x.m_as('µm')
    1.5
"""
# pylint: disable=protected-access
from __future__ import annotations

import typing as t

from fibomat.linalg import DimBoundingBox, DimTransformable, DimVector, DimVectorLike
from fibomat.mill import MillBase
from fibomat.raster_styles import RasterStyle
from fibomat.shapes import DimShape, Shape


__all__ = ['Pattern']


ShapeT = t.TypeVar('ShapeT', bound=Shape)


class Pattern(DimTransformable, t.Generic[ShapeT]):
    """A shape (with a length unit) together with the mill settings and the raster style which are used to expose it.

    A pattern is transformed like its shape. The transformation methods return new patterns: the shape is copied, but
    the mill and the raster style are shared between a pattern and its clones (e.g. the copies created by
    transformations or by lattices), because they are treated as immutable configurations.
    """

    _shared_attributes = ('_mill', '_raster_style')

    def __init__(
        self,
        dim_shape: DimShape,
        mill: t.Optional[MillBase],
        raster_style: RasterStyle,
        *,
        description: t.Optional[str] = None,
        **kwargs: t.Any
    ):
        """
        Args:
            dim_shape (DimShape): shape with a length unit. Dimensioned arrangements (e.g. a
                :class:`~fibomat.arrangements.DimGroup` or a :class:`~fibomat.arrangements.DimLattice`) are accepted,
                too; backends split them into their shapes.
            mill (MillBase, optional): mill. It may be None for patterns which are not exported (annotations).
            raster_style (RasterStyle): raster style
            description (str, optional): description
            **kwargs: additional arguments for the backends. Names with a leading underscore are used internally
                (e.g. ``_annotation`` and ``_color`` by the plotting backend).

        Raises:
            TypeError: Raised if `dim_shape` is not a dimensioned shape, `mill` is no mill or `raster_style` is no
                raster style.
        """
        super().__init__(description=description)

        if not isinstance(dim_shape, DimTransformable):
            raise TypeError(f'dim_shape must be a DimShape (shape * unit), got {type(dim_shape).__name__}.')
        if mill is not None and not isinstance(mill, MillBase):
            raise TypeError(f'mill must be a Mill, got {type(mill).__name__}.')
        if not isinstance(raster_style, RasterStyle):
            raise TypeError(f'raster_style must be a RasterStyle, got {type(raster_style).__name__}.')

        self._dim_shape = dim_shape
        self._mill = mill
        self._raster_style = raster_style
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(dim_shape={self._dim_shape!r}, mill={self._mill!r}, '
            f'raster_style={self._raster_style!r})'
        )

    @property
    def dim_shape(self) -> DimShape:
        """Dimensioned shape of the pattern.

        Access:
            get
        """
        return self._dim_shape

    @property
    def mill(self) -> t.Optional[MillBase]:
        """Mill of the pattern.

        Access:
            get
        """
        return self._mill

    @property
    def raster_style(self) -> RasterStyle:
        """Raster style of the pattern.

        Access:
            get
        """
        return self._raster_style

    @property
    def kwargs(self) -> t.Dict[str, t.Any]:
        """Additional arguments which were passed to the pattern.

        Access:
            get
        """
        return self._kwargs

    @property
    def bounding_box(self) -> DimBoundingBox:
        return self._dim_shape.bounding_box

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
