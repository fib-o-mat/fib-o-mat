"""Provides the :class:`DimShape` class.

A :class:`DimShape` is a :class:`~fibomat.shapes.shape.Shape` with a unit of length. The shape itself is unitless, its
coordinates are interpreted in units of the :attr:`DimShape.unit`.

Example::

    from fibomat.shapes import Rect, DimShape
    from fibomat.units import unit

    rect = Rect(width=1, height=2)

    dim_shape = rect * unit('µm')  # same as DimShape(rect, 'µm')

    dim_shape.center  # DimVector(x=0.0 µm, y=0.0 µm)
    dim_shape.translated(Vector(1, 1) * unit('nm'))  # translation is converted to the unit of the shape
    dim_shape.to('nm')  # new DimShape, the rect is scaled accordingly: width = 1000, height = 2000
"""
from __future__ import annotations

import typing as t

import pint  # type: ignore

from fibomat.linalg import DimBoundingBox, DimTransformable, DimVector, DimVectorLike
from fibomat.shapes.shape import Shape
from fibomat.units import DimensionError, has_length_dim, scale_factor, ureg
from fibomat.units.unit_tag import _UnitTag  # pylint: disable=protected-access


__all__ = ['DimShape']


LengthUnitLike = t.Union[str, pint.Unit, _UnitTag]  # type: ignore[type-arg]
"""Unit of length: a unit string (``'µm'``), a pint unit or ``unit('µm')``."""


def _length_unit(unit: LengthUnitLike) -> pint.Unit:
    """Convert `unit` to a pint unit and check that it is a unit of length."""
    if isinstance(unit, _UnitTag):
        pint_unit = (1. * unit).units
    elif isinstance(unit, (str, pint.Unit)):
        pint_unit = ureg.Unit(unit)
    else:
        raise TypeError(f'unit must be a unit string, pint.Unit or unit(...), got {type(unit).__name__}.')

    if not has_length_dim(pint_unit):
        raise DimensionError(f'unit must have dimension [length], got {pint_unit}.')
    return pint_unit


class DimShape(DimTransformable):
    """A shape together with a unit of length.

    :class:`DimShape` objects are immutable by convention: the transformation methods (`translated`, `rotated`, ...) and
    :meth:`DimShape.to` return new objects. The shape is **not** copied when a :class:`DimShape` is created, so
    shapes must not be modified afterwards either (all public methods of shapes return new shapes).
    """

    def __init__(self, shape: Shape, unit: LengthUnitLike, description: t.Optional[str] = None):
        """
        Args:
            shape (Shape): shape, its coordinates are given in `unit`
            unit (str, pint.Unit, unit): unit of length
            description (str, optional): description

        Raises:
            TypeError: Raised if `shape` is no :class:`Shape` or `unit` is no unit.
            DimensionError: Raised if `unit` is no unit of length.
        """
        if not isinstance(shape, Shape):
            raise TypeError(f'shape must be a Shape, got {type(shape).__name__}.')

        self._shape = shape
        self._unit = _length_unit(unit)

        super().__init__(description)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(shape={self._shape!r}, unit={self._unit!r})'

    def __getstate__(self) -> t.Dict[str, t.Any]:
        # pint units must not be pickled (they would not belong to our registry afterwards)
        state = self.__dict__.copy()
        state['_unit'] = str(self._unit)
        return state

    def __setstate__(self, state: t.Dict[str, t.Any]) -> None:
        state = dict(state)
        state['_unit'] = ureg.Unit(state['_unit'])
        self.__dict__.update(state)

    @property
    def shape(self) -> Shape:
        """The (unitless) shape.

        Access:
            get
        """
        return self._shape

    @property
    def unit(self) -> pint.Unit:
        """Unit of length of the shape.

        Access:
            get
        """
        return self._unit

    @property
    def center(self) -> DimVector:
        """Center of the shape.

        Access:
            get
        """
        return DimVector.from_vector(self._shape.center, self._unit)

    @property
    def pivot(self) -> DimVector:
        """Pivot of the shape. If no pivot is set for the :class:`DimShape` itself, the pivot of the underlying shape
        is used (which is its center by default).

        Access:
            get/set
        """
        if self._pivot is not None:
            return DimVector(self._pivot(self))
        return DimVector.from_vector(self._shape.pivot, self._unit)

    @pivot.setter
    def pivot(self, value: t.Optional[t.Callable[[DimShape], t.Any]]) -> None:
        DimTransformable.pivot.fset(self, value)  # type: ignore[attr-defined]

    @property
    def bounding_box(self) -> DimBoundingBox:
        """Bounding box of the shape.

        Access:
            get
        """
        bbox = self._shape.bounding_box
        return DimBoundingBox(
            DimVector.from_vector(bbox.lower_left, self._unit), DimVector.from_vector(bbox.upper_right, self._unit)
        )

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        trans_vec = DimVector(trans_vec)
        self._shape._impl_translate(trans_vec.vector_as(self._unit))  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        self._shape._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        self._shape._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        mirror_axis = DimVector(mirror_axis)

        # vector length does not matter, mirror axis is normalized anyway.
        self._shape._impl_mirror(mirror_axis.vector)  # pylint: disable=protected-access

    def to(self, unit: LengthUnitLike) -> DimShape:
        """Return a copy of the shape in a different unit. The shape is scaled so that its physical size stays the
        same; this object is not modified. ::

            dim_shape = Rect(width=1, height=2) * unit('µm')
            in_nm = dim_shape.to('nm')  # width = 1000, height = 2000

        Args:
            unit (str, pint.Unit, unit): new unit of length

        Returns:
            DimShape

        Raises:
            TypeError: Raised if `unit` is no unit.
            DimensionError: Raised if `unit` is no unit of length.
        """
        new_unit = _length_unit(unit)

        clone = self.clone()
        # pylint: disable=protected-access
        clone._shape._impl_scale(scale_factor(new_unit, self._unit))
        clone._unit = new_unit
        return clone
