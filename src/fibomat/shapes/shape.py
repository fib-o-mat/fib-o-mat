"""Provides the :class:`Shape` class.

:class:`Shape` is the base class of all shapes of the library. It defines the basic interface (center, bounding box,
transformations, closed or open) and creates dimensioned shapes by multiplying a shape with a unit.

Example::

    from fibomat.shapes import Shape
    from fibomat.linalg import Vector, BoundingBox
    from fibomat.units import unit


    class MyShape(Shape):
        def __repr__(self) -> str: ...
        @property
        def is_closed(self) -> bool: ...
        @property
        def center(self) -> Vector: ...
        @property
        def bounding_box(self) -> BoundingBox: ...
        def _impl_translate(self, trans_vec: Vector) -> None: ...
        def _impl_rotate(self, theta: float) -> None: ...
        def _impl_scale(self, fac: float) -> None: ...
        def _impl_mirror(self, mirror_axis: Vector) -> None: ...

    dim_shape = MyShape() * unit('µm')  # DimShape
"""
from __future__ import annotations

import abc
import typing as t

from fibomat.linalg import BoundingBox, Transformable, Vector
from fibomat.units import DimensionError, has_length_dim


__all__ = ['Shape']


class Shape(Transformable[Vector, BoundingBox], abc.ABC):
    """Base class for all shapes in the library which defines the basic interface.

    In addition to the members required by :class:`~fibomat.linalg.Transformable`, subclasses must implement
    :meth:`Shape.__repr__` and :attr:`Shape.is_closed`. :attr:`Shape.area` and :attr:`Shape.boundary_length` are
    optional and only defined for the shapes they make sense for.
    """

    def __init__(self, description: t.Optional[str] = None):
        """
        Args:
            description (str, optional): description
        """
        super().__init__(description=description)

    @abc.abstractmethod
    def __repr__(self) -> str:
        """str: repr of class"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_closed(self) -> bool:
        """
        bool: True if the shape is closed. This property should not be defined for 0-dim shapes.

        Access:
            get
        """
        raise NotImplementedError

    @property
    def area(self) -> float:
        """Area enclosed by the shape. Only implemented for closed shapes.

        Access:
            get

        Returns:
            float

        Raises:
            NotImplementedError: Raised if the area is not defined for the shape.
        """
        raise NotImplementedError(f'The area is not defined for {self.__class__.__name__}.')

    @property
    def boundary_length(self) -> float:
        """Length of the boundary (length of the curve) of the shape.

        Access:
            get

        Returns:
            float

        Raises:
            NotImplementedError: Raised if the boundary length is not defined for the shape.
        """
        raise NotImplementedError(f'The boundary length is not defined for {self.__class__.__name__}.')

    def __mul__(self, other: t.Any) -> t.Any:
        """``shape * unit('µm')`` creates a :class:`~fibomat.shapes.DimShape`.

        Args:
            other (unit): unit of length

        Returns:
            DimShape

        Raises:
            DimensionError: Raised if `other` is not a unit of length.
        """
        from fibomat.units.unit_tag import _UnitTag  # pylint: disable=import-outside-toplevel,protected-access

        if not isinstance(other, _UnitTag):
            return NotImplemented

        pint_unit = (1. * other).units
        if not has_length_dim(pint_unit):
            raise DimensionError(f'{other} is not a unit of length.')

        from fibomat.shapes.dim_shape import DimShape  # pylint: disable=import-outside-toplevel

        return DimShape(self, pint_unit)

    def __rmul__(self, other: t.Any) -> t.Any:
        """``unit('µm') * shape`` creates a :class:`~fibomat.shapes.DimShape`."""
        # (the unit tag returns NotImplemented for shapes, so this method is called)
        return self.__mul__(other)
