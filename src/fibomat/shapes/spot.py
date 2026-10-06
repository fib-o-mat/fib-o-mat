"""Provides the :class:`Spot` class.

Example::

    from fibomat.shapes import Spot
    from fibomat.units import unit

    spot = Spot((1, 2))
    spot.position  # Vector(x=1.0, y=2.0)
    spot.translated((1, 1)).position  # Vector(x=2.0, y=3.0)

    dim_spot = spot * unit('µm')
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.shape import Shape


__all__ = ['Spot']


class Spot(Shape):
    """0-dim spot."""

    def __init__(self, position: t.Optional[VectorLike] = None, *, description: t.Optional[str] = None):
        """
        Args:
            position (VectorLike, optional): position of the spot, default to (0, 0)
            description (str, optional): description

        Raises:
            VectorValueError: Raised if `position` is no vector.
            ValueError: Raised if `position` is not finite.
        """
        super().__init__(description)

        self._position: Vector = Vector(position) if position is not None else Vector()

        if not np.all(np.isfinite(np.asarray(self._position))):
            raise ValueError('position must be finite.')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(position={self._position!r})'

    @property
    def position(self) -> Vector:
        """Position of the spot. Same as :attr:`center`

        Access:
            get

        Returns:
            Vector
        """
        return self._position

    @property
    def bounding_box(self) -> BoundingBox:
        """Bounding box of the spot (a box with zero width and height).

        Access:
            get
        """
        return BoundingBox(self._position, self._position)

    @property
    def is_closed(self) -> bool:
        """Not defined for 0-dim shapes.

        Access:
            get

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('is_closed is not defined for 0-dim shapes like Spot.')

    @property
    def center(self) -> Vector:
        """Center of the spot (its position).

        Access:
            get
        """
        return self._position

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._position = self._position + trans_vec

    def _impl_rotate(self, theta: float) -> None:
        self._position = self._position.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        self._position = self._position * fac

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        self._position = self._position.mirrored(mirror_axis)
