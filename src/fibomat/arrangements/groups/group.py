"""Provide the :class:`Group` class.

Example:
    >>> from fibomat.arrangements import Group
    >>> from fibomat.shapes import Rect
    >>> group = Group([Rect(1, 1), Rect(1, 1).translated((4, 0))])
    >>> group.translated((0, 1)).bounding_box.upper_right
    Vector(x=4.5, y=1.5)
    >>> from fibomat.units import unit
    >>> dim_group = group * unit('µm')  # a DimGroup
"""
# pylint: disable=protected-access
from __future__ import annotations

import typing as t

from fibomat.arrangements.groups.group_base import GroupBase
from fibomat.linalg import BoundingBox, Transformable, Vector


__all__ = ['Group']


class Group(GroupBase[Transformable, Vector, BoundingBox], Transformable):
    """A group of transformable objects without units (shapes, other groups, ...). All elements are transformed
    together; the group can be multiplied with a unit to get a :class:`~fibomat.arrangements.DimGroup`."""

    _dimensioned = False

    def __init__(self, elements: t.Iterable[Transformable], description: t.Optional[str] = None):
        """
        Args:
            elements (Iterable[Transformable]): elements (without units)
            description (str, optional): description

        Raises:
            ValueError: Raised if there are no elements.
            TypeError: Raised if an element is not transformable or has a unit.
        """
        super().__init__(elements=elements, description=description)

    def __mul__(self, other: t.Any) -> t.Any:
        """``group * unit('µm')`` creates a :class:`~fibomat.arrangements.DimGroup` of the elements with the unit.

        Args:
            other (unit): unit of length

        Returns:
            DimGroup

        Raises:
            DimensionError: Raised if `other` is not a unit of length.
        """
        from fibomat.units.unit_tag import _UnitTag  # pylint: disable=import-outside-toplevel
        from fibomat.arrangements.groups.dim_group import DimGroup  # pylint: disable=import-outside-toplevel

        if not isinstance(other, _UnitTag):
            return NotImplemented

        return DimGroup([element * other for element in self._elements], description=self.description)

    def __rmul__(self, other: t.Any) -> t.Any:
        """``unit('µm') * group`` creates a :class:`~fibomat.arrangements.DimGroup`."""
        return self.__mul__(other)
