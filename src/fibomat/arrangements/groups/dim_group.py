"""Provide the :class:`DimGroup` class.

Example:
    >>> from fibomat.arrangements import DimGroup
    >>> from fibomat.shapes import Rect
    >>> from fibomat.units import unit
    >>> group = DimGroup([Rect(1, 1) * unit('µm'), Rect(1, 1).translated((4, 0)) * unit('µm')])
    >>> group.bounding_box.width.m_as('µm')
    5.0
"""
from __future__ import annotations

import typing as t

from fibomat.arrangements.groups.group_base import GroupBase
from fibomat.linalg import DimBoundingBox, DimTransformable, DimVector


__all__ = ['DimGroup']


class DimGroup(GroupBase[DimTransformable, DimVector, DimBoundingBox], DimTransformable):
    """A group of transformable objects with units (:class:`~fibomat.shapes.DimShape`,
    :class:`~fibomat.layout.pattern.Pattern`, other groups, ...). All elements are transformed together."""

    _dimensioned = True

    def __init__(self, elements: t.Iterable[DimTransformable], description: t.Optional[str] = None):
        """
        Args:
            elements (Iterable[DimTransformable]): elements (with units)
            description (str, optional): description

        Raises:
            ValueError: Raised if there are no elements.
            TypeError: Raised if an element is not transformable or has no unit.
        """
        super().__init__(elements=elements, description=description)
