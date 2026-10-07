"""Provide the :class:`Group` class."""
# pylint: disable=protected-access
from __future__ import annotations
from typing import Optional, List, TypeVar

from fibomat.arrangements.groups.group_base import GroupBase
from fibomat.linalg import Vector, Transformable, BoundingBox, VectorLike
from fibomat.units import U_


class Group(GroupBase[Transformable, Vector, BoundingBox], Transformable):
    def __init__(
        self,
        elements: List[Transformable],
        description: Optional[str] = None
    ):
        super().__init__(elements=elements, description=description)

    def __mul__(self, other):
        if isinstance(other, U_):
            from fibomat.arrangements.groups.dim_group import DimGroup
            return DimGroup([elem * other for elem in self.elements], description=self.description)
        raise NotImplementedError

