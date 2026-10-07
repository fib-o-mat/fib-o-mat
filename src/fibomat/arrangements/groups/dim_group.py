from __future__ import annotations
from typing import Optional, List, Any

from fibomat.arrangements.groups.group_base import GroupBase
from fibomat.linalg import DimVector, DimTransformable, DimBoundingBox


class DimGroup(GroupBase[DimTransformable, DimVector, DimBoundingBox], DimTransformable):
    def __init__(
        self,
        elements: List[DimTransformable],
        description: Optional[str] = None
    ):
        super().__init__(elements=elements, description=description)
