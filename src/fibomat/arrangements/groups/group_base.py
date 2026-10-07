from __future__ import annotations
from typing import Optional, List, Iterator, Any, TypeVar, Generic
import abc

from frozenlist import FrozenList

from fibomat.arrangements.arrangementbase import ArrangementBase
from fibomat.linalg import Transformable, DimTransformable, VectorLike, DimVectorLike, BoundingBox, DimBoundingBox
# from fibomat.arrangements.utils import _have_same_type


ElementT = TypeVar('ElementT', Transformable, DimTransformable)
VectorT = TypeVar('VectorT', VectorLike, DimVectorLike)
BBoxT = TypeVar('BBoxT', BoundingBox, DimBoundingBox)


class GroupBase(ArrangementBase[ElementT, VectorT, BBoxT], abc.ABC):
    def __init__(
        self,
        elements: List[ElementT],
        description: Optional[str] = None
    ):
        if not elements:
            raise ValueError('Length if elements must not be zero.')

        # TODO: make this works for different types (e.g. pattern, Dimgroup, ...)
        # if not _have_same_type(elements):
        #     raise TypeError('Elements in an arrangement must have same (base) type.')

        self._elements: FrozenList[ElementT] = FrozenList(elements)
        self._elements.freeze()

        # we evaluate this only if needed
        self._center = None
        self._bounding_box = None

        super().__init__(description=description)

    @property
    def elements(self):
        return self._elements

    def _arrangement_elements(self) -> Iterator[ElementT]:
        for element in self._elements:
            yield element

    @property
    def center(self) -> VectorT:
        if self._center:
            return self._center
        else:
            center = self._elements[0].center
            for element in self._elements[1:]:
                center += element.center

            center /= len(self._elements)

            self._center = center

            return center

    @property
    def bounding_box(self) -> BBoxT:
        if self._bounding_box:
            return self._bounding_box
        else:
            bbox = self._elements[0].bounding_box
            for element in self._elements[1:]:
                bbox = bbox.extended(element.bounding_box)

            self._bounding_box = bbox

            return bbox

    def _impl_translate(self, trans_vec: VectorT) -> None:
        for element in self._elements:
            element._impl_translate(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        for element in self._elements:
            element._impl_rotate(theta)

    def _impl_scale(self, fac: float) -> None:
        for element in self._elements:
            element._impl_scale(fac)

    def _impl_mirror(self, mirror_axis: VectorT) -> None:
        for element in self._elements:
            element._impl_mirror(mirror_axis)
