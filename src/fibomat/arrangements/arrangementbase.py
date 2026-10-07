"""Provide the :class:`ArrangementBase` class."""
from __future__ import annotations
from typing import Optional, Iterator, TypeVar, Generic
import abc

from fibomat.linalg import (
    Transformable, DimTransformable, VectorLike, DimVectorLike, BoundingBox, DimBoundingBox
)


ElementT = TypeVar('ElementT', Transformable, DimTransformable)
VectorT = TypeVar('VectorT', VectorLike, DimVectorLike)
BBoxT = TypeVar('BBoxT', BoundingBox, DimBoundingBox)


class ArrangementBase(Generic[ElementT, VectorT, BBoxT], abc.ABC):
    """
    ABC for all arrangement related classes.

    It can be used to arrange :class:`fibomat.layout.site.Site`, :class:`fibomat.layout.pattern.Pattern` and
    :class:`fibomat.shapes.Shape`.

    The saved elements are accessed via the :meth:`ArrangementBase.arrangement_elements` method which returns a
    generator containing all included elements.

    What kind of elements and how these are set must be specified in child classes.

    """

    def __init__(self, description: Optional[str] = None):
        """
        Args:
            description (str, optional): description
        """
        super().__init__(description=description)

    @abc.abstractmethod
    def _arrangement_elements(self) -> Iterator[ElementT]:
        raise NotImplementedError

    def arrangement_elements(self) -> Iterator[ElementT]:
        """Access to the saved elements.

        Yields:
            Any: Type depends on saved element.
        """
        for element in self._arrangement_elements():
            if isinstance(element, ArrangementBase):
                for sub_element in element._arrangement_elements():  # pylint: disable=protected-access
                    yield sub_element
            else:
                yield element

    # @property
    # def pivot(self):
    #     if self._pivot is not None:
    #         if self._is_dimensioned:
    #             return DimVector.create(self._pivot(self))
    #         else:
    #             return Vector(self._pivot(self))
    #     return self.center
    #
    # @pivot.setter
    # def pivot(self, value):
    #     self._pivot = value
    #
    # @property
    # def bounding_box(self) -> boundingbox.BoundingBox:
    #     """Bounding box of shape (getter)
    #
    #     Access:
    #         get
    #
    #     Returns:
    #         BoundingBox
    #
    #     Raises:
    #         RuntimeError: Raised if Arrangement object does not contain any elements
    #     """
    #     raise NotImplementedError
    #
    #     element_iter = self.arrangement_elements()
    #
    #     try:
    #         bbox = next(element_iter).bounding_box
    #     except StopIteration:
    #         raise RuntimeError('Cannot calculate bounding box of empty Arrangement.')  # pylint: disable=raise-missing-from
    #
    #     for element in element_iter:
    #         bbox = bbox.extended(element.bounding_box)
    #
    #     return bbox
