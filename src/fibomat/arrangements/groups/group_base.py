"""Provide the :class:`GroupBase` class."""
from __future__ import annotations

import abc
import typing as t

from fibomat.arrangements.arrangementbase import ArrangementBase, BBoxT, ElementT, VectorT
from fibomat.linalg import DimTransformable


__all__ = ['GroupBase']


class GroupBase(ArrangementBase[ElementT, VectorT, BBoxT], abc.ABC):
    """Base class of :class:`~fibomat.arrangements.Group` and :class:`~fibomat.arrangements.DimGroup`.

    A group holds a (not empty) sequence of elements, which are transformed together. The elements are not copied; the
    transformation methods of the group return new groups with transformed copies of the elements.
    """

    _dimensioned: t.ClassVar[bool]
    """True if the elements must have units (derived from :class:`~fibomat.linalg.DimTransformable`)."""

    def __init__(self, elements: t.Iterable[ElementT], description: t.Optional[str] = None):
        """
        Args:
            elements (Iterable): elements of the group
            description (str, optional): description

        Raises:
            ValueError: Raised if there are no elements.
            TypeError: Raised if an element is not transformable, or has (not) a unit although the group needs
                elements which do not (do) have one.
        """
        elements = tuple(elements)

        if not elements:
            raise ValueError('A group needs at least one element.')

        for element in elements:
            self._check_element(element)

        self._elements: t.Tuple[ElementT, ...] = elements

        super().__init__(description=description)

    @classmethod
    def _check_element(cls, element: t.Any) -> None:
        """Check that an element can be an element of the group.

        Raises:
            TypeError: Raised if it cannot.
        """
        from fibomat.linalg import Transformable  # pylint: disable=import-outside-toplevel

        if not isinstance(element, Transformable):
            raise TypeError(f'The elements of a group must be transformable, got {type(element).__name__}.')

        if cls._dimensioned and not isinstance(element, DimTransformable):
            raise TypeError(
                f'The elements of a {cls.__name__} need units (e.g. shape * unit("µm")), got {type(element).__name__}.'
            )
        if not cls._dimensioned and isinstance(element, DimTransformable):
            raise TypeError(
                f'The elements of a {cls.__name__} must not have units, got {type(element).__name__}; '
                'use a DimGroup.'
            )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n_elements={len(self._elements)})'

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> t.Iterator[ElementT]:
        return iter(self._elements)

    def __getitem__(self, index: t.Any) -> t.Any:
        return self._elements[index]

    @property
    def elements(self) -> t.Tuple[ElementT, ...]:
        """The (direct) elements of the group.

        Access:
            get
        """
        return self._elements

    def _arrangement_elements(self) -> t.Iterator[ElementT]:
        yield from self._elements

    @property
    def center(self) -> VectorT:
        """The mean of the centers of the elements.

        Access:
            get
        """
        center = self._elements[0].center  # type: ignore[attr-defined]
        for element in self._elements[1:]:
            center = center + element.center  # type: ignore[attr-defined]

        return center / len(self._elements)

    @property
    def bounding_box(self) -> BBoxT:
        """The bounding box of all elements.

        Access:
            get
        """
        bbox = self._elements[0].bounding_box  # type: ignore[attr-defined]
        for element in self._elements[1:]:
            bbox = bbox.extended(element.bounding_box)  # type: ignore[attr-defined]

        return bbox

    # (the elements of the clone are copies, so they can be transformed in-place)

    def _impl_translate(self, trans_vec: VectorT) -> None:
        for element in self._elements:
            element._impl_translate(trans_vec)  # type: ignore[attr-defined]  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        for element in self._elements:
            element._impl_rotate(theta)  # type: ignore[attr-defined]  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        for element in self._elements:
            element._impl_scale(fac)  # type: ignore[attr-defined]  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: VectorT) -> None:
        for element in self._elements:
            element._impl_mirror(mirror_axis)  # type: ignore[attr-defined]  # pylint: disable=protected-access
