"""Provide the :class:`ArrangementBase` class."""
from __future__ import annotations

import abc
import typing as t


__all__ = ['ArrangementBase']


ElementT = t.TypeVar('ElementT')
VectorT = t.TypeVar('VectorT')
BBoxT = t.TypeVar('BBoxT')


class ArrangementBase(t.Generic[ElementT, VectorT, BBoxT], abc.ABC):
    """Base class of all arrangements: objects which contain other objects (elements) and are transformed together
    with them, e.g. :class:`~fibomat.arrangements.Group` and :class:`~fibomat.arrangements.Lattice`.

    An arrangement can be used to arrange :class:`~fibomat.layout.site.Site`, :class:`~fibomat.layout.pattern.Pattern`
    and :class:`~fibomat.shapes.shape.Shape` objects. The elements are accessed via
    :meth:`ArrangementBase.arrangement_elements`, which yields all elements (also those of nested arrangements).

    What kind of elements are stored must be specified in the child classes.
    """

    def __init__(self, description: t.Optional[str] = None):
        """
        Args:
            description (str, optional): description
        """
        super().__init__(description=description)  # type: ignore[call-arg]  # (a Transformable follows in the MRO)

    @abc.abstractmethod
    def _arrangement_elements(self) -> t.Iterator[ElementT]:
        """The direct elements of the arrangement."""
        raise NotImplementedError

    def arrangement_elements(self) -> t.Iterator[ElementT]:
        """Access to the elements. Nested arrangements are resolved: only the elements which are no arrangements are
        yielded.

        Yields:
            Any: the elements; the type depends on the stored elements.
        """
        for element in self._arrangement_elements():
            if isinstance(element, ArrangementBase):
                yield from element.arrangement_elements()
            else:
                yield element
