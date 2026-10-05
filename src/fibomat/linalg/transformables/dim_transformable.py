"""Provides the :class:`DimTransformable` class."""
from __future__ import annotations

import abc
import typing as t

from fibomat.linalg.boundingboxes import DimBoundingBox
from fibomat.linalg.transformables.transformable import Transformable
from fibomat.linalg.vectors import DimVector


__all__ = ['DimTransformable']


class DimTransformable(Transformable[DimVector, DimBoundingBox], abc.ABC):
    """Base class providing the translate, rotate, scale and mirror transformations for objects with units.

    It derives from :class:`Transformable` with :class:`DimVector` and :class:`DimBoundingBox`; see there for the
    methods and properties which must be implemented. All vector arguments are converted to :class:`DimVector`.
    """

    _VectorClass = DimVector

    def __init__(self, description: t.Optional[str] = None):
        """
        Args:
            description (str, optional): optional description
        """
        super().__init__(description=description)
