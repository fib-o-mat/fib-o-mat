"""Provides the :class:`DimBoundingBox` class."""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg.boundingboxes.boundingbox import BoundingBox
from fibomat.linalg.vectors import DimVector, Vector


__all__ = ['DimBoundingBox']


class DimBoundingBox(BoundingBox[DimVector]):
    """Axis aligned rectangular bounding box with lengths as coordinates.

    Corners, width, height and area are :class:`DimVector` and :class:`~fibomat.units.DimFloat`.

    Example::

        box = DimBoundingBox(Vector(0, 0) * unit('µm'), Vector(2, 1) * unit('µm'))
        box.width  # 2 µm
    """

    _VectorClass = DimVector

    @classmethod
    def _from_vectors(cls, vectors: t.Sequence[DimVector]) -> DimBoundingBox:  # type: ignore[override]
        # all points are converted to the unit of the first one
        unit = vectors[0].unit
        array = np.array([np.asarray(vector.vector_as(unit)) for vector in vectors])
        return cls(
            DimVector.from_vector(Vector(np.min(array, axis=0)), unit),
            DimVector.from_vector(Vector(np.max(array, axis=0)), unit),
        )
