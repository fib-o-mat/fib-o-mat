"""Two dimensional vectors: :class:`Vector` (floats) and :class:`DimVector` (lengths)."""
from typing import Iterable, Union

from fibomat.linalg.vectors.vector import Vector, VectorValueError, FloatTypes
from fibomat.linalg.vectors.dim_vector import DimVector
from fibomat.linalg.vectors.vector_functions import angle_between, signed_angle_between

from fibomat.units import DimFloat, LengthDimension, LengthQuantity


__all__ = [
    'Vector', 'VectorLike', 'DimVector', 'DimVectorLike', 'VectorValueError', 'FloatTypes', 'angle_between',
    'signed_angle_between',
]


VectorLike = Union[Vector, Iterable[float]]
"""vector-like objects for type hints"""

DimVectorLike = Union[DimVector, Iterable[DimFloat[LengthDimension]], Iterable[LengthQuantity]]
"""dim-vector-like objects for type hints (iterables of pint quantities are deprecated)"""
