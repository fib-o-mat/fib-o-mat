"""Functions operating on two vectors."""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg.vectors.vector import Vector, VectorValueError
from fibomat.linalg.vectors.dim_vector import DimVector


__all__ = ['angle_between', 'signed_angle_between']


def _as_arrays(vec_1: t.Any, vec_2: t.Any) -> t.Tuple[np.ndarray, np.ndarray]:
    """Convert both vectors to float arrays in the same unit; raise ValueError for null vectors."""
    if isinstance(vec_1, DimVector) or isinstance(vec_2, DimVector):
        dim_1, dim_2 = DimVector(vec_1), DimVector(vec_2)
        arr_1, arr_2 = np.asarray(dim_1), dim_2.vector_as(dim_1.unit).__array__()
    else:
        arr_1, arr_2 = np.asarray(Vector(vec_1)), np.asarray(Vector(vec_2))

    if not np.any(arr_1) or not np.any(arr_2):
        raise ValueError('The angle to a null vector is not defined.')
    return arr_1, arr_2


def angle_between(vec_1: t.Any, vec_2: t.Any) -> float:
    """Return the (unsigned) angle between two vectors in rad (between 0 and pi).

    If one of the vectors is a :class:`DimVector`, both must be.

    Args:
        vec_1 (VectorLike, DimVectorLike): first vector
        vec_2 (VectorLike, DimVectorLike): second vector

    Returns:
        float

    Raises:
        VectorValueError: Raised if one of the arguments is no vector.
        ValueError: Raised if one of the vectors is the null vector.
    """
    arr_1, arr_2 = _as_arrays(vec_1, vec_2)
    cos = arr_1 @ arr_2 / (np.linalg.norm(arr_1) * np.linalg.norm(arr_2))
    # rounding errors can push the cosine slightly out of [-1, 1]
    return float(np.arccos(np.clip(cos, -1., 1.)))


def signed_angle_between(vec_1: t.Any, vec_2: t.Any) -> float:
    """Return the signed angle between two vectors in rad (between -pi and pi).

    The angle is positive if the rotation from `vec_1` to `vec_2` is in mathematically positive direction using the
    smaller angle between them.

    https://stackoverflow.com/a/16544330

    Args:
        vec_1 (VectorLike, DimVectorLike): first vector
        vec_2 (VectorLike, DimVectorLike): second vector

    Returns:
        float

    Raises:
        VectorValueError: Raised if one of the arguments is no vector.
        ValueError: Raised if one of the vectors is the null vector.
    """
    arr_1, arr_2 = _as_arrays(vec_1, vec_2)
    return float(np.arctan2(arr_1[0] * arr_2[1] - arr_1[1] * arr_2[0], arr_1 @ arr_2))
