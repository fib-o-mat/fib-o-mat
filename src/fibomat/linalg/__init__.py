"""
This submodule provides :class:`~fibomat.linalg.linalg.Vector` and :class:`~fibomat.linalg.vectorarray.VectorArray`
classes with some utility functions.

Example ::

    import numpy as np
    from fibomat import linalg

    v = linalg.Vector(1, 2)
    u = linalg.Vector(r=1, phi=np.pi/3)  # polar form

    # access to elements
    v_x = v.x
    # or
    v_x = v[0]

    v_r = v.r
    v_phi = v.phi

    # changing the linalg
    v_x = 1
    v_phi = 2 * np.phi/3

    v.rotate(np.pi/4, center=(1, 1)).scale((2, 2)).translate((-1, 2))

    # use it for calculations
    w = v + w
    w *= 2

    # ...

Nearly the same can be done with the :class:`~fibomat.linalg.vectorarray.VectorArray` class, which is a list of vectors.
Example ::

    import numpy as np
    from fibomat import linalg

    va = linalg.VectorArray((1, 1), (2, 2), (3, 3))
    # or
    va = linalg.VectorArray([(1, 1), (2, 2), (3, 3)])
    # or
    va = linalg.VectorArray(np.array([(1, 1), (2, 2), (3, 3)]))

    # access to elements
    va_x = va.x  # list of x values

    v = va[0]  # returns the first linalg
    v_x = va[0].x  # or v_x = va[0][0]

    # changing the linalg array
    # all operations are applied to each linalg of the list

    va.rotated(np.pi/4, center=(1, 1)).scale((2, 2)).translate((-1, 2))
"""

from fibomat.linalg.vectors import (
    Vector, VectorLike, DimVector, DimVectorLike, VectorValueError, angle_between, signed_angle_between
)
# from fibomat.linalg.vectorarray import VectorArray, VectorArrayLike, VectorArrayValueError
from fibomat.linalg.transformables.transformable import Transformable
from fibomat.linalg.transformables.dim_transformable import DimTransformable
from fibomat.linalg.transformables.transformation_builder import translate, rotate, scale, mirror
# from fibomat.linalg.dimvector import DimVector, DimVectorLike
from fibomat.linalg.boundingboxes import BoundingBox, DimBoundingBox


__all__ = [
    'Vector', 'VectorLike', 'DimVector', 'DimVectorLike', 'VectorValueError', 'angle_between', 'signed_angle_between',
    'Transformable', 'DimTransformable', 'translate', 'rotate', 'mirror', 'scale',
    'BoundingBox', 'DimBoundingBox'
]
