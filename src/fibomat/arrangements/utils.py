# from fibomat.shapes.dim_shape import DimShape
from typing import List, Tuple

import numpy as np

# from fibomat.shapes import Shape, DimShape
# from fibomat.layout.pattern import Pattern
# from fibomat.linalg import Vector


# def _have_same_type(objs: List) -> bool:
#     objs_iter = iter(objs)
#     first_elem = next(objs_iter)

#     if isinstance(first_elem, Shape):
#         base = Shape
#     elif isinstance(first_elem, DimShape):
#         base = DimShape
#     elif isinstance(first_elem, Pattern):
#         base = Pattern
#     else:
#         base = type(first_elem)

#     print(base, [isinstance(obj, base) for obj in objs_iter])
#     print(objs)

#     return all(isinstance(obj, base) for obj in objs_iter)


def _check_lattice_vectors(u, v) -> None:
    if np.isclose(abs(np.dot(np.array(u), np.array(v))), 1):
        raise ValueError('Lattice vectors may not be collinear.')


def _round_towards_mean(a: float, b: float) -> Tuple[int, int]:
    flipped = False
    if a > b:
        flipped = True
        a, b = b, a

    a = np.ceil(a)
    b = np.floor(b)

    return (b, a) if flipped else (a, b)
