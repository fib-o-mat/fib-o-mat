from typing import Iterable, Union, Callable, Optional, Tuple, List

from fibomat.composite_shapes import HollowArcSpline
from fibomat.shapes import ArcSplineCompatible, Rect
from fibomat.linalg import Transformable, Vector, VectorLike
from fibomat.arrangements.lattices.lattice_base import LatticeBaseMixin
from fibomat.arrangements.groups.group import Group
from fibomat.units import U_

import numpy as np


class Lattice(Group, LatticeBaseMixin):
    def __init__(self, elements: List[Transformable], elements_by_uv: List[Transformable], description: Optional[str] = None):
        super().__init__(elements, description)

        self.elements_by_uv = elements_by_uv

    @classmethod
    def generate_rect(
        cls,
        nu: int, nv: int,
        du: float, dv: float,
        element: Union[Transformable, Callable],
        center: Optional[VectorLike] = None,
        predicate: Optional[Union[Callable, List[Callable]]] = None,
        explode: bool = False,
        remove_outliers: bool = False,
    ):
        nu = int(nu)
        nv = int(nv)

        if nu < 1 or nv < 1:
            raise ValueError('nu and nv must be at least 1.')

        du = float(du)
        dv = float(dv)

        seed_u = du / 2 if nu % 2 == 0 else 0
        seed_v = dv / 2 if nv % 2 == 0 else 0

        seed = Vector(seed_u, seed_v)

        return cls.generate(
            Rect(width=du*nu, height=dv*nv), (du, 0), (0, -dv), element, center, predicate, explode, remove_outliers, seed
        )

    @classmethod
    def generate(
        cls,
        boundary: Union[HollowArcSpline, ArcSplineCompatible],
        u: VectorLike, v: VectorLike,
        element: Union[Transformable, Callable[[Tuple[float, float], Tuple[int, int]], Optional[Transformable]]],
        center: Optional[VectorLike] = None,
        predicate: Optional[Union[Callable, List[Callable]]] = None,
        explode: bool = False,
        remove_outliers: bool = False,
        seed: Optional[VectorLike] = None
        # break_arrangements: bool = False
    ):
        u = Vector(u)
        v = Vector(v)

        center = Vector(center)

        if callable(element):
            element_gen = element
        else:
            def element_gen(*args):
                return element

        if seed is None:
            seed = center
        else:
            seed = Vector(seed)

        elements, elements_by_uv = cls._generate_impl(
            boundary, u, v, center, element_gen, predicate, explode, remove_outliers, lambda x: x, lambda x: x, seed
        )

        return cls(elements, elements_by_uv)

    def __mul__(self, other):
        if isinstance(other, U_):
            from fibomat.arrangements.lattices.dim_lattice import DimLattice

            # mapper = np.vectorize(lambda elem: elem * other)

            def mapper_impl(elem):
                if elem:
                    return elem * other
                else:
                    return None

            mapper = np.vectorize(mapper_impl, otypes=[object])
            
            elements = list(mapper(self.elements))
            elements_by_uv = np.vectorize(mapper)(self.elements_by_uv)

            return DimLattice(elements, elements_by_uv, description=self.description)
        raise NotImplementedError

