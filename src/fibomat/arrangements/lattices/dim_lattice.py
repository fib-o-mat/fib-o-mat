from typing import Union, Callable, Optional, Tuple, List

from fibomat.shapes import DimShape
from fibomat.linalg import Transformable, DimTransformable, Vector, DimVector, VectorLike, DimVectorLike
from fibomat.units import U_, scale_factor, LengthQuantity, has_length_dim
from fibomat.arrangements.lattices.lattice_base import LatticeBaseMixin
from fibomat.arrangements.groups.dim_group import DimGroup
from fibomat.shapes import Rect


class DimLattice(DimGroup, LatticeBaseMixin):
    def __init__(self, elements: List[DimTransformable], elements_by_uv: List[DimTransformable], description: Optional[str] = None):
        super().__init__(elements, description=description)

        self.elements_by_uv = elements_by_uv

    @classmethod
    def generate_rect(
        cls,
        nu: int, nv: int,
        dim_du: LengthQuantity, dim_dv: LengthQuantity,
        dim_element: Union[Transformable, Callable],
        dim_center: Optional[DimVectorLike] = None,
        predicate: Optional[Union[Callable, List[Callable]]] = None,
        explode: bool = False,
        remove_outliers: bool = False,
    ):
        nu = int(nu)
        nv = int(nv)

        if nu < 1 or nv < 1:
            raise ValueError('nu and nv must be at least 1.')

        if not has_length_dim(dim_du) or not has_length_dim(dim_dv):
            raise ValueError('dim_du and dim_dv must be length quantities')

        unit = dim_du.u
        du = dim_du.m
        dv = dim_dv.m_as(unit)

        seed_u = du / 2 if nu % 2 == 0 else 0
        seed_v = dv / 2 if nv % 2 == 0 else 0

        seed = Vector(seed_u, seed_v)

        # TODO: seed!
        return cls.generate(
            Rect(width=du*nu, height=dv*nv) * unit, (du, 0) * unit, (0, -dv) * unit, dim_element, dim_center, predicate, explode, remove_outliers, dim_seed=seed * unit
        )

    @classmethod
    def generate(
        cls,
        dim_boundary: DimShape,
        dim_u: VectorLike, dim_v: VectorLike,
        dim_element: Union[Transformable, Callable[[Tuple[float, float], Tuple[int, int]], Optional[DimTransformable]]],
        dim_center: Optional[VectorLike] = None,
        predicate: Optional[Union[Callable, List[Callable]]] = None,
        explode: bool = False,
        remove_outliers: bool = False,
        # break_arrangements: bool = False
        dim_seed: Optional[DimVectorLike] = None
    ):
        """
        predicate get its coordinates in µm

        Args:
            dim_boundary:
            dim_u:
            dim_v:
            dim_element:
            dim_center:
            predicate:
            explode:
            remove_outliers:

        Returns:

        """
        base_unit = U_('µm')

        def dim_vec_to_vec(dim_vec: DimVector) -> Vector:
            return scale_factor(base_unit, dim_vec.unit) * dim_vec.vector

        def vec_to_dim_vec(vec: Vector) -> DimVector:
            return vec * base_unit

        u = dim_vec_to_vec(DimVector(dim_u))
        v = dim_vec_to_vec(DimVector(dim_v))

        if dim_seed is None:
            dim_seed = dim_center
        else:
            dim_seed = DimVector(dim_seed)

        seed = dim_vec_to_vec(dim_seed)

        center = dim_vec_to_vec(DimVector(dim_center))

        boundary = dim_boundary.scaled(scale_factor(base_unit, dim_boundary.unit), 'pivot').shape

        if callable(dim_element):
            element_gen = dim_element
        else:
            def element_gen(*args):
                return dim_element

        # TODO: seed!
        elements, elements_by_uv = cls._generate_impl(
            boundary, u, v, center, element_gen, predicate, explode, remove_outliers, dim_vec_to_vec, vec_to_dim_vec, seed
        )

        return cls(elements, elements_by_uv)
