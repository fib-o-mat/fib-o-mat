from typing import Optional

from fibomat.arrangements.lattices.lattice_builder_base import LatticeBuilderBase
from fibomat.arrangements.lattices.dim_lattice import DimLattice
from fibomat.linalg import VectorLike, DimVector, DimVectorLike
from fibomat.units import LengthQuantity, U_, Q_, has_length_dim


class DimLatticeBuilder(LatticeBuilderBase):
    _VectorClass = DimVector

    def __init__(self, nu: int, nv: int, u: DimVectorLike, v: DimVectorLike, center: Optional[DimVectorLike] = None):
        super().__init__(nu, nv, u, v, center)

    @classmethod
    def generate_rect(cls, nu: int, nv: int, du: LengthQuantity, dv: LengthQuantity, center: Optional[VectorLike] = None):
        if not isinstance(dv, Q_) or not has_length_dim(dv) or not isinstance(dv.m, (int, float)):
            raise TypeError('dv must be scalar with dimension [length]')

        if not isinstance(du, Q_) or not has_length_dim(du) or not isinstance(du.m, (int, float)):
            raise TypeError('du must be scalar with dimension [length]')

        return cls(nu, nv, DimVector(du, 0*U_('µm')), DimVector(0*U_('µm'), -dv), center)

    def to_lattice(self, sort_by: str = 'added') -> DimLattice:
        if sort_by == 'added':
            return DimLattice(self._elements, None)
        if sort_by == 'row_col':
            return DimLattice(sorted(
                sorted(self._elements, key=lambda elem: elem.center.x),
                key=lambda elem:elem.center.y, reverse=True
            ), None)
        if sort_by == 'col_row':
            return DimLattice(sorted(
                sorted(self._elements, key=lambda elem: elem.center.y, reverse=True),
                key=lambda elem: elem.center.x
            ), None)
        else:
            raise ValueError(f'Unknown sort type "{sort_by}"')