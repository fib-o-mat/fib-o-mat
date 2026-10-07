from typing import Optional

from fibomat.arrangements.lattices.lattice_builder_base import LatticeBuilderBase
from fibomat.arrangements.lattices.lattice import Lattice
from fibomat.linalg import VectorLike, Vector


class LatticeBuilder(LatticeBuilderBase):
    _VectorClass = Vector

    def __init__(self, nu: int, nv: int, u: VectorLike, v: VectorLike, center: Optional[VectorLike] = None):
        super().__init__(nu, nv, u, v, center)

    @classmethod
    def generate_rect(cls, nu: int, nv: int, du: float, dv: float, center: Optional[VectorLike] = None):
        return cls(nu, nv, (float(du), 0), (0, -float(dv)), center)

    def to_lattice(self) -> Lattice:
        return Lattice(self._elements)

