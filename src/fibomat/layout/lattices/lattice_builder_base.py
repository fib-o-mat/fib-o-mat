from typing import Union, Optional, Type, Tuple
import abc

from fibomat.linalg import (
    VectorLike,
    Vector,
    DimVector,
    VectorValueError,
    TransformableBase,
)
from fibomat.layout.utils import _check_lattice_vectors


class LatticeBuilderBase(abc.ABC):
    _VectorClass: Type[Union[Vector, DimVector]]

    def __init__(
        self,
        nu: int,
        nv: int,
        u: Union[VectorLike, DimVector],
        v: Union[VectorLike, DimVector],
        center: Optional[Union[Vector, DimVector]] = None,
    ):
        self._nu = int(nu)
        self._nv = int(nv)

        if self._nu < 1 or self._nv < 1:
            raise ValueError("nu and nv must be at least 1.")

        self._u = self._VectorClass(u)
        self._v = self._VectorClass(v)

        _check_lattice_vectors(self._u, self._v)

        self._offset = (
            -(self._nu - 1) / 2 * self._u
            - (self._nv - 1) / 2 * self._v
            + self._VectorClass(center)
        )

        self._elements = []
        self._added_sites = set()

    @property
    def nu(self):
        return self._nu

    @property
    def nv(self):
        return self._nv

    def pos(self, key: Tuple[int, int]):
        if (
            not isinstance(key, tuple)
            or not len(key) == 2
            or not isinstance(key[0], int)
            or not isinstance(key[1], int)
        ):
            raise TypeError("key must be Tuple[int, int].")

        if key[0] >= self._nu or key[1] >= self._nv:
            raise ValueError("key[0] >= nu or key[1] >= nv")

        return key[0] * self._u + key[1] * self._v + self._offset

    def __setitem__(self, key: Tuple[int, int], element: TransformableBase):
        if (
            not isinstance(key, tuple)
            or not len(key) == 2
            or not isinstance(key[0], int)
            or not isinstance(key[1], int)
        ):
            raise TypeError("key must be Tuple[int, int].")

        if key in self._added_sites:
            raise ValueError("Lattice site is already set.")

        if key[0] >= self._nu or key[1] >= self._nv:
            raise ValueError("key[0] >= nu or key[1] >= nv")

        pos = self.pos(key)

        try:
            self._elements.append(element.translated_to(pos))
        except VectorValueError as vec_val_error:
            raise RuntimeError(
                "Maybe you tried to add a dimensioned object to a non dimensioned lattice builder (or vice versa)?"
            ) from vec_val_error

        self._added_sites.add(key)
