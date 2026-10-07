"""Provide the :class:`LatticeBuilderBase` class, the common code of the lattice builders."""
from __future__ import annotations

import abc
import operator
import typing as t

import numpy as np

from fibomat.arrangements.utils import check_lattice_vectors


__all__ = ['LatticeBuilderBase']


class LatticeBuilderBase(abc.ABC):
    """Base class of the builders of lattices, in which the element of every lattice point is set individually.

    The lattice is centered at `center`; the elements are moved to the lattice points (``pos((i_u, i_v))``).
    """

    _VectorClass: t.Type[t.Any]
    """Vector type of the builder (:class:`~fibomat.linalg.Vector` or :class:`~fibomat.linalg.DimVector`)."""

    def __init__(self, nu: int, nv: int, u: t.Any, v: t.Any, center: t.Optional[t.Any] = None):
        """
        Args:
            nu (int): number of lattice points along `u` (at least 1)
            nv (int): number of lattice points along `v` (at least 1)
            u (VectorLike): first lattice vector
            v (VectorLike): second lattice vector
            center (VectorLike, optional): center of the lattice, default (0, 0)

        Raises:
            ValueError: Raised if a number of points is less than 1 or the lattice vectors are collinear or null
                vectors.
        """
        if int(nu) != nu or int(nv) != nv or nu < 1 or nv < 1:
            raise ValueError(f'nu and nv must be integers which are at least 1, got {nu}, {nv}.')

        self._nu = int(nu)
        self._nv = int(nv)

        self._u = self._VectorClass(u)
        self._v = self._VectorClass(v)

        check_lattice_vectors(np.asarray(self._u_unitless()), np.asarray(self._v_unitless()))

        center_vec = self._VectorClass() if center is None else self._VectorClass(center)
        self._offset = center_vec - (self._nu - 1) / 2 * self._u - (self._nv - 1) / 2 * self._v

        self._elements: t.List[t.Any] = []
        self._keys: t.List[t.Tuple[int, int]] = []

    def _u_unitless(self) -> t.Any:
        return self._u.vector if hasattr(self._u, 'vector') else self._u

    def _v_unitless(self) -> t.Any:
        return self._v.vector if hasattr(self._v, 'vector') else self._v

    @property
    def nu(self) -> int:
        """Number of lattice points along `u`.

        Access:
            get
        """
        return self._nu

    @property
    def nv(self) -> int:
        """Number of lattice points along `v`.

        Access:
            get
        """
        return self._nv

    def __len__(self) -> int:
        """Number of lattice points which have an element."""
        return len(self._elements)

    def _check_key(self, key: t.Any) -> t.Tuple[int, int]:
        """Convert a key to a tuple of two integers.

        Raises:
            TypeError: Raised if the key is no tuple of two integers.
            ValueError: Raised if the key is outside of the lattice.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError('key must be a tuple of two integers (i_u, i_v).')
        try:
            i_u, i_v = operator.index(key[0]), operator.index(key[1])
        except TypeError:
            raise TypeError('key must be a tuple of two integers (i_u, i_v).') from None

        if not (0 <= i_u < self._nu and 0 <= i_v < self._nv):
            raise ValueError(f'The lattice point {key} is outside of the lattice ({self._nu} x {self._nv} points).')

        return i_u, i_v

    def pos(self, key: t.Tuple[int, int]) -> t.Any:
        """Position of a lattice point.

        Args:
            key (Tuple[int, int]): indices (i_u, i_v) of the lattice point

        Returns:
            Vector: position

        Raises:
            TypeError: Raised if the key is no tuple of two integers.
            ValueError: Raised if the point is outside of the lattice.
        """
        i_u, i_v = self._check_key(key)
        return i_u * self._u + i_v * self._v + self._offset

    @abc.abstractmethod
    def _check_element(self, element: t.Any) -> None:
        """Raise a TypeError if `element` cannot be an element of the lattice."""

    def __setitem__(self, key: t.Tuple[int, int], element: t.Any) -> None:
        """Set the element of a lattice point; the element is moved (with its pivot) to the point.

        Args:
            key (Tuple[int, int]): indices (i_u, i_v) of the lattice point
            element (Transformable): element

        Raises:
            TypeError: Raised if the key is no tuple of two integers or the element is not valid (e.g. a dimensioned
                element for a lattice without units).
            ValueError: Raised if the point is outside of the lattice or already has an element.
        """
        index = self._check_key(key)
        self._check_element(element)

        if index in self._keys:
            raise ValueError('Lattice site is already set.')

        self._elements.append(element.translated_to(self.pos(index)))
        self._keys.append(index)

    def _sorted(self, sort_by: str) -> t.Tuple[t.List[t.Any], t.List[t.Tuple[int, int]]]:
        """The elements and their keys in the requested order."""
        if sort_by == 'added':
            order = list(range(len(self._keys)))
        elif sort_by == 'row_col':  # u changes fastest
            order = sorted(range(len(self._keys)), key=lambda i: (self._keys[i][1], self._keys[i][0]))
        elif sort_by == 'col_row':  # v changes fastest
            order = sorted(range(len(self._keys)), key=lambda i: self._keys[i])
        else:
            raise ValueError(f'Unknown sort type "{sort_by}", expected "added", "row_col" or "col_row".')

        return [self._elements[i] for i in order], [self._keys[i] for i in order]
