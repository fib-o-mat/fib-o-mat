"""Provide the :class:`LatticeBuilder` class.

Example:
    >>> from fibomat.arrangements import LatticeBuilder
    >>> from fibomat.shapes import Circle, Rect
    >>> builder = LatticeBuilder.rectangular(2, 2, 10., 10.)
    >>> builder[0, 0] = Circle(1.)
    >>> builder[1, 1] = Rect(2, 2)
    >>> lattice = builder.to_lattice()
    >>> lattice.points_uv.tolist()
    [[0, 0], [1, 1]]
"""
from __future__ import annotations

import typing as t

from fibomat.arrangements.lattices.lattice import Lattice
from fibomat.arrangements.lattices.lattice_builder_base import LatticeBuilderBase
from fibomat.linalg import Vector, VectorLike


__all__ = ['LatticeBuilder']


class LatticeBuilder(LatticeBuilderBase):
    """Builder of a :class:`~fibomat.arrangements.Lattice`, in which the element of every lattice point is set
    individually (``builder[i_u, i_v] = element``)."""

    _VectorClass = Vector

    def __init__(self, nu: int, nv: int, u: VectorLike, v: VectorLike, center: t.Optional[VectorLike] = None):
        """
        Args:
            nu (int): number of lattice points along `u` (at least 1)
            nv (int): number of lattice points along `v` (at least 1)
            u (VectorLike): first lattice vector
            v (VectorLike): second lattice vector
            center (VectorLike, optional): center of the lattice, default (0, 0)
        """
        super().__init__(nu, nv, u, v, center)

    @classmethod
    def rectangular(
        cls, nu: int, nv: int, pitch_x: float, pitch_y: float, center: t.Optional[VectorLike] = None
    ) -> LatticeBuilder:
        """Builder of a lattice with orthogonal lattice vectors (`u` to the right, `v` down, like the lattices of
        :meth:`Lattice.from_counts`).

        Args:
            nu (int): number of lattice points in x direction
            nv (int): number of lattice points in y direction
            pitch_x (float): distance of the points in x direction
            pitch_y (float): distance of the points in y direction
            center (VectorLike, optional): center of the lattice, default (0, 0)

        Returns:
            LatticeBuilder
        """
        return cls(nu, nv, (float(pitch_x), 0.), (0., -float(pitch_y)), center)

    def _check_element(self, element: t.Any) -> None:
        Lattice._check_element(element)  # pylint: disable=protected-access

    def to_lattice(self, sort_by: str = 'added', description: t.Optional[str] = None) -> Lattice:
        """Create the lattice.

        Args:
            sort_by (str): order of the elements: ``'added'`` (the order in which they were set), ``'row_col'`` (row by
                row, `u` changes fastest) or ``'col_row'`` (column by column).
            description (str, optional): description

        Returns:
            Lattice

        Raises:
            ValueError: Raised if no element was set or `sort_by` is unknown.
        """
        elements, keys = self._sorted(sort_by)
        if not elements:
            raise ValueError('No lattice point has an element.')
        return Lattice(elements, keys, self._u, self._v, description)
