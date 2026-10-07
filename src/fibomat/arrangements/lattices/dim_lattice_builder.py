"""Provide the :class:`DimLatticeBuilder` class.

Example:
    >>> from fibomat.arrangements import DimLatticeBuilder
    >>> from fibomat.shapes import Circle
    >>> from fibomat.units import unit
    >>> builder = DimLatticeBuilder.rectangular(2, 1, 10. * unit('µm'), 10. * unit('µm'))
    >>> builder[1, 0] = Circle(1.) * unit('µm')
    >>> builder.to_lattice().n_points
    1
"""
from __future__ import annotations

import typing as t

from fibomat.arrangements.lattices.dim_lattice import DimLattice
from fibomat.arrangements.lattices.lattice_builder_base import LatticeBuilderBase
from fibomat.linalg import DimVector, DimVectorLike, Vector
from fibomat.units import DimFloat, has_length_dim


__all__ = ['DimLatticeBuilder']


class DimLatticeBuilder(LatticeBuilderBase):
    """Builder of a :class:`~fibomat.arrangements.DimLattice` (a lattice with units), in which the element of every
    lattice point is set individually (``builder[i_u, i_v] = element``)."""

    _VectorClass = DimVector

    def __init__(self, nu: int, nv: int, u: DimVectorLike, v: DimVectorLike, center: t.Optional[DimVectorLike] = None):
        """
        Args:
            nu (int): number of lattice points along `u` (at least 1)
            nv (int): number of lattice points along `v` (at least 1)
            u (DimVectorLike): first lattice vector
            v (DimVectorLike): second lattice vector
            center (DimVectorLike, optional): center of the lattice, default (0, 0)
        """
        super().__init__(nu, nv, u, v, center)

    @classmethod
    def rectangular(
        cls, nu: int, nv: int, pitch_x: DimFloat[t.Any], pitch_y: DimFloat[t.Any],
        center: t.Optional[DimVectorLike] = None
    ) -> DimLatticeBuilder:
        """Builder of a lattice with orthogonal lattice vectors (`u` to the right, `v` down).

        Args:
            nu (int): number of lattice points in x direction
            nv (int): number of lattice points in y direction
            pitch_x (DimFloat): distance of the points in x direction
            pitch_y (DimFloat): distance of the points in y direction
            center (DimVectorLike, optional): center of the lattice, default (0, 0)

        Returns:
            DimLatticeBuilder

        Raises:
            TypeError: Raised if a pitch is no dimensioned value.
            ValueError: Raised if a pitch is no length.
        """
        for name, pitch in (('pitch_x', pitch_x), ('pitch_y', pitch_y)):
            if not isinstance(pitch, DimFloat):
                raise TypeError(f'{name} must be a dimensioned value like 5. * unit("µm"), got {type(pitch).__name__}.')
            if not has_length_dim(pitch):
                raise ValueError(f'{name} must have dimension [length].')

        zero = 0. * pitch_x
        return cls(nu, nv, DimVector(pitch_x, zero), DimVector(zero, -pitch_y), center)

    def _check_element(self, element: t.Any) -> None:
        DimLattice._check_element(element)  # pylint: disable=protected-access

    def to_lattice(self, sort_by: str = 'added', description: t.Optional[str] = None) -> DimLattice:
        """Create the lattice.

        Args:
            sort_by (str): order of the elements: ``'added'`` (the order in which they were set), ``'row_col'`` (row by
                row, `u` changes fastest) or ``'col_row'`` (column by column).
            description (str, optional): description

        Returns:
            DimLattice

        Raises:
            ValueError: Raised if no element was set or `sort_by` is unknown.
        """
        elements, keys = self._sorted(sort_by)
        if not elements:
            raise ValueError('No lattice point has an element.')
        return DimLattice(elements, keys, self._u, self._v, description)
