"""Provide the :class:`DimLattice` class (a lattice with units).

Example:
    >>> from fibomat.arrangements import DimLattice
    >>> from fibomat.shapes import Circle
    >>> from fibomat.units import unit
    >>> lattice = DimLattice.from_counts(3, 2, 10. * unit('µm'), 5. * unit('µm'), Circle(1.) * unit('µm'))
    >>> lattice.n_points
    6
    >>> lattice.elements[0].center.vector_as(unit('µm'))
    Vector(x=-10.0, y=2.5)
"""
from __future__ import annotations

import typing as t

from fibomat.arrangements.groups.dim_group import DimGroup
from fibomat.arrangements.lattices.fast_axis import FastAxis
from fibomat.arrangements.lattices.lattice import to_filling_shape
from fibomat.arrangements.lattices.lattice_base import LatticeMixin, place_elements
from fibomat.arrangements.lattices.lattice_points import boundary_points, count_points, grid_points
from fibomat.linalg import DimTransformable, DimVector, DimVectorLike, Vector
from fibomat.shapes import DimShape
from fibomat.units import DimFloat, LengthUnit, has_length_dim, scale_to


__all__ = ['DimLattice']


ElementOrCallback = t.Union[
    DimTransformable, t.Callable[[int, t.Tuple[int, int], DimVector, int], t.Optional[DimTransformable]]
]


def _length(value: t.Any, unit: LengthUnit, name: str, allow_zero: bool) -> float:
    """A length as float in `unit`.

    Raises:
        TypeError: Raised if `value` is not a dimensioned value.
        ValueError: Raised if it is not a length or not positive (negative).
    """
    if not isinstance(value, DimFloat):
        raise TypeError(f'{name} must be a dimensioned value like 5. * unit("µm"), got {type(value).__name__}.')
    if not has_length_dim(value):
        raise ValueError(f'{name} must have dimension [length].')

    magnitude = scale_to(unit, value)
    if magnitude < 0. or (magnitude == 0. and not allow_zero):
        raise ValueError(f'{name} must be {"not negative" if allow_zero else "positive"}, got {value!r}.')
    return magnitude


class DimLattice(LatticeMixin, DimGroup):
    """A group of elements with units at the points of a lattice (see :class:`~fibomat.arrangements.Lattice`).

    The coordinates of the lattice points which are passed to a callback that creates the elements are
    :class:`DimVectors <fibomat.linalg.DimVector>`.
    """

    def __init__(
        self,
        elements: t.Iterable[DimTransformable],
        points_uv: t.Any,
        u: DimVectorLike,
        v: DimVectorLike,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            elements (Iterable[DimTransformable]): the elements (with units)
            points_uv (array-like): indices (i_u, i_v) of the lattice point of every element, shape (n, 2)
            u (DimVectorLike): first lattice vector
            v (DimVectorLike): second lattice vector
            description (str, optional): description

        Raises:
            ValueError: Raised if there are no elements, the indices are invalid (wrong shape, negative, repeated or not
                one for every element) or the lattice vectors are collinear or null vectors.
            TypeError: Raised if an element is no transformable or has no unit.
        """
        super().__init__(elements, description)
        self._init_lattice(points_uv, DimVector(u), DimVector(v))

    @classmethod
    def _create(
        cls,
        unit: LengthUnit,
        indices: t.Any,
        points: t.Any,
        element: ElementOrCallback,
        u: DimVector,
        v: DimVector,
        description: t.Optional[str],
        keep: t.Optional[t.Callable[[t.Any], bool]] = None,
    ) -> DimLattice:
        def to_vector(xy: t.Any) -> DimVector:
            return DimVector.from_vector(Vector(xy), unit)

        elements, kept_indices = place_elements(
            indices, points, element, to_vector, to_vector, cls._check_element, keep
        )
        if not elements:
            raise ValueError('The lattice does not contain any element.')

        return cls(elements, kept_indices, u, v, description)

    @classmethod
    def from_rect(
        cls,
        width: DimFloat[t.Any],
        height: DimFloat[t.Any],
        pitch_x: DimFloat[t.Any],
        pitch_y: DimFloat[t.Any],
        element: ElementOrCallback,
        *,
        center: t.Optional[DimVectorLike] = None,
        fast_axis: t.Union[FastAxis, str] = 'x',
        description: t.Optional[str] = None,
    ) -> DimLattice:
        """Create a lattice with orthogonal lattice vectors which has as many points as fit in a rectangle.

        The lattice is centered in the rectangle and the points on the edges are included (see
        :meth:`Lattice.from_rect`).

        Args:
            width (DimFloat): width of the rectangle
            height (DimFloat): height of the rectangle
            pitch_x (DimFloat): distance of the points in x direction
            pitch_y (DimFloat): distance of the points in y direction
            element (DimTransformable, Callable): the element at each point, or a function
                ``element(i, uv, xy, n) -> Optional[DimTransformable]`` which creates the element of a point (None:
                no element). `i` is the index of the point, `uv` its lattice indices, `xy` its coordinates (a
                :class:`~fibomat.linalg.DimVector`) and `n` the total number of points.
            center (DimVectorLike, optional): center of the rectangle, default (0, 0)
            fast_axis (FastAxis, str): ``'x'`` (default): the points are numbered row by row from the upper left,
                ``'y'``: column by column.
            description (str, optional): description

        Returns:
            DimLattice

        Raises:
            TypeError: Raised if a length is no dimensioned value or an element is not valid.
            ValueError: Raised if a length has the wrong dimension, the size is negative, a pitch is not positive, or
                there are no elements.
        """
        unit = cls._unit_of(pitch_x, 'pitch_x')
        n_x = count_points(_length(width, unit, 'width', True), _length(pitch_x, unit, 'pitch_x', False))
        n_y = count_points(_length(height, unit, 'height', True), _length(pitch_y, unit, 'pitch_y', False))
        return cls.from_counts(
            n_x, n_y, pitch_x, pitch_y, element, center=center, fast_axis=fast_axis, description=description
        )

    @staticmethod
    def _unit_of(length: t.Any, name: str) -> LengthUnit:
        """The unit of a dimensioned length (used for the calculations).

        Raises:
            TypeError: Raised if `length` is no dimensioned value.
            ValueError: Raised if it is no length.
        """
        if not isinstance(length, DimFloat):
            raise TypeError(f'{name} must be a dimensioned value like 5. * unit("µm"), got {type(length).__name__}.')
        if not has_length_dim(length):
            raise ValueError(f'{name} must have dimension [length].')
        return length.units

    @classmethod
    def from_counts(
        cls,
        n_x: int,
        n_y: int,
        pitch_x: DimFloat[t.Any],
        pitch_y: DimFloat[t.Any],
        element: ElementOrCallback,
        *,
        center: t.Optional[DimVectorLike] = None,
        fast_axis: t.Union[FastAxis, str] = 'x',
        description: t.Optional[str] = None,
    ) -> DimLattice:
        """Create a lattice with orthogonal lattice vectors and a given number of points in each direction (see
        :meth:`Lattice.from_counts`).

        Args:
            n_x (int): number of points in x direction (at least 1)
            n_y (int): number of points in y direction (at least 1)
            pitch_x (DimFloat): distance of the points in x direction
            pitch_y (DimFloat): distance of the points in y direction
            element (DimTransformable, Callable): the element at each point or a function which creates it (see
                :meth:`from_rect`)
            center (DimVectorLike, optional): center of the lattice, default (0, 0)
            fast_axis (FastAxis, str): ``'x'`` (default): the points are numbered row by row from the upper left,
                ``'y'``: column by column.
            description (str, optional): description

        Returns:
            DimLattice

        Raises:
            TypeError: Raised if a pitch is no dimensioned value or an element is not valid.
            ValueError: Raised if a number of points is less than 1, a pitch is not a positive length, or there are no
                elements.
        """
        unit = cls._unit_of(pitch_x, 'pitch_x')
        du = _length(pitch_x, unit, 'pitch_x', False)
        dv = _length(pitch_y, unit, 'pitch_y', False)

        center_vec = DimVector(center).vector_as(unit) if center is not None else Vector(0, 0)
        indices, points = grid_points(n_x, n_y, du, dv, (center_vec.x, center_vec.y), FastAxis.parse(fast_axis))

        return cls._create(
            unit, indices, points, element, DimVector.from_vector(Vector(du, 0.), unit),
            DimVector.from_vector(Vector(0., -dv), unit), description
        )

    @classmethod
    def from_boundary(
        cls,
        boundary: DimShape,
        u: DimVectorLike,
        v: DimVectorLike,
        element: ElementOrCallback,
        *,
        origin: t.Optional[DimVectorLike] = None,
        fast_axis: t.Union[FastAxis, str] = 'u',
        clip_elements: bool = False,
        description: t.Optional[str] = None,
    ) -> DimLattice:
        """Create a lattice with arbitrary lattice vectors which contains all lattice points inside of a shape (see
        :meth:`Lattice.from_boundary`).

        Args:
            boundary (DimShape): closed shape (e.g. a rectangle or a shape with holes) with a unit
            u (DimVectorLike): first lattice vector
            v (DimVectorLike): second lattice vector
            element (DimTransformable, Callable): the element at each point or a function which creates it (see
                :meth:`from_rect`)
            origin (DimVectorLike, optional): position of a lattice point, default: the center of the bounding box of the
                boundary.
            fast_axis (FastAxis, str): ``'u'`` (default): the points are numbered along `u` first, ``'v'``: along `v`
                first.
            clip_elements (bool): if True, elements which are not completely (their bounding box) inside the boundary
                are dropped.
            description (str, optional): description

        Returns:
            DimLattice

        Raises:
            TypeError: Raised if the boundary or an element is not valid.
            ValueError: Raised if the boundary is not closed, the lattice vectors are collinear or null vectors, or if
                there are no elements (no lattice point is inside of the boundary).
        """
        if not isinstance(boundary, DimShape):
            raise TypeError(f'The boundary must be a DimShape (shape * unit), got {type(boundary).__name__}.')

        unit = boundary.unit
        shape = to_filling_shape(boundary.shape)

        u_vec, v_vec = DimVector(u).vector_as(unit), DimVector(v).vector_as(unit)
        origin_vec = shape.bounding_box.center if origin is None else DimVector(origin).vector_as(unit)

        indices, points = boundary_points(
            shape, (u_vec.x, u_vec.y), (v_vec.x, v_vec.y), (origin_vec.x, origin_vec.y), FastAxis.parse(fast_axis)
        )

        keep = None
        if clip_elements:
            def keep(placed: t.Any) -> bool:  # pylint: disable=function-redefined
                return all(shape.contains(corner.vector_as(unit)) for corner in placed.bounding_box.corners)

        return cls._create(
            unit, indices, points, element, DimVector.from_vector(u_vec, unit), DimVector.from_vector(v_vec, unit),
            description, keep
        )
