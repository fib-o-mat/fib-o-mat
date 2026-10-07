"""Provide the :class:`Lattice` class.

A lattice is a group of elements (shapes, groups, ...) which are placed at the points of a lattice. There are three ways
to create one; the elements are either one shape (which is copied to all points) or a function which creates an
element for every point:

* :meth:`Lattice.from_rect`: all points of a lattice with orthogonal lattice vectors which fit in a rectangle,
* :meth:`Lattice.from_counts`: a given number of points in every direction (orthogonal lattice vectors),
* :meth:`Lattice.from_boundary`: all points of a lattice (arbitrary lattice vectors) inside of a shape.

Example:
    >>> from fibomat.arrangements import Lattice
    >>> from fibomat.shapes import Circle
    >>> lattice = Lattice.from_counts(3, 2, 10., 5., Circle(1.))
    >>> lattice.n_points
    6
    >>> lattice.points_uv.tolist()  # the points are ordered row by row ("fast axis" x)
    [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    >>> lattice.elements[0].center
    Vector(x=-10.0, y=2.5)

    The element can be a function, e.g. to place different shapes:

    >>> def element(i, uv, xy, n):
    ...     return Circle(1.) if (uv[0] + uv[1]) % 2 == 0 else None  # a checkerboard
    >>> Lattice.from_counts(3, 2, 10., 5., element).n_points
    3
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.arrangements.groups.group import Group
from fibomat.arrangements.lattices.fast_axis import FastAxis
from fibomat.arrangements.lattices.lattice_base import LatticeMixin, place_elements
from fibomat.arrangements.lattices.lattice_points import boundary_points, count_points, grid_points
from fibomat.linalg import Transformable, Vector, VectorLike


__all__ = ['Lattice', 'to_filling_shape']


ElementOrCallback = t.Union[
    Transformable, t.Callable[[int, t.Tuple[int, int], t.Tuple[float, float], int], t.Optional[Transformable]]
]


def to_filling_shape(shape: t.Any) -> t.Any:
    """Convert a shape to the closed curve (with holes) which is used as boundary of a lattice.

    Args:
        shape: a :class:`~fibomat.composite_shapes.HollowArcSpline`, a shape with ``to_hollow_arc_spline()`` (e.g. a
            ring) or a shape which can be converted to an arc spline (e.g. a rectangle).

    Returns:
        ArcSpline, HollowArcSpline: closed shape

    Raises:
        TypeError: Raised if the shape cannot be used as boundary.
        ValueError: Raised if the shape is not closed.
    """
    from fibomat.composite_shapes import HollowArcSpline  # pylint: disable=import-outside-toplevel
    from fibomat.shapes import ArcSplineCompatible  # pylint: disable=import-outside-toplevel

    if isinstance(shape, HollowArcSpline):
        converted = shape
    elif callable(getattr(shape, 'to_hollow_arc_spline', None)):
        converted = shape.to_hollow_arc_spline()
    elif isinstance(shape, ArcSplineCompatible):
        converted = shape.to_arc_spline()
    else:
        raise TypeError(f'The boundary must be a closed shape (e.g. a Rect), got {type(shape).__name__}.')

    if not converted.is_closed:
        raise ValueError('The boundary must be a closed shape.')

    return converted


class Lattice(LatticeMixin, Group):
    """A group of elements at the points of a lattice (without units).

    The point (i_u, i_v) of a lattice with the lattice vectors `u` and `v` is at ``origin + i_u * u + i_v * v``. The
    indices (``points_uv``) start at 0. The elements have the order of the points; for the lattices which are created
    with :meth:`from_rect` and :meth:`from_counts` ``u`` points to the right and ``v`` down, so the point (0, 0) is the
    upper left point.

    Use the class methods to create a lattice. If a lattice is transformed, all of its elements and its lattice
    vectors are transformed.
    """

    def __init__(
        self,
        elements: t.Iterable[Transformable],
        points_uv: t.Any,
        u: VectorLike,
        v: VectorLike,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            elements (Iterable[Transformable]): the elements (without units)
            points_uv (array-like): indices (i_u, i_v) of the lattice point of every element, shape (n, 2)
            u (VectorLike): first lattice vector
            v (VectorLike): second lattice vector
            description (str, optional): description

        Raises:
            ValueError: Raised if there are no elements, the indices are invalid (wrong shape, negative, repeated or not
                one for every element) or the lattice vectors are collinear or null vectors.
            TypeError: Raised if an element is no transformable or has a unit.
        """
        super().__init__(elements, description)
        self._init_lattice(points_uv, Vector(u), Vector(v))

    # creation

    @classmethod
    def _create(
        cls,
        indices: np.ndarray,
        points: np.ndarray,
        element: ElementOrCallback,
        u: Vector,
        v: Vector,
        description: t.Optional[str],
        keep: t.Optional[t.Callable[[t.Any], bool]] = None,
    ) -> Lattice:
        elements, kept_indices = place_elements(
            indices, points, element, lambda xy: Vector(xy), lambda xy: (float(xy[0]), float(xy[1])),
            cls._check_element, keep
        )
        if not elements:
            raise ValueError('The lattice does not contain any element.')

        return cls(elements, kept_indices, u, v, description)

    @classmethod
    def from_rect(
        cls,
        width: float,
        height: float,
        pitch_x: float,
        pitch_y: float,
        element: ElementOrCallback,
        *,
        center: t.Optional[VectorLike] = None,
        fast_axis: t.Union[FastAxis, str] = 'x',
        description: t.Optional[str] = None,
    ) -> Lattice:
        """Create a lattice with orthogonal lattice vectors which has as many points as fit in a rectangle.

        The lattice is centered in the rectangle; the points on the edges of the rectangle are included, e.g. a rectangle
        with the width 10 and the pitch 2 contains 6 points (at -5, -3, -1, 1, 3 and 5 relative to the center), and with
        the pitch 3 four points (at -4.5, -1.5, 1.5 and 4.5).

        Args:
            width (float): width of the rectangle
            height (float): height of the rectangle
            pitch_x (float): distance of the points in x direction (the magnitude of the lattice vector `u`)
            pitch_y (float): distance of the points in y direction (the magnitude of the lattice vector `v`)
            element (Transformable, Callable): the element at each point, or a function
                ``element(i, uv, xy, n) -> Optional[Transformable]`` which creates the element of a point (None: no
                element). `i` is the index of the point, `uv` its lattice indices (i_u, i_v), `xy` its coordinates
                (x, y) and `n` the total number of points. Elements are moved with their pivot to their point.
            center (VectorLike, optional): center of the rectangle, default (0, 0)
            fast_axis (FastAxis, str): ``'x'`` (default): the points are numbered row by row from the upper left,
                ``'y'``: column by column.
            description (str, optional): description

        Returns:
            Lattice

        Raises:
            ValueError: Raised if the size is negative or a pitch is not positive, or if there are no elements.
            TypeError: Raised if an element is not valid.
        """
        return cls.from_counts(
            count_points(float(width), float(pitch_x)), count_points(float(height), float(pitch_y)),
            pitch_x, pitch_y, element, center=center, fast_axis=fast_axis, description=description
        )

    @classmethod
    def from_counts(
        cls,
        n_x: int,
        n_y: int,
        pitch_x: float,
        pitch_y: float,
        element: ElementOrCallback,
        *,
        center: t.Optional[VectorLike] = None,
        fast_axis: t.Union[FastAxis, str] = 'x',
        description: t.Optional[str] = None,
    ) -> Lattice:
        """Create a lattice with orthogonal lattice vectors and a given number of points in each direction.

        The lattice is centered at `center`. The lattice vector `u` points to the right and `v` down, the point (0, 0) is
        the upper left point.

        Args:
            n_x (int): number of points in x direction (at least 1)
            n_y (int): number of points in y direction (at least 1)
            pitch_x (float): distance of the points in x direction
            pitch_y (float): distance of the points in y direction
            element (Transformable, Callable): the element at each point or a function which creates it (see
                :meth:`from_rect`)
            center (VectorLike, optional): center of the lattice, default (0, 0)
            fast_axis (FastAxis, str): ``'x'`` (default): the points are numbered row by row from the upper left,
                ``'y'``: column by column.
            description (str, optional): description

        Returns:
            Lattice

        Raises:
            ValueError: Raised if a number of points is less than 1, a pitch is not positive, or if there are no
                elements.
            TypeError: Raised if an element is not valid.
        """
        center_vec = Vector(0, 0) if center is None else Vector(center)
        indices, points = grid_points(
            n_x, n_y, float(pitch_x), float(pitch_y), (center_vec.x, center_vec.y), FastAxis.parse(fast_axis)
        )
        return cls._create(
            indices, points, element, Vector(float(pitch_x), 0.), Vector(0., -float(pitch_y)), description
        )

    @classmethod
    def from_boundary(
        cls,
        boundary: t.Any,
        u: VectorLike,
        v: VectorLike,
        element: ElementOrCallback,
        *,
        origin: t.Optional[VectorLike] = None,
        fast_axis: t.Union[FastAxis, str] = 'u',
        clip_elements: bool = False,
        description: t.Optional[str] = None,
    ) -> Lattice:
        """Create a lattice with arbitrary lattice vectors which contains all lattice points inside of a shape.

        Points on the boundary of the shape are not included. The indices of the lattice points are shifted, so that the
        smallest index along each lattice vector is 0.

        Args:
            boundary (HollowArcSpline, ArcSplineCompatible): closed shape (e.g. a rectangle or a shape with holes)
            u (VectorLike): first lattice vector
            v (VectorLike): second lattice vector
            element (Transformable, Callable): the element at each point or a function which creates it (see
                :meth:`from_rect`)
            origin (VectorLike, optional): position of a lattice point, default: the center of the bounding box of the
                boundary.
            fast_axis (FastAxis, str): ``'u'`` (default): the points are numbered along `u` first (the lattice points
                along `u` follow each other), ``'v'``: along `v` first.
            clip_elements (bool): if True, elements which are not completely (their bounding box) inside the boundary
                are dropped.
            description (str, optional): description

        Returns:
            Lattice

        Raises:
            ValueError: Raised if the boundary is not closed, the lattice vectors are collinear or null vectors, or if
                there are no elements (no lattice point is inside of the boundary).
            TypeError: Raised if the boundary or an element is not valid.
        """
        shape = to_filling_shape(boundary)
        u_vec, v_vec = Vector(u), Vector(v)
        origin_vec = shape.bounding_box.center if origin is None else Vector(origin)

        indices, points = boundary_points(
            shape, (u_vec.x, u_vec.y), (v_vec.x, v_vec.y), (origin_vec.x, origin_vec.y), FastAxis.parse(fast_axis)
        )

        keep = None
        if clip_elements:
            def keep(placed: t.Any) -> bool:  # pylint: disable=function-redefined
                return all(shape.contains(corner) for corner in placed.bounding_box.corners)

        return cls._create(indices, points, element, u_vec, v_vec, description, keep)

    def __mul__(self, other: t.Any) -> t.Any:
        """``lattice * unit('µm')`` creates a :class:`~fibomat.arrangements.DimLattice`.

        Args:
            other (unit): unit of length

        Returns:
            DimLattice
        """
        from fibomat.arrangements.lattices.dim_lattice import DimLattice  # pylint: disable=import-outside-toplevel
        from fibomat.units.unit_tag import _UnitTag  # pylint: disable=import-outside-toplevel,protected-access

        if not isinstance(other, _UnitTag):
            return NotImplemented

        return DimLattice(
            [element * other for element in self._elements], self._points_uv, self._u * other, self._v * other,
            description=self.description
        )
