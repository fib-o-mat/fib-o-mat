"""Provides the :class:`BoundingBox` class.

:class:`BoundingBox` is an axis aligned rectangle. It is generic over the vector type;
:class:`~fibomat.linalg.boundingboxes.dim_boundingbox.DimBoundingBox` derives from it for boxes with units.

Example::

    box = BoundingBox((0, 0), (2, 1))
    box.width, box.height, box.area, box.center

    box.contains((1, 0.5))
    (1, 0.5) in box
    box.extended((5, 5))
    BoundingBox.from_points([(0, 0), (3, 4)])

    dim_box = box * unit('µm')
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg.vectors import Vector, VectorValueError


__all__ = ['BoundingBox']


VectorT = t.TypeVar('VectorT', bound=Vector)  # type: ignore[type-arg]
SelfT = t.TypeVar('SelfT', bound='BoundingBox')  # type: ignore[type-arg]


class BoundingBox(t.Generic[VectorT]):
    """Axis aligned rectangular bounding box.

    Boxes are immutable by convention (all operations return new boxes) and not hashable. Boxes with units and
    boxes without units cannot be mixed.
    """

    _VectorClass: t.Type[VectorT] = Vector  # type: ignore[assignment]
    """Vector type of the corners."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, lower_left: t.Any, upper_right: t.Any):
        """
        Args:
            lower_left (VectorLike): lower left corner
            upper_right (VectorLike): upper right corner

        Raises:
            VectorValueError: Raised if the corners are no vectors.
            ValueError: Raised if the coordinates are not finite or `lower_left` is larger than `upper_right`.
        """
        self._lower_left: VectorT = self._VectorClass(lower_left)
        self._upper_right: VectorT = self._VectorClass(upper_right)

        self._validate()

    def _validate(self) -> None:
        if not np.all(np.isfinite(np.asarray(self._lower_left))) or not np.all(
            np.isfinite(np.asarray(self._upper_right))
        ):
            raise ValueError('Coordinates of bounding box must be finite.')

        if self._lower_left.x > self._upper_right.x or self._lower_left.y > self._upper_right.y:
            raise ValueError('Invalid coordinates for bounding box: lower_left must not be larger than upper_right.')

    @classmethod
    def from_points(cls: t.Type[SelfT], points: t.Iterable[t.Any]) -> SelfT:
        """Construct the smallest rectangular bounding box containing all `points`.

        Args:
            points (Iterable[VectorLike]): points which should be included in the bounding box (at least one)

        Returns:
            BoundingBox

        Raises:
            ValueError: Raised if `points` is empty.
            VectorValueError: Raised if one of the points is not a valid vector.
        """
        vectors = [cls._VectorClass(point) for point in points]
        if not vectors:
            raise ValueError('points must contain at least one point.')
        return cls._from_vectors(vectors)

    @classmethod
    def _from_vectors(cls: t.Type[SelfT], vectors: t.Sequence[t.Any]) -> SelfT:
        array = np.array([np.asarray(vector) for vector in vectors])
        return cls(np.min(array, axis=0), np.max(array, axis=0))

    def _check_compatible(self, other: BoundingBox[t.Any]) -> None:
        if self._VectorClass is not other._VectorClass:  # pylint: disable=protected-access
            raise TypeError(f'Cannot combine {self.__class__.__name__} and {other.__class__.__name__}.')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return self.close_to(other)

    def close_to(self, other: object) -> bool:
        """Check if the corners of `other` are close to the corners of this box.

        Args:
            other (BoundingBox): other box

        Returns:
            bool: False if `other` is no box of the same kind.
        """
        if not isinstance(other, BoundingBox) or self._VectorClass is not other._VectorClass:
            return False
        return self._lower_left.close_to(other._lower_left) and self._upper_right.close_to(other._upper_right)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(lower_left={self._lower_left!r}, upper_right={self._upper_right!r})'

    @property
    def corners(self) -> t.Tuple[VectorT, VectorT, VectorT, VectorT]:
        """The corners in counterclockwise order: lower left, lower right, upper right, upper left.

        Access:
            get
        """
        return (
            self._lower_left,
            self._VectorClass(self._upper_right.x, self._lower_left.y),
            self._upper_right,
            self._VectorClass(self._lower_left.x, self._upper_right.y),
        )

    @property
    def lower_left(self) -> VectorT:
        """Lower left corner.

        Access:
            get
        """
        return self._lower_left

    @property
    def upper_right(self) -> VectorT:
        """Upper right corner.

        Access:
            get
        """
        return self._upper_right

    @property
    def width(self) -> t.Any:
        """Width of the box.

        Access:
            get
        """
        return self._upper_right.x - self._lower_left.x

    @property
    def height(self) -> t.Any:
        """Height of the box.

        Access:
            get
        """
        return self._upper_right.y - self._lower_left.y

    @property
    def area(self) -> t.Any:
        """Area of the box.

        Access:
            get
        """
        return self.width * self.height

    @property
    def center(self) -> VectorT:
        """Center of the box.

        Access:
            get
        """
        return (self._upper_right + self._lower_left) / 2

    def clone(self: SelfT) -> SelfT:
        """Clone the bounding box.

        Returns:
            BoundingBox: cloned box
        """
        return self.__class__(self._lower_left, self._upper_right)

    def scaled(self: SelfT, val: float) -> SelfT:
        """Return a version of the box with width and height scaled by `val`. The center stays fixed.

        Args:
            val (float): scale factor, must be positive

        Returns:
            BoundingBox

        Raises:
            ValueError: Raised if `val` is not positive and finite.
        """
        val = float(val)
        if not np.isfinite(val) or val <= 0.:
            raise ValueError('val must be positive and finite.')

        # each side moves outwards by (val - 1) / 2 of the width (height)
        shift = (self.width * ((val - 1.) / 2.), self.height * ((val - 1.) / 2.))

        return self.__class__(self._lower_left - shift, self._upper_right + shift)

    def overlaps_with(self, other: BoundingBox[VectorT]) -> bool:
        """Check if this bounding box overlaps with `other`. Boxes which only touch overlap.

        Args:
            other (BoundingBox): other bounding box

        Returns:
            bool: True if self and other overlap

        Raises:
            TypeError: Raised if `other` is no box of the same kind.
        """
        if not isinstance(other, BoundingBox):
            raise TypeError('other must be a bounding box.')
        self._check_compatible(other)

        return not (
            self._lower_left.x > other._upper_right.x
            or self._upper_right.x < other._lower_left.x
            or self._lower_left.y > other._upper_right.y
            or self._upper_right.y < other._lower_left.y
        )

    def contains(self, other: t.Union[BoundingBox[VectorT], t.Any]) -> bool:
        """Check if this bounding box contains `other` (a box or a point). Alternatively, you can use the `in`
        syntax. Note that a box always contains itself. ::

            box_1 = BoundingBox((0, 0), (4, 4))
            box_2 = BoundingBox((1, 1), (2, 2))

            print(box_1.contains(box_2))
            # or
            print(box_2 in box_1)

            print((1, 1) in box_1)

        Args:
            other (BoundingBox, VectorLike): other box or point

        Returns:
            bool: True if `self` contains `other`

        Raises:
            TypeError: Raised if `other` is neither a box of the same kind nor a point.
        """
        if isinstance(other, BoundingBox):
            self._check_compatible(other)
            return (
                self._lower_left.x <= other._lower_left.x
                and self._lower_left.y <= other._lower_left.y
                and self._upper_right.x >= other._upper_right.x
                and self._upper_right.y >= other._upper_right.y
            )

        try:
            point = self._VectorClass(other)
        except VectorValueError as error:
            raise TypeError('other must be a bounding box or a vector.') from error

        return (
            self._lower_left.x <= point.x <= self._upper_right.x
            and self._lower_left.y <= point.y <= self._upper_right.y
        )

    def __contains__(self, other: t.Any) -> bool:
        return self.contains(other)

    def extended(self: SelfT, other: t.Union[BoundingBox[VectorT], t.Any]) -> SelfT:
        """Return an extended bounding box so that `self` and `other` are contained.

        Args:
            other (BoundingBox, VectorLike, Iterable[VectorLike]): other box, point or points

        Returns:
            BoundingBox

        Raises:
            TypeError: Raised if `other` is neither a box of the same kind, a point nor an iterable of points.
        """
        if isinstance(other, BoundingBox):
            self._check_compatible(other)
            points: t.List[VectorT] = [other._lower_left, other._upper_right]
        else:
            try:
                # materialize first, a generator could only be consumed once
                items = list(other)
            except TypeError as error:
                raise TypeError('other must be a bounding box, a vector or an iterable of vectors.') from error

            try:
                points = [self._VectorClass(items)]
            except VectorValueError:
                try:
                    points = [self._VectorClass(item) for item in items]
                except VectorValueError as error:
                    raise TypeError('other must be a bounding box, a vector or an iterable of vectors.') from error

        return self.from_points([self._lower_left, self._upper_right, *points])

    def __mul__(self, other: t.Any) -> t.Any:
        """``box * unit('µm')`` creates a :class:`DimBoundingBox` from a box without units."""
        from fibomat.units.unit_tag import _UnitTag  # pylint: disable=import-outside-toplevel,protected-access

        if isinstance(other, _UnitTag):
            if self._VectorClass._has_unit:  # pylint: disable=protected-access
                raise TypeError('Cannot attach a unit to a bounding box which already has a unit.')
            from fibomat.linalg.boundingboxes.dim_boundingbox import DimBoundingBox  # pylint: disable=import-outside-toplevel

            return DimBoundingBox(self._lower_left * other, self._upper_right * other)
        return NotImplemented

    def __rmul__(self, other: t.Any) -> t.Any:
        # (the unit tag returns NotImplemented for boxes, so this method is called)
        return self.__mul__(other)
