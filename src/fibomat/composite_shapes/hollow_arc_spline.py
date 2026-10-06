"""Provides the :class:`HollowArcSpline` class.

Example::

    from fibomat.shapes import ArcSpline, Circle, Rect
    from fibomat.composite_shapes import HollowArcSpline

    boundary = Rect(width=10, height=10).to_arc_spline()
    hole = Circle(r=2, center=(1, 1)).to_arc_spline()

    hollow = HollowArcSpline(boundary, [hole])
    hollow.area  # 100 - pi * 4
    hollow.contains((1, 1))  # False
    hollow.boundary, hollow.holes
"""
from __future__ import annotations

import typing as t
from collections import deque

import numpy as np

from fibomat.curve_tools.combine import combine_curves
from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.shape import Shape


__all__ = ['HollowArcSpline']


class HollowArcSpline(Shape):
    """A closed arc spline (the boundary) with holes (closed arc splines).

    The holes are inside of the boundary and do not touch or overlap each other (unless `disable_checks` is True):
    overlapping holes are merged and holes which touch the boundary are cut out of the boundary. The shape must be
    simply connected, i.e. the holes must not separate the shape into several parts.
    """

    def __init__(
        self,
        boundary: ArcSpline,
        holes: t.Optional[t.Iterable[ArcSpline]] = None,
        description: t.Optional[str] = None,
        disable_checks: bool = False,
    ):
        """
        Args:
            boundary (ArcSpline): closed arc spline (the curves are copied)
            holes (Iterable[ArcSpline], optional): closed arc splines
            description (str, optional): description
            disable_checks (bool):
                if True, it is assumed that the holes are disjoint, do not touch each other and the boundary and lie in
                the boundary. This is faster but wrong input leads to invalid shapes. Default to False.

        Raises:
            TypeError: Raised if boundary or holes are no arc splines.
            ValueError: Raised if boundary or holes are not closed, a hole lies outside of the boundary, the boundary
                lies in a hole or the holes separate the shape into several parts.
        """
        super().__init__(description)

        holes = list(holes) if holes is not None else []

        if not isinstance(boundary, ArcSpline):
            raise TypeError('boundary must be an ArcSpline.')
        if not all(isinstance(hole, ArcSpline) for hole in holes):
            raise TypeError('holes must be ArcSplines.')
        if not boundary.is_closed:
            raise ValueError('boundary must be a closed arc spline.')
        if not all(hole.is_closed for hole in holes):
            raise ValueError('holes must be closed arc splines.')

        boundary = ArcSpline(boundary)
        holes = [ArcSpline(hole) for hole in holes]

        if disable_checks:
            self._boundary, self._holes = boundary, holes
        else:
            self._boundary, self._holes = self._exclude_holes(boundary, self._merge_holes(holes))

    @classmethod
    def from_points(
        cls,
        boundary: t.Any,
        holes: t.Optional[t.Iterable[t.Any]] = None,
        description: t.Optional[str] = None,
        disable_checks: bool = False,
    ) -> HollowArcSpline:
        """Create a hollow polygon from points.

        Args:
            boundary (np.ndarray): points of the boundary polygon, array of shape (n, 2)
            holes (Iterable[np.ndarray], optional): points of the holes, arrays of shape (n, 2)
            description (str, optional): description
            disable_checks (bool): see :meth:`HollowArcSpline.__init__`

        Returns:
            HollowArcSpline
        """

        def polygon(points: t.Any) -> ArcSpline:
            array = np.array(points, dtype=float)
            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError('points must have shape (n, 2).')
            return ArcSpline(np.c_[array, np.zeros(len(array))], True)

        return cls(polygon(boundary), [polygon(hole) for hole in holes or []], description, disable_checks)

    @staticmethod
    def _exclude_holes(boundary: ArcSpline, holes: t.Sequence[ArcSpline]) -> t.Tuple[ArcSpline, t.List[ArcSpline]]:
        """Cut holes which touch the boundary out of the boundary. Returns the new boundary and the remaining holes."""
        inner_holes = []

        for hole in holes:
            excluded = combine_curves(boundary, hole, mode='exclude')

            if not excluded['remaining']:
                raise ValueError('The boundary lies inside of a hole.')

            if excluded['subtracted']:
                # the hole is completely inside of the boundary
                inner_holes.append(hole)
            elif len(excluded['remaining']) > 1:
                raise ValueError(
                    'Shape is not simply connected. '
                    'This is most likely caused by a hole cutting the shape in two or more pieces.'
                )
            elif _same_curve(excluded['remaining'][0], boundary):
                # nothing was cut out of the boundary: the hole is not in the boundary at all
                raise ValueError('A hole lies outside of the boundary.')
            else:
                # the hole touches or intersects the boundary and is cut out of it
                boundary = excluded['remaining'][0]

        return boundary, inner_holes

    @staticmethod
    def _merge_holes(holes: t.Sequence[ArcSpline]) -> t.List[ArcSpline]:
        """Merge overlapping holes."""
        queue = deque(holes)
        disjoint_holes: t.List[ArcSpline] = []

        while queue:
            hole = queue.popleft()
            merged = False

            for i, other in enumerate(queue):
                if hole.bounding_box.overlaps_with(other.bounding_box):
                    union = combine_curves(hole, other, mode='union')

                    if union['subtracted']:
                        raise ValueError(
                            'Shape is not simply connected. '
                            'This is most likely caused by holes which separate the shape in two or more parts.'
                        )

                    if len(union['remaining']) == 1:
                        del queue[i]
                        queue.append(union['remaining'][0])
                        merged = True
                        break

            if not merged:
                disjoint_holes.append(hole)

        return disjoint_holes

    def to_hollow_arc_spline(self) -> HollowArcSpline:
        """Return a copy.

        Returns:
            HollowArcSpline
        """
        return HollowArcSpline(self._boundary, self._holes, self.description, disable_checks=True)

    @property
    def boundary(self) -> ArcSpline:
        """The outer boundary.

        Access:
            get
        """
        return self._boundary

    @property
    def holes(self) -> t.List[ArcSpline]:
        """The holes (a new list is returned).

        Access:
            get
        """
        return [*self._holes]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(boundary={self._boundary!r}, n_holes={len(self._holes)})'

    @property
    def area(self) -> float:
        """Area of the shape (the area of the boundary minus the areas of the holes).

        Access:
            get
        """
        return self._boundary.area - sum(hole.area for hole in self._holes)

    @property
    def boundary_length(self) -> float:
        """Length of all boundary curves (the boundary and the holes).

        Access:
            get
        """
        return self._boundary.length + sum(hole.length for hole in self._holes)

    @property
    def is_closed(self) -> bool:
        return True

    def contains(self, pos: VectorLike) -> bool:
        """Returns True if pos is contained in shape

        Args:
            pos (VectorLike): point to be tested

        Returns:
            bool
        """
        return self._boundary.contains(pos) and not any(hole.contains(pos) for hole in self._holes)

    @property
    def center(self) -> Vector:
        """Center of the bounding box of the boundary.

        Access:
            get
        """
        return self._boundary.center

    @property
    def bounding_box(self) -> BoundingBox:
        return self._boundary.bounding_box

    def _impl_translate(self, trans_vec: Vector) -> None:
        for curve in (self._boundary, *self._holes):
            curve._impl_translate(trans_vec)  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        for curve in (self._boundary, *self._holes):
            curve._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        for curve in (self._boundary, *self._holes):
            curve._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        for curve in (self._boundary, *self._holes):
            curve._impl_mirror(mirror_axis)  # pylint: disable=protected-access


def _same_curve(first: ArcSpline, second: ArcSpline) -> bool:
    """True if the two arc splines have the same vertices."""
    return len(first) == len(second) and bool(np.allclose(first.vertices, second.vertices))
