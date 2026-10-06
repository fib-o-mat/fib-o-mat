from collections import deque
from typing import List, Optional, Sequence

import numpy as np

from fibomat.curve_tools import combine_curves
from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.shape import Shape


class HollowArcSpline(Shape):
    def __init__(
        self,
        boundary: ArcSpline,
        holes: Optional[Sequence[ArcSpline]] = None,
        description: Optional[str] = None,
        disable_checks=False,
    ):
        super().__init__(description)

        if not isinstance(boundary, ArcSpline):
            raise TypeError("boundary must be an ArcSplines.")

        if not all(isinstance(hole, ArcSpline) for hole in holes):
            raise TypeError("holes must be ArcSplines.")

        if not boundary.is_closed:
            raise ValueError("boundary must be closed arc spline.")

        if not holes:
            holes = []

        if not all(hole.is_closed for hole in holes):
            raise ValueError("holes must be closed arc splines")

        if disable_checks:
            self._boundary, self._holes = boundary, holes
        else:
            self._boundary, self._holes = self._exclude_holes(
                boundary, self._merge_holes(holes)
            )

    @staticmethod
    def _exclude_holes(boundary: ArcSpline, holes: Sequence[ArcSpline]):
        queue = deque(holes)

        non_intersecting_holes = []

        while queue:
            hole = queue.popleft()

            # if not combine_curves(boundary, hole, mode='intersect'):
            #     raise RuntimeError('Hole is not included in boundary.')

            excluded = combine_curves(boundary, hole, mode="exclude")

            if not excluded["remaining"]:
                print(
                    boundary.bounding_box,
                    [hole.bounding_box for hole in holes],
                    excluded,
                )
                # something strange happened
                raise RuntimeError
            if len(excluded["remaining"]) == 1:
                if len(excluded["remaining"][0].vertices) == len(
                    boundary.vertices
                ) and np.allclose(excluded["remaining"][0].vertices, boundary.vertices):
                    non_intersecting_holes.append(hole)
                else:
                    boundary = excluded["remaining"][0]
            if len(excluded["remaining"]) > 1:
                raise RuntimeError(
                    "Shape is not simply connected."
                    "This is most likely caused by a hole cutting the shape in two or more pieces."
                )

        return boundary, non_intersecting_holes

    @staticmethod
    def _merge_holes(holes: Sequence[ArcSpline]):
        if holes:
            queue = deque(holes)

            non_intersection_holes = []

            while queue:
                hole = queue.popleft()
                non_intersecting = True

                for i in range(len(queue)):
                    if hole.bounding_box.overlaps_with(queue[i].bounding_box):
                        union = combine_curves(hole, queue[i], mode="union")

                        if union["subtracted"]:
                            raise RuntimeError(
                                "Shape is not simply connected. "
                                "This is most likely caused by holes which separate the shape in two or more parts."
                            )

                        if len(union["remaining"]) == 1:
                            del queue[i]
                            queue.append(union["remaining"][0])
                            non_intersecting = False
                            break

                if non_intersecting:
                    non_intersection_holes.append(hole)

            return non_intersection_holes
        else:
            return []

    @property
    def boundary(self) -> ArcSpline:
        return self._boundary

    @property
    def holes(self) -> List[ArcSpline]:
        return [*self._holes]

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def area(self) -> float:
        area = self._boundary.area

        for hole in self._holes:
            if hole.to_arc_spline().orientation:
                area += hole.area
            else:
                area -= hole.area

        return area

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
        return self._boundary.contains(pos) and not any(
            [hole.contains(pos) for hole in self._holes]
        )

    @property
    def center(self) -> Vector:
        return self._boundary.center

    @property
    def bounding_box(self) -> BoundingBox:
        return self._boundary.bounding_box

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._boundary._impl_translate(trans_vec)

        for hole in self._holes:
            hole._impl_translate(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        self._boundary._impl_rotate(theta)

        for hole in self._holes:
            hole._impl_rotate(theta)

    def _impl_scale(self, fac: float) -> None:
        self._boundary._impl_scale(fac)

        for hole in self._holes:
            hole._impl_scale(fac)

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        self._boundary._impl_mirror(mirror_axis)

        for hole in self._holes:
            hole._impl_mirror(mirror_axis)
