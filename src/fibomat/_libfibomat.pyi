from __future__ import annotations
from typing import overload, Union, List, Tuple, Iterable, Optional, Callable

import numpy as np

from fibomat.linalg import VectorLike


class ArcSpline:
    ...

    @overload
    def __init__(self, other: ArcSpline): ...

    @overload
    def __init__(
        self,
        vertices: Union[
            np.ndarray, List[List[float]], List[Tuple[float, float, float]]
        ],
        is_closed: bool,
    ): ...

    def clone(self) -> ArcSpline: ...

    @property
    def size(self) -> int: ...

    @property
    def is_closed(self) -> bool: ...

    @property
    def bounding_box(self) -> Tuple[Tuple[float, float], Tuple[float, float]]: ...

    @property
    def center(self) -> Tuple[float, float]: ...

    def impl_translate(self, trans_vec: VectorLike) -> None: ...
    def impl_rotate(self, angle: float) -> None: ...
    def impl_scale(self, fac: float) -> None: ...
    def impl_mirror(self, mirror_axis: VectorLike) -> None: ...

    @property
    def start(self) -> Tuple[float, float, float]: ...

    @property
    def end(self) -> Tuple[float, float, float]: ...

    @property
    def vertices(self) -> List[Tuple[float, float, float]]: ...

    @property
    def orientation(self) -> bool: ...

    @property
    def length(self) -> float: ...

    @property
    def area(self) -> float: ...

    def contains(self, p_x: float, p_y: float) -> bool: ...

    def closest_point(
        self, p_x: float, p_y: float
    ) -> Tuple[int, Tuple[float, float], float]: ...

    def reverse(self) -> None: ...

    # def visit(self, func: Callable[[int, Tuple[float, float, float], Tuple[float, float, float]], bool]): ...
    #
    # def __getstate__(self) -> Tuple: ...
    #
    # def __setstate__(self, state: Tuple) -> Curve: ...


IntersectionType = List[Tuple[int, int, Tuple[float, float]]]
CoincidentType = List[Tuple[int, int, Tuple[float, float], Tuple[float, float]]]


def self_intersections(curve: ArcSpline) -> IntersectionType: ...


def curve_intersections(
    curve_1: ArcSpline, curve_2: ArcSpline
) -> Tuple[IntersectionType, CoincidentType]: ...


def combine_curves(
    curve_1: ArcSpline, curve_2: ArcSpline, mode: str
) -> Tuple[Iterable[ArcSpline], Iterable[ArcSpline]]: ...


def offset_curve(curves: ArcSpline, delta: float) -> Iterable[ArcSpline]: ...


def offset_with_islands(
    islands: Iterable[ArcSpline], outer_curve: Optional[ArcSpline], delta: float
) -> Tuple[Iterable[ArcSpline], Iterable[ArcSpline]]: ...


def convert_arcs_to_lines(curve: ArcSpline, error: float) -> ArcSpline: ...
