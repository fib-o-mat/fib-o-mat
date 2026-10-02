"""Provides the :class:`ArcSpline` and :class:`ArcSplineCompatible` classes."""

from __future__ import annotations
from typing import (
    Optional,
    Union,
    Sequence,
    Iterable,
    Protocol,
    runtime_checkable,
    Tuple,
    List,
    Dict,
)

import numpy as np

from fibomat.linalg import VectorLike, Vector, BoundingBox
from fibomat.shapes.shape import Shape

from fibomat import _libfibomat


@runtime_checkable
class ArcSplineCompatible(Protocol):  # pylint: disable=too-few-public-methods
    """Abstract class can be used to mark ArcSpline compatible shapes."""

    def to_arc_spline(self) -> ArcSpline:
        """
        Transform shape to ArcSpline.

        Returns:
            ArcSpline
        """
        raise NotImplementedError


class ArcSpline(Shape, ArcSplineCompatible):
    """Class represents a spline containing circular arcs and straight line segments. The spline is C^0, hence,
    continuous but not differentiable.
    """

    def __init__(
        self,
        arc_spline: Union[_libfibomat.ArcSpline, np.ndarray],
        is_closed: Optional[bool] = None,
        description: Optional[str] = None,
    ):
        """
        Args:
            arc_spline (_libfibomat.ArcSpline, np.ndarray):
                np.ndarray must have shape = (-1, 3) where each point is given by (x, y, bulge).
            is_closed (bool, optional):
                if True, the last and first point are connected (potentially with an arc, if the bulge value of the last
                vertex is nonzero). if False, the bulge value of the last point is ignored. The argumetn must be given
                if arc_spline is np.ndarray and is ignored if arc_spline is _libfibomat.ArcSpline.
            description (str, optional): description

        Raises:
            ValueError: Raised if arc_spline is np.ndarray but is_closed not given.
        """
        super().__init__(description)

        if isinstance(arc_spline, _libfibomat.ArcSpline):
            self._arc_spline = _libfibomat.ArcSpline(arc_spline)
        else:
            if is_closed is None:
                raise ValueError(
                    "is_closed must be defined if ArcSpline is build from vertices."
                )
            self._arc_spline = _libfibomat.ArcSpline(arc_spline, is_closed)

        # self._vertices = np.array(self._arc_spline.vertices)
        # self._vertices.flags.writeable = False

    def __copy__(self):
        return self.__class__(
            arc_spline=self._arc_spline.clone(), description=self.description
        )

    def __deepcopy__(self, memodict):
        return self.__copy__()

    def __len__(self) -> int:
        return self._arc_spline.size

    # shape.Shape methods

    @classmethod
    def from_segments(
        cls, segments: Iterable[ArcSplineCompatible], description: Optional[str] = None
    ):
        """Build an ArcSpline from connected segments.

        Args:
            segments (Iterable[ArcSplineCompatible]):
                segments. the start and end point of to consecutive segments must be equal. If this also holds for the
                last and first segment, the curve ist closed.
            description (str, optional): description.

        Returns:
            ArcSpline

        Raises:
            RuntimeError: Raised if segments are not connected.
        """
        vertices: List[np.ndarray] = []

        for i_seg, seg in enumerate(segments):
            arc_spline = seg.to_arc_spline()

            if arc_spline.is_closed and i_seg > 0:
                raise RuntimeError(
                    "Cannot build ArcSpline from segments because some segments are closed."
                )

            seg_vertices = np.array(arc_spline.vertices)

            if vertices:
                if not np.allclose(vertices[-1][-1, :2], seg_vertices[0, :2]):
                    raise RuntimeError(
                        "Segments are not C^0. The distance is {}".format(
                            np.linalg.norm(vertices[-1][-1, :2] - seg_vertices[0, :2])
                        )
                    )
                vertices[-1][-1] = seg_vertices[0]
                vertices.append(seg_vertices[1:])
            else:
                vertices.append(seg_vertices)

        conc_vertices: np.ndarray = np.concatenate(vertices)

        if np.allclose(conc_vertices[0, :2], conc_vertices[-1, :2]):
            return cls(conc_vertices[:-1], True, description)

        return cls(conc_vertices, False, description)

    @classmethod
    def from_shape(cls, segment: ArcSplineCompatible):
        """Converts a single segment to an ArcSpline.

        Args:
            segment (ArcSplineCompatible): segment.

        Returns:
            ArcSpline
        """
        return segment.to_arc_spline()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(start={self.start}, end={self.end}, description={self.description})"

    @property
    def is_closed(self) -> bool:
        return self._arc_spline.is_closed

    def to_arc_spline(self) -> ArcSpline:
        return self

    # Transformable methods

    def clone(self) -> ArcSpline:
        return self.__class__(
            arc_spline=self._arc_spline.clone(),
            description=self.description,
        )

    def clone_with_new_description(self, description: Optional[str] = None):
        """Similar to :meth:`ArcSpline.clone` but set the description the passed description.

        Args:
            description (str, optional): description.

        Returns:
            ArcSpline
        """
        return self.__class__(
            arc_spline=self._arc_spline.clone(), description=description
        )

    @property
    def bounding_box(self) -> BoundingBox:
        # return BoundingBox(*self._arc_spline.bounding_box)

        # there is a bug in cavc bounding box calculation for multiple arc segments (cf. examples/fill_lines.py without this hacky code)

        segments = self.segments

        if len(segments) > 1:
            bbox = segments[0].bounding_box
            for seg in segments[1:]:
                bbox = bbox.extended(seg.bounding_box)

            return bbox
        else:
            return BoundingBox(*self._arc_spline.bounding_box)

    @property
    def center(self) -> Vector:
        return Vector(self._arc_spline.center)

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._arc_spline.impl_translate(trans_vec)

    def _impl_scale(self, fac: float) -> None:
        self._arc_spline.impl_scale(fac)

    def _impl_rotate(self, theta: float) -> None:
        self._arc_spline.impl_rotate(theta)

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        self._arc_spline.impl_mirror(mirror_axis)

    # other utility methods

    @property
    def start(self) -> Vector:
        """Start point curve

        Access:
            get

        Returns:
            Vector
        """
        return Vector(self._arc_spline.start[:2])

    @property
    def end(self) -> Vector:
        """End point curve

        Access:
            get

        Returns:
            Vector
        """
        if self._arc_spline.is_closed:
            return Vector(self._arc_spline.start[:2])

        return Vector(self._arc_spline.end[:2])

    @property
    def vertices(self) -> np.ndarray:
        """Curve vertices.

        Access:
            get

        Returns:
            np.ndarray
        """
        return np.array(self._arc_spline.vertices)

    @property
    def segments(self) -> Sequence[ArcSplineCompatible]:
        """Return a list of Line and Arc elements representing the curve.

        .. note:: This method is not bijective with regard to the added shapes. E.g. if the curve was constructed from a
                  Circle, this method will not return a Circle element but two Arcs.

        Access:
            get

        Returns:
            List[shape.Shape]
        """
        from fibomat.shapes.line import Line  # pylint: disable=import-outside-toplevel
        from fibomat.shapes.arc import Arc  # pylint: disable=import-outside-toplevel

        def _make_segment(start_vertex, end_vertex):
            if np.isclose(start_vertex[2], 0.0):
                return Line(start_vertex[:2], end_vertex[:2])

            return Arc.from_bulge(start_vertex[:2], end_vertex[:2], start_vertex[2])

        segments = []
        vertices = self.vertices
        for i, vertex in enumerate(vertices[1:], start=1):
            segments.append(_make_segment(vertices[i - 1], vertex))

        if self.is_closed:
            segments.append(_make_segment(vertices[-1], vertices[0]))

        return segments

    @property
    def arc_spline_impl(self) -> _libfibomat.ArcSpline:
        return self._arc_spline

    @property
    def orientation(self) -> bool:
        """Orientation of curve. True if curve is counterclockwise.

        .. note:: This property is only defined for closed curves.

        Access:
            get

        Returns:
            bool
        """
        return self._arc_spline.orientation

    @property
    def length(self) -> float:
        """Length of curve.

        Access:
            get

        Returns:
            float
        """
        return self._arc_spline.length

    @property
    def area(self) -> float:
        return abs(self._arc_spline.area)

    def contains(self, pos: VectorLike):
        pos = Vector(pos)
        return self._arc_spline.contains(pos.x, pos.y)

    def closest_point(self, pos: VectorLike) -> Dict:
        pos = Vector(pos)
        res = self._arc_spline.closest_point(pos.x, pos.y)

        return {"segment": pos[0], "point": Vector(res[1]), "distance": res[2]}

    def unit_tangents(self, i_vertex: int) -> Tuple[Optional[Vector], Optional[Vector]]:
        """Unit tangents at vertex i_vertex.

        Args:
            i_vertex (int): vertex index

        Returns:
            Tuple[Optional[Vector], Optional[Vector]]:
                left tangent, right tangent. If any of the two tangents does not exist, the tuple entry will be None.

        Raises:
            ValueError: Raised if i_vertex > #segmens.
        """
        from fibomat.shapes.arc import Arc  # pylint: disable=import-outside-toplevel

        if i_vertex < self._arc_spline.size:
            vertices = self.vertices
            vertex = vertices[i_vertex]

            first_tangent = None
            second_tangent = None

            if i_vertex != 0 or self.is_closed:
                vertex_before = vertices[(i_vertex - 1) % len(vertices)]

                if np.isclose(vertex_before[2], 0.0):
                    first_tangent = vertex[:2] - vertex_before[:2]
                    first_tangent /= np.linalg.norm(first_tangent)
                else:
                    arc = Arc.from_bulge(
                        vertex_before[:2], vertex[:2], vertex_before[2]
                    )
                    first_tangent = arc.unit_tangent_end

            if i_vertex != self._arc_spline.size - 1 or self.is_closed:
                vertex_next = vertices[(i_vertex + 1) % len(vertices)]

                if np.isclose(vertex[2], 0.0):
                    second_tangent = vertex_next[:2] - vertex[:2]
                    second_tangent /= np.linalg.norm(second_tangent)
                else:
                    arc = Arc.from_bulge(vertex[:2], vertex_next[:2], vertex[2])
                    second_tangent = arc.unit_tangent_start

            return first_tangent, second_tangent

        raise ValueError("i_vertex >= number of segments.")

    def kinks(self) -> List[int]:
        """Return kinks (non differentiable points) of the spline.

        Returns:
             List[int]: vertex indices of kinks.
        """
        tangents = [self.unit_tangents(i) for i in range(len(self.vertices))]

        kinks = []

        if self.is_closed:
            if not np.allclose(tangents[0][0], tangents[0][1]):
                kinks.append(0)

        for i in range(1, len(self.vertices) - 1):
            if not np.allclose(tangents[i][0], tangents[i][1]):
                kinks.append(i)

        if self.is_closed:
            if not np.allclose(tangents[-1][0], tangents[-1][1]):
                kinks.append(len(self.vertices) - 1)

        return kinks

    def segments_at_vertex(
        self, i_vertex: int
    ) -> Tuple[Optional[Shape], Optional[Shape]]:
        """Return the segments around the vertex with index i_vertex.

        Args:
            i_vertex: vertex index

        Returns:
            Tuple[Optional[Shape], Optional[Shape]]:
                left and right segments. If any of the segments is not defined, the tuple entry will be None.

        Raises:
            ValueError: Raised if i_vertex > #segmens.
        """
        from fibomat.shapes.arc import Arc  # pylint: disable=import-outside-toplevel
        from fibomat.shapes.line import Line  # pylint: disable=import-outside-toplevel

        if i_vertex < self._arc_spline.size:
            vertices = self.vertices
            vertex = vertices[i_vertex]

            first_seg = None
            second_seg = None

            if i_vertex != 0 or self.is_closed:
                vertex_before = vertices[(i_vertex - 1) % len(vertices)]

                if np.isclose(vertex_before[2], 0.0):
                    first_seg = Line(vertex_before[:2], vertex[:2])
                else:
                    first_seg = Arc.from_bulge(
                        vertex_before[:2], vertex[:2], vertex_before[2]
                    )

            if i_vertex != self._arc_spline.size - 1 or self.is_closed:
                vertex_next = vertices[(i_vertex + 1) % len(vertices)]

                if np.isclose(vertex[2], 0.0):
                    second_seg = Line(vertex[:2], vertex_next[:2])
                else:
                    second_seg = Arc.from_bulge(vertex[:2], vertex_next[:2], vertex[2])

            return first_seg, second_seg

        raise ValueError("i_vertex >= number of segments.")

    def reversed(self):
        """Return a reversed copy of the arc spline

        Returns:
            ArcSpline
        """
        clone = self.clone()
        clone._arc_spline.reverse()  # pylint: disable=protected-access
        return clone
