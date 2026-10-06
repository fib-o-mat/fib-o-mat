"""Provides the :class:`ArcSpline` class.

An arc spline is a curve of straight line segments and circular arcs, stored as list of vertices ``(x, y, bulge)``.
The bulge of a vertex describes the segment from this vertex to the next one: ``bulge = tan(angle / 4)`` where `angle`
is the sweep angle of the arc (0 for lines, positive for counterclockwise arcs, ``|bulge| <= 1``). If the curve is
closed, the last vertex is connected to the first one.

Example::

    import numpy as np
    from fibomat.shapes import ArcSpline

    # closed curve: a square with a half circle on the right hand side
    spline = ArcSpline(np.array([(0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 0)]), is_closed=True)

    spline.area, spline.length, spline.bounding_box
    spline.contains((0.5, 0.5))
    spline.segments  # [Line, Arc, Line, Line]
    spline.rotated(np.pi / 2, origin='center')
"""
from __future__ import annotations

import operator
import typing as t
import warnings

import numpy as np

from fibomat import _libfibomat
from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.shape import Shape


__all__ = ['ArcSpline', 'ArcSplineCompatible']


_BULGE_TOL = 1e-9
"""Bulge values up to 1 + _BULGE_TOL are accepted (and clipped to 1) to tolerate rounding errors."""


def _is_line(bulge: float) -> bool:
    return bool(np.isclose(bulge, 0.0))


def _as_vertex_array(vertices: t.Any) -> np.ndarray:
    """Convert and validate vertices (x, y, bulge)."""
    try:
        array = np.array(vertices, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('vertices must be an array of floats with shape (n, 3).') from error

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f'vertices must have shape (n, 3) with columns (x, y, bulge), got {array.shape}.')
    if not np.all(np.isfinite(array)):
        raise ValueError('vertices must be finite.')
    if np.any(np.abs(array[:, 2]) > 1. + _BULGE_TOL):
        raise ValueError('|bulge| must not be larger than 1 (arcs sweep at most a half circle). Split larger arcs.')

    array[:, 2] = np.clip(array[:, 2], -1., 1.)
    return array


class ArcSpline(Shape, ArcSplineCompatible):
    """Class represents a spline containing circular arcs and straight line segments. The spline is C^0, hence,
    continuous but not differentiable.

    The curve consists of at least two vertices. Arcs sweep at most a half circle (``|bulge| <= 1``).
    """

    def __init__(
        self,
        arc_spline: t.Union[_libfibomat.ArcSpline, ArcSpline, np.ndarray, t.Sequence[t.Sequence[float]]],
        is_closed: t.Optional[bool] = None,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            arc_spline (_libfibomat.ArcSpline, ArcSpline, np.ndarray):
                np.ndarray (or nested sequence) must have shape = (n, 3) where each point is given by (x, y, bulge).
            is_closed (bool, optional):
                if True, the last and first point are connected (potentially with an arc, if the bulge value of the last
                vertex is nonzero). if False, the bulge value of the last point is ignored. The argument must be given
                if arc_spline contains vertices and is ignored if arc_spline is a native or an ArcSpline.
            description (str, optional): description

        Raises:
            ValueError: Raised if arc_spline contains vertices but is_closed is not given, the vertices are invalid
                (wrong shape, not finite, |bulge| > 1) or there are less than two vertices.
        """
        super().__init__(description)

        if isinstance(arc_spline, ArcSpline):
            arc_spline = arc_spline._arc_spline  # pylint: disable=protected-access

        if isinstance(arc_spline, _libfibomat.ArcSpline):
            self._arc_spline = _libfibomat.ArcSpline(arc_spline)
        else:
            if is_closed is None:
                raise ValueError('is_closed must be defined if ArcSpline is build from vertices.')
            self._arc_spline = _libfibomat.ArcSpline(_as_vertex_array(arc_spline), bool(is_closed))

        if self._arc_spline.size < 2:
            raise ValueError('An ArcSpline needs at least two vertices.')

    def __len__(self) -> int:
        return self._arc_spline.size

    # construction

    @classmethod
    def from_segments(cls, segments: t.Iterable[ArcSplineCompatible], description: t.Optional[str] = None) -> ArcSpline:
        """Build an ArcSpline from connected segments.

        Args:
            segments (Iterable[ArcSplineCompatible]):
                segments. the start and end point of two consecutive segments must be equal. If this also holds for the
                last and first segment, the curve is closed. A single closed segment results in a closed curve.
            description (str, optional): description.

        Returns:
            ArcSpline

        Raises:
            ValueError: Raised if there are no segments.
            RuntimeError: Raised if segments are not connected or some of several segments are closed.
        """
        splines = [seg.to_arc_spline() for seg in segments]

        if not splines:
            raise ValueError('segments must not be empty.')

        if len(splines) == 1:
            return cls(splines[0], description=description)

        vertices: t.List[np.ndarray] = []

        for spline in splines:
            if spline.is_closed:
                raise RuntimeError('Cannot build ArcSpline from segments because some segments are closed.')

            seg_vertices = spline.vertices

            if vertices:
                if not np.allclose(vertices[-1][-1, :2], seg_vertices[0, :2]):
                    raise RuntimeError(
                        'Segments are not C^0. The distance is {}'.format(
                            np.linalg.norm(vertices[-1][-1, :2] - seg_vertices[0, :2])
                        )
                    )
                # the last vertex of a segment has no bulge, the bulge of the next segment's start is used instead
                vertices[-1][-1] = seg_vertices[0]
                vertices.append(seg_vertices[1:])
            else:
                vertices.append(seg_vertices)

        conc_vertices: np.ndarray = np.concatenate(vertices)

        if np.allclose(conc_vertices[0, :2], conc_vertices[-1, :2]):
            return cls(conc_vertices[:-1], True, description)

        return cls(conc_vertices, False, description)

    @classmethod
    def from_shape(cls, segment: ArcSplineCompatible) -> ArcSpline:
        """Converts a single segment to an ArcSpline.

        Args:
            segment (ArcSplineCompatible): segment.

        Returns:
            ArcSpline
        """
        return segment.to_arc_spline()

    def to_arc_spline(self) -> ArcSpline:
        return self

    def clone_with_new_description(self, description: t.Optional[str] = None) -> ArcSpline:
        """Similar to :meth:`ArcSpline.clone` but set the description to the passed description.

        .. deprecated:: 0.6.0
            Use :meth:`fibomat.describable.Describable.with_changed_description` instead.

        Args:
            description (str, optional): description.

        Returns:
            ArcSpline
        """
        warnings.warn(
            'clone_with_new_description is deprecated, use with_changed_description instead.',
            category=DeprecationWarning,
            stacklevel=2,
        )
        clone = self.clone()
        clone._description = str(description) if description else None  # pylint: disable=protected-access
        return clone

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(start={self.start}, end={self.end}, description={self.description})'

    # shape interface

    @property
    def is_closed(self) -> bool:
        return self._arc_spline.is_closed

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(*self._arc_spline.bounding_box)

    @property
    def center(self) -> Vector:
        """Center of the curve, defined as the center of its bounding box.

        .. note:: This is neither the mean of the vertices nor the centroid of the enclosed area.

        Access:
            get

        Returns:
            Vector
        """
        return Vector(self._arc_spline.center)

    def _impl_translate(self, trans_vec: Vector) -> None:
        trans_vec = Vector(trans_vec)
        self._arc_spline.impl_translate((trans_vec.x, trans_vec.y))

    def _impl_scale(self, fac: float) -> None:
        self._arc_spline.impl_scale(fac)

    def _impl_rotate(self, theta: float) -> None:
        self._arc_spline.impl_rotate(theta)

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        mirror_axis = Vector(mirror_axis)
        self._arc_spline.impl_mirror((mirror_axis.x, mirror_axis.y))

    @property
    def area(self) -> float:
        """Area enclosed by the curve (non-negative).

        Access:
            get

        Returns:
            float

        Raises:
            NotImplementedError: Raised if the curve is not closed.
        """
        if not self.is_closed:
            raise NotImplementedError('The area is only defined for closed curves.')
        return abs(self._arc_spline.area)

    @property
    def boundary_length(self) -> float:
        """Length of the curve. Same as :attr:`ArcSpline.length`.

        Access:
            get

        Returns:
            float
        """
        return self.length

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
        """End point curve. For closed curves, this is the start point.

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
        """Curve vertices as (a copy of an) array with shape (n, 3). The columns are x, y and bulge.

        Access:
            get

        Returns:
            np.ndarray
        """
        return self._arc_spline.vertices_array

    @property
    def segments(self) -> t.Sequence[ArcSplineCompatible]:
        """Return a list of Line and Arc elements representing the curve.

        .. note:: This method is not bijective with regard to the added shapes. E.g. if the curve was constructed from a
                  Circle, this method will not return a Circle element but two Arcs.

        Access:
            get

        Returns:
            List[Shape]
        """
        vertices = self.vertices
        n_vertices = len(vertices)
        n_segments = n_vertices if self.is_closed else n_vertices - 1

        return [self._make_segment(vertices[i], vertices[(i + 1) % n_vertices]) for i in range(n_segments)]

    @staticmethod
    def _make_segment(start_vertex: np.ndarray, end_vertex: np.ndarray) -> Shape:
        """Line or Arc from the vertex `start_vertex` to `end_vertex`."""
        from fibomat.shapes.line import Line  # pylint: disable=import-outside-toplevel
        from fibomat.shapes.arc import Arc  # pylint: disable=import-outside-toplevel

        if _is_line(start_vertex[2]):
            return Line(start_vertex[:2], end_vertex[:2])

        return Arc.from_bulge(start_vertex[:2], end_vertex[:2], start_vertex[2])

    @property
    def arc_spline_impl(self) -> _libfibomat.ArcSpline:
        """The underlying native curve. It must not be modified.

        Access:
            get
        """
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
        """Length of curve (including the closing segment for closed curves).

        Access:
            get

        Returns:
            float
        """
        return self._arc_spline.length

    def contains(self, pos: VectorLike) -> bool:
        """Check if the point lies in the area enclosed by the curve.

        Args:
            pos (VectorLike): point

        Returns:
            bool

        Raises:
            RuntimeError: Raised if the curve is not closed.
        """
        pos = Vector(pos)
        return bool(self._arc_spline.contains(pos.x, pos.y))

    def closest_point(self, pos: VectorLike) -> t.Dict[str, t.Any]:
        """Find the point of the curve closest to `pos`.

        Args:
            pos (VectorLike): point

        Returns:
            Dict: `segment` (index of the segment), `point` (closest point as Vector), `distance` (to `pos`)
        """
        pos = Vector(pos)
        segment, point, distance = self._arc_spline.closest_point(pos.x, pos.y)

        return {'segment': segment, 'point': Vector(point), 'distance': distance}

    # tangents, kinks, segments around vertices

    def _check_vertex_index(self, i_vertex: int) -> int:
        try:
            i_vertex = operator.index(i_vertex)
        except TypeError as error:
            raise TypeError('i_vertex must be an integer.') from error

        if not 0 <= i_vertex < self._arc_spline.size:
            raise ValueError(f'i_vertex must be in [0, {self._arc_spline.size - 1}], got {i_vertex}.')
        return i_vertex

    def _has_segment_before(self, i_vertex: int) -> bool:
        return i_vertex != 0 or self.is_closed

    def _has_segment_after(self, i_vertex: int) -> bool:
        return i_vertex != self._arc_spline.size - 1 or self.is_closed

    @staticmethod
    def _unit_tangents_of_segment(start_vertex: np.ndarray, end_vertex: np.ndarray) -> t.Tuple[Vector, Vector]:
        """Unit tangents at the start and the end of the segment."""
        from fibomat.shapes.arc import Arc  # pylint: disable=import-outside-toplevel

        if _is_line(start_vertex[2]):
            direction = Vector(end_vertex[:2] - start_vertex[:2]).normalized()
            return direction, direction

        arc = Arc.from_bulge(start_vertex[:2], end_vertex[:2], start_vertex[2])
        return Vector(arc.unit_tangent_start), Vector(arc.unit_tangent_end)

    def unit_tangents(self, i_vertex: int) -> t.Tuple[t.Optional[Vector], t.Optional[Vector]]:
        """Unit tangents at vertex i_vertex.

        Args:
            i_vertex (int): vertex index

        Returns:
            Tuple[Optional[Vector], Optional[Vector]]:
                left tangent, right tangent. If any of the two tangents does not exist, the tuple entry will be None.

        Raises:
            ValueError: Raised if i_vertex is not a vertex index.
        """
        i_vertex = self._check_vertex_index(i_vertex)
        vertices = self.vertices
        n_vertices = len(vertices)

        first_tangent = None
        second_tangent = None

        if self._has_segment_before(i_vertex):
            before = (i_vertex - 1) % n_vertices
            first_tangent = self._unit_tangents_of_segment(vertices[before], vertices[i_vertex])[1]

        if self._has_segment_after(i_vertex):
            second_tangent = self._unit_tangents_of_segment(vertices[i_vertex], vertices[(i_vertex + 1) % n_vertices])[0]

        return first_tangent, second_tangent

    def kinks(self) -> t.List[int]:
        """Return kinks (non differentiable points) of the spline.

        Returns:
             List[int]: vertex indices of kinks.
        """
        vertices = self.vertices
        n_vertices = len(vertices)
        n_segments = n_vertices if self.is_closed else n_vertices - 1

        # tangents (at start and end) of each segment are evaluated only once
        tangents = [
            self._unit_tangents_of_segment(vertices[i], vertices[(i + 1) % n_vertices]) for i in range(n_segments)
        ]

        kinks = []
        for i_vertex in range(n_vertices):
            if self._has_segment_before(i_vertex) and self._has_segment_after(i_vertex):
                tangent_before = tangents[(i_vertex - 1) % n_segments][1]
                tangent_after = tangents[i_vertex][0]
                if not np.allclose(tangent_before, tangent_after):
                    kinks.append(i_vertex)

        return kinks

    def segments_at_vertex(self, i_vertex: int) -> t.Tuple[t.Optional[Shape], t.Optional[Shape]]:
        """Return the segments around the vertex with index i_vertex.

        Args:
            i_vertex (int): vertex index

        Returns:
            Tuple[Optional[Shape], Optional[Shape]]:
                left and right segments. If any of the segments is not defined, the tuple entry will be None.

        Raises:
            ValueError: Raised if i_vertex is not a vertex index.
        """
        i_vertex = self._check_vertex_index(i_vertex)
        vertices = self.vertices
        n_vertices = len(vertices)

        first_seg = None
        second_seg = None

        if self._has_segment_before(i_vertex):
            first_seg = self._make_segment(vertices[(i_vertex - 1) % n_vertices], vertices[i_vertex])

        if self._has_segment_after(i_vertex):
            second_seg = self._make_segment(vertices[i_vertex], vertices[(i_vertex + 1) % n_vertices])

        return first_seg, second_seg

    def reversed(self) -> ArcSpline:
        """Return a reversed copy of the arc spline

        Returns:
            ArcSpline
        """
        clone = self.clone()
        clone._arc_spline.reverse()  # pylint: disable=protected-access
        return clone
