import copy
import math
import pickle
import warnings

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat import _libfibomat
from fibomat.linalg import BoundingBox, Vector, translate, rotate, scale, mirror
from fibomat.shapes.arc import Arc
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.line import Line
from fibomat.shapes.shape import Shape
from fibomat.units import unit

SQUARE = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
CIRCLE = [(1, 0, 1), (-1, 0, 1)]


def square(closed=True):
    return ArcSpline(SQUARE, closed)


def circle():
    return ArcSpline(CIRCLE, True)


def rounded():
    """Square with a half circle on the right side."""
    return ArcSpline([(0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 0)], True)


class TestConstruction:
    def test_from_array_and_sequences(self):
        for vertices in (np.array(SQUARE, dtype=float), SQUARE, [list(v) for v in SQUARE], tuple(SQUARE)):
            spline = ArcSpline(vertices, True)
            assert len(spline) == 4
            assert spline.is_closed
            assert spline.vertices.tolist() == [list(map(float, v)) for v in SQUARE]

    def test_is_closed_is_required_for_vertices(self):
        with pytest.raises(ValueError, match='is_closed'):
            ArcSpline(SQUARE)

    def test_description(self):
        assert ArcSpline(SQUARE, True, 'foo').description == 'foo'
        assert ArcSpline(SQUARE, True, description='foo').description == 'foo'
        assert square().description is None

    def test_is_shape(self):
        assert isinstance(square(), Shape)

    def test_from_native(self):
        native = _libfibomat.ArcSpline(np.array(SQUARE, dtype=float), True)
        spline = ArcSpline(native)
        assert spline.is_closed  # is_closed is not needed (and ignored)
        assert ArcSpline(native, is_closed=False).is_closed
        # the native curve is copied
        spline._impl_translate(Vector(1, 1))
        assert native.start == (0., 0., 0.)

    def test_from_arc_spline(self):
        original = square()
        spline = ArcSpline(original, description='foo')
        assert spline.is_closed
        assert spline.description == 'foo'
        spline._impl_translate(Vector(1, 1))
        assert original.start == (0., 0.)

    @pytest.mark.parametrize('vertices', [
        np.zeros((3, 2)),
        np.zeros((3, 4)),
        np.zeros(3),
        np.zeros((2, 3, 3)),
        [[0, 0, 0], [1, 1]],
        'abc',
        [['a', 'b', 'c'], ['d', 'e', 'f']],
        None,
    ])
    def test_invalid_shapes(self, vertices):
        with pytest.raises(ValueError):
            ArcSpline(vertices, False)

    def test_not_finite(self):
        for bad in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError, match='finite'):
                ArcSpline([(0, 0, 0), (bad, 1, 0)], False)
            with pytest.raises(ValueError, match='finite'):
                ArcSpline([(0, 0, bad), (1, 1, 0)], False)

    def test_at_least_two_vertices(self):
        for vertices in (np.zeros((0, 3)), [(0, 0, 0)]):
            with pytest.raises(ValueError, match='two vertices'):
                ArcSpline(vertices, False)
        with pytest.raises(ValueError, match='two vertices'):
            ArcSpline(_libfibomat.ArcSpline(np.zeros((1, 3)), False))

    def test_bulge_larger_than_one(self):
        with pytest.raises(ValueError, match='bulge'):
            ArcSpline([(0, 0, 1.5), (1, 0, 0)], False)
        with pytest.raises(ValueError, match='bulge'):
            ArcSpline([(0, 0, -1.01), (1, 0, 0)], False)

    def test_bulge_rounding_errors_are_clipped(self):
        spline = ArcSpline([(0, 0, 1 + 1e-12), (1, 0, -1 - 1e-12)], True)
        assert spline.vertices[:, 2].tolist() == [1., -1.]

    def test_input_array_is_not_modified_or_referenced(self):
        vertices = np.array(SQUARE, dtype=float)
        spline = ArcSpline(vertices, True)
        vertices[0, 0] = 100.
        assert spline.start == (0., 0.)
        spline._impl_translate(Vector(1, 1))
        assert vertices[1, 0] == 2.

    def test_len(self):
        assert len(square()) == 4
        assert len(circle()) == 2

    def test_repr(self):
        assert repr(square(False)) == 'ArcSpline(start=Vector(x=0.0, y=0.0), end=Vector(x=0.0, y=2.0), description=None)'
        assert repr(square()).startswith('ArcSpline(start=Vector(x=0.0, y=0.0), end=Vector(x=0.0, y=0.0)')


class TestCopy:
    def test_clone(self):
        spline = ArcSpline(SQUARE, True, 'foo')
        clone = spline.clone()
        assert type(clone) is ArcSpline
        assert clone is not spline
        assert clone.arc_spline_impl is not spline.arc_spline_impl
        assert clone.vertices.tolist() == spline.vertices.tolist()
        assert clone.is_closed and clone.description == 'foo'

    def test_clone_is_independent(self):
        spline = square()
        clone = spline.clone()
        clone._impl_translate(Vector(1, 1))
        assert spline.start == (0., 0.)
        assert clone.start == (1., 1.)

    def test_clone_keeps_pivot(self):
        # the former __deepcopy__ lost the pivot
        spline = square()
        spline.pivot = lambda obj: Vector(1, 1)
        assert spline.clone().pivot == (1., 1.)
        assert spline.translated((1, 1)).pivot == (1., 1.)
        assert copy.deepcopy(spline).pivot == (1., 1.)

    def test_copy_and_deepcopy(self):
        spline = ArcSpline(SQUARE, True, 'foo')
        for copied in (copy.copy(spline), copy.deepcopy(spline)):
            assert type(copied) is ArcSpline
            assert copied.vertices.tolist() == spline.vertices.tolist()
            assert copied.description == 'foo'

    def test_with_changed_description(self):
        res = square().with_changed_description('bar')
        assert res.description == 'bar'
        assert res.is_closed

    def test_clone_with_new_description_is_deprecated(self):
        spline = ArcSpline(SQUARE, True, 'foo')
        with pytest.warns(DeprecationWarning):
            res = spline.clone_with_new_description('bar')
        assert res.description == 'bar' and spline.description == 'foo'
        with pytest.warns(DeprecationWarning):
            assert spline.clone_with_new_description().description is None

    def test_pickle(self):
        pickled = pickle.loads(pickle.dumps(ArcSpline(SQUARE, True, 'foo')))
        assert pickled.vertices.tolist() == [list(map(float, v)) for v in SQUARE]
        assert pickled.is_closed and pickled.description == 'foo'


class TestFromSegments:
    def test_open_chain_of_lines(self):
        spline = ArcSpline.from_segments([Line((0, 0), (1, 0)), Line((1, 0), (1, 1)), Line((1, 1), (2, 1))])
        assert not spline.is_closed
        assert spline.vertices[:, :2].tolist() == [[0, 0], [1, 0], [1, 1], [2, 1]]

    def test_closed_chain(self):
        spline = ArcSpline.from_segments(
            [Line((0, 0), (1, 0)), Line((1, 0), (1, 1)), Line((1, 1), (0, 1)), Line((0, 1), (0, 0))], 'foo'
        )
        assert spline.is_closed
        assert len(spline) == 4
        assert spline.area == pytest.approx(1.)
        assert spline.description == 'foo'

    def test_arcs_keep_their_bulge(self):
        spline = ArcSpline.from_segments([
            Line((0, 0), (1, 0)), Arc.from_bulge((1, 0), (1, 1), 1.), Line((1, 1), (0, 1)), Line((0, 1), (0, 0))
        ])
        assert spline.is_closed
        assert spline.vertices[:, 2] == pytest.approx([0., 1., 0., 0.])
        assert spline.area == pytest.approx(1. + math.pi / 8)

    def test_single_closed_segment_stays_closed(self):
        # the former implementation opened a lone closed segment
        spline = ArcSpline.from_segments([circle()], 'circle')
        assert spline.is_closed
        assert spline.vertices.tolist() == [[1, 0, 1], [-1, 0, 1]]
        assert spline.description == 'circle'
        assert ArcSpline.from_segments([square()]).is_closed

    def test_single_open_segment(self):
        spline = ArcSpline.from_segments([Line((0, 0), (1, 1))])
        assert not spline.is_closed
        assert len(spline) == 2

    def test_accepts_generators_and_splines(self):
        spline = ArcSpline.from_segments(seg for seg in [Line((0, 0), (1, 0)), ArcSpline([(1, 0, 0), (2, 0, 0)], False)])
        assert spline.vertices[:, :2].tolist() == [[0, 0], [1, 0], [2, 0]]

    def test_result_is_independent_of_single_segment(self):
        original = square()
        spline = ArcSpline.from_segments([original])
        spline._impl_translate(Vector(1, 1))
        assert original.start == (0., 0.)

    def test_not_connected(self):
        with pytest.raises(RuntimeError, match='C\\^0'):
            ArcSpline.from_segments([Line((0, 0), (1, 0)), Line((2, 0), (3, 0))])

    def test_closed_segment_among_others(self):
        with pytest.raises(RuntimeError, match='closed'):
            ArcSpline.from_segments([Line((0, 0), (1, 0)), square()])
        # the former implementation did not check the first segment
        with pytest.raises(RuntimeError, match='closed'):
            ArcSpline.from_segments([square(), Line((0, 0), (1, 0))])

    def test_empty(self):
        with pytest.raises(ValueError):
            ArcSpline.from_segments([])
        with pytest.raises(ValueError):
            ArcSpline.from_segments(iter([]))

    def test_from_shape(self):
        spline = ArcSpline.from_shape(Line((0, 0), (1, 1)))
        assert isinstance(spline, ArcSpline)
        assert spline.vertices[:, :2].tolist() == [[0, 0], [1, 1]]

    def test_to_arc_spline_returns_self(self):
        spline = square()
        assert spline.to_arc_spline() is spline


class TestProperties:
    def test_start_end(self):
        assert square(False).start == (0., 0.)
        assert square(False).end == (0., 2.)
        assert square().end == (0., 0.)  # closed curves end at their start
        assert type(square().start) is Vector

    def test_vertices_is_a_copy(self):
        spline = square()
        vertices = spline.vertices
        assert isinstance(vertices, np.ndarray) and vertices.shape == (4, 3)
        vertices[0, 0] = 100.
        assert spline.start == (0., 0.)

    def test_length_and_boundary_length(self):
        assert square().length == pytest.approx(8.)
        assert square(False).length == pytest.approx(6.)
        assert circle().length == pytest.approx(2 * math.pi)
        # the former implementation did not define boundary_length
        assert square().boundary_length == pytest.approx(8.)
        assert square(False).boundary_length == pytest.approx(6.)

    def test_area(self):
        assert square().area == pytest.approx(4.)
        assert circle().area == pytest.approx(math.pi)
        assert rounded().area == pytest.approx(1. + math.pi / 8)
        reversed_ = square().reversed()
        assert reversed_.area == pytest.approx(4.)  # area is not signed

    def test_area_of_open_curve(self):
        with pytest.raises(NotImplementedError, match='closed'):
            square(False).area

    def test_orientation(self):
        assert square().orientation is True
        assert square().reversed().orientation is False
        with pytest.raises(RuntimeError):
            square(False).orientation

    def test_center_is_center_of_bounding_box(self):
        assert square().center == (1., 1.)
        assert type(square().center) is Vector
        # the mean of the vertices would be (2, 1)
        assert ArcSpline([(0, 0, 0), (3, 0, 0), (3, 3, 0)], False).center == (1.5, 1.5)
        # the mean of the vertices would be (0.5, 0.5); the arc extends the box to x = 1.5
        assert rounded().center == (0.75, 0.5)
        assert rounded().center == rounded().bounding_box.center

    def test_center_is_invariant_under_vertex_density(self):
        coarse = ArcSpline([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], True)
        dense = ArcSpline([(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], True)
        assert coarse.center == dense.center

    def test_rotation_about_center(self):
        spline = ArcSpline([(0, 0, 0), (3, 0, 0), (3, 3, 0)], False)
        rotated = spline.rotated(math.pi, 'center')
        assert rotated.bounding_box == spline.bounding_box

    def test_bounding_box_lines(self):
        assert square().bounding_box == BoundingBox((0, 0), (2, 2))
        assert square(False).bounding_box == BoundingBox((0, 0), (2, 2))

    def test_bounding_box_arcs(self):
        assert circle().bounding_box == BoundingBox((-1, -1), (1, 1))
        # half circle on the right side: x extends to 1.5
        assert rounded().bounding_box == BoundingBox((0, 0), (1.5, 1))

    def test_bounding_box_of_many_arcs_matches_sampling(self):
        # the former implementation worked around a "bug" by creating one python object per segment
        angles = np.linspace(0, 2 * np.pi, 41)[:-1]
        points = np.c_[3 * np.cos(angles), 2 * np.sin(angles)]
        vertices = np.c_[points, np.full(len(points), 0.3)]
        bbox = ArcSpline(vertices, True).bounding_box
        assert bbox.lower_left.x < points[:, 0].min() and bbox.upper_right.x > points[:, 0].max()
        assert bbox.contains(BoundingBox.from_points(points))

    def test_arc_spline_impl(self):
        assert isinstance(square().arc_spline_impl, _libfibomat.ArcSpline)

    def test_contains(self):
        assert square().contains((1, 1)) is True
        assert square().contains((3, 1)) is False
        assert circle().contains(Vector(0.5, 0.5))
        assert not circle().contains((1, 1))
        assert rounded().contains((1.2, 0.5))
        with pytest.raises(RuntimeError):
            square(False).contains((1, 1))

    def test_closest_point(self):
        res = square().closest_point((1, -1))
        assert set(res) == {'segment', 'point', 'distance'}
        assert res['segment'] == 0  # the former implementation returned the x coordinate of the position
        assert res['point'] == (1., 0.)
        assert res['distance'] == pytest.approx(1.)
        assert type(res['point']) is Vector
        assert square().closest_point((5, 1))['segment'] == 1
        assert square().closest_point((-3, 1.5))['segment'] == 3
        assert square().closest_point((5, 1))['distance'] == pytest.approx(3.)

    def test_closest_point_of_arc(self):
        res = circle().closest_point((3, 0))
        assert res['point'] == (1., 0.)
        assert res['distance'] == pytest.approx(2.)

    def test_reversed(self):
        spline = rounded()
        res = spline.reversed()
        assert res is not spline
        assert res.is_closed
        assert res.area == pytest.approx(spline.area)
        assert res.reversed().vertices.tolist() == spline.vertices.tolist()
        assert spline.vertices[1, 2] == 1.  # original not modified


class TestSegments:
    def test_segments_of_closed_curve(self):
        segments = rounded().segments
        assert len(segments) == 4
        assert [type(seg) for seg in segments] == [Line, Arc, Line, Line]
        assert segments[0].start == (0., 0.)
        assert segments[3].end == (0., 0.)  # closing segment

    def test_segments_of_open_curve(self):
        segments = square(False).segments
        assert len(segments) == 3
        assert all(isinstance(seg, Line) for seg in segments)

    def test_segments_roundtrip(self):
        spline = rounded()
        again = ArcSpline.from_segments(spline.segments)
        assert again.is_closed
        assert again.area == pytest.approx(spline.area)
        assert again.length == pytest.approx(spline.length)

    def test_segments_at_vertex(self):
        before, after = rounded().segments_at_vertex(1)
        assert isinstance(before, Line) and isinstance(after, Arc)
        before, after = rounded().segments_at_vertex(0)  # closed curve
        assert isinstance(before, Line) and isinstance(after, Line)
        assert before.end == (0., 0.)
        assert after.start == (0., 0.)

    def test_segments_at_vertex_of_open_curve(self):
        spline = square(False)
        assert spline.segments_at_vertex(0)[0] is None
        assert spline.segments_at_vertex(0)[1] is not None
        assert spline.segments_at_vertex(3)[1] is None
        assert spline.segments_at_vertex(3)[0] is not None
        assert None not in spline.segments_at_vertex(1)

    @pytest.mark.parametrize('method', ['segments_at_vertex', 'unit_tangents'])
    def test_invalid_vertex_index(self, method):
        spline = square()
        for index in (4, 100, -1, -4):
            with pytest.raises(ValueError):
                getattr(spline, method)(index)
        with pytest.raises(TypeError):
            getattr(spline, method)(1.5)
        with pytest.raises(TypeError):
            getattr(spline, method)('a')
        getattr(spline, method)(np.int64(1))


class TestTangentsAndKinks:
    def test_unit_tangents_lines(self):
        before, after = square().unit_tangents(1)
        assert before == (1., 0.) and after == (0., 1.)
        assert type(before) is Vector and type(after) is Vector

    def test_unit_tangents_closed_curve_wraps(self):
        before, after = square().unit_tangents(0)
        assert before == (0., -1.)
        assert after == (1., 0.)
        before, after = square().unit_tangents(3)
        assert before == (-1., 0.)
        assert after == (0., -1.)

    def test_unit_tangents_open_curve(self):
        spline = square(False)
        assert spline.unit_tangents(0)[0] is None
        assert spline.unit_tangents(0)[1] == (1., 0.)
        assert spline.unit_tangents(3)[1] is None
        assert spline.unit_tangents(3)[0] == (-1., 0.)

    def test_unit_tangents_arc(self):
        # two counterclockwise half circles around the origin: the circle is smooth in both vertices
        before, after = circle().unit_tangents(0)
        assert after == pytest.approx((0., 1.))
        assert before == pytest.approx((0., 1.))
        before, after = circle().unit_tangents(1)
        assert after == pytest.approx((0., -1.))
        assert before == pytest.approx((0., -1.))

    def test_unit_tangents_are_unit_vectors(self):
        for i in range(4):
            for tangent in rounded().unit_tangents(i):
                assert tangent.mag == pytest.approx(1.)

    def test_kinks_of_square(self):
        assert square().kinks() == [0, 1, 2, 3]
        assert square(False).kinks() == [1, 2]

    def test_kinks_of_smooth_curve(self):
        assert circle().kinks() == []
        # four quarter arcs are C^1
        bulge = math.tan(math.pi / 8)
        quarter = ArcSpline([(1, 0, bulge), (0, 1, bulge), (-1, 0, bulge), (0, -1, bulge)], True)
        assert quarter.kinks() == []

    def test_kinks_collinear_vertices(self):
        spline = ArcSpline([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)], False)
        assert spline.kinks() == [2]

    def test_kinks_agree_with_unit_tangents(self):
        spline = rounded()
        kinks = spline.kinks()
        for i in range(len(spline)):
            before, after = spline.unit_tangents(i)
            assert (i in kinks) == (not np.allclose(before, after))

    def test_kinks_of_big_spline_is_fast_enough(self):
        # the former implementation was quadratic (the vertex array was rebuilt for every vertex)
        n = 4000
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        spline = ArcSpline(np.c_[np.cos(angles), np.sin(angles), np.zeros(n)], True)
        assert len(spline.kinks()) == n  # every vertex of the polygon is a (small) kink
