import copy
import math

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, Vector, VectorValueError, mirror, rotate, scale, translate
from fibomat.shapes.arc import Arc
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.biarc import Biarc
from fibomat.shapes.line import Line
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi

# (p_1, p_2, t_1, t_2, types of the segments)
EXAMPLES = {
    'two arcs': ((0, 1), (3, 0), (-1, 0), (0, 1), [Arc, Arc]),
    'one arc': ((-1, 0), (1, 0), (1, -1), (-1, -1), [Arc]),
    'line and arc': ((0, 0), (3, -1), (1, 0), (0, -1), [Line, Arc]),
    'two lines': ((0, 0), (3, 0), (1, 0), (1, 0), [Line, Line]),
    'two semicircles': ((0, 1), (0, -1), (1, 0), (1, 0), [Arc, Arc]),
    'equal tangents': ((0, 1), (1, -1), (1, 0), (1, 0), [Arc, Arc]),
}


def check_interpolation(biarc, p_1, p_2, t_1, t_2):
    """The biarc interpolates the points and the tangents and is G1."""
    spline = biarc.to_arc_spline()
    assert spline.start == pytest.approx(p_1, abs=1e-9)
    assert spline.end == pytest.approx(p_2, abs=1e-9)
    t_1, t_2 = Vector(t_1).normalized(), Vector(t_2).normalized()
    end_tangent = spline.unit_tangents(len(spline) - 1)[0]
    if len(biarc.segments) == 1:
        # the joint coincides with an end point; then the interpolation is a single arc (or line) which has the given
        # tangent lines but possibly the opposite direction at the other end
        assert np.allclose(end_tangent, t_2, atol=1e-7) or np.allclose(end_tangent, -t_2, atol=1e-7)
        assert np.allclose(spline.unit_tangents(0)[1], t_1, atol=1e-7) or np.allclose(
            spline.unit_tangents(0)[1], -t_1, atol=1e-7)
        return
    assert spline.unit_tangents(0)[1] == pytest.approx(t_1, abs=1e-7)
    assert end_tangent == pytest.approx(t_2, abs=1e-7)
    # smooth in all inner vertices (also in the joint)
    assert spline.kinks() == []


class TestConstruction:
    @pytest.mark.parametrize('name', EXAMPLES)
    def test_examples(self, name):
        p_1, p_2, t_1, t_2, types = EXAMPLES[name]
        biarc = Biarc(p_1, p_2, t_1, t_2)
        assert [type(seg) for seg in biarc.segments] == types
        check_interpolation(biarc, p_1, p_2, t_1, t_2)

    def test_two_semicircles(self):
        biarc = Biarc((0, 1), (0, -1), (1, 0), (1, 0))
        assert [seg.radius for seg in biarc.segments] == pytest.approx([0.5, 0.5])
        assert [seg.theta for seg in biarc.segments] == pytest.approx([PI, PI])
        assert biarc.length == pytest.approx(PI)
        # S-shaped: the arcs turn in different directions
        assert biarc.segments[0].sweep_dir != biarc.segments[1].sweep_dir

    def test_two_semicircles_other_orientation(self):
        biarc = Biarc((0, 1), (0, -1), (-1, 0), (-1, 0))
        assert biarc.segments[0].sweep_dir != biarc.segments[1].sweep_dir
        check_interpolation(biarc, (0, 1), (0, -1), (-1, 0), (-1, 0))

    def test_points_of_a_circle(self):
        # the interpolation of two points of a circle with its tangents lies on this circle (the joint is not unique)
        biarc = Biarc((1, 0), (0, 1), (0, 1), (-1, 0))
        arcs = biarc.segments
        assert all(arc.radius == pytest.approx(1.) and arc.center == pytest.approx((0., 0.)) for arc in arcs)
        assert sum(arc.theta for arc in arcs) == pytest.approx(PI / 2)
        check_interpolation(biarc, (1, 0), (0, 1), (0, 1), (-1, 0))

    def test_one_arc_if_the_joint_is_an_end_point(self):
        biarc = Biarc((-1, 0), (1, 0), (1, -1), (-1, -1))
        assert len(biarc.segments) == 1
        assert biarc.segments[0].start == pytest.approx((-1., 0.)) and biarc.segments[0].end == pytest.approx((1., 0.))

    def test_line(self):
        biarc = Biarc((0, 0), (4, 2), (2, 1), (4, 2))
        assert all(isinstance(seg, Line) for seg in biarc.segments)
        assert biarc.length == pytest.approx(math.sqrt(20))

    def test_tangent_lengths_are_ignored(self):
        a = Biarc((0, 0), (3, 1), (1, 0), (1, 1))
        b = Biarc((0, 0), (3, 1), (7, 0), (0.1, 0.1))
        assert a.to_arc_spline().vertices == pytest.approx(b.to_arc_spline().vertices)

    @pytest.mark.parametrize('seed', range(200))
    def test_random_configurations(self, seed):
        rng = np.random.default_rng(seed)
        p_1, p_2 = rng.uniform(-5, 5, (2, 2))
        t_1, t_2 = rng.uniform(-1, 1, (2, 2))
        if np.linalg.norm(t_1) < 0.05 or np.linalg.norm(t_2) < 0.05:
            return
        # (arcs of more than a half circle occur, e.g. if the tangents point away from the other point)
        check_interpolation(Biarc(p_1, p_2, t_1, t_2), p_1, p_2, t_1, t_2)

    @pytest.mark.parametrize('angle', np.linspace(0, 2 * PI, 13))
    def test_all_directions_of_second_tangent(self, angle):
        t_2 = (math.cos(angle), math.sin(angle))
        check_interpolation(Biarc((0, 0), (2, 1), (1, 0), t_2), (0, 0), (2, 1), (1, 0), t_2)

    def test_nearly_equal_tangents(self):
        for eps in (1e-3, 1e-6, 1e-9, 1e-12):
            check_interpolation(Biarc((0, 0), (3, 1), (1, 0), (1, eps)), (0, 0), (3, 1), (1, 0), (1, eps))

    def test_nearly_antiparallel_tangents(self):
        check_interpolation(Biarc((0, 0), (3, 1), (1, 0), (-1, 1e-9)), (0, 0), (3, 1), (1, 0), (-1, 1e-9))

    def test_scale_and_translation_invariance(self):
        base = Biarc((0, 0), (3, 1), (1, 0.3), (0.5, 1))
        big = Biarc((1e6, 2e6), (1e6 + 3e3, 2e6 + 1e3), (1, 0.3), (0.5, 1))
        assert big.length == pytest.approx(1e3 * base.length, rel=1e-9)
        assert len(big.segments) == len(base.segments)

    def test_description(self):
        assert Biarc((0, 0), (1, 1), (1, 0), (0, 1), 'foo').description == 'foo'
        assert Biarc((0, 0), (1, 1), (1, 0), (0, 1), 'foo').to_arc_spline().description == 'foo'
        assert Biarc((0, 0), (1, 1), (1, 0), (0, 1)).description is None

    def test_vector_likes(self):
        biarc = Biarc(Vector(0, 0), np.array([3., 1.]), [1, 0], Vector(1, 1))
        assert biarc.start == (0., 0.) and biarc.end == pytest.approx((3., 1.))

    def test_invalid(self):
        with pytest.raises(ValueError, match='p_1 == p_2'):
            Biarc((1, 1), (1, 1), (1, 0), (1, 0))
        with pytest.raises(ValueError):
            Biarc((0, 0), (1, 1), (0, 0), (1, 0))
        with pytest.raises(ValueError):
            Biarc((0, 0), (1, 1), (1, 0), (0, 0))
        with pytest.raises(VectorValueError):
            Biarc((0, 0, 0), (1, 1), (1, 0), (1, 0))
        with pytest.raises(VectorValueError):
            Biarc((0, 0), (1, 1), 1, (1, 0))


class TestSegments:
    def test_make_biarc_seg_arc(self):
        # arc with tangent (0, 1) at (1, 0) which ends at (0, 1)
        arc = Biarc._make_biarc_seg(Vector(1, 0), Vector(0, 1), Vector(0, 1))
        assert isinstance(arc, Arc)
        assert arc.center == pytest.approx((0., 0.)) and arc.radius == pytest.approx(1.)
        assert arc.sweep_dir is True
        assert arc.theta == pytest.approx(PI / 2)
        arc = Biarc._make_biarc_seg(Vector(1, 0), Vector(0, 1), Vector(0, -1))
        assert arc.sweep_dir is False
        assert arc.theta == pytest.approx(3 * PI / 2)  # (the former implementation always used the smaller angle)

    def test_make_biarc_seg_line_and_none(self):
        line = Biarc._make_biarc_seg(Vector(0, 0), Vector(2, 2), Vector(1, 1).normalized())
        assert isinstance(line, Line) and line.end == (2., 2.)
        assert Biarc._make_biarc_seg(Vector(1, 1), Vector(1, 1), Vector(1, 0)) is None

    def test_make_biarc_seg_has_the_given_tangent(self):
        for seed in range(30):
            rng = np.random.default_rng(seed)
            start, end = (Vector(*p) for p in rng.uniform(-3, 3, (2, 2)))
            tangent = Vector(*rng.uniform(-1, 1, 2)).normalized()
            arc = Biarc._make_biarc_seg(start, end, tangent)
            assert arc.start == pytest.approx(start) and arc.end == pytest.approx(end)
            assert arc.unit_tangent_start == pytest.approx(tangent)

    def test_segments_are_connected(self):
        segments = Biarc((0, 1), (3, 0), (-1, 0), (0, 1)).segments
        for first, second in zip(segments, segments[1:]):
            assert first.end == pytest.approx(second.start)

    def test_segments_with_more_than_a_half_circle(self):
        biarc = Biarc((0, 1), (3, 0), (-1, 0), (0, 1))
        assert sum(seg.theta for seg in biarc.segments) > PI
        assert biarc.length == pytest.approx(sum(seg.length for seg in biarc.segments))


class TestShapeInterface:
    def test_is_a_shape(self):
        biarc = Biarc((0, 0), (3, 1), (1, 0), (1, 0))
        assert isinstance(biarc, Shape) and isinstance(biarc, ArcSplineCompatible)
        assert biarc.is_closed is False
        with pytest.raises(NotImplementedError):
            biarc.area

    def test_to_arc_spline(self):
        biarc = Biarc((0, 0), (3, 1), (1, 0), (1, 0))
        spline = biarc.to_arc_spline()
        assert isinstance(spline, ArcSpline) and not spline.is_closed
        assert spline.length == pytest.approx(biarc.length) == pytest.approx(biarc.boundary_length)

    def test_to_arc_spline_is_a_copy(self):
        # the former implementation returned the internal arc spline
        biarc = Biarc((0, 0), (3, 1), (1, 0), (1, 0))
        spline = biarc.to_arc_spline()
        spline._impl_translate(Vector(10, 10))
        assert biarc.start == (0., 0.)
        assert biarc.to_arc_spline().start == (0., 0.)

    def test_start_end_center_bounding_box(self):
        biarc = Biarc((0, 0), (4, 0), (1, 0), (1, 0))
        assert biarc.start == (0., 0.) and biarc.end == pytest.approx((4., 0.))
        assert biarc.bounding_box == BoundingBox((0, 0), (4, 0))
        assert biarc.center == (2., 0.)

    def test_bounding_box_contains_the_curve(self):
        biarc = Biarc((0, 1), (3, 0), (-1, 0), (0, 1))
        bbox = biarc.bounding_box
        for seg in biarc.segments:
            assert bbox.contains(seg.bounding_box.extended(seg.bounding_box))
        assert bbox == biarc.to_arc_spline().bounding_box

    def test_repr(self):
        assert repr(Biarc((0, 0), (3, 0), (1, 0), (1, 0))) == (
            'Biarc(start=Vector(x=0.0, y=0.0), end=Vector(x=3.0, y=0.0), segments=2)'
        )

    def test_dim_shape(self):
        # the former Biarc was no Shape and could not be used in a pattern
        dim_biarc = Biarc((0, 0), (3, 1), (1, 0), (1, 0)) * unit('µm')
        assert dim_biarc.bounding_box.width.m_as('µm') == pytest.approx(
            Biarc((0, 0), (3, 1), (1, 0), (1, 0)).bounding_box.width)
        assert dim_biarc.to('nm').shape.end == pytest.approx((3000., 1000.))


class TestTransformations:
    def test_translate(self):
        biarc = Biarc((0, 0), (3, 1), (1, 0), (1, 0))
        res = biarc.translated((1, -1))
        assert isinstance(res, Biarc)
        assert res.start == pytest.approx((1., -1.)) and res.end == pytest.approx((4., 0.))
        assert biarc.start == (0., 0.)

    def test_rotate_scale_mirror(self):
        biarc = Biarc((0, 0), (3, 1), (1, 0.2), (1, 0.5))
        res = biarc.rotated(PI / 2)
        assert res.end == pytest.approx((-1., 3.))
        assert res.length == pytest.approx(biarc.length)
        assert biarc.scaled(2.).length == pytest.approx(2 * biarc.length)
        assert biarc.mirrored((1, 0)).end == pytest.approx((3., -1.))
        assert biarc.mirrored((1, 0)).length == pytest.approx(biarc.length)

    def test_transformed_is_still_g1(self):
        biarc = Biarc((0, 0), (3, 1), (1, 0.2), (1, 0.5))
        res = biarc.transformed(translate((1, 1)) | rotate(0.4, (2, 2)) | scale(1.5) | mirror((1, 2)))
        assert res.to_arc_spline().kinks() == []

    def test_clone(self):
        biarc = Biarc((0, 0), (3, 1), (1, 0.2), (1, 0.5), 'foo')
        clone = biarc.clone()
        assert clone is not biarc and clone.description == 'foo'
        assert clone.to_arc_spline().vertices == pytest.approx(biarc.to_arc_spline().vertices)
        assert copy.deepcopy(biarc).end == biarc.end
