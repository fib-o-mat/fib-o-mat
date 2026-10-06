import copy
import math

import numpy as np
import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.linalg import BoundingBox, Vector, mirror, rotate, scale, translate
from fibomat.shapes.arc import Arc
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.shape import Shape
from fibomat.units import unit

PI = math.pi


def quarter():
    """Counterclockwise quarter circle of radius 2 around (1, 1)."""
    return Arc(2, 0, PI / 2, True, (1, 1))


def points_of(arc, n=7):
    """Points of the arc, sampled along the arc."""
    sign = 1. if arc.sweep_dir else -1.
    angles = arc.start_angle + sign * np.linspace(0, arc.theta, n)
    return np.asarray(arc.center) + arc.radius * np.c_[np.cos(angles), np.sin(angles)]


class TestConstruction:
    def test_basic(self):
        arc = Arc(2, 0, PI / 2, True, (1, 1), description='foo')
        assert arc.radius == 2.
        assert arc.start_angle == 0.
        assert arc.end_angle == pytest.approx(PI / 2)
        assert arc.sweep_dir is True
        assert arc.center == (1., 1.)
        assert arc.description == 'foo'
        assert isinstance(arc, Shape)

    def test_default_center(self):
        assert Arc(1, 0, 1, True).center == (0., 0.)

    def test_angles_are_normalized(self):
        arc = Arc(1, -PI / 2, 5 * PI, True)
        assert arc.start_angle == pytest.approx(3 * PI / 2)
        assert arc.end_angle == pytest.approx(PI)
        assert type(arc.start_angle) is float

    def test_numpy_arguments(self):
        arc = Arc(np.float64(2), np.float32(0), np.int64(1), np.bool_(True), np.array([1., 1.]))
        assert arc.radius == 2. and arc.sweep_dir is True
        assert type(arc.sweep_dir) is bool

    @pytest.mark.parametrize('kwargs', [
        {'radius': 0.}, {'radius': -1.}, {'radius': np.nan}, {'radius': np.inf},
        {'start_angle': np.nan}, {'end_angle': np.inf}, {'center': (np.nan, 0)},
    ])
    def test_invalid(self, kwargs):
        args = {'radius': 1., 'start_angle': 0., 'end_angle': 1., 'sweep_dir': True, 'center': (0, 0)}
        args.update(kwargs)
        with pytest.raises(ValueError):
            Arc(**args)

    def test_invalid_types(self):
        with pytest.raises((TypeError, ValueError)):
            Arc('a', 0, 1, True)

    def test_repr(self):
        assert repr(quarter()) == (
            f'Arc(r=2.0, start_angle=0.0, end_angle={PI / 2!r}, sweep_dir=True, center=Vector(x=1.0, y=1.0))'
        )


class TestAngles:
    @pytest.mark.parametrize('start, end, sweep, theta', [
        (0, PI / 2, True, PI / 2),
        (PI / 2, 0, True, 3 * PI / 2),
        (PI / 2, 0, False, PI / 2),
        (0, PI / 2, False, 3 * PI / 2),
        (3 * PI / 2, PI / 2, True, PI),
        (PI / 4, PI / 4, True, 2 * PI),
        (PI / 4, PI / 4, False, 2 * PI),
        (0., 2 * PI, True, 2 * PI),
        (0., 2 * PI, False, 2 * PI),
        (2 * PI, 0., True, 2 * PI),
        (2 * PI, 0., False, 2 * PI),
        (PI, 2 * PI, True, PI),
        (2 * PI, PI, False, PI),
        (2 * PI, PI / 2, True, PI / 2),
    ])
    def test_theta(self, start, end, sweep, theta):
        arc = Arc(1, start, end, sweep)
        assert arc.theta == pytest.approx(theta)
        assert arc.length == pytest.approx(theta)
        assert arc.boundary_length == arc.length
        assert arc.bulge == pytest.approx((1 if sweep else -1) * math.tan(theta / 4))

    def test_start_end_points(self):
        arc = quarter()
        assert arc.start == (3., 1.)
        assert arc.end == pytest.approx((1., 3.))

    def test_midpoint(self):
        assert quarter().midpoint == pytest.approx((1 + math.sqrt(2), 1 + math.sqrt(2)))
        # clockwise from 0 to pi / 2 is a 3/2 pi arc, its midpoint is at the angle -3/4 pi
        assert Arc(2, 0, PI / 2, False, (1, 1)).midpoint == pytest.approx((1 - math.sqrt(2), 1 - math.sqrt(2)))
        assert Arc(2, PI / 2, 0, False, (1, 1)).midpoint == pytest.approx((1 + math.sqrt(2), 1 + math.sqrt(2)))
        assert Arc(1, 0, 0, True).midpoint == pytest.approx((-1., 0.))

    def test_is_closed(self):
        assert not quarter().is_closed
        assert Arc(1, 0, 0, True).is_closed is True
        assert Arc(1, PI, PI, False).is_closed is True
        assert Arc(1, 0, 2 * PI, True).is_closed is True
        assert isinstance(Arc(1, 0, 0, True).is_closed, bool)

    def test_length_and_area(self):
        assert quarter().length == pytest.approx(PI)
        assert Arc(2, 0, 0, True).area == pytest.approx(4 * PI)
        with pytest.raises(NotImplementedError, match='closed'):
            quarter().area

    def test_unit_tangents(self):
        arc = quarter()
        assert arc.unit_tangent_start == pytest.approx((0., 1.))
        assert arc.unit_tangent_end == pytest.approx((-1., 0.))
        arc = Arc(2, 0, PI / 2, False, (1, 1))
        assert arc.unit_tangent_start == pytest.approx((0., -1.))
        assert type(arc.unit_tangent_end) is Vector

    def test_unit_tangent_at(self):
        arc = quarter()
        assert arc.unit_tangent_at(0.) == pytest.approx(arc.unit_tangent_start)
        assert arc.unit_tangent_at(arc.theta) == pytest.approx(arc.unit_tangent_end)
        assert arc.unit_tangent_at(PI / 4) == pytest.approx((-math.sqrt(0.5), math.sqrt(0.5)))
        clockwise = Arc(2, PI / 2, 0, False, (1, 1))
        assert clockwise.unit_tangent_at(PI / 4) == pytest.approx((math.sqrt(0.5), -math.sqrt(0.5)))
        assert clockwise.unit_tangent_at(clockwise.theta) == pytest.approx(clockwise.unit_tangent_end)

    def test_unit_tangent_at_invalid(self):
        with pytest.raises(ValueError):
            quarter().unit_tangent_at(PI)
        with pytest.raises(ValueError):
            quarter().unit_tangent_at(-0.1)


class TestFromBulge:
    def test_half_circle(self):
        arc = Arc.from_bulge((0, 0), (2, 0), 1)
        assert arc.center == pytest.approx((1., 0.))
        assert arc.radius == pytest.approx(1.)
        assert arc.sweep_dir is True
        assert arc.start == pytest.approx((0., 0.))
        assert arc.end == pytest.approx((2., 0.))
        assert arc.midpoint == pytest.approx((1., -1.))

    def test_negative_bulge_is_clockwise(self):
        arc = Arc.from_bulge((0, 0), (2, 0), -1)
        assert arc.sweep_dir is False
        assert arc.midpoint == pytest.approx((1., 1.))

    @pytest.mark.parametrize('bulge', [-3., -1., -0.5, -0.1, 0.1, 0.5, 1., 2.5])
    @pytest.mark.parametrize('start, end', [((0, 0), (2, 0)), ((1, 1), (-2, 3)), ((-5, 2), (-5, 1))])
    def test_roundtrip(self, start, end, bulge):
        arc = Arc.from_bulge(start, end, bulge)
        assert arc.start == pytest.approx(start)
        assert arc.end == pytest.approx(end)
        assert arc.bulge == pytest.approx(bulge)
        assert arc.sweep_dir is (bulge > 0)

    def test_quarter_circle(self):
        arc = Arc.from_bulge((1, 0), (0, 1), math.tan(PI / 8))
        assert arc.center == pytest.approx((0., 0.))
        assert arc.radius == pytest.approx(1.)
        assert arc.theta == pytest.approx(PI / 2)

    def test_invalid(self):
        with pytest.raises(ValueError):
            Arc.from_bulge((0, 0), (1, 0), 0.)
        with pytest.raises(ValueError):
            Arc.from_bulge((0, 0), (0, 0), 1.)
        with pytest.raises(ValueError):
            Arc.from_bulge((0, 0), (1, 0), np.nan)
        with pytest.raises(ValueError):
            Arc.from_bulge((0, 0), (1, 0), np.inf)

    def test_no_deprecation_warnings(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Arc.from_bulge((0, 0), (1, 0), 0.5)


class TestFromPoints:
    def test_counterclockwise(self):
        arc = Arc.from_points((1, 0), (0, 1), (-1, 0))
        assert arc.center == pytest.approx((0., 0.))
        assert arc.radius == pytest.approx(1.)
        assert arc.sweep_dir is True
        assert arc.theta == pytest.approx(PI)
        assert arc.midpoint == pytest.approx((0., 1.))

    def test_clockwise(self):
        arc = Arc.from_points((1, 0), (0, -1), (-1, 0))
        assert arc.sweep_dir is False
        assert arc.midpoint == pytest.approx((0., -1.))

    def test_large_arc(self):
        # p2 on the far side: going from p1 over p2 to p3 is clockwise and more than a half circle
        arc = Arc.from_points((1, 0), (-1, 0), (0, 1))
        assert arc.center == pytest.approx((0., 0.))
        assert arc.theta == pytest.approx(3 * PI / 2)
        assert arc.sweep_dir is False

    @pytest.mark.parametrize('seed', range(10))
    def test_random_points(self, seed):
        rng = np.random.default_rng(seed)
        p1, p2, p3 = rng.uniform(-5, 5, (3, 2))
        arc = Arc.from_points(p1, p2, p3)
        assert arc.start == pytest.approx(p1)
        assert arc.end == pytest.approx(p3)
        # p2 lies on the circle and on the arc
        assert np.linalg.norm(p2 - np.asarray(arc.center)) == pytest.approx(arc.radius)
        angle = (Vector(p2) - arc.center).angle_about_x_axis
        sweep = 1. if arc.sweep_dir else -1.
        assert (angle - arc.start_angle) * sweep % (2 * PI) <= arc.theta + 1e-9

    def test_invalid(self):
        with pytest.raises(ValueError, match='different'):
            Arc.from_points((0, 0), (0, 0), (1, 1))
        with pytest.raises(ValueError, match='different'):
            Arc.from_points((0, 0), (1, 1), (0, 0))
        with pytest.raises(ValueError, match='collinear'):
            Arc.from_points((0, 0), (1, 1), (2, 2))
        with pytest.raises(ValueError, match='collinear'):
            Arc.from_points((0, 0), (2, 0), (1, 0))


class TestFromPointsCenter:
    def test_basic(self):
        arc = Arc.from_points_center((1, 0), (0, 1), (0, 0), True)
        assert arc.radius == 1.
        assert arc.theta == pytest.approx(PI / 2)
        assert Arc.from_points_center((1, 0), (0, 1), (0, 0), False).theta == pytest.approx(3 * PI / 2)

    def test_full_circle(self):
        assert Arc.from_points_center((1, 0), (1, 0), (0, 0), True).is_closed

    def test_inconsistent_radii(self):
        with pytest.raises(ValueError):
            Arc.from_points_center((1, 0), (0, 2), (0, 0), True)
        with pytest.raises(ValueError):
            Arc.from_points_center((0, 0), (0, 0), (0, 0), True)

    def test_from_points_center_tangent(self):
        arc = Arc.from_points_center_tangent((1, 0), (0, 1), (0, 0), unit_tangent_start=(0, 1))
        assert arc.sweep_dir is True
        assert arc.theta == pytest.approx(PI / 2)

        arc = Arc.from_points_center_tangent((1, 0), (0, 1), (0, 0), unit_tangent_start=(0, -1))
        assert arc.sweep_dir is False
        assert arc.theta == pytest.approx(3 * PI / 2)

        arc = Arc.from_points_center_tangent((1, 0), (0, 1), (0, 0), unit_tangent_end=(-1, 0))
        assert arc.sweep_dir is True
        arc = Arc.from_points_center_tangent((1, 0), (0, 1), (0, 0), unit_tangent_end=(1, 0))
        assert arc.sweep_dir is False

    def test_from_points_center_tangent_invalid(self):
        with pytest.raises(ValueError):
            Arc.from_points_center_tangent((1, 0), (0, 1), (0, 0))
        with pytest.raises(ValueError):
            Arc.from_points_center_tangent((1, 0), (0, 1), (0, 0), unit_tangent_start=(0, 1), unit_tangent_end=(-1, 0))
        with pytest.raises(RuntimeError, match='tangent'):
            Arc.from_points_center_tangent((1, 0), (0, 1), (0, 0), unit_tangent_start=(1, 0))
        with pytest.raises(RuntimeError, match='tangent'):
            Arc.from_points_center_tangent((1, 0), (0, 1), (0, 0), unit_tangent_end=(0, 1))


class TestBoundingBox:
    def test_quarter(self):
        # 1st quadrant arc around (1, 1)
        assert quarter().bounding_box == BoundingBox((1, 1), (3, 3))

    def test_half_circles(self):
        assert Arc(1, 0, PI, True).bounding_box == BoundingBox((-1, 0), (1, 1))
        assert Arc(1, 0, PI, False).bounding_box == BoundingBox((-1, -1), (1, 0))
        assert Arc(1, PI / 2, 3 * PI / 2, True).bounding_box == BoundingBox((-1, -1), (0, 1))

    def test_full_circle(self):
        assert Arc(2, 0, 0, True, (1, 1)).bounding_box == BoundingBox((-1, -1), (3, 3))

    def test_arc_not_containing_axis_points(self):
        arc = Arc(1, PI / 8, PI / 4, True)
        points = np.array([arc.start, arc.end])
        assert arc.bounding_box == BoundingBox.from_points(points)

    def test_arc_through_zero_angle(self):
        arc = Arc(1, 7 * PI / 4, PI / 4, True)
        assert arc.bounding_box == BoundingBox((math.sqrt(0.5), -math.sqrt(0.5)), (1, math.sqrt(0.5)))

    @pytest.mark.parametrize('seed', range(40))
    def test_matches_sampled_arc(self, seed):
        rng = np.random.default_rng(seed)
        arc = Arc(rng.uniform(0.5, 3), rng.uniform(0, 2 * PI), rng.uniform(0, 2 * PI), bool(seed % 2),
                  rng.uniform(-3, 3, 2))
        points = points_of(arc, 4001)
        bbox = arc.bounding_box
        assert (*bbox.lower_left, *bbox.upper_right) == pytest.approx(
            (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()), abs=1e-4
        )


class TestSplitAndReverse:
    def test_split_small_arc(self):
        parts = quarter().split()
        assert len(parts) == 1
        assert parts[0] is not None
        assert parts[0].theta == pytest.approx(PI / 2)

    @pytest.mark.parametrize('sweep', [True, False])
    def test_split_large_arc(self, sweep):
        arc = Arc(1, 0, PI / 2, sweep, (1, 1))  # 3/2 pi if sweep is False
        if sweep:
            arc = Arc(1, PI / 2, 0, sweep, (1, 1))
        parts = arc.split()
        assert len(parts) == 2
        assert all(abs(part.bulge) <= 1 + 1e-12 for part in parts)
        assert parts[0].theta + parts[1].theta == pytest.approx(arc.theta)
        assert parts[0].end == pytest.approx(parts[1].start)
        assert parts[0].start == pytest.approx(arc.start)
        assert parts[1].end == pytest.approx(arc.end)

    def test_split_full_circle(self):
        parts = Arc(1, 0, 0, True).split()
        assert [part.theta for part in parts] == pytest.approx([PI, PI])
        assert parts[0].end == pytest.approx((-1., 0.))

    def test_split_at(self):
        parts = quarter().split_at(PI / 8)
        assert parts[0].theta == pytest.approx(PI / 8)
        assert parts[1].theta == pytest.approx(PI / 2 - PI / 8)
        assert parts[0].start == pytest.approx(quarter().start)
        assert parts[1].end == pytest.approx(quarter().end)
        assert parts[0].sweep_dir and parts[1].sweep_dir

        parts = Arc(1, PI / 2, 0, False).split_at(PI / 4)
        assert parts[0].theta == pytest.approx(PI / 4)
        assert parts[0].end == pytest.approx(parts[1].start)

    @pytest.mark.parametrize('angle', [0., -1., PI / 2, 2., 10.])
    def test_split_at_invalid(self, angle):
        # an empty part would be interpreted as full circle
        with pytest.raises(ValueError):
            quarter().split_at(angle)

    def test_split_does_not_modify(self):
        arc = quarter()
        arc.split_at(0.3)
        assert arc.theta == pytest.approx(PI / 2)

    def test_reversed(self):
        arc = quarter()
        rev = arc.reversed()
        assert rev.start == pytest.approx(arc.end)
        assert rev.end == pytest.approx(arc.start)
        assert rev.sweep_dir is False
        assert rev.theta == pytest.approx(arc.theta)
        assert rev.bulge == pytest.approx(-arc.bulge)
        assert rev.reversed().start == pytest.approx(arc.start)
        assert arc.sweep_dir is True


class TestToArcSpline:
    def test_small_arc(self):
        spline = quarter().to_arc_spline()
        assert isinstance(spline, ArcSpline)
        assert not spline.is_closed
        assert len(spline) == 2
        assert spline.vertices[0, 2] == pytest.approx(math.tan(PI / 8))
        assert spline.start == pytest.approx(quarter().start)
        assert spline.end == pytest.approx(quarter().end)
        assert spline.length == pytest.approx(quarter().length)

    def test_large_arc_is_split(self):
        arc = Arc(1, PI / 2, 0, True, (1, 1))
        spline = arc.to_arc_spline()
        assert len(spline) == 3
        assert not spline.is_closed
        assert spline.length == pytest.approx(arc.length)
        assert np.all(np.abs(spline.vertices[:, 2]) <= 1)
        assert spline.start == pytest.approx(arc.start)
        assert spline.end == pytest.approx(arc.end)

    def test_full_circle(self):
        arc = Arc(2, 0, 0, False, (1, 1), description='circle')
        spline = arc.to_arc_spline()
        assert spline.is_closed
        assert len(spline) == 2
        assert spline.area == pytest.approx(4 * PI)
        assert spline.length == pytest.approx(4 * PI)
        assert spline.description == 'circle'
        assert spline.bounding_box == arc.bounding_box

    @pytest.mark.parametrize('seed', range(10))
    def test_roundtrip_via_segments(self, seed):
        rng = np.random.default_rng(seed)
        arc = Arc(rng.uniform(0.5, 3), rng.uniform(0, 2 * PI), rng.uniform(0, 2 * PI), bool(seed % 2),
                  rng.uniform(-3, 3, 2))
        spline = arc.to_arc_spline()
        assert spline.length == pytest.approx(arc.length)
        assert spline.bounding_box == arc.bounding_box
        again = ArcSpline.from_segments(spline.segments)
        assert again.length == pytest.approx(arc.length)


class TestTransformations:
    @staticmethod
    def check(arc, transformed, func):
        """The transformed arc must be the arc through the transformed points of the original one."""
        expected = np.array([func(Vector(p)) for p in points_of(arc, 9)])
        assert points_of(transformed, 9) == pytest.approx(expected, abs=1e-9)

    def test_translate(self):
        arc = quarter()
        res = arc.translated((1, -2))
        self.check(arc, res, lambda p: p + (1, -2))
        assert res.center == (2., -1.)
        assert arc.center == (1., 1.)

    def test_rotate_around_origin(self):
        # the former implementation did not rotate the center
        arc = quarter()
        res = arc.rotated(PI / 2)
        assert res.center == pytest.approx((-1., 1.))
        self.check(arc, res, lambda p: p.rotated(PI / 2))

    def test_rotate_around_other_origins(self):
        arc = quarter()
        self.check(arc, arc.rotated(0.7, (3, -1)), lambda p: p.rotated(0.7, (3, -1)))
        self.check(arc, arc.rotated(0.7, 'center'), lambda p: p.rotated(0.7, arc.center))
        assert arc.rotated(0.7, 'center').center == pytest.approx(arc.center)

    def test_rotate_keeps_shape(self):
        arc = Arc(2, 1., 4., False, (1, 1))
        res = arc.rotated(2.5)
        assert res.theta == pytest.approx(arc.theta)
        assert res.radius == arc.radius
        assert res.sweep_dir == arc.sweep_dir

    def test_scale(self):
        # the former implementation did not scale the center
        arc = quarter()
        res = arc.scaled(2.)
        assert res.radius == 4.
        assert res.center == (2., 2.)
        self.check(arc, res, lambda p: p * 2.)
        self.check(arc, arc.scaled(0.5, (3, 3)), lambda p: (p - (3, 3)) * 0.5 + (3, 3))

    def test_scale_negative(self):
        arc = quarter()
        res = arc.scaled(-2.)
        assert res.radius == 4.
        assert res.sweep_dir is True
        self.check(arc, res, lambda p: p * -2.)

    @pytest.mark.parametrize('axis', [(1, 0), (0, 1), (1, 1), (-2, 1), (3, 0.5)])
    @pytest.mark.parametrize('sweep', [True, False])
    def test_mirror(self, axis, sweep):
        arc = Arc(2, 0.4, 2.2, sweep, (1, 1))
        res = arc.mirrored(axis)
        assert res.sweep_dir is (not sweep)
        assert res.theta == pytest.approx(arc.theta)
        self.check(arc, res, lambda p: p.mirrored(axis))

    def test_mirror_full_circle(self):
        res = Arc(1, 0, 0, True, (2, 0)).mirrored((0, 1))
        assert res.is_closed
        assert res.center == pytest.approx((-2., 0.))

    def test_transformed_chain(self):
        arc = quarter()
        res = arc.transformed(translate((1, 1)) | rotate(0.3, (2, 2)) | scale(1.5, 'center') | mirror((1, 2)))
        expected = lambda p: (  # noqa: E731
            (Vector(p) + (1, 1)).rotated(0.3, (2, 2))
        )
        center_1 = (arc.center + (1, 1)).rotated(0.3, (2, 2))
        scaled_center = center_1
        self.check(
            arc, res,
            lambda p: ((expected(p) - scaled_center) * 1.5 + scaled_center).mirrored((1, 2)),
        )

    def test_transformations_do_not_modify_original(self):
        arc = quarter()
        arc.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert arc.center == (1., 1.) and arc.radius == 2. and arc.sweep_dir is True

    def test_bounding_box_after_transformations(self):
        arc = quarter()
        res = arc.rotated(PI / 2, (0, 0))
        points = points_of(res, 2001)
        bbox = res.bounding_box
        assert (*bbox.lower_left, *bbox.upper_right) == pytest.approx(
            (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()), abs=1e-4
        )


class TestCopyAndDimShape:
    def test_clone(self):
        arc = Arc(2, 0, 1, True, (1, 1), description='foo')
        clone = arc.clone()
        assert clone is not arc
        assert (clone.radius, clone.start_angle, clone.end_angle, clone.sweep_dir) == (2., 0., 1., True)
        assert clone.center == (1., 1.) and clone.description == 'foo'
        assert copy.deepcopy(arc).center == arc.center

    def test_dim_shape(self):
        dim_arc = quarter() * unit('µm')
        assert dim_arc.center == Vector(1, 1) * unit('µm')
        assert dim_arc.bounding_box.width.m_as('µm') == pytest.approx(2.)
