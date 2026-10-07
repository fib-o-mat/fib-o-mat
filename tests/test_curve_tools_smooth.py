"""Tests of `fibomat.curve_tools.smooth`."""
import math

import numpy as np
import pytest

from fibomat.curve_tools import NonSmoothableError, smooth
from fibomat.curve_tools import rasterize
from fibomat.shapes import Arc, ArcSpline, Line


def square(size=2.):
    return ArcSpline([(0, 0, 0), (size, 0, 0), (size, size, 0), (0, size, 0)], True)


def corner_loss(radius):
    """Area between a right angle corner and its fillet."""
    return (1. - math.pi / 4.) * radius ** 2


def arcs_of(curve):
    return [segment for segment in curve.segments if isinstance(segment, Arc)]


class TestSmooth:
    def test_square(self):
        result = smooth(square(2.), 0.5)
        assert isinstance(result, ArcSpline)
        assert result.is_closed
        assert result.kinks() == []
        assert abs(result.area) == pytest.approx(4. - 4 * corner_loss(0.5))
        assert result.length == pytest.approx(4 * (2. - 2 * 0.5) + 2 * math.pi * 0.5)

    def test_fillet_radius_and_center(self):
        arcs = arcs_of(smooth(square(2.), 0.5))
        assert len(arcs) == 4
        centers = sorted(tuple(np.round(arc.center, 9)) for arc in arcs)
        assert centers == [(0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5)]
        assert all(arc.radius == pytest.approx(0.5) for arc in arcs)

    def test_orientation_is_kept(self):
        assert smooth(square(2.), 0.5).orientation
        assert not smooth(square(2.).reversed(), 0.5).orientation

    def test_negative_orientation_gives_same_area(self):
        assert abs(smooth(square(2.).reversed(), 0.5).area) == pytest.approx(abs(smooth(square(2.), 0.5).area))

    def test_radius_equal_to_half_side(self):
        # the result is a circle
        result = smooth(square(2.), 1.)
        assert result.kinks() == []
        assert abs(result.area) == pytest.approx(math.pi, rel=1e-9)

    def test_no_kinks_returns_curve(self):
        circle = ArcSpline([(1, 0, 1), (-1, 0, 1)], True)
        assert smooth(circle, 0.1) is circle

    def test_triangle(self):
        triangle = ArcSpline([(0, 0, 0), (4, 0, 0), (0, 3, 0)], True)
        result = smooth(triangle, 0.25)
        assert result.kinks() == []
        # the 3-4-5 triangle has the inradius 1; the fillet centers form the similar triangle with the inradius
        # 1 - r (the inner parallel curve). The result is the Minkowski sum of this triangle and a disc of radius r.
        scale = 1. - 0.25
        expected = 6. * scale ** 2 + 12. * scale * 0.25 + math.pi * 0.25 ** 2
        assert abs(result.area) == pytest.approx(expected)

    def test_concave_corner(self):
        l_shape = ArcSpline([(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)], True)
        radius = 0.25
        result = smooth(l_shape, radius)
        assert result.kinks() == []
        # five convex corners lose and the reflex corner gains area
        assert abs(result.area) == pytest.approx(3. - 5 * corner_loss(radius) + corner_loss(radius))
        # the fillet of the reflex corner is turned the other way around
        arcs = arcs_of(result)
        assert len(arcs) == 6
        assert sum(1 for arc in arcs if arc.sweep_dir) == 5

    def test_obtuse_corner(self):
        # regular hexagon (interior angle 120 deg): a fillet takes r * tan(30 deg) from every side
        angles = np.arange(6) * math.pi / 3
        hexagon = ArcSpline([(math.cos(a), math.sin(a), 0) for a in angles], True)
        radius = 0.2
        result = smooth(hexagon, radius)
        assert result.kinks() == []
        # all fillet arcs sweep 60 degrees
        for arc in arcs_of(result):
            assert arc.radius == pytest.approx(radius)
            assert arc.theta == pytest.approx(math.pi / 3)

    def test_open_curve(self):
        curve = ArcSpline([(0, 0, 0), (2, 0, 0), (2, 2, 0)], False)
        result = smooth(curve, 0.5)
        assert not result.is_closed
        assert result.kinks() == []
        assert np.allclose(result.start, (0, 0)) and np.allclose(result.end, (2, 2))
        assert result.length == pytest.approx(2 + 2 - 2 * 0.5 + math.pi * 0.5 / 2)

    def test_line_and_arc(self):
        # half disc: kinks of 90 degrees between line and arc
        half_disc = ArcSpline([(1, 0, 1), (-1, 0, 0)], True)
        radius = 0.2
        result = smooth(half_disc, radius)
        assert result.kinks() == []
        fillets = [arc for arc in arcs_of(result) if arc.radius == pytest.approx(radius)]
        assert len(fillets) == 2
        # the center has the distance r to the line and 1 - r to the origin (it lies inside of the half disc)
        expected = sorted(((math.sqrt((1 - radius) ** 2 - radius ** 2) * s, radius) for s in (-1, 1)))
        centers = sorted(tuple(arc.center) for arc in fillets)
        assert np.allclose(centers, expected)

    def test_arc_and_arc(self):
        # lens of two unit circles with the centers (0, 0) and (1, 0); the kinks are in (0.5, +-sqrt(3) / 2)
        sweep = 2 * math.pi / 3
        bulge = math.tan(sweep / 4)
        lens = ArcSpline([(0.5, -math.sqrt(3) / 2, bulge), (0.5, math.sqrt(3) / 2, bulge)], True)
        assert len(arcs_of(lens)) == 2
        result = smooth(lens, 0.1)
        assert result.kinks() == []
        arcs = arcs_of(result)
        assert len(arcs) == 4
        fillets = [arc for arc in arcs if arc.radius == pytest.approx(0.1)]
        assert len(fillets) == 2
        # the fillet centers have the distance 1 - r from the circle centers (inside of the lens)
        for fillet in fillets:
            distances = sorted([np.linalg.norm(fillet.center), np.linalg.norm(np.asarray(fillet.center) - (1, 0))])
            assert distances == pytest.approx([0.9, 0.9])

    def test_arc_trimming_keeps_geometry(self):
        # the trimmed arcs lie on the original circles
        half_disc = ArcSpline([(1, 0, 1), (-1, 0, 0)], True)
        result = smooth(half_disc, 0.2)
        for arc in arcs_of(result):
            if arc.radius != pytest.approx(0.2):
                assert arc.radius == pytest.approx(1.)
                assert np.allclose(arc.center, (0, 0))

    def test_distance_to_original_curve(self):
        radius = 0.1
        result = smooth(square(2.), radius)
        distances = [square(2.).closest_point(position)['distance'] for position in rasterize(result, 0.005).positions]
        # the largest deviation is in the middle of the fillets: its distance to the corner is r (sqrt(2) - 1) and its
        # distance to the (closest) edge is that divided by sqrt(2)
        assert max(distances) == pytest.approx(radius * (math.sqrt(2.) - 1.) / math.sqrt(2.), rel=1e-2)

    def test_radius_too_large(self):
        with pytest.raises(NonSmoothableError):
            smooth(square(2.), 1.5)

    def test_neighbouring_fillets_overlap(self):
        rectangle = ArcSpline([(0, 0, 0), (4, 0, 0), (4, 1, 0), (0, 1, 0)], True)
        smooth(rectangle, 0.5)  # fillets touch each other on the short sides
        with pytest.raises(NonSmoothableError):
            smooth(rectangle, 0.6)

    def test_cusp(self):
        cusp = ArcSpline([(0, 0, 0), (2, 0, 0), (1, 0, 0), (1, 1, 0)], False)
        with pytest.raises(NonSmoothableError):
            smooth(cusp, 0.1)

    def test_error_is_a_runtime_error(self):
        assert issubclass(NonSmoothableError, RuntimeError)

    @pytest.mark.parametrize('radius', [0., -1., math.inf, math.nan])
    def test_invalid_radius(self, radius):
        with pytest.raises(ValueError):
            smooth(square(), radius)

    def test_type_error(self):
        with pytest.raises(TypeError):
            smooth(Line((0, 0), (1, 1)), 0.1)

    def test_original_is_not_modified(self):
        curve = square()
        before = curve.vertices
        smooth(curve, 0.5)
        assert np.array_equal(curve.vertices, before)
