"""Tests of the native extension `fibomat._libfibomat` (Rust, cavalier_contours)."""
import math
import pickle

import numpy as np
import pytest

from fibomat import _libfibomat as lib
from fibomat._libfibomat import ArcSpline


def square(size=2., closed=True):
    return ArcSpline([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], closed)


def circle(radius=1.):
    return ArcSpline([(radius, 0, 1), (-radius, 0, 1)], True)


def line(x0, y0, x1, y1):
    return ArcSpline([(x0, y0, 0), (x1, y1, 0)], False)


class TestArcSplineConstruction:
    def test_from_numpy(self):
        c = ArcSpline(np.array([[0., 0., 0.], [1., 0., 0.]]), False)
        assert c.size == 2
        assert not c.is_closed
        assert c.vertices == [(0., 0., 0.), (1., 0., 0.)]

    def test_from_lists_and_tuples(self):
        assert ArcSpline([[0, 0, 0], [1, 1, 0]], False).vertices == [(0., 0., 0.), (1., 1., 0.)]
        assert ArcSpline([(0, 0, 0), (1, 1, 0)], True).is_closed

    def test_copy_constructor(self):
        c = square()
        copy = ArcSpline(c)
        assert copy.vertices == c.vertices
        assert copy.is_closed
        copy.impl_translate((1, 1))
        assert copy.vertices != c.vertices

    def test_clone(self):
        c = square()
        assert c.clone().vertices == c.vertices

    def test_invalid_input(self):
        with pytest.raises(RuntimeError):
            ArcSpline(np.zeros((2, 2)), False)
        with pytest.raises(RuntimeError):
            ArcSpline(np.zeros(3), False)
        with pytest.raises(RuntimeError):
            ArcSpline([[0, 0]], False)
        with pytest.raises(RuntimeError):
            ArcSpline([[0, 0, 0]])  # is_closed missing

    def test_pickle(self):
        c = circle()
        restored = pickle.loads(pickle.dumps(c))
        assert restored.vertices == c.vertices
        assert restored.is_closed


class TestArcSplineProperties:
    def test_square(self):
        c = square()
        assert c.size == 4
        assert c.length == pytest.approx(8)
        assert c.area == pytest.approx(4)
        assert c.orientation
        assert c.center == pytest.approx((1, 1))
        # center of the bounding box, not the mean of the vertices
        assert ArcSpline([(0, 0, 0), (3, 0, 0), (3, 3, 0)], False).center == pytest.approx((1.5, 1.5))
        assert c.bounding_box == ((0, 0), (2, 2))
        assert c.start == (0, 0, 0)
        assert c.end == (0, 2, 0)

    def test_circle(self):
        c = circle(2.)
        assert c.length == pytest.approx(4 * math.pi)
        assert c.area == pytest.approx(4 * math.pi)
        (x0, y0), (x1, y1) = c.bounding_box
        assert (x0, y0, x1, y1) == pytest.approx((-2, -2, 2, 2))

    def test_reverse(self):
        c = square()
        c.reverse()
        assert not c.orientation
        assert c.area == pytest.approx(-4)
        assert c.start == (0, 2, 0)

    def test_empty_curve(self):
        c = ArcSpline(np.zeros((0, 3)), False)
        assert c.size == 0
        for name in ('start', 'end', 'bounding_box', 'center'):
            with pytest.raises(RuntimeError):
                getattr(c, name)

    def test_open_curve(self):
        c = line(0, 0, 1, 0)
        with pytest.raises(RuntimeError):
            c.orientation
        with pytest.raises(RuntimeError):
            c.contains(0, 0)

    def test_contains(self):
        c = square()
        assert c.contains(1, 1)
        assert not c.contains(3, 1)

    def test_closest_point(self):
        index, point, distance = square().closest_point(1, -1)
        assert index == 0
        assert point == pytest.approx((1, 0))
        assert distance == pytest.approx(1)

    def test_visit(self):
        calls = []

        def visitor(index, start, end):
            calls.append((index, start, end))
            return True

        square().visit(visitor)
        assert [c[0] for c in calls] == [0, 1, 2, 3]
        assert calls[3][2] == (0, 0, 0)  # closing segment

        # the segment index starts at zero in every call (was a static counter in the C++ version)
        calls.clear()
        square().visit(visitor)
        assert [c[0] for c in calls] == [0, 1, 2, 3]

    def test_visit_stops(self):
        calls = []

        def visitor(index, start, end):
            calls.append(index)
            return index < 1

        square().visit(visitor)
        assert calls == [0, 1]


class TestArcSplineTransformations:
    def test_translate(self):
        c = square()
        c.impl_translate((1, -1))
        assert c.start == (1, -1, 0)
        c.impl_translate(np.array([1., 1.]))
        assert c.start == (2, 0, 0)

    def test_scale(self):
        c = square()
        c.impl_scale(2.)
        assert c.area == pytest.approx(16)

    def test_rotate(self):
        c = ArcSpline([(1, 0, 0.5), (2, 0, 0)], False)
        c.impl_rotate(math.pi / 2)
        assert c.start == pytest.approx((0, 1, 0.5))
        assert c.end == pytest.approx((0, 2, 0))

    def test_mirror(self):
        c = ArcSpline([(1, 2, 0.5), (3, 4, 0)], False)
        c.impl_mirror((1, 0))
        assert c.start == pytest.approx((1, -2, -0.5))

        c = ArcSpline([(1, 0, 0), (2, 0, 0)], False)
        c.impl_mirror((1, 1))
        assert c.start == pytest.approx((0, 1, 0))

    def test_mirror_invalid_axis(self):
        with pytest.raises(RuntimeError):
            square().impl_mirror((0, 0))
        with pytest.raises(RuntimeError):
            square().impl_mirror((1, 0, 0))

    def test_translate_invalid_vector(self):
        with pytest.raises(RuntimeError):
            square().impl_translate((1, 2, 3))


class TestIntersections:
    def test_self_intersections(self):
        bow_tie = ArcSpline([(0, 0, 0), (2, 2, 0), (2, 0, 0), (0, 2, 0)], True)
        res = lib.self_intersections(bow_tie)
        assert len(res) == 1
        assert res[0][2] == pytest.approx((1, 1))
        assert lib.self_intersections(square()) == []

    def test_curve_intersections(self):
        crossings, coincidences = lib.curve_intersections(line(-1, 0, 1, 0), line(0, -1, 0, 1))
        assert len(crossings) == 1
        assert crossings[0][:2] == (0, 0)
        assert crossings[0][2] == pytest.approx((0, 0))
        assert coincidences == []

    def test_curve_intersections_arcs(self):
        crossings, _ = lib.curve_intersections(circle(), line(-2, 0, 2, 0))
        points = sorted(p for _, _, p in crossings)
        assert points == pytest.approx([(-1, 0), (1, 0)])

    def test_coincident(self):
        crossings, coincidences = lib.curve_intersections(line(0, 0, 2, 0), line(1, 0, 3, 0))
        assert crossings == []
        assert len(coincidences) == 1
        _, _, p1, p2 = coincidences[0]
        assert sorted([p1, p2]) == pytest.approx([(1, 0), (2, 0)])

    def test_no_intersection(self):
        assert lib.curve_intersections(line(0, 0, 1, 0), line(0, 1, 1, 1)) == ([], [])


class TestCombine:
    @staticmethod
    def shifted_squares():
        a = square(2.)
        b = square(2.)
        b.impl_translate((1, 1))
        return a, b

    @staticmethod
    def total_area(curves):
        return sum(abs(c.area) for c in curves)

    @pytest.mark.parametrize('mode, area', [('union', 7), ('intersect', 1), ('exclude', 3), ('xor', 6)])
    def test_modes(self, mode, area):
        a, b = self.shifted_squares()
        remaining, _ = lib.combine_curves(a, b, mode)
        assert self.total_area(remaining) == pytest.approx(area)

    def test_unknown_mode(self):
        a, b = self.shifted_squares()
        with pytest.raises(RuntimeError, match='Unknown combining mode'):
            lib.combine_curves(a, b, 'foo')

    def test_open_curve(self):
        with pytest.raises(RuntimeError, match='closed'):
            lib.combine_curves(square(), line(0, 0, 1, 1), 'union')


class TestOffset:
    def test_offset_square(self):
        res = lib.offset_curve(square(4.), 1.)
        assert len(res) == 1
        assert res[0].area == pytest.approx(4)

    def test_offset_vanishes(self):
        assert lib.offset_curve(square(4.), 3.) == []

    def test_offset_outwards(self):
        res = lib.offset_curve(square(2.), -1.)
        assert len(res) == 1
        assert res[0].area == pytest.approx(4 + 4 * 2 + math.pi)

    def test_offset_with_islands(self):
        outer = square(10.)
        island = square(2.)
        island.impl_translate((4, 4))
        islands, outers = lib.offset_with_islands([island], outer, 1.)
        assert len(outers) == 1 and len(islands) == 1
        assert outers[0].area == pytest.approx(64)
        assert abs(islands[0].area) == pytest.approx(12 + math.pi)

    def test_offset_with_islands_without_outer(self):
        island = square(2.)
        # islands are holes, so positive offsets grow them
        islands, outers = lib.offset_with_islands([island], None, 0.5)
        assert outers == []
        assert len(islands) == 1
        assert abs(islands[0].area) == pytest.approx(4 + 4 + math.pi * 0.25)

    def test_offset_with_open_island(self):
        with pytest.raises(RuntimeError):
            lib.offset_with_islands([line(0, 0, 1, 1)], None, 1.)


class TestConvertArcsToLines:
    def test_circle(self):
        res = lib.convert_arcs_to_lines(circle(), 0.01)
        assert res.size > 8
        assert res.is_closed
        assert all(v[2] == 0 for v in res.vertices)
        assert res.area == pytest.approx(math.pi, abs=0.05)

    def test_lines_are_unchanged(self):
        res = lib.convert_arcs_to_lines(square(), 0.01)
        assert res.vertices == square().vertices


class TestCopy:
    def test_copy_and_deepcopy(self):
        import copy

        c = circle()
        for copied in (copy.copy(c), copy.deepcopy(c), copy.deepcopy({'curve': c})['curve']):
            assert isinstance(copied, ArcSpline)
            assert copied.vertices == c.vertices
            assert copied.is_closed

    def test_copies_are_independent(self):
        import copy

        c = square()
        for copied in (copy.copy(c), copy.deepcopy(c), c.clone()):
            copied.impl_translate((1, 1))
            assert copied.start == (1, 1, 0)
            assert c.start == (0, 0, 0)

    def test_deepcopy_keeps_shared_references_of_containers(self):
        import copy

        c = square()
        res = copy.deepcopy([c, c])
        assert res[0] is res[1]
        assert res[0] is not c

    def test_deepcopy_does_not_use_reduce(self):
        # __reduce__ builds python lists of all vertices and is very slow for big curves
        import copy
        import time

        n = 200_000
        arr = np.c_[np.cos(np.linspace(0, 6.28, n)), np.sin(np.linspace(0, 6.28, n)), np.zeros(n)]
        big = ArcSpline(arr, True)
        start = time.perf_counter()
        copy.deepcopy(big)
        copy_time = time.perf_counter() - start
        start = time.perf_counter()
        pickle.dumps(big)
        pickle_time = time.perf_counter() - start
        assert copy_time < pickle_time / 5


class TestVerticesArray:
    def test_values_and_type(self):
        c = ArcSpline([(0, 0, 0.5), (1, 0, 0), (1, 1, -1)], False)
        arr = c.vertices_array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 3)
        assert arr.dtype == np.float64
        assert arr.tolist() == [list(v) for v in c.vertices]

    def test_empty_curve(self):
        assert ArcSpline(np.zeros((0, 3)), False).vertices_array.shape == (0, 3)

    def test_array_is_a_copy(self):
        c = square()
        arr = c.vertices_array
        arr[0, 0] = 100.
        assert c.start == (0, 0, 0)


class TestBoundingBoxWithArcs:
    # arcs sweep at most a half circle (|bulge| <= 1), as supported by cavalier_contours
    @pytest.mark.parametrize('seed', range(20))
    def test_matches_sampled_curve(self, seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(2, 8)
        vertices = np.c_[rng.uniform(-5, 5, n), rng.uniform(-5, 5, n), rng.uniform(-1, 1, n)]
        c = ArcSpline(vertices, bool(seed % 2))

        # sample all segments
        samples = []
        n_segments = n if c.is_closed else n - 1
        for i in range(n_segments):
            p0, p1, bulge = vertices[i, :2], vertices[(i + 1) % n, :2], vertices[i, 2]
            t = np.linspace(0, 1, 2001)[:, None]
            if abs(bulge) < 1e-12:
                samples.append(p0 + t * (p1 - p0))
                continue
            chord = p1 - p0
            length = np.linalg.norm(chord)
            theta = 4 * np.arctan(bulge)
            radius = length / (2 * np.sin(abs(theta) / 2))
            mid = (p0 + p1) / 2
            normal = np.array([-chord[1], chord[0]]) / length
            center = mid + normal * radius * np.cos(theta / 2) * np.sign(bulge)
            angle0 = np.arctan2(*(p0 - center)[::-1])
            angles = angle0 + t[:, 0] * theta
            samples.append(center + radius * np.c_[np.cos(angles), np.sin(angles)])
        pts = np.concatenate(samples)
        (x0, y0), (x1, y1) = c.bounding_box
        assert (x0, y0, x1, y1) == pytest.approx((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()), abs=2e-3)

