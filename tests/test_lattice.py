"""Tests of `fibomat.arrangements.lattices` (`Lattice`, `DimLattice`, the builders and the helper functions)."""
import math

import numpy as np
import pytest

from fibomat.arrangements import (
    DimGroup, DimLattice, DimLatticeBuilder, FastAxis, Group, Lattice, LatticeBuilder
)
from fibomat.arrangements.lattices.lattice_points import boundary_points, count_points, grid_points
from fibomat.arrangements.utils import check_lattice_vectors
from fibomat.composite_shapes import HollowArcSpline, Ring
from fibomat.layout import Pattern, Site
from fibomat.linalg import DimVector, Vector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d
from fibomat.shapes import ArcSpline, Circle, DimShape, Line, Polygon, Rect
from fibomat.units import unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def centers(lattice):
    return np.array([np.asarray(element.center) for element in lattice.elements])


def dim_centers(lattice, length_unit='µm'):
    return np.array([np.asarray(element.center.vector_as(unit(length_unit))) for element in lattice.elements])


class TestHelpers:
    def test_fast_axis(self):
        assert FastAxis.parse('u') is FastAxis.U and FastAxis.parse('x') is FastAxis.U
        assert FastAxis.parse('V') is FastAxis.V and FastAxis.parse('y') is FastAxis.V
        assert FastAxis.parse(FastAxis.V) is FastAxis.V
        for value in ('z', 1, None):
            with pytest.raises(ValueError, match='fast axis'):
                FastAxis.parse(value)

    @pytest.mark.parametrize('size, pitch, expected', [
        (10., 2., 6), (10., 3., 4), (0., 1., 1), (0.5, 1., 1), (1., 1., 2), (0.3, 0.1, 4), (0.7, 0.1, 8), (1e-3, 1e-4, 11),
    ])
    def test_count_points(self, size, pitch, expected):
        assert count_points(size, pitch) == expected

    @pytest.mark.parametrize('size, pitch', [(-1., 1.), (math.inf, 1.), (1., 0.), (1., -1.), (1., math.nan)])
    def test_count_points_errors(self, size, pitch):
        with pytest.raises(ValueError):
            count_points(size, pitch)

    def test_grid_points(self):
        indices, xy = grid_points(3, 2, 2., 1., (10., 20.), FastAxis.U)
        assert indices.tolist() == [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
        assert xy.tolist() == [[8., 20.5], [10., 20.5], [12., 20.5], [8., 19.5], [10., 19.5], [12., 19.5]]

    def test_grid_points_column_by_column(self):
        indices, xy = grid_points(3, 2, 2., 1., (0., 0.), FastAxis.V)
        assert indices.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
        assert xy[:, 0].tolist() == [-2., -2., 0., 0., 2., 2.]

    @pytest.mark.parametrize('arguments', [(0, 1, 1., 1.), (1, 0, 1., 1.), (1.5, 1, 1., 1.), (1, 1, 0., 1.), (1, 1, 1., -1.)])
    def test_grid_points_errors(self, arguments):
        with pytest.raises(ValueError):
            grid_points(*arguments, (0., 0.), FastAxis.U)

    def test_check_lattice_vectors(self):
        check_lattice_vectors((1, 0), (0, 1))
        check_lattice_vectors((1, 0), (1, 1e-3))
        check_lattice_vectors((1e-6, 0), (0, 1e-6))  # small, but fine
        for u, v in (((1, 0), (2, 0)), ((1, 1), (-2, -2)), ((0, 0), (1, 0)), ((1, 0), (0, 0))):
            with pytest.raises(ValueError):
                check_lattice_vectors(u, v)

    def test_collinear_vectors_with_length_one_are_detected(self):
        # (the old check compared the dot product with 1)
        with pytest.raises(ValueError, match='collinear'):
            check_lattice_vectors((3, 0), (5, 0))
        check_lattice_vectors((1, 0), (math.cos(0.5), math.sin(0.5)))  # (dot product is not 1, but nearly)


class TestFromCounts:
    def test_points_and_order(self):
        lattice = Lattice.from_counts(3, 2, 10., 5., Circle(1.))
        assert centers(lattice).tolist() == [
            [-10., 2.5], [0., 2.5], [10., 2.5], [-10., -2.5], [0., -2.5], [10., -2.5]
        ]
        assert lattice.points_uv.tolist() == [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
        assert lattice.n_points == len(lattice) == 6

    def test_fast_axis_y(self):
        lattice = Lattice.from_counts(3, 2, 10., 5., Circle(1.), fast_axis='y')
        assert lattice.points_uv.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
        assert centers(lattice)[:2].tolist() == [[-10., 2.5], [-10., -2.5]]

    def test_fast_axis_enum_and_u_v_names(self):
        a = Lattice.from_counts(2, 2, 1., 1., Circle(0.1), fast_axis=FastAxis.V)
        b = Lattice.from_counts(2, 2, 1., 1., Circle(0.1), fast_axis='v')
        assert a.points_uv.tolist() == b.points_uv.tolist()

    def test_center(self):
        lattice = Lattice.from_counts(2, 2, 2., 2., Circle(1.), center=(100, -50))
        assert centers(lattice).tolist() == [[99., -49.], [101., -49.], [99., -51.], [101., -51.]]

    def test_odd_and_even_numbers_are_centered(self):
        assert centers(Lattice.from_counts(1, 1, 1., 1., Circle(0.1), center=(3, 4))).tolist() == [[3., 4.]]
        assert centers(Lattice.from_counts(2, 1, 4., 1., Circle(0.1)))[:, 0].tolist() == [-2., 2.]

    def test_lattice_vectors(self):
        lattice = Lattice.from_counts(2, 2, 3., 4., Circle(1.))
        assert lattice.u == Vector(3, 0) and lattice.v == Vector(0, -4)

    def test_elements_are_moved_with_their_pivot(self):
        lattice = Lattice.from_counts(2, 1, 10., 5., Rect(2, 2).translated((100, 100)))
        assert centers(lattice).tolist() == [[-5., 0.], [5., 0.]]
        assert lattice.bounding_box.lower_left == Vector(-6, -1)

    def test_every_element_is_a_copy(self):
        shape = Circle(1.)
        lattice = Lattice.from_counts(2, 1, 10., 5., shape)
        assert lattice.elements[0] is not shape and lattice.elements[0] is not lattice.elements[1]
        assert shape.center == Vector(0, 0)  # the original is not moved

    def test_group_elements(self):
        element = Group([Circle(1., center=(-1, 0)), Circle(1., center=(1, 0))])
        lattice = Lattice.from_counts(2, 1, 10., 5., element)
        assert lattice.bounding_box.width == pytest.approx(10 + 4)
        assert len(list(lattice.arrangement_elements())) == 4

    def test_description(self):
        assert Lattice.from_counts(1, 1, 1., 1., Circle(1.), description='lattice').description == 'lattice'

    def test_elements_by_uv(self):
        lattice = Lattice.from_counts(3, 2, 10., 5., Circle(1.))
        by_uv = lattice.elements_by_uv
        assert by_uv.shape == (2, 3)  # (n_v, n_u)
        assert by_uv[0, 0] is lattice.elements[0] and by_uv[1, 2] is lattice.elements[5]
        assert by_uv[1, 0].center == Vector(-10, -2.5)

    @pytest.mark.parametrize('arguments', [(0, 1), (1, 0), (-1, 2), (1.5, 1)])
    def test_invalid_counts(self, arguments):
        with pytest.raises(ValueError):
            Lattice.from_counts(*arguments, 1., 1., Circle(1.))

    @pytest.mark.parametrize('pitch', [0., -1., math.nan, math.inf])
    def test_invalid_pitch(self, pitch):
        with pytest.raises(ValueError):
            Lattice.from_counts(2, 2, pitch, 1., Circle(1.))
        with pytest.raises(ValueError):
            Lattice.from_counts(2, 2, 1., pitch, Circle(1.))

    @pytest.mark.parametrize('element', ['foo', 1, Circle(1.) * unit('µm')])
    def test_invalid_element(self, element):
        with pytest.raises(TypeError):
            Lattice.from_counts(2, 2, 1., 1., element)

    def test_is_a_group(self):
        lattice = Lattice.from_counts(2, 2, 1., 1., Circle(0.1))
        assert isinstance(lattice, Group) and repr(lattice) == 'Lattice(n_points=4)'


class TestCallback:
    def test_arguments(self):
        calls = []

        def element(i, uv, xy, n):
            calls.append((i, uv, xy, n))
            return Circle(0.1)

        Lattice.from_counts(3, 2, 2., 2., element, center=(1, 1))
        assert [call[0] for call in calls] == [0, 1, 2, 3, 4, 5]
        assert [call[1] for call in calls] == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        assert calls[0][2] == (-1., 2.) and calls[4][2] == (1., 0.)
        assert all(call[3] == 6 for call in calls)
        assert all(isinstance(value, int) for call in calls for value in (call[0], call[3], *call[1]))
        assert all(isinstance(value, float) for call in calls for value in call[2])

    def test_fast_axis_changes_the_order_of_the_calls(self):
        order = []
        Lattice.from_counts(2, 2, 1., 1., lambda i, uv, xy, n: order.append(uv) or Circle(0.1), fast_axis='y')
        assert order == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_none_means_no_element(self):
        lattice = Lattice.from_counts(
            3, 3, 1., 1., lambda i, uv, xy, n: Circle(0.1) if (uv[0] + uv[1]) % 2 == 0 else None
        )
        assert lattice.n_points == 5
        assert lattice.points_uv.tolist() == [[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]]
        by_uv = lattice.elements_by_uv
        assert by_uv[0, 1] is None and by_uv[1, 1] is lattice.elements[2]

    def test_the_returned_element_is_moved_to_the_point(self):
        lattice = Lattice.from_counts(
            2, 1, 10., 5., lambda i, uv, xy, n: Circle(1., center=(1000, 1000)))
        assert centers(lattice).tolist() == [[-5., 0.], [5., 0.]]

    def test_different_elements(self):
        lattice = Lattice.from_counts(
            2, 1, 10., 5., lambda i, uv, xy, n: Circle(1.) if i == 0 else Rect(2, 2)
        )
        assert isinstance(lattice.elements[0], Circle) and isinstance(lattice.elements[1], Rect)

    def test_the_index_counts_all_points(self):
        indices = []
        Lattice.from_counts(2, 2, 1., 1., lambda i, uv, xy, n: indices.append(i) or None if i < 3 else Circle(0.1))
        assert indices == [0, 1, 2]

    def test_no_element_at_all(self):
        with pytest.raises(ValueError, match='any element'):
            Lattice.from_counts(2, 2, 1., 1., lambda i, uv, xy, n: None)

    def test_invalid_return_value(self):
        with pytest.raises(TypeError):
            Lattice.from_counts(2, 2, 1., 1., lambda i, uv, xy, n: 'foo')
        with pytest.raises(TypeError):
            Lattice.from_counts(2, 2, 1., 1., lambda i, uv, xy, n: Circle(1.) * unit('µm'))


class TestFromRect:
    def test_example_of_the_documentation(self):
        assert centers(Lattice.from_rect(10., 0., 2., 1., Circle(0.1)))[:, 0].tolist() == [-5., -3., -1., 1., 3., 5.]
        assert centers(Lattice.from_rect(10., 0., 3., 1., Circle(0.1)))[:, 0].tolist() == [-4.5, -1.5, 1.5, 4.5]

    def test_all_points_are_in_the_rectangle(self):
        lattice = Lattice.from_rect(10., 6., 3., 2.5, Circle(0.1), center=(100, 50))
        points = centers(lattice)
        assert points[:, 0].min() >= 95 and points[:, 0].max() <= 105
        assert points[:, 1].min() >= 47 and points[:, 1].max() <= 53
        assert lattice.points_uv.max(axis=0).tolist() == [3, 2]

    def test_pitch_larger_than_size(self):
        lattice = Lattice.from_rect(1., 1., 5., 5., Circle(0.1), center=(3, 3))
        assert centers(lattice).tolist() == [[3., 3.]]

    def test_floating_point_sizes(self):
        assert Lattice.from_rect(0.3, 0.1, 0.1, 0.1, Circle(0.01)).n_points == 4 * 2

    def test_fast_axis_and_callback(self):
        lattice = Lattice.from_rect(2., 1., 1., 1., lambda i, uv, xy, n: Circle(0.1), fast_axis='y')
        assert lattice.points_uv.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

    def test_the_same_as_from_counts(self):
        a = Lattice.from_rect(10., 6., 2., 3., Circle(0.1))
        b = Lattice.from_counts(6, 3, 2., 3., Circle(0.1))
        assert np.allclose(centers(a), centers(b))

    @pytest.mark.parametrize('arguments', [(-1., 1., 1., 1.), (1., -1., 1., 1.), (1., 1., 0., 1.), (1., 1., 1., 0.)])
    def test_invalid_arguments(self, arguments):
        with pytest.raises(ValueError):
            Lattice.from_rect(*arguments, Circle(0.1))


class TestFromBoundary:
    def test_rectangle_with_orthogonal_vectors(self):
        lattice = Lattice.from_boundary(Rect(10, 6), (2, 0), (0, -2), Circle(0.1), origin=(0, 0))
        points = centers(lattice)
        assert points[:, 0].tolist() == [-4., -2., 0., 2., 4.] * 3 and sorted(set(points[:, 1])) == [-2., 0., 2.]
        assert lattice.points_uv.max(axis=0).tolist() == [4, 2]
        assert lattice.points_uv[0].tolist() == [0, 0]
        # (the point at y = 3 would be on the boundary: boundary points are not included)

    def test_origin_defaults_to_the_center_of_the_boundary(self):
        lattice = Lattice.from_boundary(Rect(10, 6, center=(100, 50)), (4, 0), (0, 3), Circle(0.1))
        assert (100, 50) in {tuple(point) for point in np.round(centers(lattice), 9)}

    def test_origin(self):
        lattice = Lattice.from_boundary(Rect(10, 10), (4, 0), (0, 4), Circle(0.1), origin=(0.5, 0.5))
        assert sorted(set(np.round(centers(lattice)[:, 0], 9))) == [-3.5, 0.5, 4.5]

    def test_fast_axis(self):
        u_fast = Lattice.from_boundary(Rect(5, 5), (2, 0), (0, 2), Circle(0.1), origin=(0, 0), fast_axis='u')
        v_fast = Lattice.from_boundary(Rect(5, 5), (2, 0), (0, 2), Circle(0.1), origin=(0, 0), fast_axis='v')
        assert u_fast.points_uv.tolist() == [[i_u, i_v] for i_v in range(3) for i_u in range(3)]
        assert v_fast.points_uv.tolist() == [[i_u, i_v] for i_u in range(3) for i_v in range(3)]
        assert sorted(map(tuple, centers(u_fast))) == sorted(map(tuple, centers(v_fast)))

    def test_hexagonal_lattice_in_a_circle(self):
        u = (2., 0.)
        v = (1., math.sqrt(3))
        lattice = Lattice.from_boundary(Circle(5.), u, v, Circle(0.1), origin=(0, 0))
        points = centers(lattice)
        assert np.all(np.linalg.norm(points, axis=1) < 5.)
        # brute force
        expected = [
            (i * u[0] + j * v[0], i * u[1] + j * v[1])
            for i in range(-10, 11) for j in range(-10, 11)
            if math.hypot(i * u[0] + j * v[0], i * u[1] + j * v[1]) < 5. - 1e-9
        ]
        assert sorted(map(lambda p: tuple(np.round(p, 6)), points)) == sorted(map(lambda p: tuple(np.round(p, 6)), expected))
        assert lattice.u == Vector(*u) and lattice.v == Vector(*v)

    @pytest.mark.parametrize('seed', range(5))
    def test_random_lattices_match_a_brute_force_search(self, seed):
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.5, 2., 2) * rng.choice([-1, 1], 2)
        v = rng.uniform(0.5, 2., 2) * rng.choice([-1, 1], 2)
        if abs(u[0] * v[1] - u[1] * v[0]) < 0.3:
            v = np.array([-u[1], u[0]])
        origin = rng.uniform(-1, 1, 2)
        boundary = Polygon([(-6, -4), (5, -5), (7, 3), (0, 6), (-5, 4)])

        lattice = Lattice.from_boundary(boundary, tuple(u), tuple(v), Circle(0.01), origin=tuple(origin))
        polygon = boundary.to_arc_spline()
        expected = {
            tuple(np.round(origin + i * u + j * v, 6))
            for i in range(-40, 41) for j in range(-40, 41)
            if polygon.contains(origin + i * u + j * v)
        }
        assert {tuple(np.round(point, 6)) for point in centers(lattice)} == expected
        # the indices belong to the points
        base = centers(lattice)[0] - lattice.points_uv[0, 0] * u - lattice.points_uv[0, 1] * v
        for point, (i_u, i_v) in zip(centers(lattice), lattice.points_uv):
            assert np.allclose(base + i_u * u + i_v * v, point, atol=1e-9)
        assert lattice.points_uv.min() == 0

    def test_boundary_with_holes(self):
        hollow = HollowArcSpline(
            ArcSpline([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)], True),
            [Circle(2.5).to_arc_spline()]
        )
        lattice = Lattice.from_boundary(hollow, (1, 0), (0, 1), Circle(0.01), origin=(0.5, 0.5))
        points = centers(lattice)
        assert np.all(np.linalg.norm(points, axis=1) > 2.5)
        assert len(points) == 100 - len([1 for x in range(-5, 5) for y in range(-5, 5)
                                         if math.hypot(x + 0.5, y + 0.5) <= 2.5])

    def test_ring_as_boundary(self):
        lattice = Lattice.from_boundary(Ring(5., 2.), (1, 0), (0, 1), Circle(0.01), origin=(0.5, 0.5))
        distances = np.linalg.norm(centers(lattice), axis=1)
        assert distances.min() > 3. and distances.max() < 5.

    def test_clip_elements(self):
        boundary = Rect(10, 10)
        unclipped = Lattice.from_boundary(boundary, (2, 0), (0, 2), Circle(0.9), origin=(0, 0))
        clipped = Lattice.from_boundary(boundary, (2, 0), (0, 2), Circle(0.9), origin=(0, 0), clip_elements=True)
        # the outer circles touch (stick out of) the boundary: the points at +-4 are inside with a radius of 0.9, the ones
        # at +-4 with a larger circle would not be
        assert clipped.n_points == unclipped.n_points  # r = 0.9 < 1: all circles fit
        wide = Lattice.from_boundary(boundary, (2, 0), (0, 2), Circle(1.5), origin=(0, 0), clip_elements=True)
        assert wide.n_points == 3 * 3  # only the points at 0 and +-2 have a distance of at least 1.5 from the boundary
        assert wide.bounding_box.upper_right.x < 5

    def test_callback(self):
        calls = []
        lattice = Lattice.from_boundary(
            Rect(5, 5), (2, 0), (0, 2), lambda i, uv, xy, n: calls.append((i, uv, xy, n)) or Circle(0.1),
            origin=(0, 0)
        )
        assert [call[0] for call in calls] == list(range(lattice.n_points))
        assert all(call[3] == lattice.n_points for call in calls) and calls[0][1] == (0, 0)

    def test_no_point_inside(self):
        with pytest.raises(ValueError, match='any element'):
            Lattice.from_boundary(Rect(1, 1), (5, 0), (0, 5), Circle(0.1), origin=(10.7, 10.7))

    def test_open_boundary(self):
        with pytest.raises(ValueError, match='closed'):
            Lattice.from_boundary(ArcSpline([(0, 0, 0), (1, 0, 0), (1, 1, 0)], False), (1, 0), (0, 1), Circle(0.1))

    def test_invalid_boundary(self):
        with pytest.raises(TypeError, match='boundary'):
            Lattice.from_boundary('foo', (1, 0), (0, 1), Circle(0.1))
        with pytest.raises(TypeError):
            Lattice.from_boundary(Rect(1, 1) * unit('µm'), (1, 0), (0, 1), Circle(0.1))

    def test_collinear_vectors(self):
        with pytest.raises(ValueError, match='collinear'):
            Lattice.from_boundary(Rect(5, 5), (1, 0), (2, 0), Circle(0.1))


class TestConstructor:
    @staticmethod
    def elements(n=2):
        return [Circle(1.).translated((i, 0)) for i in range(n)]

    def test_properties(self):
        lattice = Lattice(self.elements(), [[0, 0], [1, 0]], (1, 0), (0, 1), description='x')
        assert lattice.points_uv.tolist() == [[0, 0], [1, 0]] and lattice.description == 'x'
        assert lattice.u == Vector(1, 0)

    def test_points_uv_is_a_copy(self):
        lattice = Lattice(self.elements(), [[0, 0], [1, 0]], (1, 0), (0, 1))
        lattice.points_uv[0, 0] = 5
        assert lattice.points_uv[0, 0] == 0

    @pytest.mark.parametrize('points_uv, message', [
        ([[0, 0]], 'one lattice point'),
        ([[0, 0], [0, 0]], 'different'),
        ([[0, 0], [-1, 0]], 'negative'),
        ([0, 1], 'shape'),
        ([[0, 0, 0], [1, 0, 0]], 'shape'),
    ])
    def test_invalid_points(self, points_uv, message):
        with pytest.raises(ValueError, match=message):
            Lattice(self.elements(), points_uv, (1, 0), (0, 1))

    def test_invalid_lattice_vectors(self):
        with pytest.raises(ValueError, match='collinear'):
            Lattice(self.elements(), [[0, 0], [1, 0]], (1, 0), (2, 0))

    def test_elements_with_units(self):
        with pytest.raises(TypeError, match='DimGroup'):
            Lattice([Circle(1.) * unit('µm')], [[0, 0]], (1, 0), (0, 1))

    def test_no_elements(self):
        with pytest.raises(ValueError):
            Lattice([], np.empty((0, 2)), (1, 0), (0, 1))


class TestTransformations:
    @staticmethod
    def lattice():
        return Lattice.from_counts(3, 2, 10., 5., Rect(2, 1))

    def test_translated(self):
        lattice = self.lattice()
        moved = lattice.translated((100, 0))
        assert isinstance(moved, Lattice)
        assert np.allclose(centers(moved), centers(lattice) + (100, 0))
        assert moved.points_uv.tolist() == lattice.points_uv.tolist()
        assert moved.u == lattice.u and moved.v == lattice.v
        assert np.allclose(centers(lattice)[0], (-10, 2.5))  # the original is not changed

    def test_elements_by_uv_belongs_to_the_transformed_elements(self):
        moved = self.lattice().translated((100, 0))
        by_uv = moved.elements_by_uv
        assert all(by_uv[i_v, i_u] is element for element, (i_u, i_v) in zip(moved.elements, moved.points_uv))
        assert by_uv[0, 0].center == Vector(90, 2.5)

    def test_rotated(self):
        rotated = self.lattice().rotated(math.pi / 2)
        assert rotated.u.close_to((0, 10)) and rotated.v.close_to((5, 0))
        assert np.allclose(centers(rotated)[0], (-2.5, -10))
        # the lattice vectors are still the lattice vectors of the elements
        assert np.allclose(centers(rotated)[1] - centers(rotated)[0], np.asarray(rotated.u))

    def test_rotated_about_the_center(self):
        lattice = self.lattice()
        rotated = lattice.rotated(math.pi, origin='center')
        assert np.allclose(sorted(map(tuple, centers(rotated))), sorted(map(tuple, centers(lattice))))

    def test_scaled(self):
        scaled = self.lattice().scaled(2.)
        assert scaled.u == Vector(20, 0) and scaled.v == Vector(0, -10)
        assert np.allclose(centers(scaled)[1] - centers(scaled)[0], (20, 0))
        assert scaled.elements[0].width == pytest.approx(4)

    def test_mirrored(self):
        mirrored = self.lattice().mirrored((0, 1))
        assert mirrored.u.close_to((-10, 0)) and mirrored.v.close_to((0, -5))
        assert np.allclose(centers(mirrored)[0], (10, 2.5))

    def test_transformations_keep_the_description(self):
        assert Lattice.from_counts(1, 1, 1., 1., Circle(1.), description='x').translated((1, 1)).description == 'x'


class TestDimLattice:
    def test_from_counts(self):
        lattice = DimLattice.from_counts(3, 2, 10. * unit('µm'), 5. * unit('µm'), Circle(1.) * unit('µm'))
        assert isinstance(lattice, DimLattice) and isinstance(lattice, DimGroup)
        assert dim_centers(lattice).tolist() == [[-10., 2.5], [0., 2.5], [10., 2.5], [-10., -2.5], [0., -2.5], [10., -2.5]]
        assert lattice.points_uv.tolist() == [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
        assert lattice.u.vector_as(unit('µm')).close_to((10, 0)) and lattice.v.vector_as(unit('µm')).close_to((0, -5))
        assert repr(lattice) == 'DimLattice(n_points=6)'

    def test_mixed_units(self):
        lattice = DimLattice.from_counts(
            2, 2, 10. * unit('µm'), 5000. * unit('nm'), Circle(1.) * unit('nm'), center=um(100, 0)
        )
        assert np.allclose(dim_centers(lattice), [[95., 2.5], [105., 2.5], [95., -2.5], [105., -2.5]])
        assert lattice.elements[0].bounding_box.width.m_as('nm') == pytest.approx(2.)

    def test_shapes_in_another_unit_than_the_pitch(self):
        lattice = DimLattice.from_counts(2, 1, 1. * unit('mm'), 1. * unit('mm'), Circle(100.) * unit('µm'))
        assert dim_centers(lattice, 'mm').tolist() == [[-0.5, 0.], [0.5, 0.]]

    def test_from_rect(self):
        lattice = DimLattice.from_rect(
            10. * unit('µm'), 5000. * unit('nm'), 3. * unit('µm'), 2.5 * unit('µm'), Circle(1.) * unit('µm')
        )
        assert lattice.n_points == 4 * 3
        assert dim_centers(lattice)[:4, 0].tolist() == [-4.5, -1.5, 1.5, 4.5]

    def test_from_rect_with_fast_axis_and_center(self):
        lattice = DimLattice.from_rect(
            2. * unit('µm'), 1. * unit('µm'), 1. * unit('µm'), 1. * unit('µm'), Circle(0.1) * unit('µm'),
            center=DimVector(1 * unit('mm'), 0 * unit('mm')), fast_axis='y'
        )
        assert lattice.points_uv.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
        assert dim_centers(lattice)[0].tolist() == [999., 0.5]

    def test_callback(self):
        calls = []

        def element(i, uv, xy, n):
            calls.append((i, uv, xy, n))
            return Circle(1.) * unit('µm') if i % 2 == 0 else None

        lattice = DimLattice.from_counts(2, 2, 10. * unit('µm'), 10. * unit('µm'), element)
        assert lattice.n_points == 2 and lattice.points_uv.tolist() == [[0, 0], [0, 1]]
        assert [call[1] for call in calls] == [(0, 0), (1, 0), (0, 1), (1, 1)]
        assert all(isinstance(call[2], DimVector) for call in calls)
        assert calls[0][2].vector_as(unit('µm')).close_to((-5, 5)) and calls[0][3] == 4

    def test_from_boundary(self):
        lattice = DimLattice.from_boundary(
            Rect(10000, 6000) * unit('nm'), um(2, 0), um(0, -2), Circle(0.1) * unit('µm'), origin=um(0, 0)
        )
        points = dim_centers(lattice)
        assert np.allclose(sorted(set(np.round(points[:, 1], 9))), [-2., 0., 2.])
        assert np.allclose(sorted(set(np.round(points[:, 0], 9))), [-4., -2., 0., 2., 4.])

    def test_from_boundary_default_origin_and_clip(self):
        lattice = DimLattice.from_boundary(
            Rect(10, 10, center=(1000, 0)) * unit('µm'), um(2000, 0).to('nm'), um(0, 2), Circle(1.5) * unit('µm'),
            clip_elements=True
        )
        points = dim_centers(lattice)
        assert (1000, 0) in {tuple(np.round(point, 6)) for point in points}
        assert points[:, 0].max() < 1005 - 1.5 + 1e-9

    def test_from_boundary_with_a_hollow_shape(self):
        hollow = HollowArcSpline(Rect(10, 10).to_arc_spline(), [Circle(2.5).to_arc_spline()])
        lattice = DimLattice.from_boundary(
            hollow * unit('µm'), um(1, 0), um(0, 1), Circle(0.01) * unit('µm'), origin=um(0.5, 0.5)
        )
        assert np.all(np.linalg.norm(dim_centers(lattice), axis=1) > 2.5)

    def test_elements_by_uv_and_patterns(self):
        pattern = Pattern(
            Rect(1, 1) * unit('µm'), Mill(1. * unit('ms'), 1), one_d.Curve(0.1 * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        lattice = DimLattice.from_counts(2, 2, 10. * unit('µm'), 10. * unit('µm'), pattern)
        assert lattice.elements_by_uv.shape == (2, 2) and isinstance(lattice.elements_by_uv[1, 1], Pattern)
        assert lattice.elements_by_uv[1, 1].center.vector_as(unit('µm')).close_to((5, -5))

    def test_lattice_of_sites(self):
        lattice = DimLattice.from_counts(2, 2, 100. * unit('µm'), 100. * unit('µm'), Site(um(0, 0), um(10, 10)))
        assert [site.center.vector_as(unit('µm')).x for site in lattice.elements] == [-50, 50, -50, 50]

    def test_transformations(self):
        lattice = DimLattice.from_counts(2, 1, 10. * unit('µm'), 5. * unit('µm'), Rect(2, 1) * unit('µm'))
        moved = lattice.translated(um(1, 1))
        assert dim_centers(moved).tolist() == [[-4., 1.], [6., 1.]]
        assert moved.u.vector_as(unit('µm')).close_to((10, 0))
        rotated = lattice.rotated(math.pi / 2)
        assert rotated.u.vector_as(unit('µm')).close_to((0, 10)) and rotated.v.vector_as(unit('µm')).close_to((5, 0))
        assert lattice.scaled(2.).u.vector_as(unit('µm')).close_to((20, 0))
        assert lattice.mirrored(um(0, 1)).u.vector_as(unit('µm')).close_to((-10, 0))

    def test_multiplication_of_a_lattice_with_a_unit(self):
        lattice = Lattice.from_counts(3, 2, 10., 5., Circle(1.), description='lattice')
        dim_lattice = lattice * unit('µm')
        assert isinstance(dim_lattice, DimLattice) and dim_lattice.description == 'lattice'
        assert dim_lattice.points_uv.tolist() == lattice.points_uv.tolist()
        assert dim_lattice.u.vector_as(unit('µm')).close_to((10, 0))
        assert np.allclose(dim_centers(dim_lattice), centers(lattice))
        assert isinstance(unit('nm') * lattice, DimLattice)
        assert dim_lattice.elements_by_uv.shape == (2, 3)

    def test_multiplication_with_other_operands(self):
        with pytest.raises(TypeError):
            Lattice.from_counts(1, 1, 1., 1., Circle(1.)) * 2

    @pytest.mark.parametrize('kwargs, exception', [
        (dict(pitch_x=5., pitch_y=5. * unit('µm')), TypeError),
        (dict(pitch_x=5. * unit('s'), pitch_y=5. * unit('µm')), ValueError),
        (dict(pitch_x=0. * unit('µm'), pitch_y=5. * unit('µm')), ValueError),
        (dict(pitch_x=5. * unit('µm'), pitch_y=-5. * unit('µm')), ValueError),
        (dict(pitch_x=5. * unit('µm'), pitch_y=5. * unit('µm'), element=Circle(1.)), TypeError),
        (dict(pitch_x=5. * unit('µm'), pitch_y=5. * unit('µm'), element='foo'), TypeError),
    ])
    def test_invalid_arguments(self, kwargs, exception):
        kwargs = {'element': Circle(1.) * unit('µm'), **kwargs}
        with pytest.raises(exception):
            DimLattice.from_counts(2, 2, **kwargs)

    def test_invalid_rect_arguments(self):
        with pytest.raises(TypeError):
            DimLattice.from_rect(10., 5. * unit('µm'), 1. * unit('µm'), 1. * unit('µm'), Circle(1.) * unit('µm'))
        with pytest.raises(ValueError):
            DimLattice.from_rect(-1. * unit('µm'), 5. * unit('µm'), 1. * unit('µm'), 1. * unit('µm'), Circle(1.) * unit('µm'))

    def test_invalid_boundary(self):
        with pytest.raises(TypeError, match='DimShape'):
            DimLattice.from_boundary(Rect(1, 1), um(1, 0), um(0, 1), Circle(0.1) * unit('µm'))
        with pytest.raises(ValueError, match='collinear'):
            DimLattice.from_boundary(Rect(5, 5) * unit('µm'), um(1, 0), um(2, 0), Circle(0.1) * unit('µm'))

    def test_elements_without_units(self):
        with pytest.raises(TypeError, match='need units'):
            DimLattice([Circle(1.)], [[0, 0]], um(1, 0), um(0, 1))


class TestBuilders:
    def test_builder(self):
        builder = LatticeBuilder.rectangular(3, 2, 10., 5.)
        assert (builder.nu, builder.nv) == (3, 2)
        builder[0, 0] = Circle(1.)
        builder[2, 1] = Rect(2, 2)
        builder[1, 0] = Circle(2.)
        assert len(builder) == 3
        lattice = builder.to_lattice()
        assert isinstance(lattice, Lattice)
        assert lattice.points_uv.tolist() == [[0, 0], [2, 1], [1, 0]]
        assert centers(lattice).tolist() == [[-10., 2.5], [10., -2.5], [0., 2.5]]
        assert lattice.u == Vector(10, 0) and lattice.v == Vector(0, -5)

    def test_positions(self):
        builder = LatticeBuilder(2, 2, (2, 0), (1, 1), center=(10, 10))
        assert builder.pos((0, 0)) == Vector(9.5, 9.5) - Vector(0, 0) or builder.pos((0, 0)).close_to((8.5, 9.5))
        assert builder.pos((1, 1)).close_to((11.5, 10.5))

    def test_sorting(self):
        def build():
            builder = LatticeBuilder.rectangular(2, 2, 1., 1.)
            for key in [(1, 1), (0, 1), (1, 0), (0, 0)]:
                builder[key] = Circle(0.1)
            return builder

        assert build().to_lattice().points_uv.tolist() == [[1, 1], [0, 1], [1, 0], [0, 0]]
        assert build().to_lattice('row_col').points_uv.tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
        assert build().to_lattice('col_row').points_uv.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
        with pytest.raises(ValueError, match='sort type'):
            build().to_lattice('foo')

    def test_numpy_integer_keys(self):
        builder = LatticeBuilder.rectangular(2, 2, 1., 1.)
        builder[np.int64(1), np.int32(0)] = Circle(0.1)
        assert builder.to_lattice().points_uv.tolist() == [[1, 0]]

    @pytest.mark.parametrize('key, exception', [
        ((2, 0), ValueError), ((0, 2), ValueError), ((-1, 0), ValueError), ((0.5, 0), TypeError), ((0,), TypeError),
        (0, TypeError), ('ab', TypeError),
    ])
    def test_invalid_keys(self, key, exception):
        builder = LatticeBuilder.rectangular(2, 2, 1., 1.)
        with pytest.raises(exception):
            builder[key] = Circle(0.1)
        with pytest.raises(exception):
            builder.pos(key)

    def test_a_site_can_only_be_set_once(self):
        builder = LatticeBuilder.rectangular(2, 2, 1., 1.)
        builder[0, 0] = Circle(0.1)
        with pytest.raises(ValueError, match='already set'):
            builder[0, 0] = Circle(0.1)

    def test_invalid_elements(self):
        builder = LatticeBuilder.rectangular(2, 2, 1., 1.)
        with pytest.raises(TypeError):
            builder[0, 0] = 'foo'
        with pytest.raises(TypeError, match='DimGroup'):
            builder[0, 0] = Circle(0.1) * unit('µm')
        assert len(builder) == 0

    def test_empty_builder(self):
        with pytest.raises(ValueError, match='element'):
            LatticeBuilder.rectangular(2, 2, 1., 1.).to_lattice()

    @pytest.mark.parametrize('arguments', [(0, 1), (1, 0), (1.5, 1)])
    def test_invalid_sizes(self, arguments):
        with pytest.raises(ValueError):
            LatticeBuilder(*arguments, (1, 0), (0, 1))

    def test_collinear_vectors(self):
        with pytest.raises(ValueError, match='collinear'):
            LatticeBuilder(2, 2, (1, 0), (2, 0))

    def test_center_defaults_to_the_origin(self):
        assert LatticeBuilder(1, 1, (1, 0), (0, 1)).pos((0, 0)) == Vector(0, 0)

    def test_dim_builder(self):
        builder = DimLatticeBuilder.rectangular(2, 2, 10. * unit('µm'), 5000. * unit('nm'), center=um(100, 0))
        builder[0, 0] = Circle(1.) * unit('µm')
        builder[1, 1] = Circle(1.) * unit('nm')
        lattice = builder.to_lattice('row_col', description='dim')
        assert isinstance(lattice, DimLattice) and lattice.description == 'dim'
        assert np.allclose(dim_centers(lattice), [[95., 2.5], [105., -2.5]])
        assert builder.pos((1, 0)).vector_as(unit('µm')).close_to((105, 2.5))

    def test_dim_builder_errors(self):
        builder = DimLatticeBuilder.rectangular(2, 2, 10. * unit('µm'), 10. * unit('µm'))
        with pytest.raises(TypeError, match='need units'):
            builder[0, 0] = Circle(1.)
        with pytest.raises(TypeError):
            DimLatticeBuilder.rectangular(2, 2, 10., 10. * unit('µm'))
        with pytest.raises(ValueError):
            DimLatticeBuilder.rectangular(2, 2, 10. * unit('s'), 10. * unit('µm'))
        with pytest.raises(ValueError, match='element'):
            builder.to_lattice()


class TestWithBackends:
    def test_lattice_is_exported_element_by_element(self):
        from fibomat.default_backends import SpotListBackend
        from fibomat.layout import Layout
        from fibomat.raster_styles import zero_d

        lattice = DimLattice.from_counts(3, 2, 10. * unit('µm'), 5. * unit('µm'), Rect(1, 1).translated((0, 0)) * unit('µm'))
        layout = Layout()
        layout.create_site(um(0, 0), um(100, 100)).create_pattern(
            lattice, Mill(1. * unit('ms'), 1), one_d.Curve(0.5 * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        points = layout.export(SpotListBackend).dwell_points()
        # 6 squares with the side length 1 and 0.5 µm pitch: 8 points each
        assert len(points) == 6 * 8

    def test_lattice_in_a_site(self):
        pattern = Pattern(
            Rect(1, 1) * unit('µm'), Mill(1. * unit('ms'), 1), one_d.Curve(0.5 * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        site = Site(um(0, 0), um(100, 100))
        site += DimLattice.from_counts(2, 2, 10. * unit('µm'), 10. * unit('µm'), pattern)
        assert len(site.patterns) == 4

    def test_the_order_of_the_points_is_the_order_of_the_exposure(self):
        from fibomat.default_backends import SpotListBackend
        from fibomat.layout import Layout
        from fibomat.raster_styles import zero_d
        from fibomat.shapes import Spot

        lattice = DimLattice.from_counts(2, 2, 10. * unit('µm'), 10. * unit('µm'), Spot((0, 0)) * unit('µm'), fast_axis='y')
        layout = Layout()
        layout.create_site(um(0, 0), um(100, 100)).create_pattern(lattice, Mill(1. * unit('ms'), 1), zero_d.SingleSpot())
        points = layout.export(SpotListBackend).dwell_points()
        assert points[:, :2].tolist() == [[-5, 5], [-5, -5], [5, 5], [5, -5]]
