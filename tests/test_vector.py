import math
import warnings

import numpy as np
import pytest

from fibomat.linalg.vectors import Vector, VectorValueError, angle_between, signed_angle_between
from fibomat.units import unit


class Dummy:
    pass


class TestVectorConstruction:
    @pytest.mark.parametrize('args, kwargs, expected', [
        ((), {}, (0., 0.)),
        ((1., 2.), {}, (1., 2.)),
        ((1, 2), {}, (1., 2.)),
        (([1., 2.],), {}, (1., 2.)),
        (((1., 2.),), {}, (1., 2.)),
        ((np.array([1., 2.]),), {}, (1., 2.)),
        ((iter([1., 2.]),), {}, (1., 2.)),
        ((Vector(1., 2.),), {}, (1., 2.)),
        ((), {'x': 1., 'y': 2.}, (1., 2.)),
        ((np.float32(1.), np.int64(2)), {}, (1., 2.)),
        ((), {'r': 2., 'phi': math.pi / 2}, (0., 2.)),
        ((), {'r': 1, 'phi': np.float64(math.pi)}, (-1., 0.)),
    ])
    def test_valid(self, args, kwargs, expected):
        v = Vector(*args, **kwargs)
        assert v.x == pytest.approx(expected[0], abs=1e-12)
        assert v.y == pytest.approx(expected[1], abs=1e-12)

    @pytest.mark.parametrize('args, kwargs', [
        (([1.],), {}),
        (([1., 2., 3.],), {}),
        ((['foo', 2],), {}),
        (([2, 'foo'],), {}),
        (('ab',), {}),
        ((Dummy(),), {}),
        ((None, 1.), {}),
        ((1.,), {}),
        ((1., 2., 3.), {}),
        ((1., 2., 3., 4.), {}),
        ((1., 'a'), {}),
        ((), {'x': 1.}),
        ((), {'y': 1.}),
        ((), {'r': 1.}),
        ((), {'phi': 1.}),
        ((), {'r': 1., 'phi': 'a'}),
        ((), {'r': 'a', 'phi': 1.}),
        ((1., 2.), {'r': 1., 'phi': 1.}),
        ((1.,), {'r': 1., 'phi': 1.}),
        (([1., 2.],), {'y': 1.}),
        (([1., 2.],), {'r': 1.}),
        ((), {'y': 2., 'r': 1., 'phi': 1.}),
    ])
    def test_invalid(self, args, kwargs):
        with pytest.raises(VectorValueError):
            Vector(*args, **kwargs)

    def test_vector_value_error_is_value_error(self):
        assert issubclass(VectorValueError, ValueError)

    def test_dim_values_rejected(self):
        with pytest.raises(VectorValueError):
            Vector(1 * unit('m'), 2 * unit('m'))
        with pytest.raises(VectorValueError):
            Vector([1 * unit('m'), 2 * unit('m')])


class TestVectorSequence:
    def test_access(self):
        v = Vector(1., 2.)
        assert len(v) == 2
        assert (v[0], v[1], v[-1]) == (1., 2., 2.)
        assert v[np.int64(1)] == 2.
        assert v[:] == [1., 2.]
        assert v[1:] == [2.]
        assert list(v) == [1., 2.]
        assert 2. in v
        x, y = v
        assert (x, y) == (1., 2.)

    def test_invalid_index(self):
        v = Vector(1., 2.)
        with pytest.raises(IndexError):
            v[2]
        with pytest.raises(TypeError):
            v['x']
        with pytest.raises(TypeError):
            v[1.]

    def test_array(self):
        v = Vector(1., 2.)
        arr = np.asarray(v)
        assert arr.shape == (2,)
        assert arr.dtype == float
        arr[0] = 100.  # must not change the vector
        assert v.x == 1.
        assert np.asarray(v, dtype=np.float32).dtype == np.float32

    def test_repr(self):
        assert repr(Vector(1., 2.)) == 'Vector(x=1.0, y=2.0)'

    def test_unhashable(self):
        with pytest.raises(TypeError):
            hash(Vector(1., 2.))


class TestVectorComponents:
    def test_properties(self):
        v = Vector(3., 4.)
        assert v.x == 3. and v.y == 4.
        assert v.r == pytest.approx(5.)
        assert v.mag == pytest.approx(5.)
        assert v.magnitude == pytest.approx(5.)
        assert v.phi == pytest.approx(math.atan2(4, 3))
        assert isinstance(v.x, float)

    def test_length_deprecated(self):
        with pytest.warns(DeprecationWarning):
            assert Vector(3., 4.).length == pytest.approx(5.)

    @pytest.mark.parametrize('vec, angle', [
        ((1, 0), 0.),
        ((0, 1), math.pi / 2),
        ((-1, 0), math.pi),
        ((0, -1), 3 * math.pi / 2),
        ((1, -1e-300), 2 * math.pi),
    ])
    def test_angle_about_x_axis(self, vec, angle):
        assert Vector(vec).angle_about_x_axis == pytest.approx(angle)

    def test_angle_of_null_vector(self):
        with pytest.raises(ValueError):
            Vector().angle_about_x_axis

    def test_tiny_vector_is_not_null(self):
        assert Vector(1e-12, 0.).angle_about_x_axis == 0.

    def test_polar_roundtrip(self):
        v = Vector(r=2., phi=0.3)
        assert v.r == pytest.approx(2.)
        assert v.phi == pytest.approx(0.3)


class TestVectorComparison:
    def test_eq(self):
        assert Vector(1., 2.) == Vector(1., 2.)
        assert Vector(1., 2.) == (1., 2.)
        assert Vector(1., 2.) == [1., 2.]
        assert Vector(1., 2.) == np.array([1., 2.])
        assert Vector(1., 2.) != Vector(1., 2.1)
        assert Vector(1., 2.) == Vector(1., 2. + 1e-10)

    def test_eq_with_non_vectors(self):
        assert not Vector(1., 2.) == 1.
        assert Vector(1., 2.) != 'ab'
        assert Vector(1., 2.) != (1., 2., 3.)
        assert Vector(1., 2.) != Dummy()
        assert Vector(1., 2.) != None  # noqa: E711

    def test_eq_with_dim_vector(self):
        assert Vector(1., 2.) != (1 * unit('m'), 2 * unit('m'))

    def test_close_to(self):
        assert Vector(1., 2.).close_to((1., 2.))
        assert not Vector(1., 2.).close_to((1., 3.))
        with pytest.raises(VectorValueError):
            Vector(1., 2.).close_to((1., 2., 3.))


class TestVectorGeometry:
    def test_normalized(self):
        v = Vector(3., 4.).normalized()
        assert v == (0.6, 0.8)
        assert v.r == pytest.approx(1.)

    def test_normalized_null_vector(self):
        with pytest.raises(ValueError):
            Vector().normalized()
        with pytest.raises(ValueError):
            Vector().normalized_to(2.)

    def test_normalized_to(self):
        v = Vector(3., 4.).normalized_to(10.)
        assert v == (6., 8.)
        with pytest.raises(VectorValueError):
            Vector(3., 4.).normalized_to('a')
        with pytest.raises(VectorValueError):
            Vector(3., 4.).normalized_to(1 * unit('m'))

    def test_rotated(self):
        assert Vector(1., 0.).rotated(math.pi / 2) == (0., 1.)
        assert Vector(1., 0.).rotated(math.pi) == (-1., 0.)
        assert Vector(1., 0.).rotated(math.pi / 2, origin=(1., 1.)) == (2., 1.)
        assert Vector(2., 1.).rotated(-math.pi / 2, origin=Vector(1., 1.)) == (1., 0.)
        assert Vector(1., 2.).rotated(0.) == (1., 2.)
        assert Vector(1., 2.).rotated(np.float64(math.pi / 2)) == (-2., 1.)

    def test_rotation_keeps_length(self):
        v = Vector(1.5, -2.5)
        assert v.rotated(1.234, origin=(3., 4.)).r != 0
        assert (v.rotated(1.234) .r) == pytest.approx(v.r)

    def test_rotated_does_not_modify(self):
        v = Vector(1., 0.)
        v.rotated(1., origin=(1., 1.))
        assert v == (1., 0.)

    def test_mirrored(self):
        assert Vector(1., 2.).mirrored((1., 0.)) == (1., -2.)
        assert Vector(1., 2.).mirrored((0., 1.)) == (-1., 2.)
        assert Vector(1., 2.).mirrored((1., 1.)) == (2., 1.)
        assert Vector(1., 2.).mirrored((2., 2.)) == (2., 1.)  # only the direction of the axis matters
        assert Vector(1., 2.).mirrored((-1., -1.)) == (2., 1.)

    def test_mirrored_null_axis(self):
        with pytest.raises(ValueError):
            Vector(1., 2.).mirrored((0., 0.))

    def test_projected(self):
        assert Vector(2., 0.).projected((1., 1.)) == (1., 0.)
        assert Vector(0., 3.).projected((5., 2.)) == (0., 2.)
        assert Vector(1., 1.).projected((1., -1.)) == (0., 0.)

    def test_projected_onto_null_vector(self):
        with pytest.raises(ValueError):
            Vector().projected((1., 1.))

    def test_dot_cross(self):
        assert Vector(1., 2.).dot((3., 4.)) == pytest.approx(11.)
        assert Vector(1., 0.).cross((0., 1.)) == pytest.approx(1.)
        assert Vector(0., 1.).cross((1., 0.)) == pytest.approx(-1.)
        assert isinstance(Vector(1., 2.).dot((3., 4.)), float)
        with pytest.raises(VectorValueError):
            Vector(1., 2.).dot((1., 2., 3.))


class TestVectorArithmetic:
    def test_add_sub(self):
        assert Vector(1., 2.) + Vector(3., 4.) == (4., 6.)
        assert Vector(1., 2.) + (3., 4.) == (4., 6.)
        assert (3., 4.) + Vector(1., 2.) == (4., 6.)
        assert Vector(1., 2.) - (3., 5.) == (-2., -3.)
        assert (3., 5.) - Vector(1., 2.) == (2., 3.)
        assert isinstance((3., 4.) + Vector(1., 2.), Vector)
        assert isinstance((3., 4.) - Vector(1., 2.), Vector)

    def test_sum(self):
        assert sum([Vector(1., 2.), Vector(3., 4.), Vector(5., 6.)]) == (9., 12.)

    def test_invalid_add(self):
        with pytest.raises(VectorValueError):
            Vector(1., 2.) + 1.
        with pytest.raises(VectorValueError):
            Vector(1., 2.) + (1., 2., 3.)

    def test_mul_div(self):
        assert Vector(1., 2.) * 2 == (2., 4.)
        assert 2 * Vector(1., 2.) == (2., 4.)
        assert Vector(1., 2.) * 0.5 == (0.5, 1.)
        assert np.float64(2.) * Vector(1., 2.) == (2., 4.)
        assert isinstance(np.float64(2.) * Vector(1., 2.), Vector)
        assert np.int64(2) * Vector(1., 2.) == (2., 4.)
        assert Vector(1., 2.) * np.float32(2.) == (2., 4.)
        assert Vector(2., 4.) / 2 == (1., 2.)
        assert -Vector(1., -2.) == (-1., 2.)

    def test_invalid_mul(self):
        with pytest.raises(TypeError):
            Vector(1., 2.) * 'a'
        with pytest.raises(TypeError):
            Vector(1., 2.) * Vector(1., 2.)
        with pytest.raises(TypeError):
            Vector(1., 2.) / Vector(1., 2.)
        with pytest.raises(TypeError):
            'a' * Vector(1., 2.)

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            Vector(1., 2.) / 0

    def test_operations_do_not_modify(self):
        v = Vector(1., 2.)
        _ = v + (1., 1.), v - (1., 1.), 2 * v, v / 2, -v, v.normalized()
        assert v == (1., 2.)

    def test_mul_unit(self):
        from fibomat.linalg.vectors import DimVector

        v = Vector(1., 2.) * unit('µm')
        assert isinstance(v, DimVector)
        assert v.x.m_as('µm') == pytest.approx(1.)
        assert v.y.m_as('nm') == pytest.approx(2000.)


class TestAngles:
    def test_angle_between(self):
        assert angle_between((1, 0), (0, 1)) == pytest.approx(math.pi / 2)
        assert angle_between((1, 0), (0, -1)) == pytest.approx(math.pi / 2)
        assert angle_between((1, 0), (-1, 0)) == pytest.approx(math.pi)
        assert angle_between((1, 0), (2, 0)) == pytest.approx(0.)
        assert angle_between(Vector(1, 1), Vector(1, 0)) == pytest.approx(math.pi / 4)
        assert isinstance(angle_between((1, 0), (0, 1)), float)

    def test_angle_between_rounding(self):
        # cosine is slightly larger than 1 for these vectors, must not return nan
        v = (0.1, 0.1)
        assert angle_between(v, (0.3, 0.3)) == pytest.approx(0., abs=1e-6)
        assert not np.isnan(angle_between((1e-3, 1e-3), (1., 1.)))

    def test_angle_between_invalid(self):
        with pytest.raises(VectorValueError):
            angle_between((1, 0, 0), (0, 1))
        with pytest.raises(VectorValueError):
            angle_between((1, 0), (0, 1, 0))
        with pytest.raises(VectorValueError):
            angle_between((1, 0), 'ab')
        with pytest.raises(ValueError):
            angle_between((1, 0), (0, 0))
        with pytest.raises(ValueError):
            angle_between((0, 0), (1, 0))

    def test_signed_angle_between(self):
        assert signed_angle_between((1, 0), (0, 1)) == pytest.approx(math.pi / 2)
        assert signed_angle_between((1, 0), (0, -1)) == pytest.approx(-math.pi / 2)
        assert signed_angle_between((0, 1), (1, 0)) == pytest.approx(-math.pi / 2)
        assert signed_angle_between((1, 0), (1, 0)) == pytest.approx(0.)
        assert abs(signed_angle_between((1, 0), (-1, 0))) == pytest.approx(math.pi)
        with pytest.raises(VectorValueError):
            signed_angle_between((1, 0, 0), (0, 1))
        with pytest.raises(ValueError):
            signed_angle_between((0, 0), (0, 1))

    def test_no_numpy_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            angle_between((1, 0), (0, 1))
            Vector(1., 2.).normalized()
