import pytest

from fibomat.linalg import BoundingBox, Transformable, Vector
from fibomat.units import DimensionError, unit

from helpers_shapes import Dot, Shape, dim_shape_module


class RecordingDimShape:
    """Replacement of DimShape which records its arguments."""

    def __init__(self, shape, unit_, description=None):
        self.shape = shape
        self.unit = unit_


@pytest.fixture
def recording_dim_shape(monkeypatch):
    monkeypatch.setattr(dim_shape_module, 'DimShape', RecordingDimShape)


class TestInterface:
    def test_is_transformable(self):
        assert issubclass(Shape, Transformable)
        assert isinstance(Dot(), Transformable)

    def test_abstract_members(self):
        with pytest.raises(TypeError):
            Shape()

        with pytest.raises(TypeError):
            # is_closed is missing
            class Incomplete(Shape):
                def __repr__(self):
                    return ''
                center = Vector()
                bounding_box = BoundingBox((0, 0), (0, 0))
                _impl_translate = _impl_rotate = _impl_scale = _impl_mirror = lambda *args: None
            Incomplete()

    def test_description(self):
        assert Dot().description is None
        assert Dot(description='foo').description == 'foo'

    def test_repr_and_closed(self):
        assert repr(Dot((1, 2))) == 'Dot(Vector(x=1.0, y=2.0))'
        assert Dot().is_closed is False

    def test_transformations_work_on_clones(self):
        dot = Dot((1., 0.))
        assert dot.translated((1, 1)).position == (2., 1.)
        assert dot.rotated(3.141592653589793 / 2).position == (0., 1.)
        assert dot.position == (1., 0.)
        assert isinstance(dot.translated((1, 1)), Dot)


class TestOptionalProperties:
    def test_area_not_implemented(self):
        with pytest.raises(NotImplementedError, match='area'):
            Dot().area

    def test_boundary_length_not_implemented(self):
        # the former implementation raised NotADirectoryError
        with pytest.raises(NotImplementedError, match='boundary length'):
            Dot().boundary_length

    def test_can_be_overridden(self):
        class Square(Dot):
            area = 4.
            boundary_length = 8.

        assert Square().area == 4.
        assert Square().boundary_length == 8.


class TestMultiplyWithUnit:
    def test_mul(self, recording_dim_shape):
        dot = Dot()
        res = dot * unit('µm')
        assert isinstance(res, RecordingDimShape)
        assert res.shape is dot
        assert res.unit == (1. * unit('µm')).units

    def test_rmul(self, recording_dim_shape):
        dot = Dot()
        res = unit('nm') * dot
        assert isinstance(res, RecordingDimShape)
        assert res.shape is dot
        assert res.unit == (1. * unit('nm')).units

    def test_unit_strings_with_prefix_and_custom(self, recording_dim_shape):
        assert (Dot() * unit('mm')).unit == (1. * unit('mm')).units
        assert (Dot() * unit('m', None)).unit == (1. * unit('m')).units

    def test_not_a_length(self, recording_dim_shape):
        with pytest.raises(DimensionError):
            Dot() * unit('s')
        with pytest.raises(DimensionError):
            unit('kg') * Dot()

    @pytest.mark.parametrize('other', [2, 2., 'a', None, (1, 2), Vector(1, 2)])
    def test_other_operands_are_refused(self, other):
        with pytest.raises(TypeError):
            Dot() * other
        with pytest.raises(TypeError):
            other * Dot()

    def test_dim_float_is_refused(self):
        with pytest.raises(TypeError):
            Dot() * (1. * unit('m'))
        with pytest.raises(TypeError):
            (1. * unit('m')) * Dot()
