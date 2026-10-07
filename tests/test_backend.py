"""Tests of `fibomat.backend`."""
import importlib

import numpy as np
import pytest

from fibomat import backend as backend_package
from fibomat import shapes
from fibomat.arrangements import DimGroup
from fibomat.backend import BackendBase, ShapeNotSupportedError, shape_type
from fibomat.composite_shapes import Text
from fibomat.layout import Layout, Pattern, Site
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d
from fibomat.shapes import Circle, Line, Polygon, Polyline, Rect, Shape, Spot
from fibomat.units import unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def make_pattern(shape, **kwargs):
    return Pattern(
        shape * unit('µm'), Mill(1. * unit('ms'), 1), one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE), **kwargs
    )


class Recorder(BackendBase):
    """Supports lines and polylines (polygons through the base class)."""

    def __init__(self, description=None):
        super().__init__(description)
        self.calls = []

    def line(self, ptn):
        self.calls.append(('line', ptn))

    def polyline(self, ptn):
        self.calls.append(('polyline', ptn))


class TestPackage:
    def test_exports(self):
        assert set(backend_package.__all__) == {'BackendBase', 'ShapeNotSupportedError', 'shape_type'}

    @pytest.mark.parametrize('module', ['fibomat.backend.registry', 'fibomat.backend.backendbasemeta'])
    def test_registry_and_metaclass_are_gone(self, module):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
        assert not hasattr(backend_package, 'registry')

    def test_no_metaclass(self):
        assert type(BackendBase) is type
        assert type(Recorder) is type


class TestConstruction:
    def test_description(self):
        assert BackendBase().description is None
        assert BackendBase('layout').description == 'layout'
        assert BackendBase(description='layout').description == 'layout'

    def test_unknown_arguments_are_an_error(self):
        with pytest.raises(TypeError):
            BackendBase(foo=1)


class TestShapeMethods:
    def test_all_shapes_have_a_declared_method(self):
        declared = BackendBase().shape_methods
        for shape_class in (
            shapes.Spot, shapes.Line, shapes.Rect, shapes.Ellipse, shapes.Circle, shapes.Arc, shapes.ArcSpline,
            shapes.ParametricCurve, shapes.RasterizedPoints, shapes.Polyline, shapes.Polygon,
        ):
            assert shape_class in declared
        assert declared[shapes.Line] == 'line'
        assert declared[shapes.Polygon] == 'polygon'

    def test_nothing_is_implemented_by_the_base_class(self):
        assert BackendBase().implemented_shape_methods == {}

    def test_implemented_shape_methods(self):
        implemented = Recorder().implemented_shape_methods
        assert set(implemented) == {Line, Polyline}
        assert implemented[Line] is Recorder.line

    def test_inheritance(self):
        class Derived(Recorder):
            def circle(self, ptn):
                pass

        assert set(Derived().implemented_shape_methods) == {Line, Polyline, Circle}
        assert set(Recorder().implemented_shape_methods) == {Line, Polyline}  # the base is not changed

    def test_overriding_in_a_subclass(self):
        class Derived(Recorder):
            def line(self, ptn):
                self.calls.append(('derived line', ptn))

        backend = Derived()
        backend.process_pattern(make_pattern(Line((0, 0), (1, 0))))
        assert backend.calls[0][0] == 'derived line'


class TestProcessPattern:
    def test_dispatch_by_shape_type(self):
        backend = Recorder()
        pattern = make_pattern(Line((0, 0), (1, 0)))
        backend.process_pattern(pattern)
        assert backend.calls == [('line', pattern)]

    def test_dispatch_falls_back_to_the_base_classes_of_the_shape(self):
        backend = Recorder()
        pattern = make_pattern(Polygon([(0, 0), (1, 0), (1, 1)]))
        backend.process_pattern(pattern)  # a Polygon is a Polyline
        assert backend.calls == [('polyline', pattern)]

    def test_unsupported_shape(self):
        with pytest.raises(ShapeNotSupportedError, match='Recorder.*Spot'):
            Recorder().process_pattern(make_pattern(Spot((0, 0))))

    def test_shape_not_supported_is_a_type_error(self):
        assert issubclass(ShapeNotSupportedError, TypeError)

    def test_base_stubs_raise(self):
        with pytest.raises(ShapeNotSupportedError, match='Rect'):
            BackendBase().rect(make_pattern(Rect(1, 1)))

    def test_process_unknown_can_be_overwritten(self):
        class Lenient(Recorder):
            def process_unknown(self, ptn):
                self.calls.append(('unknown', ptn))

        backend = Lenient()
        pattern = make_pattern(Spot((0, 0)))
        backend.process_pattern(pattern)
        assert backend.calls == [('unknown', pattern)]

    def test_custom_shape_type(self):
        class MyShape(Shape):
            def __init__(self):
                super().__init__()

            def __repr__(self):
                return 'MyShape()'

            @property
            def is_closed(self):
                return False

            @property
            def bounding_box(self):
                from fibomat.linalg import BoundingBox
                return BoundingBox((0, 0), (1, 1))

            @property
            def center(self):
                from fibomat.linalg import Vector
                return Vector(0.5, 0.5)

            def _impl_translate(self, trans_vec):
                pass

            def _impl_rotate(self, theta):
                pass

            def _impl_scale(self, fac):
                pass

            def _impl_mirror(self, mirror_axis):
                pass

        class CustomBackend(Recorder):
            @shape_type(MyShape)
            def my_shape(self, ptn):
                self.calls.append(('my_shape', ptn))

        backend = CustomBackend()
        assert MyShape in backend.implemented_shape_methods
        pattern = make_pattern(MyShape())
        backend.process_pattern(pattern)
        assert backend.calls == [('my_shape', pattern)]
        with pytest.raises(ShapeNotSupportedError):
            Recorder().process_pattern(pattern)

    def test_declared_but_unimplemented_custom_shape(self):
        class Stubbed(BackendBase):
            pass

        assert Stubbed().implemented_shape_methods == {}

    def test_group_is_split_into_shapes(self):
        backend = Recorder()
        group = DimGroup([Line((0, 0), (1, 0)) * unit('µm'), Line((0, 1), (1, 1)) * unit('µm')])
        pattern = Pattern(
            group, Mill(1. * unit('ms'), 1), one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE),
            description='group', _color='red'
        )
        backend.process_pattern(pattern)
        assert [name for name, _ in backend.calls] == ['line', 'line']
        for _, ptn in backend.calls:
            assert ptn.mill is pattern.mill and ptn.raster_style is pattern.raster_style
            assert ptn.description == 'group' and ptn.kwargs == {'_color': 'red'}

    def test_nested_groups(self):
        backend = Recorder()
        inner = DimGroup([Line((0, 0), (1, 0)) * unit('µm')])
        outer = DimGroup([inner, Line((0, 1), (1, 1)) * unit('µm')])
        backend.process_pattern(Pattern(
            outer, Mill(1. * unit('ms'), 1), one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE)
        ))
        assert [name for name, _ in backend.calls] == ['line', 'line']

    def test_text_is_split_into_shapes(self):
        class Collector(BackendBase):
            def __init__(self):
                super().__init__()
                self.shapes = []

            def polyline(self, ptn):
                self.shapes.append(ptn.dim_shape.shape)

        backend = Collector()
        text = Text('I', font_size=1.) * unit('µm')
        backend.process_pattern(make_pattern_dim(text))
        assert len(backend.shapes) == len(list(text.arrangement_elements())) >= 1
        assert all(isinstance(shape, Polyline) for shape in backend.shapes)

    def test_unsupported_shape_in_group(self):
        group = DimGroup([Spot((0, 0)) * unit('µm')])
        with pytest.raises(ShapeNotSupportedError):
            Recorder().process_pattern(Pattern(
                group, Mill(1. * unit('ms'), 1), one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE)
            ))


def make_pattern_dim(dim_object):
    return Pattern(dim_object, Mill(1. * unit('ms'), 1), one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE))


class TestProcessSite:
    def test_all_patterns_are_processed(self):
        site = Site(um(5, 5), um(10, 10))
        first = site.create_pattern(
            Line((0, 0), (1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1),
            one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        second = site.create_pattern(
            Line((0, 1), (1, 1)) * unit('µm'), Mill(1. * unit('ms'), 1),
            one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        backend = Recorder()
        backend.process_site(site)
        assert [ptn for _, ptn in backend.calls] == [first, second]

    def test_overwriting_process_site(self):
        class SiteBackend(Recorder):
            def process_site(self, new_site):
                self.calls.append(('site', new_site))
                super().process_site(new_site)

        site = Site(um(0, 0), um(1, 1))
        backend = SiteBackend()
        backend.process_site(site)
        assert backend.calls == [('site', site)]


class TestSave:
    def test_save_is_not_implemented_by_default(self):
        with pytest.raises(NotImplementedError):
            BackendBase().save('file')


class TestWithLayout:
    def test_export(self):
        layout = Layout(description='design')
        site = layout.create_site(um(0, 0), um(10, 10))
        site.create_pattern(
            Line((0, 0), (1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1),
            one_d.Curve(1. * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        backend = layout.export(Recorder)
        assert isinstance(backend, Recorder)
        assert backend.description == 'design'
        assert [name for name, _ in backend.calls] == ['line']
