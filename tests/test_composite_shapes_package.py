import subprocess
import sys
import types

import pytest

import fibomat
import fibomat.shapes as shapes


class TestLazyAccessToCompositeShapes:
    """The composite shapes moved to `fibomat.composite_shapes`; `fibomat.shapes` still provides them lazily."""

    def test_names_are_not_imported_eagerly(self):
        for name in ('HollowArcSpline', 'Ring', 'Text', 'DimText'):
            assert name not in vars(shapes)
            assert name not in shapes.__all__
        assert shapes._COMPOSITE_SHAPES == ('HollowArcSpline', 'Ring', 'Text', 'DimText')

    def test_attribute_access_is_delegated(self, monkeypatch):
        fake = types.ModuleType('fibomat.composite_shapes')
        for name in shapes._COMPOSITE_SHAPES:
            setattr(fake, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, 'fibomat.composite_shapes', fake)

        for name in shapes._COMPOSITE_SHAPES:
            assert getattr(shapes, name) is getattr(fake, name)

    def test_from_import_works(self, monkeypatch):
        fake = types.ModuleType('fibomat.composite_shapes')
        fake.Ring = type('Ring', (), {})
        monkeypatch.setitem(sys.modules, 'fibomat.composite_shapes', fake)

        from fibomat.shapes import Ring

        assert Ring is fake.Ring

    def test_unknown_names(self):
        with pytest.raises(AttributeError, match="no attribute 'Foo'"):
            shapes.Foo
        assert not hasattr(shapes, 'Foo')

    def test_primitives_do_not_depend_on_higher_level_packages(self):
        # `fibomat.shapes` must not import `curve_tools`, `arrangements` or the composite shapes (they depend on it)
        code = (
            'import sys, fibomat.shapes as shapes\n'
            'for name in ("Shape", "ArcSpline", "Line", "Arc", "Circle", "Rect", "Polygon", "ParametricCurve", "Biarc"):\n'
            '    assert hasattr(shapes, name), name\n'
            'for package in ("curve_tools", "arrangements", "composite_shapes"):\n'
            '    assert "fibomat." + package not in sys.modules, package\n'
        )
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr

    def test_curve_tools_can_be_imported(self):
        # (`curve_tools` handles hollow arc splines but imports them lazily to avoid a circular import)
        result = subprocess.run(
            [sys.executable, '-c', 'import fibomat.curve_tools, fibomat.shapes'], capture_output=True, text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
