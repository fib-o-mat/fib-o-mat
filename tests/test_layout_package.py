"""Tests of the package structure of `fibomat.layout` and `fibomat.arrangements`."""
import importlib
import subprocess
import sys

import pytest

from fibomat.layout import Layout, Pattern, Site
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d
from fibomat.shapes import Line
from fibomat.units import unit


def center(x=0., y=0.):
    return DimVector(x * unit('µm'), y * unit('µm'))


def pattern():
    return Pattern(
        Line((0, 0), (1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1), one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
    )


class TestStructure:
    def test_exports(self):
        import fibomat.layout as layout_package
        assert set(layout_package.__all__) == {'Layout', 'Site', 'Pattern'}
        assert Layout.__module__ == 'fibomat.layout.layout'
        assert Site.__module__ == 'fibomat.layout.site'
        assert Pattern.__module__ == 'fibomat.layout.pattern'

    @pytest.mark.parametrize('module', ['fibomat.sample', 'fibomat.site', 'fibomat.pattern', 'fibomat.layout.layoutbase'])
    def test_old_modules_are_gone(self, module):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)

    def test_old_layout_names_are_gone(self):
        import fibomat.arrangements as arrangements
        import fibomat.layout as layout_package
        assert not hasattr(layout_package, 'Sample')
        assert not hasattr(layout_package, 'Group') and not hasattr(layout_package, 'LayoutBase')
        assert hasattr(arrangements, 'ArrangementBase') and not hasattr(arrangements, 'LayoutBase')

    @pytest.mark.parametrize('first', [
        'fibomat.layout', 'fibomat.layout.site', 'fibomat.layout.layout', 'fibomat.backend', 'fibomat.default_backends',
        'fibomat.arrangements',
    ])
    def test_import_order(self, first):
        # the backends depend on `Site` and `Pattern`, `Layout` on the backends: no import order may fail
        code = f'import {first}\nimport fibomat.layout, fibomat.backend, fibomat.default_backends\n'
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr

    def test_shapes_do_not_import_the_layout(self):
        code = 'import sys, fibomat.shapes\nassert "fibomat.layout" not in sys.modules\n'
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


class TestLayout:
    def test_empty(self):
        layout = Layout(description='design')
        assert layout.number_of_sites == 0
        assert layout.description == 'design'
        assert layout.bounding_box is None

    def test_create_site(self):
        layout = Layout()
        site = layout.create_site(center(1., 2.), description='first')
        assert isinstance(site, Site)
        assert layout.number_of_sites == 1
        assert site.description == 'first'

    def test_add_site_and_iadd(self):
        layout = Layout()
        layout.add_site(Site(center()))
        layout += Site(center(5., 5.))
        assert layout.number_of_sites == 2

    def test_site_holds_patterns(self):
        site = Site(center())
        created = site.create_pattern(
            Line((0, 0), (1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1),
            one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
        )
        site += pattern()
        assert isinstance(created, Pattern)
