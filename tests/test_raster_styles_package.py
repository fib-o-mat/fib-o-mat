"""Tests of the package `fibomat.raster_styles`."""
import importlib

import pytest

from fibomat import raster_styles
from fibomat.raster_styles import RasterStyle, ScanSequence, one_d, two_d, zero_d


def test_exports():
    assert set(raster_styles.__all__) == {'RasterStyle', 'zero_d', 'one_d', 'two_d', 'ScanSequence'}
    assert set(zero_d.__all__) == {'SingleSpot', 'PreRasterized'}
    assert one_d.__all__ == ['Curve']
    assert set(two_d.__all__) == {'ContourParallel', 'LineByLine', 'Spiral'}


def test_star_import_works():
    namespace = {}
    exec('from fibomat.raster_styles.two_d import *', namespace)  # pylint: disable=exec-used
    assert {'ContourParallel', 'LineByLine', 'Spiral'} <= set(namespace)


def test_linear_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('fibomat.raster_styles.two_d.linear')


def test_raster_style_is_abstract():
    with pytest.raises(TypeError):
        RasterStyle()


def test_contour_parallel_without_experimental_dependencies():
    try:
        import numba  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ModuleNotFoundError:
        with pytest.raises(ImportError, match='experimental'):
            two_d.ContourParallel()
    else:
        assert issubclass(two_d.ContourParallel, RasterStyle)


def test_all_styles_are_raster_styles():
    for cls in (zero_d.SingleSpot, zero_d.PreRasterized, one_d.Curve, two_d.LineByLine, two_d.Spiral):
        assert issubclass(cls, RasterStyle)


def test_scan_sequence_is_exported():
    assert ScanSequence.SERPENTINE.value == 'serpentine'
