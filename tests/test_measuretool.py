"""Tests of the bokeh `MeasureTool` (`fibomat.default_backends.measuretool`)."""
import hashlib
import pathlib

import bokeh.models as bm
import pytest
from bokeh.util.compiler import CustomModel, get_cache_hook, set_cache_hook

from fibomat.default_backends import measuretool
from fibomat.default_backends.measuretool import MeasureTool


SOURCE = pathlib.Path(measuretool.__file__).with_name('measuretool.ts')
COMPILED = pathlib.Path(measuretool.__file__).with_name('measuretool.js')


def test_compiled_file_belongs_to_the_source():
    # fails if measuretool.ts was changed without running `python scripts/build_measuretool.py`
    assert COMPILED.exists(), 'run python scripts/build_measuretool.py'
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert COMPILED.read_text(encoding='utf-8').startswith(f'// source-sha256: {digest}\n')
    assert measuretool._compiled_code() is not None  # pylint: disable=protected-access


def test_compiled_code_is_a_bokeh_module():
    code = measuretool._compiled_code()  # pylint: disable=protected-access
    assert 'exports.MeasureTool' in code
    assert 'require("models/tools/gestures/gesture_tool")' in code


def test_outdated_compiled_file_is_not_used(monkeypatch, tmp_path):
    outdated = tmp_path / 'measuretool.js'
    outdated.write_text('// source-sha256: 0000\ncode', encoding='utf-8')
    monkeypatch.setattr(measuretool, '_COMPILED', outdated)
    assert measuretool._compiled_code() is None  # pylint: disable=protected-access

    monkeypatch.setattr(measuretool, '_COMPILED', tmp_path / 'missing.js')
    assert measuretool._compiled_code() is None  # pylint: disable=protected-access


def test_cache_hook_returns_the_compiled_code_without_node():
    hook = get_cache_hook()
    compiled = hook(CustomModel(MeasureTool), None)
    assert compiled is not None and compiled.deps == [] and 'exports.MeasureTool' in compiled.code


def test_cache_hook_delegates_for_other_models(monkeypatch):
    calls = []
    monkeypatch.setattr(measuretool, '_previous_hook', lambda model, impl: calls.append(model) or 'previous')

    class Other(bm.Tool):
        pass

    assert measuretool._cache_hook(CustomModel(Other), None) == 'previous'  # pylint: disable=protected-access
    assert len(calls) == 1


def test_model_properties():
    source = bm.ColumnDataSource(data=dict(x=[], y=[]))
    label = bm.Label(text='')
    tool = MeasureTool(source=source, label=label, measure_unit='nm')
    assert tool.source is source and tool.label is label and tool.measure_unit == 'nm'
    assert MeasureTool().measure_unit == ''


def test_old_names_are_gone():
    import fibomat.default_backends.bokeh_backend as backend_module
    assert not hasattr(backend_module, 'DrawTool')
    assert not hasattr(backend_module, '_ugly_patched_bundle_models')
