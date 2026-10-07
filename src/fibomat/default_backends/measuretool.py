"""Provides the bokeh tool :class:`MeasureTool`.

The tool is implemented in TypeScript (``measuretool.ts``). The TypeScript source is compiled to ``measuretool.js`` by
``scripts/build_measuretool.py`` and that file is committed: the compiled code is used if it belongs to the current
source, so neither node nor an internet connection is needed to create plots. If ``measuretool.ts`` is changed without
rebuilding, bokeh compiles it when a plot is created (node is needed for this).

Example::

    source = bokeh.models.ColumnDataSource(data=dict(x=[], y=[]))
    label = bokeh.models.Label(x=10, y=10, x_units='screen', y_units='screen', text='')

    figure.add_tools(MeasureTool(source=source, label=label, measure_unit='µm'))
    figure.add_layout(label)
    figure.line('x', 'y', source=source)
"""
import hashlib
import pathlib
import typing as t

from bokeh.core.properties import Instance, String
from bokeh.models import ColumnDataSource, Label, Tool
from bokeh.util.compiler import AttrDict, FromFile, get_cache_hook, set_cache_hook


__all__ = ['MeasureTool']


_HERE = pathlib.Path(__file__).resolve().parent
_SOURCE = _HERE / 'measuretool.ts'
_COMPILED = _HERE / 'measuretool.js'


def _compiled_code() -> t.Optional[str]:
    """The compiled tool or None if it does not exist or does not belong to the TypeScript source."""
    if not _COMPILED.exists():
        return None

    header, _, code = _COMPILED.read_text(encoding='utf-8').partition('\n')
    digest = hashlib.sha256(_SOURCE.read_bytes().decode('utf-8').encode('utf-8')).hexdigest()
    if header != f'// source-sha256: {digest}':
        return None

    return code


class MeasureTool(Tool):
    """A tool to measure distances and angles: drag with the mouse to draw a line, the length and the angle of the line
    are shown in `label`. A click removes the line.

    The line is stored in `source` (columns ``x`` and ``y``, in data coordinates) and has to be added to the plot, as
    well as the `label`, by the user (see the module example).
    """

    __implementation__ = FromFile(str(_SOURCE))

    source = Instance(ColumnDataSource, help='Data source in which the measured line is stored.')
    label = Instance(Label, help='Label which shows the distance and the angle.')
    measure_unit = String(default='', help='Unit which is appended to the measured distance.')


_previous_hook = get_cache_hook()


def _cache_hook(custom_model: t.Any, implementation: t.Any) -> t.Optional[AttrDict]:
    """Return the compiled code of the measure tool (so node is not needed); other models are handled as before."""
    if custom_model.cls is MeasureTool:
        code = _compiled_code()
        if code is not None:
            return AttrDict({'deps': [], 'code': code})

    return _previous_hook(custom_model, implementation)


set_cache_hook(_cache_hook)
