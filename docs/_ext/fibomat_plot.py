"""Sphinx extension which embeds interactive plots of fib-o-mat scripts.

The directive ``fibomat-plot`` runs a script (usually one of the scripts in ``examples/``), takes the layout which the
script plots with :meth:`fibomat.layout.Layout.plot` and embeds the plot (the bokeh plot with all its tools, e.g. the
measure tool) in the page. Every plot is a self-contained HTML document (no JavaScript is loaded from the internet) which
is shown in an ``iframe``.

Example::

    .. fibomat-plot:: ../examples/getting_started.py

The script is run in a temporary directory with ``sphinx-build`` in ``sys.argv`` (the examples use this to switch off
the full screen mode of the plots). The plot is not opened in a browser.

Options:
    height: height of the plot, default 500px
    url: link to the source code of the script. By default it is created from the config value
        ``fibomat_source_url`` (the URL of the folder with the script).
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import runpy
import shutil
import sys
import tempfile
import typing as t
import warnings

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.util import logging


LOGGER = logging.getLogger(__name__)

_PLOTS_DIRECTORY = '_plots'


def _run_script(path: pathlib.Path) -> t.Any:
    """Run a script and return the plotting backend of the first plot it creates.

    Raises:
        SphinxError: Raised if the script does not plot a layout or fails.
    """
    from fibomat.default_backends import BokehBackend  # pylint: disable=import-outside-toplevel

    plotted: t.List[BokehBackend] = []

    original_show = BokehBackend.show
    BokehBackend.show = lambda self: plotted.append(self)  # type: ignore[method-assign]

    old_argv, old_cwd = sys.argv, os.getcwd()
    with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            os.chdir(directory)
            sys.argv = ['sphinx-build']
            runpy.run_path(str(path), run_name='__main__')
        except Exception as error:  # pylint: disable=broad-except
            raise SphinxError(f'fibomat-plot: the script {path} failed: {error!r}') from error
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
            BokehBackend.show = original_show  # type: ignore[method-assign]

    if not plotted:
        raise SphinxError(f'fibomat-plot: the script {path} does not plot a layout (Layout.plot()).')

    return plotted[0]


class FibomatPlotDirective(Directive):
    """Embed the plot of a script."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    option_spec = {
        'height': directives.unchanged,
        'url': directives.unchanged,
    }

    def run(self) -> t.List[nodes.Node]:
        env = self.state.document.settings.env
        app: Sphinx = env.app

        path = (pathlib.Path(env.srcdir) / self.arguments[0]).resolve()
        if not path.exists():
            raise SphinxError(f'fibomat-plot: {path} does not exist.')

        name = f'{path.stem}-{hashlib.sha1(str(path).encode()).hexdigest()[:8]}'
        cache = pathlib.Path(app.doctreedir) / _PLOTS_DIRECTORY
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f'{name}.html'

        source_mtime = path.stat().st_mtime
        if not target.exists() or target.stat().st_mtime < source_mtime:
            LOGGER.info('fibomat-plot: running %s', path.name)
            backend = _run_script(path)
            target.write_text(backend.html(), encoding='utf-8')

        env.note_dependency(str(path))

        depth = env.docname.count('/')
        src = '../' * depth + f'{_PLOTS_DIRECTORY}/{name}.html'
        height = self.options.get('height', '500px')

        iframe = (
            f'<iframe src="{src}" class="fibomat-plot" loading="lazy" '
            f'style="width: 100%; height: {height}; border: 1px solid var(--pst-color-border, #ccc);" '
            f'title="plot of {path.name}"></iframe>'
        )
        result: t.List[nodes.Node] = [nodes.raw('', iframe, format='html')]

        url = self.options.get('url')
        if url is None and app.config.fibomat_source_url:
            url = f'{app.config.fibomat_source_url.rstrip("/")}/{path.name}'

        if url:
            paragraph = nodes.paragraph()
            paragraph += nodes.Text('Source: ')
            paragraph += nodes.reference('', url, internal=False, refuri=url)
            result.append(paragraph)

        return result


def _copy_plots(app: Sphinx, exception: t.Optional[Exception]) -> None:
    """Copy the plots to the output directory (the html files are not created by the directive during the build)."""
    if exception is not None or app.builder.name not in ('html', 'dirhtml'):
        return

    cache = pathlib.Path(app.doctreedir) / _PLOTS_DIRECTORY
    if cache.exists():
        shutil.copytree(cache, pathlib.Path(app.outdir) / _PLOTS_DIRECTORY, dirs_exist_ok=True)


def setup(app: Sphinx) -> t.Dict[str, t.Any]:
    app.add_config_value('fibomat_source_url', '', 'env')
    app.add_directive('fibomat-plot', FibomatPlotDirective)
    app.connect('build-finished', _copy_plots)
    return {'version': '1.0', 'parallel_read_safe': False, 'parallel_write_safe': True}
