"""Default backends.

* :class:`BokehBackend` plots a layout (interactive, self-contained HTML files). It needs bokeh and pillow (the
  ``exporting`` extra).
* :class:`SpotListBackend` rasterizes all shapes and creates a list of dwell points.
* :class:`PatterningDurationCalculator` estimates the patterning time.

Backends are classes which are passed to :meth:`fibomat.layout.Layout.export`.
"""
from fibomat.default_backends.stub_raster_style import StubRasterStyle
from fibomat.default_backends.spotlist_backend import SpotListBackend
from fibomat.default_backends.patterning_duration_calculator import PatterningDurationCalculator

try:
    from fibomat.default_backends.bokeh_backend import BokehBackend, BokehImage
except ModuleNotFoundError as _error:
    # The plotting dependencies (bokeh, pillow) are optional. The placeholders make the package importable and tell the
    # user what is missing as soon as they are used.
    _MISSING = _error

    from fibomat.backend import BackendBase as _BackendBase

    class BokehBackend(_BackendBase):  # type: ignore[no-redef]
        """Placeholder for the plotting backend (the optional dependencies are not installed)."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'The plotting backend needs the optional dependencies; install fibomat with the "exporting" extra '
                f'(pip install fibomat[exporting]). Original error: {_MISSING}'
            )

    class BokehImage:  # type: ignore[no-redef]
        """Placeholder for the image shape of the plotting backend."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'The plotting backend needs the optional dependencies; install fibomat with the "exporting" extra '
                f'(pip install fibomat[exporting]). Original error: {_MISSING}'
            )

__all__ = ['BokehBackend', 'BokehImage', 'StubRasterStyle', 'SpotListBackend', 'PatterningDurationCalculator']
