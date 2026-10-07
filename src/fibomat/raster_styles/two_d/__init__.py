"""Raster styles for two dimensional shapes."""
from fibomat.raster_styles.two_d.linebyline import LineByLine
from fibomat.raster_styles.two_d.spiral import Spiral

try:
    from fibomat.raster_styles.two_d.contour_parallel import ContourParallel
except RuntimeError as _error:  # pragma: no cover
    # `ContourParallel` needs the optional (experimental) package `numba` (see `fibomat.optimize`); a stub keeps the
    # rest of the package usable and tells the user what is missing as soon as the style is used.
    _OPTIMIZE_ERROR = _error

    class ContourParallel:  # type: ignore[no-redef]
        """Placeholder for :class:`~fibomat.raster_styles.two_d.contour_parallel.ContourParallel`."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'ContourParallel needs the experimental dependencies. Install fibomat with the "experimental" extra '
                f'(pip install fibomat[experimental]). Original error: {_OPTIMIZE_ERROR}'
            )

__all__ = ['ContourParallel', 'LineByLine', 'Spiral']
