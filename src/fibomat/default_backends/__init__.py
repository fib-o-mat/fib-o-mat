"""
This submodule implements two exemplary backends.
The first one, the bokeh backend, can be used to visualize a project. It uses the bokeh library to create a plot which
can be viewed with a web browser.
The second one, the spotlist backend, rasterizes all shapes and creates a list of dwell points and times.
"""



try:
    from fibomat.default_backends.bokeh_backend import BokehBackend, StubRasterStyle
except ModuleNotFoundError:
    import warnings
    warnings.warn("You need to install the package with 'exporting' as optional dependency to be able to use the plotting feature.")

    # we currently need stub classed to not break the Sample class
    # TODO: fixme
    class BokehBackend:
        pass

    class StubRasterStyle:
        pass

# from fibomat.default_backends.donothing_backend import DoNothingBackend
from fibomat.default_backends.spotlist_backend import SpotListBackend
# from fibomat.default_backends.svg_backend import SVGBackend
# from fibomat.default_backends.bitmap_backend import BitmapBackend
from fibomat.default_backends.patterning_duration_calculator import (
    PatterningDurationCalculator,
)



__all__ = [
    "BokehBackend",
    "StubRasterStyle",
    # "SpotListBackend",
    # "BitmapBackend",
    "PatterningDurationCalculator",
]
