# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.units import unit
from fibomat import default_backends, shapes, curve_tools

sample = Layout()

# a closed arc spline with six corners. Its vertices are (x, y, bulge); the segments are circular arcs which bulge
# inwards.
angles = np.arange(6) * np.pi / 3
spline = shapes.ArcSpline(
    [(.866 * np.cos(angle), .866 * np.sin(angle), -.25) for angle in angles],
    is_closed=True
)

# replace the corners with arcs with a radius of 0.05 (the result has no corners anymore).
# curve_tools.smooth raises a NonSmoothableError if a corner cannot be replaced, e.g. if the radius is too large.
spline_smoothed = curve_tools.smooth(spline, .05)

unsmoothed_site = sample.create_site(
    dim_position=Vector(-1.5, 0) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='unsmoothed'
)

unsmoothed_site.create_pattern(
    dim_shape=spline * unit('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

smoothed_site = sample.create_site(
    dim_position=Vector(1.5, 0) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='smoothed'
)

smoothed_site.create_pattern(
    dim_shape=spline_smoothed * unit('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * unit('µm'))
