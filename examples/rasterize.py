# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.units import unit
from fibomat import default_backends, shapes, curve_tools

sample = Layout()

site = sample.create_site(
    dim_position=Vector(0, 0) * unit('µm'),
    dim_fov=Vector(5, 5) * unit('µm'),
    description='curve intersections'
)

circle = shapes.Circle(r=2.4)

# deflate the circle a few times and then rasterize it

for defl_rect in curve_tools.deflate(circle.to_arc_spline(), 0.5):
    rasterized = curve_tools.rasterize(defl_rect, 0.25)

    for point in rasterized.positions:
        site.create_pattern(
            dim_shape=shapes.Spot(point) * unit('µm'),
            mill=None,
            raster_style=default_backends.StubRasterStyle(1)
        )

    site.create_pattern(
        dim_shape=defl_rect * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(1)
    )

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * unit('µm'))
