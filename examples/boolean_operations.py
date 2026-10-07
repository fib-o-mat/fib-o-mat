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
from fibomat.curve_tools import CombineMode

sample = Layout()

c1 = shapes.ArcSpline.from_shape(shapes.Circle(r=1, center=(-.75, 0)))
c2 = shapes.ArcSpline.from_shape(shapes.Circle(r=1, center=(.75, 0)))


# Union
union_site = sample.create_site(
    dim_position=Vector(-2.5, 2.5) * unit('µm'),
    dim_fov=Vector(4, 4) * unit('µm'),
    description='Union'
)

union_curves = curve_tools.combine_curves(c1, c2, CombineMode.UNION)
for curve in union_curves.boundaries:
    union_site.create_pattern(
        dim_shape=curve * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(2)
    )

# Xor
xor_site = sample.create_site(
    dim_position=Vector(2.5, 2.5) * unit('µm'),
    dim_fov=Vector(4, 4) * unit('µm'),
    description='Xor'
)

xor_curves = curve_tools.combine_curves(c1, c2, CombineMode.XOR)
for curve in xor_curves.boundaries:
    xor_site.create_pattern(
        dim_shape=curve * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(2)
    )

# Exclude
exclude_site = sample.create_site(
    dim_position=Vector(-2.5, -2.5) * unit('µm'),
    dim_fov=Vector(4, 4) * unit('µm'),
    description='Exclude'
)

exclude_curves = curve_tools.combine_curves(c1, c2, CombineMode.EXCLUDE)
for curve in exclude_curves.boundaries:
    exclude_site.create_pattern(
        dim_shape=curve * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(2)
    )

# Intersect
intersect_site = sample.create_site(
    dim_position=Vector(2.5, -2.5) * unit('µm'),
    dim_fov=Vector(4, 4) * unit('µm'),
    description='Intersect'
)

intersect_curves = curve_tools.combine_curves(c1, c2, CombineMode.INTERSECT)
for curve in intersect_curves.boundaries:
    intersect_site.create_pattern(
        dim_shape=curve * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(2)
    )

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * unit('µm'))
