# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import sympy
from sympy.abc import t
from sympy import sin, cos

import numpy as np

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.units import unit
from fibomat import default_backends, shapes, curve_tools

sample = Layout()


starfish = shapes.ParametricCurve.from_sympy_curve(
    sympy.Curve([3*cos(t)+cos(t)*cos(5*t), 3*sin(t)+sin(t)*cos(5*t)], (t, 0, 2*np.pi))
).to_arc_spline()


# inflate
inflate_site = sample.create_site(
    dim_position=Vector(-8, 0) * unit('µm'),
    dim_fov=Vector(14, 30) * unit('µm'),
    description='inflating'
)

inflating_curve_1 = starfish.translated((0, 8))

# defining the number of steps explicitly
inflated_curves_1 = curve_tools.inflate(inflating_curve_1, 0.5, n_steps=5)
for inflated in inflated_curves_1:
    inflate_site.create_pattern(
        dim_shape=inflated * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(1)
    )

inflate_site.create_pattern(
    dim_shape=inflating_curve_1 * unit('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

inflating_curve_2 = starfish.translated((0, -8))

# defining the offset distance
inflated_curves_2 = curve_tools.inflate(inflating_curve_2, 0.25, distance=2)
for inflated in inflated_curves_2:
    inflate_site.create_pattern(
        dim_shape=inflated * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(1)
    )

inflate_site.create_pattern(
    dim_shape=inflating_curve_2 * unit('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

# deflate
deflate_site = sample.create_site(
    dim_position=Vector(8, 0) * unit('µm'),
    dim_fov=Vector(14, 30) * unit('µm'),
    description='deflating'
)

deflating_curve_1 = starfish.translated((0, 8))

# setting no number of steps or direction will deflate the curve until no curve is left.
deflated_curves_1 = curve_tools.deflate(deflating_curve_1, 0.5)
for deflated in deflated_curves_1:
    deflate_site.create_pattern(
        dim_shape=deflated * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(1)
    )

deflate_site.create_pattern(
    dim_shape=deflating_curve_1 * unit('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

deflating_curve_2 = starfish.translated((0, -8))

# defining the offset distance
deflated_curves_2 = curve_tools.deflate(deflating_curve_2, 0.25, distance=2)
for deflated in deflated_curves_2:
    deflate_site.create_pattern(
        dim_shape=deflated * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(1)
    )

deflate_site.create_pattern(
    dim_shape=deflating_curve_2 * unit('µm'),
    mill=None,
    raster_style=default_backends.StubRasterStyle(1)
)

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * unit('µm'))
