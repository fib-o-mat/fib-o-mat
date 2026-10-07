# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

import sympy
from sympy.abc import t
from sympy import sin

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.units import unit
from fibomat import default_backends
from fibomat import composite_shapes, shapes


all_shapes = Layout(description='All shapes')

mill = None

# Spot
spot = shapes.Spot(position=(0, 0))

spot_site = all_shapes.create_site(
    dim_position=Vector(0, 0) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Spot'
)

spot_site.create_pattern(
    dim_shape=spot * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(0)
)

# RasterizedPoints

# dwell_point = (x, y, w) = tuple of position and weighting factor
# dwell_points = List[dwell_point]
# is_closed = whether the points should be treated as a closed shape
points = shapes.RasterizedPoints(
    dwell_points=np.array([(0, 0, 1), (.3, .9, 1), (-.75, .5, 1), (-.25, -.6, 1)]),
    is_closed=False
)

raster_site = all_shapes.create_site(
    dim_position=Vector(2.5, 0) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='RasterizedPoints'
)

raster_site.create_pattern(
    dim_shape=points * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(0)
)

# Line
line = shapes.Line(start=(-.5, -.5), end=(.5, .5))

line_site = all_shapes.create_site(
    dim_position=Vector(5, 0) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Line'
)

line_site.create_pattern(
    dim_shape=line * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

# Polyline
polyline = shapes.Polyline(
    points=[(-.9, -.9), (-.3, .2), (-.3, -.2), (.4, -.5), (.7, .7)]
)

polyline_site = all_shapes.create_site(
    dim_position=Vector(0, -2.5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Polyline'
)

polyline_site.create_pattern(
    dim_shape=polyline * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

# Polygon

polygon_1 = shapes.Polygon(
    points=[(.6, .9), (.2, .7), (.2, .1), (.4, .3), (.8, .3)]
)

polygon_2 = shapes.Polygon.regular_ngon(n=6, radius=.35, center=(-.5, -.5))

polygon_site = all_shapes.create_site(
    dim_position=Vector(2.5, -2.5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Polygon'
)

polygon_site.create_pattern(
    dim_shape=polygon_1 * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(2)
)

polygon_site.create_pattern(
    dim_shape=polygon_2 * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(2)
)

# Arc

arc_1 = shapes.Arc.from_points((.5, 0), (.4, .4), (0., .8))
arc_2 = shapes.Arc.from_bulge((-.1, -.1), (-.1, -.2), 15)


arc_site = all_shapes.create_site(
    dim_position=Vector(5, -2.5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Arc'
)

arc_site.create_pattern(
    dim_shape=arc_1 * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

arc_site.create_pattern(
    dim_shape=arc_2 * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

# ArcSpline
arc_spline_1 = shapes.ArcSpline.from_segments([
    shapes.Arc.from_points((1, -1), (4, 0), (1, 1)),
    shapes.Line((1, 1), (-1, 1)),
    shapes.Arc.from_points((-1, 1), (-4, 0), (-1, -1)),
    shapes.Line((-1, -1), (1, -1)),
]).scaled(.2).translated((0, -.4))

# points are tuples of (x, y, bulge). The module reference for info about bulge.
arc_spline_2 = shapes.ArcSpline(
    [(-.6, .3, -.2), (-.4, .8, -.2), (-.2, .3, .2), (0, .8, .2), (.2, .3, -.2), (.4, .8, -.2), (.6, .3, 0)],
    is_closed=False
)

arc_spline_site = all_shapes.create_site(
    dim_position=Vector(0, -5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='ArcSpline'
)

arc_spline_site.create_pattern(
    dim_shape=arc_spline_1 * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(2)
)

arc_spline_site.create_pattern(
    dim_shape=arc_spline_2 * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

# ParametricCurve


# all derivatives are calculated automatically if sympy curves are used
parametric_1 = shapes.ParametricCurve.from_sympy_curve(
    curve=sympy.Curve([sin(3*np.pi*t), sin(2*np.pi*t)], (t, 0, 2))
)
parametric_1_as_arc_spline = parametric_1.to_arc_spline().scaled(.5).translated((-.5, -.5))


def f(t_):
    t_ = np.asarray(t_)
    return 0.25 * np.array(
        (np.cos(2*t_) - 3 * np.sin(t_) - 1, np.sin(2*t_) - 2*np.cos(t_))
    ).T

def df(t_):
    t_ = np.asarray(t_)

    return 0.25 * np.array(
        (-2*np.sin(2*t_) - 3 * np.cos(t_), 2*np.cos(2*t_) + 2*np.sin(t_))
    ).T

def d2f(t_):
    t_ = np.asarray(t_)

    return 0.25 * np.array(
        (-4*np.cos(2*t_) + 3 * np.sin(t_), -4*np.sin(2*t_) + 2*np.cos(t_))
    ).T

parametric_2 = shapes.ParametricCurve(f, df, d2f, domain=(np.pi/2, 5*np.pi/2))

parametric_2_as_arc_spline = parametric_2.to_arc_spline().translated((.6, .3))

parametric_site = all_shapes.create_site(
    dim_position=Vector(2.5, -5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='ParametricCurve'
)

parametric_site.create_pattern(
    dim_shape=parametric_1_as_arc_spline * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

parametric_site.create_pattern(
    dim_shape=parametric_2_as_arc_spline * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

# Rect
rect = shapes.Rect(width=1.2, height=.9, theta=np.pi/3)

rect_site = all_shapes.create_site(
    dim_position=Vector(5, -5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Rect'
)

rect_site.create_pattern(
    dim_shape=rect * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

# Circle
circle = shapes.Circle(r=.8)

circle_site = all_shapes.create_site(
    dim_position=Vector(0, -7.5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Circle'
)

circle_site.create_pattern(
    dim_shape=circle * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

# Ellipse
ellipse = shapes.Ellipse(a=.6, b=1.1, theta=np.pi/4)

ellipse_site = all_shapes.create_site(
    dim_position=Vector(2.5, -7.5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Ellipse'
)

ellipse_site.create_pattern(
    dim_shape=ellipse * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(1)
)

# HollowArcSpline

boundary = shapes.ArcSpline.from_shape(shapes.Circle(r=.75))

holes = []

# add holes which are partly outside of the boundary. the boundary and this holes will be combined via Boolean
# operations automatically
for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
    holes.append(shapes.Circle(r=.25, center=(0, .75)).rotated(phi).to_arc_spline())

# add some partially overlapping holes, which are in the inside of the boundary curve
holes.append(shapes.Rect(width=.25, height=.25, theta=np.pi/4, center=(.15, .15)).to_arc_spline())
holes.append(shapes.Rect(width=.25, height=.25, theta=np.pi/4, center=(-.15, .15)).to_arc_spline())
holes.append(shapes.Rect(width=.25, height=.25, theta=np.pi/4, center=(.15, -.15)).to_arc_spline())
# note: we cannot add a fourth rect in the third quadrant. this would create a not simply connected shape which his not
# supported.

hollow_spline = composite_shapes.HollowArcSpline(boundary=boundary, holes=holes)

hollow_spline_site = all_shapes.create_site(
    dim_position=Vector(5, -7.5) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm'),
    description='Ellipse'
)

hollow_spline_site.create_pattern(
    dim_shape=hollow_spline * unit('µm'),
    mill=mill,
    raster_style=default_backends.StubRasterStyle(2)
)

# plot all
all_shapes.plot(rasterize_pitch=.001 * unit('µm'), legend=False, fullscreen=_fullscreen)
