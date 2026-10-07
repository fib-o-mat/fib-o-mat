# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

from fibomat import raster_styles
from fibomat.arrangements import DimGroup, Group
from fibomat.layout import Layout, Pattern
from fibomat.linalg import Vector, rotate, translate
from fibomat.mill import Mill
from fibomat.shapes import Circle, Line, Spot
from fibomat.units import unit


sample = Layout()

center_group = Group([
    Line((-1, 0), (1, 0)),
    Line((0, -1), (0, 1)),
    Circle(r=.5, center=(0, 0))
])

marker_pattern = DimGroup([
    Pattern(
        dim_shape=center_group * unit('µm'),
        mill=Mill(dwell_time=1 * unit('ms'), repeats=1),
        raster_style=raster_styles.one_d.Curve(pitch=1 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    ),
    Pattern(
        dim_shape=Spot((.75, .75)) * unit('µm'),
        mill=Mill(dwell_time=1 * unit('ms'), repeats=1),
        raster_style=raster_styles.zero_d.SingleSpot()
    )
])

# we chose again the center of circle located at third element of the first pattern (a DimShape, its center has a unit)
marker_pattern.pivot = lambda self: self.elements[0].dim_shape.elements[2].center


single_marker_site = sample.create_site(
    dim_position=Vector(0, 0) * unit('µm'),
    dim_fov=Vector(2.5, 2.5) * unit('µm')
)

single_marker_site += marker_pattern


# generate four copies and translate and rotate them
first_marker_pattern = marker_pattern.translated(Vector(4, 4) * unit('µm'))
second_marker_pattern = marker_pattern.transformed(translate(Vector(-4, 4) * unit('µm')) | rotate(np.pi/2, origin='pivot'))
third_marker_pattern = marker_pattern.transformed(translate(Vector(-4, -4) * unit('µm')) | rotate(np.pi, origin='pivot'))
fourth_marker_pattern = marker_pattern.transformed(translate(Vector(4, -4) * unit('µm')) | rotate(3*np.pi/2, origin='pivot'))

corner_marker_site = sample.create_site(
    dim_position=Vector(8, 0) * unit('µm'),
    dim_fov=Vector(10, 10) * unit('µm')
)

for corner_marker_pattern in [first_marker_pattern, second_marker_pattern, third_marker_pattern, fourth_marker_pattern]:
    corner_marker_site += corner_marker_pattern

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * unit('µm'))
