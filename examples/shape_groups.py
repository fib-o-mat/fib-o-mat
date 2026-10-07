# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

from fibomat.layout import Layout
from fibomat import raster_styles
from fibomat.arrangements import Group
from fibomat.linalg import Vector, rotate, translate
from fibomat.mill import Mill
from fibomat.shapes import Circle, Line
from fibomat.units import unit


sample = Layout()

marker = Group(
    [
        Line((-1, 0), (1, 0)),
        Line((0, -1), (0, 1)),
        Circle(r=.5, center=(0, 0)),
        Circle(r=.05, center=(.75, .75))
    ],
)

marker.pivot = lambda self: self.elements[2].center

single_marker_site = sample.create_site(
    dim_position=Vector(0, 0) * unit('µm'),
    dim_fov=Vector(2.5, 2.5) * unit('µm')
)

single_marker_site.create_pattern(
    dim_shape=marker * unit('µm'),
    mill=Mill(dwell_time=1 * unit('ms'), repeats=1),
    raster_style=raster_styles.one_d.Curve(pitch=1 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
)

# generate four copies and translate and rotate them
first_marker = marker.translated((4, 4))
second_marker = marker.transformed(translate((-4, 4)) | rotate(np.pi/2, origin='pivot'))
third_marker = marker.transformed(translate((-4, -4)) | rotate(np.pi, origin='pivot'))
fourth_marker = marker.transformed(translate((4, -4)) | rotate(3*np.pi/2, origin='pivot'))


corner_marker_site = sample.create_site(
    dim_position=Vector(8, 0) * unit('µm'),
    dim_fov=Vector(10, 10) * unit('µm')
)

for corner_marker in [first_marker, second_marker, third_marker, fourth_marker]:
    corner_marker_site.create_pattern(
        dim_shape=corner_marker * unit('µm'),
        mill=Mill(dwell_time=1 * unit('ms'), repeats=1),
        raster_style=raster_styles.one_d.Curve(pitch=1 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * unit('µm'))
