import numpy as np

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.units import unit
from fibomat import default_backends, shapes, curve_tools, raster_styles

sample = Layout()

mill = Mill(dwell_time=1 * unit('ms'), repeats=5)

site_consecutive = sample.create_site(
    dim_position=Vector(0, 0) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_consecutive.create_pattern(
    dim_shape=shapes.Line(start=(-.75, -.75), end=(.75, .75)) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.one_d.Curve(pitch=1 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
)

# ----------------------------------------------------------------------------------------------------------------------

site_back_stitch = sample.create_site(
    dim_position=Vector(1, 0) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_back_stitch.create_pattern(
    dim_shape=shapes.Circle(r=.5) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.one_d.Curve(pitch=100 * unit('nm'), scan_sequence=raster_styles.ScanSequence.BACKSTITCH)
)

# ----------------------------------------------------------------------------------------------------------------------

site_back_and_forth = sample.create_site(
    dim_position=Vector(2, 0) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_back_and_forth.create_pattern(
    dim_shape=shapes.Line(start=(-.75, -.75), end=(.75, .75)) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.one_d.Curve(pitch=1 * unit('nm'), scan_sequence=raster_styles.ScanSequence.BACK_AND_FORTH)
)


sample.export(default_backends.SpotListBackend).save('rasterized.txt')

