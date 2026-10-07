# NOTE: ContourParallel is not reworked yet (it needs `pip install fibomat[experimental]` and, for the optimization,
# a mill with a beam). Only the syntax of this example is adapted to the new API; it does not run for now.
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
    dim_shape=shapes.Circle(r=.4) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.ContourParallel(
        offset_pitch=10 * unit('nm'),
        offset_direction='outwards',
        offset_distance=60 * unit('nm'),
        start_direction='inwards',
        scan_sequence=raster_styles.ScanSequence.DOUBLE_SERPENTINE,
        line_style=raster_styles.one_d.Curve(pitch=50 * unit('nm'), scan_sequence=raster_styles.ScanSequence.BACKSTITCH),
        include_original_curve=True,
        optimize=False
    )
)

sample.export(default_backends.SpotListBackend).save('rasterized.txt')
