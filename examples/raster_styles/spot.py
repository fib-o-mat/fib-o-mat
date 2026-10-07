import numpy as np

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.units import unit
from fibomat import default_backends, shapes, curve_tools, raster_styles

sample = Layout()

site = sample.create_site(
    dim_position=Vector(0, 0) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

# create 100 randomly distributed spots and add them to the sample
# each spot is repeated 5 times (which is actually the same as setting the repeats=1, dwell_time=5 ms)
mill = Mill(dwell_time=1 * unit('ms'), repeats=5)

for pos in np.random.uniform(-1, 1, 100).reshape(-1, 2):
    site.create_pattern(
        dim_shape=shapes.Spot(pos) * unit('µm'),
        mill=mill,
        raster_style=raster_styles.zero_d.SingleSpot()
    )

sample.export(default_backends.SpotListBackend).save('rasterized.txt')

