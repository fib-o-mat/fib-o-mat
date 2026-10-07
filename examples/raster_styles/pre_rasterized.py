import numpy as np

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.units import unit
from fibomat import default_backends, shapes, curve_tools, raster_styles
from fibomat.rasterizedpattern import RasterizedPattern

sample = Layout()

rasterized_points_site = sample.create_site(
    dim_position=Vector(-1, 0) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

mill = Mill(dwell_time=1 * unit('ms'), repeats=5)

line = shapes.Line(start=(-.75, -.75), end=(.75, .75))

# first example: rasterize the line manually with a pitch of 0.001 which will eventually become a pitch of 50 nm
# rasterize returns an object of type RasterizedPoints
rasterized_line = curve_tools.rasterize(line, pitch=.05)

# dwell point is a numpy array with N elements where each element contains x, y, weight
# in the PreRasterized style, the dwell_time set in the mill object will be multiplied by the weight parameter
dwell_points = np.array(rasterized_line.dwell_points)

# assigned linearly interpolated values from 1 to 2 to the weights
dwell_points[:, 2] = np.linspace(1, 2, len(dwell_points))

# rasterized_ramp contains now the rasterized lines with increasing weights
rasterized_ramp = shapes.RasterizedPoints(dwell_points, is_closed=False)

# on exporting, the dwell_time of the spots is given by mill.dwell_time * weight of spot
# hence, the last spot has twice the dwell time of the first spot
# the pattern is repeated 5 times because mill.repeats == 5
rasterized_points_site.create_pattern(
    dim_shape=rasterized_ramp * unit('µm'),
    mill=mill,
    raster_style=raster_styles.zero_d.PreRasterized()
)

# ----------------------------------------------------------------------------------------------------------------------

rasterized_pattern_site = sample.create_site(
    dim_position=Vector(1, 0) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

# another approach is to call the rasterize method of the raster style manually.
# note, this mehtod is different from curve_tools.rasterize!
# it will return an object of type RasterizedPattern. this is very similar to the RasterizedPoints object, but contains
# dimensioned position and well times
# in this case, the 1-dim Curve raster style is used to manually rasterize the line
rasterized_pattern = raster_styles.one_d.Curve(
    pitch=50 * unit('nm'), scan_sequence=raster_styles.ScanSequence.BACKSTITCH
).rasterize(
    dim_shape=line * unit('µm'), mill=mill, out_length_unit=unit('µm'), out_time_unit=unit('µs')
)

# access the units of rasterized_pattern
print(rasterized_pattern.length_unit, rasterized_pattern.time_unit)

# access the dwell_points
dwell_points = np.array(rasterized_pattern.dwell_points)

# we con modify the absolute dwell_times here
# for no reason we assign random values between 0 an 5 for for the dwell times here.
# Note that the time unit is set to µs.
dwell_points[:, 2] = np.random.uniform(0, 1, len(dwell_points))

# rebuild the rasterized_pattern
# rasterized_random_pattern = RasterizedPattern(
#     dwell_points,
#     time_unit=rasterized_pattern.time_unit,
#     length_unit=rasterized_pattern.length_unit
# )
#
# # and add it to the site
# # note that we set the the rasterized_pattern_site explicitly to None
# # (it will be ignored when passing a RasterizedPattern)
# # the same is the case for the unit of dim_shape
# rasterized_pattern_site.create_pattern(
#     dim_shape=rasterized_random_pattern,
#     mill=None,
#     raster_style=raster_styles.zero_d.PreRasterized()
# )

sample.export(default_backends.SpotListBackend).save('rasterized.txt')
