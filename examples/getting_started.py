# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys

if "sphinx-build" in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from fibomat import default_backends, linalg, raster_styles, shapes
from fibomat.layout import Layout, Pattern
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.units import unit

sample = Layout(description="an optional description for yourself")

site = sample.create_site(
    dim_position=Vector(123.0, 456.0) * unit("µm"),
    dim_fov=Vector(5.0, 5.0) * unit("µm"),
    description="another description",
)

# a mill object with defines the dwell time per spot in the rasterized shape and the number of repeats
single_repeat_mill = Mill(dwell_time=5 * unit("ms"), repeats=1)

# and a line shape
line = shapes.Line(start=(-2, 2), end=(2, 0.5))

# and finally rasterizing style. In this case, the line will be rasterized consecutive from start to end.
line_style = raster_styles.one_d.Curve(
    pitch=1 * unit("nm"), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
)

# everything is collected in a pattern
line_pattern = Pattern(
    dim_shape=line * unit("µm"), mill=single_repeat_mill, raster_style=line_style
)

# and added to the site.
site += line_pattern
# or
# site.add_pattern(line_pattern)

# secondly, add a square

square = shapes.Rect(width=2, height=2, center=(0, -1))

# rasterize the square line-by-line. see text for details
square_style = raster_styles.two_d.LineByLine(
    line_pitch=10 * unit("nm"),
    scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
    alpha=0,
    invert=False,
    line_style=raster_styles.one_d.Curve(
        pitch=10 * unit("nm"), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
    ),
)

# we can also create the pattern in-place
site.create_pattern(
    dim_shape=square * unit("µm"), mill=single_repeat_mill, raster_style=square_style
)

# plot the patterning layout and save the plot
# if you run the script for yourself, uncomment the following line and delete the line below.
# sample.plot()
sample.plot(fullscreen=_fullscreen)

# export as text file
sample.export(default_backends.SpotListBackend).save("getting_started.txt")
