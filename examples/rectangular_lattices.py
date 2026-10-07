# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from typing import Tuple

import numpy as np

from fibomat import raster_styles
from fibomat.arrangements import DimLattice, Group, Lattice
from fibomat.default_backends import StubRasterStyle
from fibomat.layout import Layout, Pattern, Site
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.shapes import Circle, Ellipse, Line, Rect, Spot
from fibomat.units import unit


def ellipse_generator(i: int, uv: Tuple[int, int], xy: Tuple[float, float], n: int):
    """Create the element of a lattice point.

    Args:
        i: index of the lattice point
        uv: indices (u, v) of the lattice point
        xy: coordinates of the lattice point
        n: total number of lattice points
    """
    ellipse_base_radius = .1
    u, v = uv

    # the position of the shape does not matter here, the lattice moves it to the lattice point.
    return Ellipse(a=ellipse_base_radius + 0.05 * u, b=ellipse_base_radius + 0.05 * v)


def checkerboard_generator(i: int, uv: Tuple[int, int], xy: Tuple[float, float], n: int):
    size = .2
    u, v = uv

    if (u + v) % 2 == 0:
        return Rect(width=size, height=size)
    else:
        return None  # no element at this lattice point


sample = Layout()

spot_lattice_site = sample.create_site(
    dim_position=Vector(-2.5, 0) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm')
)

# a lattice with 3 x 4 points and a distance of .75 in x and .5 in y direction. The element is placed at each lattice
# point (the position of the spot does not matter).
spot_lattice = Lattice.from_counts(n_x=3, n_y=4, pitch_x=.75, pitch_y=.5, element=Spot((0, 0)))

spot_lattice_site.create_pattern(
    dim_shape=spot_lattice * unit('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=1)
)

# ----------------------------------------------------------------------------------------------------------------------

ellipse_lattice_site = sample.create_site(
    dim_position=Vector(0, 0) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm')
)

# the elements can be created by a function
ellipse_lattice = Lattice.from_counts(
    n_x=4, n_y=4, pitch_x=.5, pitch_y=.5, element=ellipse_generator, center=(-.15 / 2, .15 / 2)
)

ellipse_lattice_site.create_pattern(
    dim_shape=ellipse_lattice * unit('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=1)
)

# ----------------------------------------------------------------------------------------------------------------------

checkerboard_lattice_site = sample.create_site(
    dim_position=Vector(2.5, 0) * unit('µm'),
    dim_fov=Vector(2, 2) * unit('µm')
)

# a lattice can also be created from the size of a rectangle: all points which fit in it are used (here 1.6 x 1.6 and a
# pitch of .2: 9 x 9 points). If the function returns None, the lattice point is empty.
checkerboard_lattice = Lattice.from_rect(
    width=1.6, height=1.6, pitch_x=.2, pitch_y=.2, element=checkerboard_generator
)

checkerboard_lattice_site.create_pattern(
    dim_shape=checkerboard_lattice * unit('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=2)
)

# ----------------------------------------------------------------------------------------------------------------------

non_square_lattice_site = sample.create_site(
    dim_position=Vector(0.5, -5) * unit('µm'),
    dim_fov=Vector(7, 7) * unit('µm')
)

# lattices with arbitrary lattice vectors fill a shape
non_square_lattice = Lattice.from_boundary(
    boundary=Rect(4, 4),
    u=(np.sqrt(5) / 2, 0), v=(0, -np.sqrt(5) / 2),
    element=Circle(r=.25),
).rotated(np.pi / 4 - np.deg2rad(60))

non_square_lattice_site.create_pattern(
    dim_shape=non_square_lattice * unit('µm'),
    mill=None,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=1 * unit('nm'),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
        alpha=0,
        invert=False,
        line_style=raster_styles.one_d.Curve(
            pitch=1 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
        )
    )
)

# ----------------------------------------------------------------------------------------------------------------------

square_pattern_site = sample.create_site(
    dim_position=Vector(8.5, 0) * unit('µm'),
    dim_fov=Vector(7, 7) * unit('µm')
)

marker = Group([
    Line((-1, 0), (1, 0)),
    Line((0, -1), (0, 1)),
    Circle(r=.5, center=(0, 0)),
    Circle(r=.05, center=(.75, .75))
]).scaled(.75)

marker.pivot = lambda self: self.elements[2].center

pattern = Pattern(
    dim_shape=marker * unit('µm'),
    mill=Mill(dwell_time=1 * unit('ms'), repeats=1),
    raster_style=raster_styles.one_d.Curve(pitch=1 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
)

# lattices with units: the elements (here a pattern), the pitches and the center have units. A lattice of patterns can
# be added to a site.
square_pattern_lattice = DimLattice.from_counts(
    n_x=3, n_y=3,
    pitch_x=2 * unit('µm'), pitch_y=2 * unit('µm'),
    element=pattern
)

square_pattern_site.add_pattern(square_pattern_lattice)

# ----------------------------------------------------------------------------------------------------------------------

# lattices of sites: all sites are added to the sample
site_lattice = DimLattice.from_counts(
    n_x=4, n_y=4,
    pitch_x=4 * unit('µm'), pitch_y=4 * unit('µm'),
    element=Site(dim_center=Vector(0, 0) * unit('µm'), dim_fov=Vector(3, 3) * unit('µm')),
    center=Vector(4.5, -17) * unit('µm')
)

sample.add_site(site_lattice)

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * unit('µm'))
