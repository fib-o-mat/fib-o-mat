# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

import numpy as np

from fibomat.arrangements import Group, Lattice
from fibomat.default_backends import StubRasterStyle
from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.shapes import Circle, Rect, Spot
from fibomat.units import unit


sample = Layout()

# The three spots of the unit cell of a kagome lattice. The lattice vectors are u and v (the angle between the lattice
# vectors is 60 degree).
spots = [(-.25 / 2, 0), (.25 / 2, 0), (0, .25 * np.sqrt(3) / 2)]
u = Vector(.5, 0)
v = Vector(.5, 0).rotated(np.pi / 3)

# ----------------------------------------------------------------------------------------------------------------------

kagome_lattice_site = sample.create_site(
    dim_position=Vector(-3, 0) * unit('µm'),
    dim_fov=Vector(5, 5) * unit('µm')
)

unit_cell = Group([Spot(position) for position in spots])
# the spots are given relative to the lattice point, so the pivot (the point which is placed at the lattice point) is the
# origin
unit_cell.pivot = lambda _: Vector(0, 0)

# `from_boundary` fills a shape (a circle in this case) with all lattice points. The unit cell is placed at each of
# them.
kagome_lattice = Lattice.from_boundary(boundary=Circle(r=2.5), u=u, v=v, element=unit_cell)

kagome_lattice_site.create_pattern(
    dim_shape=kagome_lattice * unit('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=1)
)

# ----------------------------------------------------------------------------------------------------------------------

kagome_lattice_fixed_site = sample.create_site(
    dim_position=Vector(3, 0) * unit('µm'),
    dim_fov=Vector(5, 5) * unit('µm')
)

boundary = Rect(5, 5)


# The elements of a lattice can be created by a function. It is called for every lattice point (`i`: index of the
# lattice point, `uv`: its lattice indices, `xy`: its coordinates and `n`: the total number of lattice points).
# Here, only the spots of the unit cells which lie inside of the rectangle are used (the function returns None, i.e. an
# empty lattice site, if there is no spot inside).
def unit_cell_inside_boundary(i, uv, xy, n):
    spots_inside = [
        Spot(position) for position in spots
        if abs(xy[0] + position[0]) < 5 / 2 and abs(xy[1] + position[1]) < 5 / 2
    ]
    if not spots_inside:
        return None

    group = Group(spots_inside)
    # the elements of a lattice are placed with their pivot at the lattice point; the spots are given relative to the
    # lattice point, so the pivot is the origin.
    group.pivot = lambda _: Vector(0, 0)
    return group


kagome_lattice_fixed = Lattice.from_boundary(boundary=boundary, u=u, v=v, element=unit_cell_inside_boundary)

kagome_lattice_fixed_site.create_pattern(
    dim_shape=kagome_lattice_fixed * unit('µm'),
    mill=None,
    raster_style=StubRasterStyle(dimension=1)
)

sample.plot(fullscreen=_fullscreen, legend=False, rasterize_pitch=0.001 * unit('µm'))
