"""Example how to implement a new shape.

A shape is a class derived from :class:`fibomat.shapes.Shape`. It has to provide its bounding box, its center, whether it
is closed, and the methods which translate, rotate, scale and mirror it in-place (the public methods `translated`,
`rotated`, ... are provided by the base class and work on copies).

To be plotted (and exported by backends which support arc splines), the shape should convert itself to an arc spline
(``to_arc_spline``) or, if it has holes, to a hollow arc spline (``to_hollow_arc_spline``). The ring of this example
is a simplified version of :class:`fibomat.composite_shapes.Ring`.
"""
# Ignore the following lines. These are used to adjust the plot for the documentation.
import sys
if 'sphinx-build' in sys.argv:
    _fullscreen = False
else:
    _fullscreen = True

from typing import Optional

import numpy as np

from fibomat.composite_shapes import HollowArcSpline
from fibomat.default_backends import StubRasterStyle
from fibomat.layout import Layout
from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes import Circle, Shape
from fibomat.units import unit


class MyRing(Shape):
    def __init__(
        self, inner_r: float, outer_r: float, center: Optional[VectorLike] = None, description: Optional[str] = None
    ):
        super().__init__(description)

        self._inner_r = float(inner_r)
        self._outer_r = float(outer_r)

        if not 0 < self._inner_r < self._outer_r:
            raise ValueError('0 < inner_r < outer_r is required.')

        self._center = Vector(center) if center is not None else Vector(0, 0)

    @property
    def inner_r(self) -> float:
        return self._inner_r

    @property
    def outer_r(self) -> float:
        return self._outer_r

    def __repr__(self) -> str:
        return '{}(inner_r={!r}, outer_r={!r}, center={!r})'.format(
            self.__class__.__name__, self._inner_r, self._outer_r, self._center
        )

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(
            self._center - (self._outer_r, self._outer_r), self._center + (self._outer_r, self._outer_r)
        )

    @property
    def is_closed(self) -> bool:
        return True

    @property
    def center(self) -> Vector:
        return self._center

    # The in-place transformations. They are called for a copy of the shape by the public methods (`translated`, ...).
    def _impl_translate(self, trans_vec: Vector) -> None:
        self._center = self._center + trans_vec

    def _impl_rotate(self, theta: float) -> None:
        # the ring is rotated about the origin: only its center moves
        self._center = self._center.rotated(theta)

    def _impl_scale(self, fac: float) -> None:
        self._inner_r *= abs(fac)
        self._outer_r *= abs(fac)
        self._center = self._center * fac

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        self._center = self._center.mirrored(mirror_axis)

    # Backends (like the plotting backend) use this method if they do not know the shape.
    def to_hollow_arc_spline(self) -> HollowArcSpline:
        return HollowArcSpline(
            Circle(self._outer_r, center=self._center).to_arc_spline(),
            [Circle(self._inner_r, center=self._center).to_arc_spline()]
        )


# plot an example

ring = MyRing(inner_r=1, outer_r=2)

ring_sample = Layout(description=f'{ring}')

site = ring_sample.create_site(Vector(0, 0) * unit('µm'), dim_fov=Vector(5, 5) * unit('µm'))

site.create_pattern(dim_shape=ring * unit('µm'), mill=None, raster_style=StubRasterStyle(2))

ring_sample.plot(fullscreen=_fullscreen, legend=False)
