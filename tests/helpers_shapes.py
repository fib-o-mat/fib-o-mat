"""Helpers to test the shapes sub module."""
import importlib

from fibomat.linalg import BoundingBox, Vector


def import_shape_modules():
    """Import fibomat.shapes.shape and fibomat.shapes.dim_shape."""
    return importlib.import_module('fibomat.shapes.shape'), importlib.import_module('fibomat.shapes.dim_shape')


shape_module, dim_shape_module = import_shape_modules()
Shape = shape_module.Shape
DimShape = dim_shape_module.DimShape


class Dot(Shape):
    """Minimal concrete shape."""

    def __init__(self, position=(0., 0.), description=None):
        super().__init__(description)
        self.position = Vector(position)

    def __repr__(self):
        return f'Dot({self.position!r})'

    @property
    def is_closed(self):
        return False

    @property
    def center(self):
        return self.position

    @property
    def bounding_box(self):
        return BoundingBox(self.position, self.position)

    def _impl_translate(self, trans_vec):
        assert type(trans_vec) is Vector
        self.position = self.position + trans_vec

    def _impl_rotate(self, theta):
        self.position = self.position.rotated(theta)

    def _impl_scale(self, fac):
        self.position = self.position * fac

    def _impl_mirror(self, mirror_axis):
        assert type(mirror_axis) is Vector
        self.position = self.position.mirrored(mirror_axis)


class Box(Dot):
    """Shape with an extent."""

    def __init__(self, ll=(0., 0.), ur=(1., 1.), description=None):
        super().__init__(Vector(ll), description)
        self.ll, self.ur = Vector(ll), Vector(ur)

    @property
    def center(self):
        return (self.ll + self.ur) / 2

    @property
    def bounding_box(self):
        return BoundingBox(self.ll, self.ur)

    def _impl_translate(self, trans_vec):
        self.ll, self.ur = self.ll + trans_vec, self.ur + trans_vec

    def _impl_rotate(self, theta):
        raise NotImplementedError

    def _impl_scale(self, fac):
        self.ll, self.ur = self.ll * fac, self.ur * fac

    def _impl_mirror(self, mirror_axis):
        raise NotImplementedError
