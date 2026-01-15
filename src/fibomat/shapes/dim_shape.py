from typing import Optional

from fibomat.linalg import DimTransformable, DimVector, DimVectorLike, DimBoundingBox
from fibomat.shapes.shape import Shape
from fibomat.units import U_, has_length_dim, scale_factor


class DimShape(DimTransformable):
    def __init__(self, shape: Shape, unit: U_, description: Optional[str] = None):
        self._shape = shape

        if not isinstance(unit, U_):
            raise TypeError('unit must be of type pint.Unit (aka U_).')

        if not has_length_dim(unit):
            raise ValueError('unit must have dimension [lenght].')

        self._unit = unit

        super().__init__(description)

    def __repr__(self):
        return '{}(shape={!r}, unit={!r})'.format(self.__class__.__name__, self._shape, self._unit)

    @property
    def shape(self):
        return self._shape

    @property
    def unit(self):
        return self._unit

    @property
    def center(self) -> DimVector:
        return DimVector(self._shape.center * self._unit)

    @property
    def bounding_box(self) -> DimBoundingBox:
        return self._shape.bounding_box * self._unit

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        trans_vec = DimVector(trans_vec)
        self._shape._impl_translate(scale_factor(self._unit, trans_vec.unit) * trans_vec.vector)

    def _impl_rotate(self, theta: float) -> None:
        self._shape._impl_rotate(theta)

    def _impl_scale(self, fac: float) -> None:
        self._shape._impl_scale(fac)

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        mirror_axis = DimVector(mirror_axis)

        # vector length does not matter, mirror axis is normalized anyway.
        self._shape._impl_mirror(mirror_axis.vector)
    
    def set_unit(self, unit: U_) -> None:
        if not isinstance(unit, U_):
            raise TypeError('unit must be of type pint.Unit (aka U_).')

        if not has_length_dim(unit):
            raise ValueError('unit must have dimension [lenght].')
        factor=scale_factor(unit, self._unit)
        self._unit=unit
        self.shape._impl_scale(factor)
