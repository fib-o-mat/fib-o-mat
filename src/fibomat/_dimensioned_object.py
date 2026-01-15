"""Provide the :class:`DimensionedObj` class."""
from __future__ import annotations
from typing import Generic, TypeVar, Union, Tuple, Type

from dataclasses import dataclass

from fibomat.units import UnitType, scale_factor
from fibomat.linalg import DimVector, DimTransformable, Transformable, DimVectorLike


ObjT = TypeVar('ObjT', bound=Transformable)
UnitT = TypeVar('UnitT', bound=UnitType)
DimObjT = TypeVar('DimObjT', bound='DimensionedObj')


@dataclass(frozen=True)
class DimensionedObj(DimTransformable, Generic[ObjT, UnitT]):
    """Class should be used if any kind of object must have an associated unit.

    Examples::

        dim_shape = DimensionedObj((Line([0, 0], [1, 1]), U_('µm'))

        # DimObj is an alias of DimensionedObj
        dim_shape = DimObj((Line([0, 0], [1, 1]), U_('µm'))
    """

    obj: ObjT
    unit: UnitT

    @classmethod
    def create(cls: Type[DimObjT], dim_obj: Union[DimObjT, Tuple[ObjT, UnitT]]) -> DimensionedObj[ObjT, UnitT]:
        """Parse a object and create a :class:`DimensionedObj` from a tuple or class instance.

        No checks are performed at the arguments.

        Args:
            dim_obj (DimensionedObj[[ObjT, UnitT], Tuple[ObjT, UnitT]): dimensioned object

        Returns:
            DimensionedObj[ObjT, UnitT]

        Raises:
            TypeError: Raised if `dim_obj` cannot be parsed.
        """
        if isinstance(dim_obj, cls):
            return dim_obj

        if isinstance(dim_obj, tuple):
            if len(dim_obj) != 2:
                raise ValueError('len(dim_obj) != 2. Maybe you forgot to pass a unit?')
            return cls(*dim_obj)

        raise TypeError('Cannot understand passed "dim_obj". Maybe you forgot to pass a unit?')

    @property
    def center(self) -> DimVector:
        return DimVector(self.obj.center, self.unit)

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        trans_vec = DimVector.create(trans_vec)
        self.obj._impl_translate(scale_factor(self.unit, trans_vec.unit) * trans_vec.vector)

    def _impl_rotate(self, theta: float) -> None:
        self.obj._impl_rotate(theta)

    def _impl_scale(self, fac: float) -> None:
        self.obj._impl_scale(fac)

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        mirror_axis = DimVector.create(mirror_axis)
        self.obj._impl_mirror(scale_factor(self.unit, mirror_axis.unit) * mirror_axis.vector)


DimObj = DimensionedObj

DimObjLike = Union[DimensionedObj[ObjT, UnitT], Tuple[ObjT, UnitT]]
