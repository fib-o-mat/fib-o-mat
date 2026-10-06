"""Base class of the composite objects with a unit (:class:`DimGlyph` and :class:`DimText`)."""
from __future__ import annotations

import typing as t

import pint  # type: ignore

from fibomat.linalg import DimBoundingBox, DimTransformable, DimVector, DimVectorLike, Transformable
from fibomat.shapes.dim_shape import LengthUnitLike, _length_unit
from fibomat.units import ureg, scale_factor


__all__ = ['DimComposite']


class DimComposite(DimTransformable):
    """A (unitless) transformable object together with a unit of length.

    Like :class:`~fibomat.shapes.dim_shape.DimShape` for composite objects: the wrapped object is not copied (objects
    with unit are immutable by convention); the transformation methods and :meth:`DimComposite.to` return new objects.
    """

    def __init__(self, obj: Transformable, unit: LengthUnitLike, description: t.Optional[str] = None):
        """
        Args:
            obj (Transformable): wrapped object, its coordinates are given in `unit`
            unit (str, pint.Unit, unit): unit of length
            description (str, optional): description

        Raises:
            TypeError: Raised if `unit` is no unit.
            DimensionError: Raised if `unit` is no unit of length.
        """
        self._obj = obj
        self._unit = _length_unit(unit)

        super().__init__(description)

    def __getstate__(self) -> t.Dict[str, t.Any]:
        # pint units must not be pickled (they would not belong to our registry afterwards)
        state = self.__dict__.copy()
        state['_unit'] = str(self._unit)
        return state

    def __setstate__(self, state: t.Dict[str, t.Any]) -> None:
        state = dict(state)
        state['_unit'] = ureg.Unit(state['_unit'])
        self.__dict__.update(state)

    @property
    def unit(self) -> pint.Unit:
        """Unit of length of the object.

        Access:
            get
        """
        return self._unit

    @property
    def center(self) -> DimVector:
        """Center of the object.

        Access:
            get
        """
        return DimVector.from_vector(self._obj.center, self._unit)

    @property
    def pivot(self) -> DimVector:
        """Pivot of the object. If no pivot is set for the object itself, the pivot of the wrapped object is used.

        Access:
            get/set
        """
        if self._pivot is not None:
            return DimVector(self._pivot(self))
        return DimVector.from_vector(self._obj.pivot, self._unit)

    @pivot.setter
    def pivot(self, value: t.Optional[t.Callable[[t.Any], t.Any]]) -> None:
        DimTransformable.pivot.fset(self, value)  # type: ignore[attr-defined]

    @property
    def bounding_box(self) -> DimBoundingBox:
        """Bounding box of the object.

        Access:
            get
        """
        bbox = self._obj.bounding_box
        return DimBoundingBox(
            DimVector.from_vector(bbox.lower_left, self._unit), DimVector.from_vector(bbox.upper_right, self._unit)
        )

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        trans_vec = DimVector(trans_vec)
        self._obj._impl_translate(trans_vec.vector_as(self._unit))  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        self._obj._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        self._obj._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        # the length of the mirror axis does not matter
        self._obj._impl_mirror(DimVector(mirror_axis).vector)  # pylint: disable=protected-access

    def to(self, unit: LengthUnitLike) -> DimComposite:
        """Return a copy of the object in a different unit. The wrapped object is scaled so that its physical size
        stays the same; this object is not modified.

        Args:
            unit (str, pint.Unit, unit): new unit of length

        Returns:
            DimComposite

        Raises:
            TypeError: Raised if `unit` is no unit.
            DimensionError: Raised if `unit` is no unit of length.
        """
        new_unit = _length_unit(unit)

        clone = self.clone()
        # pylint: disable=protected-access
        clone._obj._impl_scale(scale_factor(new_unit, self._unit))
        clone._unit = new_unit
        return clone
