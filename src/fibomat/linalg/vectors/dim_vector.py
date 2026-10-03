"""Two dimensional vectors of lengths.

Example::

    from fibomat.linalg import DimVector
    from fibomat.units import unit

    v = DimVector(1 * unit('µm'), 2 * unit('µm'))
    v = DimVector([1 * unit('µm'), 2 * unit('µm')])
    v = DimVector(r=1 * unit('nm'), phi=np.pi)
    v = Vector(1, 2) * unit('µm')

    v.x  # DimFloat[LengthDimension]
    v.vector_as('nm')  # Vector(1000, 2000)
"""
from __future__ import annotations

import typing as t
import warnings

import numpy as np
import pint  # type: ignore

from fibomat.linalg.vectors.vector import Vector, VectorValueError, FloatTypes
from fibomat.units import DimFloat, DimensionError, LengthDimension, scale_factor, has_length_dim, ureg
from fibomat.units.unit_tag import _UnitTag  # pylint: disable=protected-access


__all__ = ['DimVector']


# absolute tolerance of DimVector.close_to
_ATOL = 1e-12 * ureg.meter


def _warn_quantity() -> None:
    warnings.warn(
        'Passing pint.Quantity to DimVector is deprecated. Use unit(...) / U_(...), e.g. 1. * unit("um").',
        category=DeprecationWarning,
        stacklevel=4,
    )


def _rebuild_dim_vector(cls: t.Type[DimVector], array: t.List[float], unit: str) -> DimVector:
    """Helper for unpickling."""
    return cls.from_vector(Vector(array), unit)


class DimVector(Vector[DimFloat[LengthDimension]]):
    """A two dimensional vector of lengths.

    Components are :class:`fibomat.units.DimFloat` of dimension ``[length]``. Internally, the vector stores floats and
    one unit (the unit of the first component, if created from components). All operations between dim-vectors
    convert units as needed. Mixing a :class:`DimVector` and a plain :class:`Vector` raises
    :class:`VectorValueError`.

    .. note:: Components given as `pint.Quantity` are accepted, but this is deprecated and warns.
    """

    _has_unit = True

    # ------------------------------------------------------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _is_scalar(val: t.Any) -> bool:
        if isinstance(val, DimFloat):
            return LengthDimension.matches(val.quantity) and isinstance(val.magnitude, FloatTypes)
        if isinstance(val, pint.Quantity):
            return has_length_dim(val) and isinstance(val.magnitude, FloatTypes)
        return False

    @staticmethod
    def _to_dim_float(val: t.Any) -> DimFloat[LengthDimension]:
        """Convert a valid scalar to DimFloat (warns for pint quantities)."""
        if isinstance(val, DimFloat):
            return val
        _warn_quantity()
        return DimFloat(ureg.Quantity(val.magnitude, val.units))

    def _set_from_scalars(self, x: t.Any, y: t.Any) -> None:  # pylint: disable=invalid-name
        x_dim, y_dim = self._to_dim_float(x), self._to_dim_float(y)
        self._unit: pint.Unit = x_dim.units
        # y is converted to the unit of x; the arguments are not modified
        self._array = np.array([x_dim.magnitude, y_dim.m_as(self._unit)], dtype=float)

    def _set_zero(self) -> None:
        self._unit = ureg.Unit('µm')
        self._array = np.zeros(2)

    def _scalar(self, value: float) -> DimFloat[LengthDimension]:
        return DimFloat(ureg.Quantity(float(value), self._unit))

    def _product(self, value: float) -> DimFloat[t.Any]:
        return DimFloat(ureg.Quantity(float(value), self._unit ** 2))

    def _length_to_float(self, length: t.Any) -> float:
        if not self._is_scalar(length):
            raise VectorValueError('length not compatible with vector values.')
        return self._to_dim_float(length).m_as(self._unit)

    def _other_array(self, other: t.Any) -> np.ndarray:
        other_vec = DimVector(other)
        return other_vec._array * scale_factor(self._unit, other_vec._unit)  # pylint: disable=protected-access

    def _copy_meta(self, other: Vector[t.Any]) -> None:
        self._unit = other._unit  # type: ignore[attr-defined]  # pylint: disable=protected-access

    def _atol(self) -> float:
        return float(_ATOL.m_as(self._unit))

    # ------------------------------------------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_vector(cls, vector: Vector[float], unit: t.Union[str, pint.Unit, _UnitTag]) -> DimVector:
        """Create a dim-vector from a unitless vector and a unit.

        Args:
            vector (Vector): unitless vector
            unit (str, pint.Unit, unit): length unit, e.g. ``unit('µm')`` or ``'µm'``

        Returns:
            DimVector

        Raises:
            VectorValueError: Raised if `vector` has a unit.
            DimensionError: Raised if `unit` is no unit of length.
        """
        if vector._has_unit:  # pylint: disable=protected-access
            raise VectorValueError('vector must not have a unit.')

        if isinstance(unit, _UnitTag):
            pint_unit = (1. * unit).units
        else:
            pint_unit = ureg.Unit(unit)
        if not has_length_dim(pint_unit):
            raise DimensionError(f'{unit} is not a unit of length.')

        new = cls.__new__(cls)
        new._array = np.array(np.asarray(vector), dtype=float)  # pylint: disable=protected-access
        new._unit = pint_unit  # pylint: disable=protected-access
        return new

    def __reduce__(self) -> t.Tuple[t.Any, ...]:
        """Support for pickle. The unit is stored as string, because pint units cannot be pickled across registries."""
        return _rebuild_dim_vector, (self.__class__, self._array.tolist(), str(self._unit))

    # ------------------------------------------------------------------------------------------------------------------
    # properties and conversion
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(x={self.x.quantity:~}, y={self.y.quantity:~})'

    @property
    def vector(self) -> Vector[float]:
        """The unitless vector in units of :attr:`DimVector.unit`."""
        return Vector(self._array)

    @property
    def unit(self) -> pint.Unit:
        """Unit of the internally stored components."""
        return self._unit

    def vector_as(self, unit: t.Union[str, pint.Unit, _UnitTag]) -> Vector[float]:
        """Return the unitless vector in units of `unit`.

        Args:
            unit (str, pint.Unit, unit): length unit

        Returns:
            Vector

        Raises:
            DimensionError: Raised if `unit` is no unit of length.
        """
        pint_unit = (1. * unit).units if isinstance(unit, _UnitTag) else ureg.Unit(unit)
        if not has_length_dim(pint_unit):
            raise DimensionError(f'{unit} is not a unit of length.')
        return Vector(self._array * scale_factor(pint_unit, self._unit))

    def to(self, unit: t.Union[str, pint.Unit, _UnitTag]) -> DimVector:
        """Return the vector converted to `unit` as internal unit.

        Args:
            unit (str, pint.Unit, unit): length unit

        Returns:
            DimVector
        """
        return DimVector.from_vector(self.vector_as(unit), unit)
