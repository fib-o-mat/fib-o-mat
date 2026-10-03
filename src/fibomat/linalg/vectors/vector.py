"""Two dimensional vectors.

:class:`Vector` is a generic 2D vector. The plain class holds floats, :class:`fibomat.linalg.vectors.DimVector`
derives from it and holds lengths (:class:`fibomat.units.DimFloat`). All of the logic (arithmetics, rotation, mirroring,
...) lives here; subclasses only provide a few hooks (see "Hooks" below).

Example::

    from fibomat.linalg import Vector

    v = Vector(1, 2)
    v = Vector([1, 2])
    v = Vector(r=2, phi=np.pi / 2)

    v.x, v.y, v.r, v.phi
    v + (1, 1)
    2 * v
    v.rotated(np.pi / 2, origin=(1, 1))
    v.dot((1, 0))

    # attach a unit, see DimVector
    dim_v = v * unit('µm')

Hooks (for subclasses):

* ``_is_scalar``: check if a value is a valid component,
* ``_set_from_scalars``: set the internal state from two components,
* ``_scalar`` / ``_product``: wrap a float into the component type / the type of a dot product,
* ``_length_to_float``: convert a length argument (e.g. of :meth:`Vector.normalized_to`) to a float,
* ``_other_array``: convert a vector-like object to the internal float array,
* ``_copy_meta``: copy additional state (e.g. the unit) to a new vector,
* ``_atol``: absolute tolerance of :meth:`Vector.close_to`.
"""
from __future__ import annotations

import typing as t
import warnings

import numpy as np


__all__ = ['Vector', 'VectorValueError', 'FloatTypes']


FloatTypes = (int, float, np.integer, np.floating)
"""Types which are accepted as components of :class:`Vector`."""

S = t.TypeVar('S')
"""Component type."""
SelfT = t.TypeVar('SelfT', bound='Vector[t.Any]')


class VectorValueError(ValueError):
    """Exception of this type is raised if any non supported value type is passed to Vector.__init__."""


class Vector(t.Sequence[S]):
    """A two dimensional vector.

    Vectors are immutable by convention: all operations return new vectors. Vectors are not hashable.

    A vector can be created from

    * two components: ``Vector(1, 2)``, ``Vector(x=1, y=2)``,
    * an iterable with exactly two items: ``Vector([1, 2])``, ``Vector(Vector(1, 2))``,
    * polar coordinates: ``Vector(r=1, phi=np.pi)``,
    * nothing: ``Vector()`` is the null vector.
    """

    # make numpy return NotImplemented if a numpy scalar is the left operand (`np.float64(2) * vector`), so that
    # the reflected operators of this class are used.
    __array_ufunc__ = None
    __hash__ = None  # type: ignore[assignment]

    _has_unit = False
    """True if the vector has a physical unit."""

    # ------------------------------------------------------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _is_scalar(val: t.Any) -> bool:
        return isinstance(val, FloatTypes)

    def _set_from_scalars(self, x: t.Any, y: t.Any) -> None:  # pylint: disable=invalid-name
        self._array = np.array([x, y], dtype=float)

    def _scalar(self, value: float) -> t.Any:
        return float(value)

    def _product(self, value: float) -> t.Any:
        """Wrap the result of a product of two components (dot and cross product)."""
        return float(value)

    def _length_to_float(self, length: t.Any) -> float:
        if not self._is_scalar(length):
            raise VectorValueError('length not compatible with vector values.')
        return float(length)

    def _other_array(self, other: t.Any) -> np.ndarray:
        """Convert a vector-like object to the internal float array (in the unit of self)."""
        # pylint: disable=protected-access
        return self.__class__(other)._array

    def _copy_meta(self, other: Vector[t.Any]) -> None:
        """Copy additional state of `other` (e.g. the unit) to self."""

    def _atol(self) -> float:
        return 1e-8

    # ------------------------------------------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        x: t.Any = None,  # pylint: disable=invalid-name
        y: t.Any = None,  # pylint: disable=invalid-name
        r: t.Any = None,  # pylint: disable=invalid-name
        phi: t.Optional[float] = None,
    ):
        """
        Args:
            x: x component or an iterable with two items
            y: y component
            r: radial component
            phi: angular component in rad

        Raises:
            VectorValueError: Raised if the arguments are incompatible or have unsupported types.
        """
        self._array: np.ndarray
        if x is None and y is None and r is None and phi is None:
            self._set_zero()
        elif x is not None:
            if self._is_scalar(x):
                if r is not None or phi is not None:
                    raise VectorValueError('Define (x, y) or (r, phi) but not both.')
                if y is None or not self._is_scalar(y):
                    raise VectorValueError(
                        f'if x is a scalar, y must be defined and must have the same type. Got x = {x} and y = {y}.'
                    )
                self._set_from_scalars(x, y)
            else:  # x is some kind of list, tuple, Vector, np.ndarray, ...
                if y is not None or r is not None or phi is not None:
                    raise VectorValueError('when x is an object, you must not specify y, r, or phi')
                self._set_from_scalars(*self._from_iterable(x))
        elif r is not None and phi is not None:  # polar case
            if y is not None:
                raise VectorValueError('Define (x, y) or (r, phi) but not both.')
            self._set_from_scalars(*self._from_polar(r, phi))
        else:
            raise VectorValueError('Incompatible combination of arguments.')

    def _set_zero(self) -> None:
        self._set_from_scalars(0., 0.)

    @classmethod
    def _from_iterable(cls, iterable: t.Iterable[t.Any]) -> t.Tuple[t.Any, t.Any]:
        if not hasattr(iterable, '__iter__'):
            raise VectorValueError('x must be a scalar or an iterable.')

        items = list(iterable)
        if len(items) != 2:
            raise VectorValueError(f'if x is an iterable, it must contain exactly 2 items. Got {len(items)}.')

        x_val, y_val = items
        if not cls._is_scalar(x_val):
            raise VectorValueError('x not compatible with vector values.')
        if not cls._is_scalar(y_val):
            raise VectorValueError('y not compatible with vector values.')

        return x_val, y_val

    @classmethod
    def _from_polar(cls, r: t.Any, phi: float) -> t.Tuple[t.Any, t.Any]:  # pylint: disable=invalid-name
        if not cls._is_scalar(r):
            raise VectorValueError('r not compatible with vector values.')
        if not isinstance(phi, FloatTypes):
            raise VectorValueError('phi not compatible with vector values.')

        phi = float(phi)
        return r * float(np.cos(phi)), r * float(np.sin(phi))

    def _new(self: SelfT, array: np.ndarray) -> SelfT:
        """Create a new vector of the same type (and unit) from a float array."""
        new = self.__class__.__new__(self.__class__)
        new._array = np.array(array, dtype=float)  # pylint: disable=protected-access
        new._copy_meta(self)  # pylint: disable=protected-access
        return new

    # ------------------------------------------------------------------------------------------------------------------
    # sequence protocol
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(x={self.x}, y={self.y})'

    @t.overload
    def __getitem__(self, index: int) -> S: ...  # pragma: no cover
    @t.overload
    def __getitem__(self, index: slice) -> t.List[S]: ...  # pragma: no cover
    def __getitem__(self, index: t.Union[int, slice]) -> t.Union[S, t.List[S]]:
        if isinstance(index, (int, np.integer)):
            return self._scalar(self._array[index])
        if isinstance(index, slice):
            return [self._scalar(val) for val in self._array[index]]
        raise TypeError('index must be int or slice.')

    def __len__(self) -> int:
        return 2

    def __array__(self, dtype: t.Any = None, copy: t.Optional[bool] = None) -> np.ndarray:  # pylint: disable=unused-argument
        """Allow conversion to a numpy array (a unit, if any, is dropped).

        ::

            np_array = np.asarray(Vector(1, 2))

        Returns:
            numpy.ndarray
        """
        return np.array(self._array, dtype=dtype)

    # ------------------------------------------------------------------------------------------------------------------
    # components
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def x(self) -> S:  # pylint: disable=invalid-name
        """X component of the vector."""
        return self._scalar(self._array[0])

    @property
    def y(self) -> S:  # pylint: disable=invalid-name
        """Y component of the vector."""
        return self._scalar(self._array[1])

    @property
    def r(self) -> S:  # pylint: disable=invalid-name
        """Radial component of the vector (same as :attr:`Vector.mag`)."""
        return self.mag

    @property
    def phi(self) -> float:
        """Angular component of vector (angle between vector and x axis in rad between -pi and pi)."""
        return float(np.arctan2(self._array[1], self._array[0]))

    @property
    def length(self) -> S:
        """Length (magnitude) of vector.

        .. deprecated:: 0.6.0
            Use :attr:`Vector.mag` or :attr:`Vector.magnitude` instead.
        """
        warnings.warn('This property is deprecated. Use mag or magnitude instead', category=DeprecationWarning,
                      stacklevel=2)
        return self.mag

    @property
    def magnitude(self) -> S:
        """Magnitude of vector."""
        return self._scalar(np.hypot(self._array[0], self._array[1]))

    @property
    def mag(self) -> S:
        """Magnitude of vector (short form of :attr:`Vector.magnitude`)."""
        return self.magnitude

    @property
    def angle_about_x_axis(self) -> float:
        """Angle between vector and positive x axis (angle will be in [0, 2pi]).

        Raises:
            ValueError: Raised if self is the null vector.
        """
        if not np.any(self._array):
            raise ValueError('Cannot calculate angle of null vector')

        return float(np.clip(self.phi % (2 * np.pi), 0., 2 * np.pi))

    # ------------------------------------------------------------------------------------------------------------------
    # comparison
    # ------------------------------------------------------------------------------------------------------------------

    def close_to(self, other: t.Iterable[t.Any]) -> bool:
        """Check if `other` is close to `self` component wise (relative tolerance 1e-5, small absolute tolerance).

        Args:
            other (Vector, Iterable): other vector(like)

        Returns:
            bool

        Raises:
            VectorValueError: Raised if `other` is not vector-like.
        """
        return bool(np.allclose(self._array, self._other_array(other), atol=self._atol()))

    def __eq__(self, other: object) -> bool:
        if not hasattr(other, '__iter__'):
            return NotImplemented
        try:
            return self.close_to(t.cast(t.Iterable[t.Any], other))
        except VectorValueError:
            return False

    # ------------------------------------------------------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------------------------------------------------------

    def normalized(self: SelfT) -> SelfT:
        """Create a new vector with same :attr:`Vector.phi` but :attr:`Vector.r` = 1.

        Raises:
            ValueError: Raised if self is the null vector.
        """
        norm = np.hypot(self._array[0], self._array[1])
        if norm == 0.:
            raise ValueError('Cannot normalize the null vector.')
        return self._new(self._array / norm)

    def normalized_to(self: SelfT, length: t.Any) -> SelfT:
        """Create a new vector with same :attr:`Vector.phi` but :attr:`Vector.r` = `length`.

        Args:
            length: new length of vector

        Raises:
            ValueError: Raised if self is the null vector.
        """
        return self._new(self.normalized()._array * self._length_to_float(length))  # pylint: disable=protected-access

    def rotated(self: SelfT, theta: float, origin: t.Optional[t.Iterable[t.Any]] = None) -> SelfT:
        """Return a rotated copy of the vector around `origin` with angle `theta` in math. positive direction.

        Args:
            theta (float): rotation angle in rad
            origin (VectorLike): rotation center, defaults to the null vector

        Returns:
            Vector
        """
        cos = np.cos(float(theta))
        sin = np.sin(float(theta))
        matrix = np.array([[cos, -sin], [sin, cos]])

        if origin is None:
            return self._new(matrix @ self._array)

        origin_array = self._other_array(origin)
        return self._new(matrix @ (self._array - origin_array) + origin_array)

    def mirrored(self: SelfT, mirror_axis: t.Iterable[t.Any]) -> SelfT:
        """Return a copy of the vector mirrored on a line through the origin.

        Args:
            mirror_axis (VectorLike): direction of the mirror axis

        Returns:
            Vector

        Raises:
            ValueError: Raised if the mirror axis is the null vector.
        """
        l_x, l_y = self._other_array(mirror_axis)
        norm2 = l_x ** 2 + l_y ** 2
        if norm2 == 0.:
            raise ValueError('The mirror axis must not be the null vector.')

        mirror_matrix = np.array([[l_x * l_x - l_y * l_y, 2 * l_x * l_y], [2 * l_x * l_y, l_y * l_y - l_x * l_x]]) / norm2
        return self._new(mirror_matrix @ self._array)

    def projected(self: SelfT, other: t.Iterable[t.Any]) -> SelfT:
        """Project `other` onto self.

        https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line

        Args:
            other (VectorLike): vector to be projected

        Returns:
            Vector

        Raises:
            ValueError: Raised if self is the null vector.
        """
        other_array = self._other_array(other)
        norm2 = self._array @ self._array
        if norm2 == 0.:
            raise ValueError('Cannot project onto the null vector.')

        return self._new(self._array * ((self._array @ other_array) / norm2))

    def dot(self, other: t.Iterable[t.Any]) -> t.Any:
        """Calculate dot product with other vector.

        Args:
            other (VectorLike): other vector

        Returns:
            float (the square of the unit if the vector has a unit)
        """
        return self._product(self._array @ self._other_array(other))

    def cross(self, other: t.Iterable[t.Any]) -> t.Any:
        """Cross product in 2d.

        a x b = a.x * b.y - a.y * b.x

        a = self, b = other

        Args:
            other (VectorLike): other vector

        Returns:
            float (the square of the unit if the vector has a unit)
        """
        other_array = self._other_array(other)
        return self._product(self._array[0] * other_array[1] - self._array[1] * other_array[0])

    # ------------------------------------------------------------------------------------------------------------------
    # arithmetics
    # ------------------------------------------------------------------------------------------------------------------

    def __add__(self: SelfT, other: t.Iterable[t.Any]) -> SelfT:
        """Add operation: self + other."""
        return self._new(self._array + self._other_array(other))

    def __radd__(self: SelfT, other: t.Any) -> SelfT:
        """Add operation: other + self. ``0 + vector`` is supported, so that `sum` works."""
        if isinstance(other, FloatTypes) and other == 0:
            return self._new(self._array)
        return self._new(self._other_array(other) + self._array)

    def __sub__(self: SelfT, other: t.Iterable[t.Any]) -> SelfT:
        """Subtraction operation: self - other."""
        return self._new(self._array - self._other_array(other))

    def __rsub__(self: SelfT, other: t.Iterable[t.Any]) -> SelfT:
        """Subtraction operation: other - self."""
        return self._new(self._other_array(other) - self._array)

    def _attach_unit(self, tag: t.Any) -> t.Any:
        """Create a :class:`DimVector` from self and a unit tag, e.g. ``unit('µm')``."""
        if self._has_unit:
            raise TypeError('Cannot attach a unit to a vector which already has a unit.')
        from fibomat.linalg.vectors.dim_vector import DimVector  # pylint: disable=import-outside-toplevel

        return DimVector.from_vector(self, tag)

    def __mul__(self, other: t.Any) -> t.Any:
        """Scalar multiplication: self * other.

        If `other` is a unit (``unit('µm')``) and self has no unit, a :class:`DimVector` is returned.
        """
        if isinstance(other, FloatTypes):
            return self._new(float(other) * self._array)

        from fibomat.units.unit_tag import _UnitTag  # pylint: disable=import-outside-toplevel,protected-access

        if isinstance(other, _UnitTag):
            return self._attach_unit(other)
        return NotImplemented

    def __rmul__(self, other: t.Any) -> t.Any:
        """Scalar multiplication: other * self. ``unit('µm') * vector`` returns a :class:`DimVector`."""
        # (the unit tag returns NotImplemented for vectors, so this method is called)
        return self.__mul__(other)

    def __truediv__(self: SelfT, other: t.Any) -> SelfT:
        """Scalar division: self / other."""
        if not isinstance(other, FloatTypes):
            return NotImplemented
        if other == 0:
            raise ZeroDivisionError('Division of a vector by zero.')
        return self._new(self._array / float(other))

    def __neg__(self: SelfT) -> SelfT:
        """Negation: -vector."""
        return self._new(-self._array)
