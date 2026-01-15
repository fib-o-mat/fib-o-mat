from __future__ import annotations
from typing import (
    TypeVar,
    overload,
    Union,
    Sequence,
    Iterable,
    Optional,
    Any,
    cast,
    List,
)
import warnings
import abc

import numpy as np
import pint

FloatTypes = int, float, np.integer
FloatTypesUnion = Union[int, float, np.integer]
T = TypeVar("T", int, float, np.integer, pint.Quantity)  # LengthQuantity
SelfT = TypeVar("SelfT", bound="VectorBase")


class VectorValueError(ValueError):
    """Exception of this type is raised if any non supported value type is passed to Vector.__init__."""


class VectorBase(Sequence[T], abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def _validate_type(val: Any) -> bool:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _build_np_array(val: Sequence[T]) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _make_zero_vector() -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _from_iterable(
        cls, x: Iterable[Any]
    ) -> List[T]:  # pylint: disable=invalid-name
        # np.empty(shape=(2,), dtype=float)

        if not hasattr(x, "__iter__"):
            raise VectorValueError("x must be int, float, or iterable")

        i = iter(x)
        try:
            x_val: T = next(i)
            y_val: T = next(i)
        except StopIteration:
            raise VectorValueError(
                "if x is an iterable, it must contain exactly 2 items and not less"
            )

        if not cls._validate_type(x_val):
            raise VectorValueError("x not compatible with vector values.")
        if not cls._validate_type(y_val):
            raise VectorValueError("y not compatible with vector values.")
        try:
            next(i)
            raise VectorValueError(
                "if x is an iterable, it must contain exactly 2 items and not more"
            )
        except StopIteration:
            pass

        return [x_val, y_val]

    @classmethod
    def _from_polar(cls, r: T, phi: float) -> List[T]:  # pylint: disable=invalid-name
        if not cls._validate_type(r):
            raise VectorValueError("r not compatible with vector values.")
        if not isinstance(phi, FloatTypes):
            raise VectorValueError("phi not compatible with vector values.")

        return [r * np.cos(phi), r * np.sin(phi)]

    def __init__(
        self,
        x: Optional[Union[T, VectorBase, Iterable[T]]] = None,
        y: Optional[T] = None,
        r: Optional[T] = None,
        phi: Optional[T] = None,
    ):
        vector: List[T]

        if x is None and y is None and r is None and phi is None:
            vector = self._make_zero_vector()
        elif x is not None:
            if self._validate_type(x):
                if (r is not None) or (phi is not None):
                    raise VectorValueError("Define (x, y) or (r, theta) but not both.")
                if y is not None and self._validate_type(y):
                    x = cast(T, x)
                    vector = self._from_iterable([x, y])
                else:
                    raise VectorValueError(
                        f"if x is int or float, y must be defined and must have the same type. Got x = {x} and y = {y}."
                    )
            else:  # x is some kind of list, tuple, Vector, np.ndarray, ...
                if (y is not None) or (r is not None) or (phi is not None):
                    raise VectorValueError(
                        "when x is an object, you must not specify y, r, or theta"
                    )
                x = cast(Iterable[T], x)
                vector = self._from_iterable(x)
        elif (
            r is not None and phi is not None
        ):  # polar case, r and phi should be defined.
            if y is not None:
                raise VectorValueError("Define (x, y) or (r, theta) but not both.")

            vector = self._from_polar(r, phi)
        else:
            raise VectorValueError("Incompatible combination of arguments.")

        self._vector = self._build_np_array(vector)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self._vector[0]}, y={self._vector[1]})"

    @overload
    def __getitem__(self, index: int) -> T:
        ...  # pragma: no cover

    @overload
    def __getitem__(self, slice_: slice) -> Sequence[T]:
        ...  # pragma: no cover

    def __getitem__(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        if isinstance(index, int):
            return self._vector[index]
        elif isinstance(index, slice):
            sliced = self._vector[index]
            # needed so that if [T = QuantityType] [Q_(..), Q_(..)] is returned and not Q_([.., ..]).
            return [val for val in sliced]
        else:
            raise TypeError("index must be int or slice.")

    def __len__(self) -> int:
        return 2

    def __array__(self, dtype=None) -> np.ndarray:
        """Helper to allow conversion to numpy array.

        ::

            np_array = np.asarray(Vector(1, 2))

        Returns:
            numpy.ndarray
        """
        # to suppress UnitStrippedWarning in pint module
        with warnings.catch_warnings(record=True):
            return np.array(self._vector, dtype=(dtype or self._vector.dtype))

    @property
    def x(self) -> T:  # pylint: disable=invalid-name
        """X component the vector

        Access:
            get

        Returns:
            T
        """
        return self._vector[0]

    @property
    def y(self) -> T:  # pylint: disable=invalid-name
        """X component the vector

        Access:
            get

        Returns:
            T
        """
        return self._vector[1]

    @property
    def r(self) -> T:  # pylint: disable=invalid-name
        """Radial component the vector.

        Access:
            get

        Returns:
            float
        """
        return self.mag

    @property
    def phi(self) -> float:
        """Angular component of vector (angle between vector and x axis in radiant between -pi and pi).

        Access:
            get

        Returns:
            float
        """
        return float(np.arctan2(self._vector[1], self._vector[0]))

    @property
    def length(self) -> T:
        """Length (magnitude) of vector.

        Access:
            get

        Returns:
            T
        """
        warnings.warn(
            "This method is deprecated. Use mag or magnitude instead",
            category=DeprecationWarning,
        )
        return np.sqrt((self._vector * self._vector).sum())

    @property
    def magnitude(self) -> T:
        """Magnitude of vector.

        Access:
            get

        Returns:
            T
        """
        return np.sqrt((self._vector * self._vector).sum())

    @property
    def mag(self) -> T:
        """Magnitude of vector (short form of :attr:`VectorBase.magnitude`).

        Access:
            get

        Returns:
            T
        """
        return np.sqrt((self._vector * self._vector).sum())

    @property
    def angle_about_x_axis(self) -> float:
        """Angle between vector and positive x axis (angle will be in [0, 2pi]).

        Access:
            get

        Returns:
            float

        Raises:
            RuntimeError: Raised if self is null vector.
        """
        if np.allclose(self, 0.0):
            raise ValueError("Cannot calculate angle of null vector")

        angle = self.phi
        while angle < 0.0:
            angle += 2 * np.pi

        assert np.isclose(np.clip(angle, 0.0, 2 * np.pi), angle)
        return np.clip(angle, 0.0, 2 * np.pi)

    def close_to(self, other: Iterable[T]) -> bool:
        """Checks if `other` is close to `self` component wise.

        Args:
            other (Vector, Iterable[float]): other vector(like)

        Returns:
            bool
        """
        return np.allclose(self._vector, self.__class__(other))

    def __eq__(self, other: Iterable[T]) -> bool:
        return self.close_to(other)

    def normalized(self: SelfT) -> SelfT:
        """Create a new vector with same :attr:`Vector.phi` but :attr:`Vector.r` = 1.

        Returns:
            SelfT
        """
        return self.__class__(self._vector / np.linalg.norm(self._vector))

    def normalized_to(self: SelfT, length: T) -> SelfT:
        """Create a new vector :attr:`Vector.phi` but :attr:`Vector.r` = length.

        Args:
            length (float): new length of vector

        Returns:
            Vector
        """
        return self.normalized() * length

    def rotated(
        self: SelfT, theta: float, origin: Optional[Iterable[T]] = None
    ) -> SelfT:
        """Return a rotated copy the vector around `center` with angle `theta` in math. positive direction.

        Args:
            theta (float): rotation angle in rad
            origin (VectorLike): rotation center, default to [0., 0.]

        Returns:
            Vector
        """
        theta = float(theta)
        cos = np.cos(theta)
        sin = np.sin(theta)

        # m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        rotated_vec = self._vector.copy()

        if origin is not None:
            origin_array = self.__class__(origin)._vector

            # rotated_vec -= origin_array
            rotated_vec[0] = (self._vector[0] - origin_array[0]) * cos - (
                self._vector[1] - origin_array[1]
            ) * sin
            rotated_vec[1] = (self._vector[1] - origin_array[1]) * cos + (
                self._vector[0] - origin_array[0]
            ) * sin
            rotated_vec += origin_array
        else:
            rotated_vec[0] = self._vector[0] * cos - self._vector[1] * sin
            rotated_vec[1] = self._vector[1] * cos + self._vector[0] * sin

        return self.__class__(rotated_vec)

    def mirrored(self: SelfT, mirror_axis: Iterable[T]) -> SelfT:
        """Return a mirrored version of the vector.

        Args:
            mirror_axis (VectorLike): mirror axis

        Returns:
            Vector
        """
        # pylint: disable=invalid-name
        mirror_axis = self.__class__(mirror_axis)

        lx, ly = np.asarray(mirror_axis)
        mirror_matrix = np.array(
            [[lx * lx - ly * ly, 2 * lx * ly], [2 * lx * ly, ly * ly - lx * lx]]
        ) / (lx**2 + ly**2)

        return self.__class__(mirror_matrix @ self._vector)

    def projected(self, other: Iterable[T]) -> SelfT:
        """Project other onto self.
        https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line

        Args:
            other:

        Returns:
            Vector
        """
        other = self.__class__(other)

        return self * (self.dot(other) / self.dot(self))

    def dot(self: SelfT, other: Iterable[T]) -> T:
        """Calculate dot product with other vector.

        Args:
            other(VectorLike): other vector

        Returns:
            float
        """
        # pylint: disable=protected-access
        return np.dot(self._vector, self.__class__(other)._vector)

    def cross(self: SelfT, other: Iterable[T]) -> T:
        """Cross product in 2d.

        a x b = a.x * b.y - a.y * b.x

        a = self, b = other

        Args:
            other(VectorLike): other vector

        Returns:
            T
        """
        other = self.__class__(other)

        return self.x * other.y - self.y * other.x

    def __add__(self: SelfT, other: Iterable[T]) -> SelfT:
        """Add operation: self + other.

        Args:
            other (VectorLike): summand

        Returns:
            Vector
        """
        return self.__class__(self._vector + self.__class__(other)._vector)

    def __radd__(self: SelfT, other: Iterable[T]) -> SelfT:
        """Add operation: self + other.

        Args:
            other (VectorLike): summand

        Returns:
            SelfT
        """
        return self.__class__(self._vector + self.__class__(other)._vector)

    def __sub__(self: SelfT, other: Iterable[T]) -> SelfT:
        """Subtraction operation: self - other.

        Args:
            other (VectorLike): subtrahend

        Returns:
            SelfT
        """
        return self.__class__(self._vector - self.__class__(other)._vector)

    def __rsub__(self: SelfT, other: Iterable[T]) -> SelfT:
        """Subtraction operation: other - self.

        Args:
            other (VectorLike): minuend

        Returns:
            SelfT
        """
        return self.__class__(self.__class__(other)._vector - self._vector)

    def __mul__(self: SelfT, other: FloatTypesUnion) -> SelfT:
        """Scalar multiplication: self * other.

        Args:
            other (float): multiplicand

        Returns:
            Vector
        """
        return self.__class__(float(other) * self._vector)

    def __rmul__(self: SelfT, other: FloatTypesUnion) -> SelfT:
        """Scalar multiplication: other * self.

        Args:
            other (float): multiplicand

        Returns:
            SelfT
        """
        return self.__mul__(other)

    def __truediv__(self: SelfT, other: float) -> SelfT:
        """Scalar division: self / other.

        Args:
            other (float): divisor

        Returns:
            SelfT
        """
        return self.__class__(self._vector / float(other))

    def __neg__(self: SelfT) -> SelfT:
        """Negation: -vector.

        Returns:
            SelfT
        """
        return self.__class__(-self._vector)
