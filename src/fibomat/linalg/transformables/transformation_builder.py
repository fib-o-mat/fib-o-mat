"""Provides the functions :func:`translate`, :func:`rotate`, :func:`scale` and :func:`mirror` to build chained
transformations.

Example::

    from fibomat.linalg import translate, rotate, mirror

    # transformations are applied from left to right
    trafo = translate([1, 2]) | rotate(np.pi / 3, 'center') | mirror([3, 4])

    transformable_obj.transformed(trafo)

The ``|`` operator never modifies its operands, so partial transformations can be reused.
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg.vectors import Vector


__all__ = ['translate', 'rotate', 'scale', 'mirror']


VectorT = t.TypeVar('VectorT', bound=Vector)  # type: ignore[type-arg]


def _to_float(value: t.Any, name: str) -> float:
    """Convert `value` to a finite float."""
    try:
        res = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{name} must be a number, got {value!r}.') from error
    if not np.isfinite(res):
        raise ValueError(f'{name} must be finite.')
    return res


class _TransformationBuilder(t.Generic[VectorT]):  # pylint: disable=too-few-public-methods
    """Base class of all transformations. Instances are immutable by convention."""

    def __init__(self) -> None:
        self._transformations: t.List[_TransformationBuilder[VectorT]] = [self]

    @property
    def transformations(self) -> t.List[_TransformationBuilder[VectorT]]:
        """The elementary transformations (a copy of the internal list).

        Access:
            get

        Returns:
            List[_TransformationBuilder]
        """
        return list(self._transformations)

    def __or__(self, other: _TransformationBuilder[VectorT]) -> _TransformationBuilder[VectorT]:
        if not isinstance(other, _TransformationBuilder):
            raise TypeError('other is not a TransformationBuilder object.')

        return _TransformationChain(self.transformations + other.transformations)


class _TransformationChain(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    """Sequence of elementary transformations created by the ``|`` operator."""

    def __init__(self, transformations: t.Iterable[_TransformationBuilder[VectorT]]) -> None:
        super().__init__()
        self._transformations = list(transformations)


class _TranslationBuilder(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self, trans_vec: t.Any):
        super().__init__()
        self.trans_vec = trans_vec


class _RotationBuilder(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self, theta: float, origin: t.Optional[t.Union[t.Any, str]] = None):
        super().__init__()
        self.theta = _to_float(theta, 'theta')
        self.origin = origin


class _ScaleBuilder(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self, fac: float, origin: t.Optional[t.Union[t.Any, str]] = None):
        super().__init__()
        self.fac = _to_float(fac, 'fac')
        if self.fac == 0.:
            raise ValueError('fac must not be zero.')
        self.origin = origin


class _MirrorBuilder(_TransformationBuilder[VectorT]):  # pylint: disable=too-few-public-methods
    def __init__(self, mirror_plane: t.Any):
        super().__init__()
        self.mirror_plane = mirror_plane


def translate(trans_vec: t.Any) -> _TransformationBuilder[t.Any]:
    """Translate an object by `trans_vec`.

    Args:
        trans_vec (VectorLike, DimVectorLike): translation vector

    Returns:
        _TranslationBuilder
    """
    return _TranslationBuilder(trans_vec)


def rotate(theta: float, origin: t.Optional[t.Union[t.Any, str]] = None) -> _TransformationBuilder[t.Any]:
    """Rotate an object around `origin` with angle `theta` in math. positive direction (counterclockwise).

    Args:
        theta (float): rotation angle in rad
        origin (VectorLike, DimVectorLike, str, optional):
            origin of rotation. If not set, (0, 0) is used as origin. If origin == 'center', the
            :attr:`Transformable.center` of the object will be used. If origin == 'pivot', its
            :attr:`Transformable.pivot` will be used.

    Returns:
        _RotationBuilder

    Raises:
        TypeError: Raised if `theta` is no number.
        ValueError: Raised if `theta` is not finite.
    """
    return _RotationBuilder(theta, origin)


def scale(fac: float, origin: t.Optional[t.Union[t.Any, str]] = None) -> _TransformationBuilder[t.Any]:
    """Scale an object homogeneously about `origin` with factor `fac`.

    Args:
        fac (float): scale factor
        origin (VectorLike, DimVectorLike, str, optional):
            origin of the scaling. If not set, (0, 0) is used as origin. If origin == 'center', the
            :attr:`Transformable.center` of the object will be used. If origin == 'pivot', its
            :attr:`Transformable.pivot` will be used.

    Returns:
        _ScaleBuilder

    Raises:
        TypeError: Raised if `fac` is no number.
        ValueError: Raised if `fac` is not finite or zero.
    """
    return _ScaleBuilder(fac, origin)


def mirror(mirror_plane: t.Any) -> _TransformationBuilder[t.Any]:
    """Mirror an object about the line through the origin which is spanned by `mirror_plane`.

    Args:
        mirror_plane (VectorLike, DimVectorLike): direction of the mirror axis

    Returns:
        _MirrorBuilder
    """
    return _MirrorBuilder(mirror_plane)
