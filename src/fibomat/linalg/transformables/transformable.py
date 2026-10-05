"""Provides the :class:`Transformable` class.

:class:`Transformable` is a generic base class providing the translate, rotate, scale and mirror transformations for
objects without units. :class:`~fibomat.linalg.transformables.dim_transformable.DimTransformable` derives from it for
objects with units (lengths).

Example::

    class MyShape(Transformable):
        @property
        def center(self) -> Vector: ...
        @property
        def bounding_box(self) -> BoundingBox: ...
        def _impl_translate(self, trans_vec: Vector) -> None: ...
        def _impl_rotate(self, theta: float) -> None: ...
        def _impl_scale(self, fac: float) -> None: ...
        def _impl_mirror(self, mirror_axis: Vector) -> None: ...

    shape = MyShape()
    shape.rotated(np.pi / 2, origin='center')
    shape.transformed(translate([1, 2]) | rotate(np.pi / 3))
"""
from __future__ import annotations

import abc
import typing as t

import numpy as np

from fibomat.describable import Describable
from fibomat.linalg.boundingboxes import BoundingBox
from fibomat.linalg.transformables.transformation_builder import (
    _TransformationBuilder,
    _TranslationBuilder,
    _RotationBuilder,
    _ScaleBuilder,
    _MirrorBuilder,
    _to_float,
)
from fibomat.linalg.vectors import Vector


__all__ = ['Transformable']


VectorT = t.TypeVar('VectorT', bound=Vector)  # type: ignore[type-arg]
BBoxT = t.TypeVar('BBoxT', bound=BoundingBox)  # type: ignore[type-arg]
SelfT = t.TypeVar('SelfT', bound='Transformable')  # type: ignore[type-arg]


class Transformable(Describable, t.Generic[VectorT, BBoxT], abc.ABC):
    """Base class providing the translate, rotate, scale and mirror transformations.

    In order to use this class as base of a child class, the following methods and properties must be implemented:

        * :attr:`Transformable.center`
        * :attr:`Transformable.bounding_box`
        * :meth:`Transformable._impl_translate`
        * :meth:`Transformable._impl_rotate`
        * :meth:`Transformable._impl_scale`
        * :meth:`Transformable._impl_mirror`

    All public transformation methods work on a clone and return it. Arguments are converted to
    :attr:`Transformable._VectorClass` before the ``_impl_*`` methods are called.
    """

    _VectorClass: t.Type[VectorT] = Vector  # type: ignore[assignment]
    """Vector type used by the object (:class:`Vector` or :class:`DimVector`)."""

    def __init__(self, description: t.Optional[str] = None):
        """
        Args:
            description (str, optional): optional description
        """
        super().__init__(description)

        self._pivot: t.Optional[t.Callable[[t.Any], t.Any]] = None

    @property
    @abc.abstractmethod
    def center(self) -> VectorT:
        """Center of the (geometric) object.

        Access:
            get

        Returns:
            Vector
        """
        raise NotImplementedError

    @property
    def pivot(self) -> VectorT:
        """Origin of the (geometric) object. If no pivot is set, :attr:`Transformable.center` is returned.

        The pivot must be set to a callable which takes the object and returns a vector(like). ::

            transformable_obj = ...
            transformable_obj.pivot = lambda obj: Vector(1, 2)
            print(transformable_obj.pivot)  # will print Vector(1, 2)

        Set the pivot to `None` to reset it to the center.

        Access:
            get/set

        Returns:
            Vector
        """
        if self._pivot is not None:
            return self._VectorClass(self._pivot(self))
        return self.center

    @pivot.setter
    def pivot(self: SelfT, value: t.Optional[t.Callable[[SelfT], t.Any]]) -> None:
        if value is not None and not callable(value):
            raise TypeError('pivot must be a callable (or None).')
        self._pivot = value

    @property
    @abc.abstractmethod
    def bounding_box(self) -> BBoxT:
        """Bounding box of the object.

        Access:
            get

        Returns:
            BoundingBox
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _impl_translate(self, trans_vec: VectorT) -> None:
        """Translate the object in-place by `trans_vec`.

        Args:
            trans_vec (Vector): translation vector
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _impl_rotate(self, theta: float) -> None:
        """Rotate the object in-place around the origin by `theta`.

        Args:
            theta (float): rotation angle in rad
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _impl_scale(self, fac: float) -> None:
        """Scale the object in-place about the origin by `fac`.

        Args:
            fac (float): scale factor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _impl_mirror(self, mirror_axis: VectorT) -> None:
        """Mirror the object in-place at the line through the origin spanned by `mirror_axis`.

        Args:
            mirror_axis (Vector): direction of the mirror axis
        """
        raise NotImplementedError

    def _resolve_origin(self, origin: t.Union[t.Any, str]) -> VectorT:
        """Convert the origin of a rotation or scaling to a vector."""
        if isinstance(origin, str):
            if origin == 'center':
                return self.center
            if origin == 'pivot':
                return self.pivot
            raise ValueError(f'Unknown origin `{origin}`')
        return self._VectorClass(origin)

    def _apply_shifted_trafo(
        self: SelfT,
        trafo: t.Callable[[float], None],
        arg: float,
        origin: t.Optional[t.Union[t.Any, str]] = None,
    ) -> SelfT:
        """Apply the in-place transformation `trafo` about `origin` (the null vector if `None`)."""
        if origin is None:
            trafo(float(arg))
        else:
            origin_vec = self._resolve_origin(origin)

            self._impl_translate(-origin_vec)
            trafo(float(arg))
            self._impl_translate(origin_vec)

        return self

    def translated_to(self: SelfT, pos: t.Any) -> SelfT:
        """Return a translated copy of the object so that ``self.pivot == pos``.

        Args:
            pos (VectorLike): new position of object

        Returns:
            Transformable
        """
        return self.translated(self._VectorClass(pos) - self.pivot)

    def translated(self: SelfT, trans_vec: t.Any) -> SelfT:
        """Return a translated copy of the object by `trans_vec`.

        Args:
            trans_vec (VectorLike): translation vector

        Returns:
            Transformable
        """
        # pylint: disable=protected-access
        trans_vec = self._VectorClass(trans_vec)
        clone: SelfT = self.clone()
        clone._impl_translate(trans_vec)
        return clone

    def rotated(self: SelfT, theta: float, origin: t.Optional[t.Union[t.Any, str]] = None) -> SelfT:
        """Return a rotated copy around `origin` with angle `theta` in math. positive direction (counterclockwise).

        Args:
            theta (float): rotation angle in rad
            origin (VectorLike, str, optional):
                origin of rotation. If not set, (0, 0) is used as origin. If origin == 'center', the
                :attr:`Transformable.center` of the object will be used. If origin == 'pivot', its
                :attr:`Transformable.pivot` will be used.

        Returns:
            Transformable (always a new object)

        Raises:
            ValueError: Raised if `theta` is not finite or `origin` is an unknown string.
        """
        theta = _to_float(theta, 'theta')
        clone: SelfT = self.clone()
        # pylint: disable=protected-access
        return clone._apply_shifted_trafo(clone._impl_rotate, theta, origin)

    def scaled(self: SelfT, fac: float, origin: t.Optional[t.Union[t.Any, str]] = None) -> SelfT:
        """Return a copy of the object scaled homogeneously about `origin` with factor `fac`.

        Args:
            fac (float): scale factor
            origin (VectorLike, str, optional):
                origin of the scaling. If not set, (0, 0) is used as origin. If origin == 'center', the
                :attr:`Transformable.center` of the object will be used. If origin == 'pivot', its
                :attr:`Transformable.pivot` will be used.

        Returns:
            Transformable (always a new object)

        Raises:
            ValueError: Raised if `fac` is zero or not finite or `origin` is an unknown string.
        """
        fac = _to_float(fac, 'fac')
        if fac == 0.:
            raise ValueError('fac must not be zero.')
        clone: SelfT = self.clone()
        # pylint: disable=protected-access
        return clone._apply_shifted_trafo(clone._impl_scale, fac, origin)

    def mirrored(self: SelfT, mirror_plane: t.Any) -> SelfT:
        """Return a copy of the object mirrored at the line through the origin spanned by `mirror_plane`.

        Args:
            mirror_plane (VectorLike): direction of the mirror axis

        Returns:
            Transformable

        Raises:
            ValueError: Raised if `mirror_plane` is the null vector.
        """
        mirror_axis = self._VectorClass(mirror_plane)
        if not np.any(np.asarray(mirror_axis)):
            raise ValueError('The mirror axis must not be the null vector.')

        clone: SelfT = self.clone()
        clone._impl_mirror(mirror_axis)  # pylint: disable=protected-access
        return clone

    def transformed(self: SelfT, transformations: _TransformationBuilder[t.Any]) -> SelfT:
        """Return a transformed copy of the object. The transformation can be built by the following functions:
            - :func:`~fibomat.linalg.transformables.transformation_builder.translate`
            - :func:`~fibomat.linalg.transformables.transformation_builder.rotate`
            - :func:`~fibomat.linalg.transformables.transformation_builder.scale`
            - :func:`~fibomat.linalg.transformables.transformation_builder.mirror`

        E.g. ::

            transformable_obj.transformed(translate([1, 2]) | rotate(np.pi/3) | mirror([3, 4]))

        Args:
            transformations (_TransformationBuilder): transformation

        Returns:
            Transformable

        Raises:
            TypeError: Raised if `transformations` is no transformation or contains an unknown one.
        """
        # pylint: disable=protected-access
        if not isinstance(transformations, _TransformationBuilder):
            raise TypeError('transformations must be built with translate, rotate, scale and mirror.')

        clone: SelfT = self.clone()
        for trafo in transformations.transformations:
            if isinstance(trafo, _TranslationBuilder):
                clone._impl_translate(clone._VectorClass(trafo.trans_vec))
            elif isinstance(trafo, _RotationBuilder):
                clone._apply_shifted_trafo(clone._impl_rotate, trafo.theta, trafo.origin)
            elif isinstance(trafo, _ScaleBuilder):
                clone._apply_shifted_trafo(clone._impl_scale, trafo.fac, trafo.origin)
            elif isinstance(trafo, _MirrorBuilder):
                mirror_axis = clone._VectorClass(trafo.mirror_plane)
                if not np.any(np.asarray(mirror_axis)):
                    raise ValueError('The mirror axis must not be the null vector.')
                clone._impl_mirror(mirror_axis)
            else:
                raise TypeError(f'{trafo.__class__} is an unknown transformation.')

        return clone
