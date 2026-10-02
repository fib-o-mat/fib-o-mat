"""
Provides the :class:`TransformableBase` class.
"""

from __future__ import annotations
from typing import Optional, Union, TypeVar, Protocol, Type, Generic, Callable
import abc

import numpy as np

from fibomat.linalg.vectors import Vector, VectorLike, DimVector
from fibomat.describable import Describable
from fibomat.linalg.transformables.transformation_builder import (
    _TransformationBuilder,
    _TranslationBuilder,
    _RotationBuilder,
    _ScaleBuilder,
    _MirrorBuilder,
)
from fibomat.linalg.boundingboxes import BoundingBox, DimBoundingBox
from fibomat.units import LengthQuantity


VectorT = TypeVar("VectorT", Vector, DimVector)
ScalarT = TypeVar("ScalarT", int, float, np.integer, LengthQuantity)
BBoxT = TypeVar("BBoxT", BoundingBox, DimBoundingBox)
SelfT = TypeVar("SelfT", bound="TransformableBase")


class TransformableBase(Describable, Generic[VectorT, ScalarT, BBoxT], abc.ABC):
    """
    :class:`Transformable` is a base class providing the translate, rotate and uniform scale
    transformations.

    In order to use this mixin in a child class, the following methods and properties must be implemented:
        * :attr:`Transformable.center`
        * :meth:`Transformable.translate`
        * :meth:`Transformable.simple_rotate`
        * :meth:`Transformable.simple_scale`

    """

    _VectorClass: Type[VectorT]

    def __init__(self: SelfT, description: Optional[str] = None):
        """
        Args:
            pivot (VectorLike, optional): if set, the :attr:`Transformable.pivot` is set to `pivot`. If not set,
                                           :attr:`Transformable.center` is used as default.

            description (str, optional): optional description
        """
        super().__init__(description)

        self._pivot: Optional[Callable[[SelfT], VectorT]] = None

    @property
    @abc.abstractmethod
    def center(self) -> VectorT:
        """center of the (geometric) object

        Access:
            get

        Returns:
            Any
        """
        raise NotImplementedError

    @property
    def pivot(self) -> VectorT:
        """Origin of the (geometric) object. If origin is set to `None`, :attr:`Transformable.center` will be
        returned.

        Pivot must be set to a callable function without parameters. ::

            transformable_obj = ...
            transformable_obj.pivot = lambda: return Vector(1, 2)
            print(transformable_obj.pivot)  # will print Vector(1, 2)

        Access:
            get/set

        Returns:
            Vector
        """
        # if self._pivot is not None:
        #     return Vector(self._pivot(self))
        # return self.center
        if self._pivot is not None:
            return self._VectorClass(self._pivot(self))
        return self.center

    @pivot.setter
    def pivot(self: SelfT, value: Callable[[SelfT], VectorLike]):
        self._pivot = value

    @property
    @abc.abstractmethod
    def bounding_box(self) -> BBoxT:
        """
        :class:`~fibomat.boundingbox.BoundingBox`: bounding box of transformable

        Access:
            get
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _impl_translate(self, trans_vec: VectorT) -> None:
        """Translate the object by `trans_vec`.

        Args:
            trans_vec (VectorLike): translation linalg

        Returns:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _impl_rotate(self, theta: float) -> None:
        """Rotate the object by theta.

        Args:
            theta: rotation angle

        Returns:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _impl_scale(self, fac: float) -> None:
        """Scale the object by fac.

        Args:
            fac: scale factor

        Returns:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _impl_mirror(self, mirror_axis: VectorT) -> None:
        """Mirror the object at mirror_axis.

        Args:
            mirror_axis (VectorLike): mirror_plane

        Returns:
            None
        """
        raise NotImplementedError

    def _apply_shifted_trafo(
        self: SelfT,
        trafo: Callable[[float], None],
        arg: float,
        origin: Optional[Union[VectorT, str]] = None,
    ) -> SelfT:
        if origin:
            if isinstance(origin, str):
                if origin == "center":
                    origin = self.center
                elif origin == "pivot":
                    origin = self.pivot
                else:
                    raise ValueError(f"Unknown origin `{origin}`")
            else:
                origin = self._VectorClass(origin)

            self._impl_translate(-origin)  # pylint: disable=invalid-unary-operand-type
            trafo(float(arg))
            self._impl_translate(origin)
        else:
            trafo(float(arg))

        return self

    def translated_to(self: SelfT, pos: VectorT) -> SelfT:
        """Return a translated copy of the object so that self.pivot == pos

        Args:
            pos: new position of object

        Returns:
            TransformableBase
        """
        return self.translated(self._VectorClass(pos) - self.pivot)

    def translated(self: SelfT, trans_vec: VectorT) -> SelfT:
        """Return a translated copy of the object by `trans_vec`.

        Args:
            trans_vec (VectorLike): translation vector

        Returns:
            TransformableBase
        """
        # pylint: disable=protected-access

        # if not np.isclose(float(self._VectorClass(trans_vec).mag), 0.):
        clone: SelfT = self.clone()
        clone._impl_translate(trans_vec)
        return clone
        # return self

    def rotated(
        self: SelfT, theta: float, origin: Optional[Union[VectorT, str]] = None
    ) -> SelfT:
        """Return a rotated copy around `origin` with angle `theta` in math. positive direction (counterclockwise).

        Args:
            theta (float): rotation angle in rad
            origin (Optional[Union[linalg.VectorLike, str]], optional):
                origin of rotation. If not set, (0,0) is used as
                origin. If origin == 'center', the
                :attr:`Transformable.center` of the object will
                be used. The same applies for the case that
                origin == 'origin' with the
                :attr:`Transformable.origin` property. Default
                to None.

        Returns:
            TransformableBase
        """
        # pylint: disable=protected-access
        if not np.isclose(float(theta), 0.0):
            clone: SelfT = self.clone()
            clone._apply_shifted_trafo(clone._impl_rotate, theta, origin)
            return clone
        return self

    def scaled(
        self: SelfT, fac: float, origin: Optional[Union[VectorT, str]] = None
    ) -> SelfT:
        """Return a scale object homogeneously about `origin` with factor `s`.

        Args:
            fac (float): rotation angle in rad
            origin (Optional[Union[linalg.VectorLike, str]], optional):
                origin of rotation. If not set, (0,0) is used as
                origin. If origin == 'center', the
                :attr:`Transformable.center` of the object will
                be used. The same applies for the case that
                origin == 'origin' with the
                :attr:`Transformable.origin` property. Default
                to None.

        Returns:
            TransformableBase
        """
        if not np.isclose(float(fac), 1.0):
            clone: SelfT = self.clone()
            clone._apply_shifted_trafo(clone._impl_scale, fac, origin)  # pylint: disable=protected-access
            return clone
        return self

    def mirrored(self: SelfT, mirror_plane: VectorT) -> SelfT:
        """Return a mirrored object mirrored about `mirror_plane`.

        Args:
            mirror_plane (VectorLike): mirror plane to be used.

        Returns:
            TransformableBase
        """
        clone: T = self.clone()
        clone._impl_mirror(mirror_plane)  # pylint: disable=protected-access
        return clone

    def transformed(
        self: SelfT, transformations: _TransformationBuilder[VectorT]
    ) -> SelfT:
        """Return a transformed object. the transformation can be build by the following functions:
            - :func:`~fibomat.linalg.transformation_builder.translate`
            - :func:`~fibomat.linalg.transformation_builder.rotate`
            - :func:`~fibomat.linalg.transformation_builder.scale`
            - :func:`~fibomat.linalg.transformation_builder.mirror`

        E.g. ::

            transformable_obj.transform(translate([1, 2]) | rotate(np.pi/3) | mirror([3,4])

        Args:
            transformations (_TransformationBuilder): transformation

        Returns:
            TransformableBase
        """
        # pylint: disable=protected-access
        clone: SelfT = self.clone()
        for trafo in transformations.transformations:
            if isinstance(trafo, _TranslationBuilder):
                clone._impl_translate(trafo.trans_vec)
            elif isinstance(trafo, _RotationBuilder):
                clone._apply_shifted_trafo(
                    clone._impl_rotate, trafo.theta, trafo.origin
                )
            elif isinstance(trafo, _ScaleBuilder):
                clone._apply_shifted_trafo(clone._impl_scale, trafo.fac, trafo.origin)
            elif isinstance(trafo, _MirrorBuilder):
                clone._impl_mirror(trafo.mirror_plane)
            else:
                raise TypeError(f"{trafo.__class__} is an unknown transforamtion.")

        return clone
