"""Provides the :class:`ArcSplineCompatible` class.

Example::

    class MyShape(Shape, ArcSplineCompatible):
        def to_arc_spline(self) -> ArcSpline:
            ...

    isinstance(MyShape(), ArcSplineCompatible)  # True
    isinstance(Circle(radius=1), ArcSplineCompatible)  # True

Shapes which derive from :class:`ArcSplineCompatible` must implement :meth:`ArcSplineCompatible.to_arc_spline`
(otherwise they cannot be instantiated). Other objects are also treated as arc spline compatible if their class
defines a method `to_arc_spline` (duck typing).
"""
from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:  # pragma: no cover
    from fibomat.shapes.arc_spline import ArcSpline


__all__ = ['ArcSplineCompatible']


class ArcSplineCompatible(abc.ABC):  # pylint: disable=too-few-public-methods
    """Mixin marking shapes which can be converted to an :class:`~fibomat.shapes.arc_spline.ArcSpline`.

    Algorithms which work on arc splines (offsetting, rasterization, ...) accept all shapes of this type.
    """

    @abc.abstractmethod
    def to_arc_spline(self) -> ArcSpline:
        """Transform the shape to an :class:`~fibomat.shapes.arc_spline.ArcSpline`.

        Returns:
            ArcSpline
        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass: type) -> t.Any:
        # structural check: every class defining a method `to_arc_spline` is compatible
        if cls is ArcSplineCompatible:
            for base in subclass.__mro__:
                if 'to_arc_spline' in base.__dict__:
                    return callable(base.__dict__['to_arc_spline'])
        return NotImplemented
