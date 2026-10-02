"""Provides the `class:`Shape` class."""

from __future__ import annotations

import abc
from typing import Any, Optional

from fibomat.linalg import Transformable
from fibomat.units import U_


class Shape(Transformable, abc.ABC):
    """Baseclass for all shapes in the library which defines the basic interface."""

    def __init__(self, description: Optional[str] = None):
        """
        Args:
            description (str, optional): description
        """
        super().__init__(description=description)

    @abc.abstractmethod
    def __repr__(self) -> str:
        """str: repr of class"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_closed(self) -> bool:
        """
        bool: True if shape is closes. This property should not be defined for 0-dim shapes

        Access:
            get
        """
        raise NotImplementedError

    @property
    def area(self) -> float:
        # only implemented for closed shapes
        raise NotImplementedError

    @property
    def boundary_length(self) -> float:
        raise NotADirectoryError

    def __mul__(self, other: Any):
        if isinstance(other, U_):
            from .dim_shape import DimShape

            return DimShape(self, other)
        else:
            raise NotImplementedError
