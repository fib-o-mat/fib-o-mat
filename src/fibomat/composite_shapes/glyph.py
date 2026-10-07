"""Provides the :class:`Glyph` and :class:`DimGlyph` classes.

A glyph is the shape of one character of a :class:`~fibomat.composite_shapes.text.Text`. It consists of several shapes,
e.g. the polylines of a letter or the polygons (with holes) of a stroked letter.

Example::

    from fibomat.composite_shapes import Text

    text = Text('Hello', stroke_width=0.1)
    for glyph in text:
        glyph.character  # 'H', 'e', ...
        for shape in glyph:  # Polygon or HollowArcSpline of the stroked glyph
            ...
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.composite_shapes._dim_composite import DimComposite
from fibomat.linalg import BoundingBox, Transformable, Vector, VectorLike
from fibomat.shapes.dim_shape import DimShape, LengthUnitLike
from fibomat.shapes.shape import Shape
from fibomat.units import DimFloat, ureg


__all__ = ['Glyph', 'DimGlyph']


class Glyph(Transformable[Vector, BoundingBox]):
    """The shapes of one character.

    Glyphs are transformable. The :attr:`Glyph.origin` (the start of the baseline of the character) and the
    :attr:`Glyph.advance_width` are transformed together with the shapes. A glyph without shapes (e.g. a space) is
    empty (``bool(glyph)`` is False); it has no bounding box.
    """

    def __init__(
        self,
        shapes: t.Iterable[Shape],
        character: str,
        origin: t.Optional[VectorLike] = None,
        advance_width: float = 0.,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            shapes (Iterable[Shape]): shapes of the glyph (not copied)
            character (str): the character
            origin (VectorLike, optional): start of the baseline of the character, default to (0, 0)
            advance_width (float): distance to the origin of the next character
            description (str, optional): description

        Raises:
            TypeError: Raised if the shapes are no shapes.
            ValueError: Raised if the advance width or the origin are not finite.
        """
        super().__init__(description)

        shapes = tuple(shapes)
        if not all(isinstance(shape, Shape) for shape in shapes):
            raise TypeError('shapes must be Shapes.')

        self._shapes = shapes
        self._character = str(character)
        self._origin = Vector(origin) if origin is not None else Vector()
        self._advance_width = float(advance_width)

        if not np.isfinite(self._advance_width) or not np.all(np.isfinite(np.asarray(self._origin))):
            raise ValueError('origin and advance_width must be finite.')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._character!r}, n_shapes={len(self._shapes)})'

    @property
    def shapes(self) -> t.Tuple[Shape, ...]:
        """The shapes of the glyph.

        Access:
            get
        """
        return self._shapes

    def __iter__(self) -> t.Iterator[Shape]:
        return iter(self._shapes)

    def __len__(self) -> int:
        return len(self._shapes)

    @property
    def character(self) -> str:
        """The character.

        Access:
            get
        """
        return self._character

    @property
    def origin(self) -> Vector:
        """Start of the baseline of the character.

        Access:
            get
        """
        return self._origin

    @property
    def advance_width(self) -> float:
        """Distance to the origin of the next character (along the baseline).

        Access:
            get
        """
        return self._advance_width

    @property
    def bounding_box(self) -> BoundingBox:
        """Bounding box of all shapes of the glyph.

        Access:
            get

        Raises:
            ValueError: Raised if the glyph is empty.
        """
        if not self._shapes:
            raise ValueError('An empty glyph has no bounding box.')

        bbox = self._shapes[0].bounding_box
        for shape in self._shapes[1:]:
            bbox = bbox.extended(shape.bounding_box)
        return bbox

    @property
    def center(self) -> Vector:
        """Center of the bounding box of the glyph.

        Access:
            get

        Raises:
            ValueError: Raised if the glyph is empty.
        """
        return self.bounding_box.center

    def _impl_translate(self, trans_vec: Vector) -> None:
        trans_vec = Vector(trans_vec)
        self._origin = self._origin + trans_vec
        for shape in self._shapes:
            shape._impl_translate(trans_vec)  # pylint: disable=protected-access

    def _impl_rotate(self, theta: float) -> None:
        self._origin = self._origin.rotated(theta)
        for shape in self._shapes:
            shape._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)
        self._origin = self._origin * fac
        self._advance_width *= abs(fac)
        for shape in self._shapes:
            shape._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        mirror_axis = Vector(mirror_axis)
        self._origin = self._origin.mirrored(mirror_axis)
        for shape in self._shapes:
            shape._impl_mirror(mirror_axis)  # pylint: disable=protected-access

    def __mul__(self, other: t.Any) -> t.Any:
        """``glyph * unit('µm')`` creates a :class:`DimGlyph`.

        Raises:
            DimensionError: Raised if `other` is not a unit of length.
        """
        from fibomat.units.unit_tag import _UnitTag  # pylint: disable=import-outside-toplevel,protected-access

        if not isinstance(other, _UnitTag):
            return NotImplemented
        return DimGlyph(self, other)

    def __rmul__(self, other: t.Any) -> t.Any:
        return self.__mul__(other)


class DimGlyph(DimComposite):
    """A :class:`Glyph` with a unit of length."""

    def __init__(self, glyph: Glyph, unit: LengthUnitLike, description: t.Optional[str] = None):
        """
        Args:
            glyph (Glyph): glyph, its coordinates are given in `unit`
            unit (str, pint.Unit, unit): unit of length
            description (str, optional): description

        Raises:
            TypeError: Raised if `glyph` is no :class:`Glyph` or `unit` is no unit.
            DimensionError: Raised if `unit` is no unit of length.
        """
        if not isinstance(glyph, Glyph):
            raise TypeError(f'glyph must be a Glyph, got {type(glyph).__name__}.')
        super().__init__(glyph, unit, description)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._obj.character!r}, n_shapes={len(self._obj)}, unit={self._unit!r})'

    @property
    def glyph(self) -> Glyph:
        """The (unitless) glyph.

        Access:
            get
        """
        return self._obj

    @property
    def character(self) -> str:
        """The character.

        Access:
            get
        """
        return self._obj.character

    @property
    def shapes(self) -> t.Tuple[DimShape, ...]:
        """The shapes of the glyph with the unit of the glyph.

        Access:
            get
        """
        return tuple(DimShape(shape, self._unit) for shape in self._obj.shapes)

    def __iter__(self) -> t.Iterator[DimShape]:
        return iter(self.shapes)

    def __len__(self) -> int:
        return len(self._obj)

    def arrangement_elements(self) -> t.Iterator[DimShape]:
        """The shapes of the glyph (so that backends can export a glyph like a group of shapes).

        Yields:
            DimShape
        """
        yield from self.shapes

    @property
    def advance_width(self) -> DimFloat:
        """Distance to the origin of the next character.

        Access:
            get
        """
        return DimFloat(ureg.Quantity(self._obj.advance_width, self._unit))

    @property
    def origin(self) -> t.Any:
        """Start of the baseline of the character.

        Access:
            get
        """
        from fibomat.linalg import DimVector  # pylint: disable=import-outside-toplevel

        return DimVector.from_vector(self._obj.origin, self._unit)
