"""Provides the :class:`Text` and :class:`DimText` classes.

Text is shaped with the Hershey fonts of `pyhershey`. Every character is a :class:`~fibomat.composite_shapes.glyph.Glyph`
which consists of the polylines of the character or, if a stroke width is given, of the outlines of the strokes
(polygons, characters with enclosed areas like 'O' or 'B' have polygons with holes).

Example::

    from fibomat.composite_shapes import Text
    from fibomat.units import unit

    text = Text('Hello fib-o-mat!\\nsecond line', font_size=2, stroke_width=0.2, text_alignment='center')

    for glyph in text:  # one glyph per character (including spaces)
        if glyph:  # glyphs of spaces are empty
            for shape in glyph:
                ...

    text.shapes()  # all shapes of all glyphs
    text.baseline_anchor('left', i_line=1)

    dim_text = text * unit('µm')  # DimText
"""
from __future__ import annotations

import typing as t

import numpy as np
from pyhershey.glyph_factory import glyph_factory
import pyhershey

from fibomat.composite_shapes._dim_composite import DimComposite
from fibomat.composite_shapes.glyph import DimGlyph, Glyph
from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline
from fibomat.linalg import BoundingBox, DimVector, Transformable, Vector
from fibomat.shapes.dim_shape import DimShape, LengthUnitLike
from fibomat.shapes.polygon import Polygon
from fibomat.shapes.polyline import Polyline
from fibomat.shapes.shape import Shape
from fibomat import _libfibomat


__all__ = ['Text', 'DimText']


_ALIGNMENTS = ('left', 'center', 'right')
_ANCHORS = ('left', 'center', 'right')
_JOINS = ('bevel', 'miter', 'round')
_CAPS = ('butt', 'square', 'round')
_HERSHEY_FONT_SIZE = 21
"""The height of upper case letters of the Hershey fonts (before scaling)."""
_LINE_SPACING = 1.6
"""Distance of the baselines in units of the font size."""


def _merge_segments(segments: t.Sequence[np.ndarray]) -> t.List[np.ndarray]:
    """Chain segments which share an end point (if exactly two segments end in the point).

    The Hershey fonts store e.g. a 'V' as two separate strokes. Stroked separately they have butt caps and leave a gap on
    the outer side of their common end point; as one polyline they are connected by a proper join.
    """
    segments = [np.asarray(segment, dtype=float) for segment in segments]
    tol = 1e-9 * max((float(np.max(np.abs(segment))) for segment in segments), default=1.)

    while True:
        ends: t.List[t.Tuple[int, int]] = []  # (segment index, 0 for the start and -1 for the end)
        for i_segment, segment in enumerate(segments):
            if not np.allclose(segment[0], segment[-1], rtol=0., atol=tol):  # (closed segments are not chained)
                ends.extend([(i_segment, 0), (i_segment, -1)])

        merged = False
        for i, (i_first, side_first) in enumerate(ends):
            partners = [
                (i_second, side_second) for (i_second, side_second) in ends
                if i_second != i_first and np.allclose(
                    segments[i_second][side_second], segments[i_first][side_first], rtol=0., atol=tol
                )
            ]
            # no junction of three or more segments, and no further segment ending in this point twice
            same_segment = [e for e in ends[i + 1:] if e[0] == i_first and np.allclose(
                segments[e[0]][e[1]], segments[i_first][side_first], rtol=0., atol=tol)]
            if len(partners) == 1 and not same_segment:
                i_second, side_second = partners[0]
                first = segments[i_first] if side_first == -1 else segments[i_first][::-1]
                second = segments[i_second] if side_second == 0 else segments[i_second][::-1]
                chain = np.concatenate((first, second[1:]))
                segments = [segment for k, segment in enumerate(segments) if k not in (i_first, i_second)] + [chain]
                merged = True
                break

        if not merged:
            return segments


def _stroked_shapes(segments: t.Sequence[np.ndarray], width: float, join: str, cap: str) -> t.List[Shape]:
    """Outlines of the united strokes of the segments of a glyph: polygons, or polygons with holes."""
    paths, closed = [], []
    for segment in _merge_segments(segments):
        is_closed = len(segment) > 2 and bool(np.allclose(segment[0], segment[-1]))
        # (the first point must not be repeated for closed paths)
        paths.append(np.ascontiguousarray(segment[:-1] if is_closed else segment, dtype=float))
        closed.append(is_closed)

    shapes: t.List[Shape] = []
    for contours in _libfibomat.stroke_paths(paths, closed, width, join, cap):
        outer, holes = contours[0], contours[1:]
        if holes:
            shapes.append(HollowArcSpline.from_points(outer, holes, disable_checks=True))
        else:
            shapes.append(Polygon(outer))
    return shapes


class Text(Transformable[Vector, BoundingBox]):
    """Text, shaped with a Hershey font.

    The text is a sequence of glyphs (one for each character). The shapes of all glyphs are accessible with
    :meth:`Text.shapes`. Text is transformable.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        text: str,
        font_size: float = 1.,
        stroke_width: t.Optional[float] = None,
        text_alignment: t.Optional[str] = None,
        description: t.Optional[str] = None,
        mapping: t.Optional[str] = None,
        stroke_join: str = 'bevel',
        stroke_cap: str = 'butt',
    ):
        """
        The first baseline starts at (0, 0) (for the alignments "left", "center" and "right" its left, center and
        right side, respectively, is at x = 0). Further lines are placed below.

        Args:
            text (str): text. Lines are separated by ``\\n``. Only printable ASCII characters can be used (cf.
                `mapping`).
            font_size (float, optional):
                font size (directly correspond to the height of upper case letters). Default to 1.
            stroke_width (float, optional):
                the (total) width of the strokes of the glyphs. If None, the glyphs consist of 1d polylines. Otherwise,
                the outlines of the strokes are used. Default to None.
            text_alignment (str, optional): the text alignment. Can be "left", "center", "right". Default to "left".
            description (str, optional): description
            mapping (str, optional): the ascii to hershey mapping (font). Default to "roman_simplex".
            stroke_join (str, optional): "bevel", "miter" or "round". Only used if `stroke_width` is given.
            stroke_cap (str, optional): "butt", "square" or "round". Only used if `stroke_width` is given.

        Raises:
            ValueError: Raised if arguments are invalid, the text contains characters which are not available in the
                font or contains no glyph with shapes (e.g. only spaces).
        """
        super().__init__(description)

        mapping = mapping or 'roman_simplex'
        text_alignment = text_alignment or 'left'

        font_size = float(font_size)
        if not np.isfinite(font_size) or font_size <= 0.:
            raise ValueError('font_size must be positive and finite.')
        if stroke_width is not None:
            stroke_width = float(stroke_width)
            if not np.isfinite(stroke_width) or stroke_width <= 0.:
                raise ValueError('stroke_width must be positive and finite.')
        if stroke_join not in _JOINS:
            raise ValueError(f'stroke_join must be one of {_JOINS}, got {stroke_join!r}.')
        if stroke_cap not in _CAPS:
            raise ValueError(f'stroke_cap must be one of {_CAPS}, got {stroke_cap!r}.')
        if text_alignment not in _ALIGNMENTS:
            raise ValueError(f'text_alignment must be one of {_ALIGNMENTS}, got {text_alignment!r}.')
        if mapping not in glyph_factory.ascii_mappings:
            raise ValueError(f'Unknown mapping {mapping!r}, available mappings: {glyph_factory.ascii_mappings}.')
        text = str(text)
        unsupported = []
        for character in sorted(set(text) - {'\n'}):
            try:
                glyph_factory.from_ascii(character, mapping)
            except Exception:  # pylint: disable=broad-except
                unsupported.append(character)
        if unsupported:
            raise ValueError(f'The characters {unsupported} are not available in the font {mapping!r}.')

        self._text = text
        self._font_size = font_size
        self._stroke_width = stroke_width

        scale = font_size / _HERSHEY_FONT_SIZE
        line_height = _LINE_SPACING * font_size

        glyphs: t.List[Glyph] = []
        anchors: t.List[t.Dict[str, Vector]] = []

        for i_line, line in enumerate(text.split('\n')):
            y_pos = -i_line * line_height
            shaped_line = (
                pyhershey.shape_text(line, advance_height=0, mapping=mapping, font_size=scale, text_align='left')
                if line else []
            )

            width = shaped_line[-1]['pos'][0] + shaped_line[-1]['glyph'].advance_width if shaped_line else 0.
            shift = {'left': 0., 'center': -width / 2, 'right': -width}[text_alignment]

            for character, shaped_glyph in zip(line, shaped_line):
                position = np.array([shaped_glyph['pos'][0] + shift, y_pos])
                segments = [np.asarray(segment, dtype=float) + position for segment in shaped_glyph['glyph'].segments]
                segments = [segment for segment in segments if len(segment) >= 2]

                if stroke_width and segments:
                    shapes = _stroked_shapes(segments, stroke_width, stroke_join, stroke_cap)
                else:
                    shapes = [Polyline(segment) for segment in segments]

                glyphs.append(Glyph(shapes, character, position, shaped_glyph['glyph'].advance_width))

            anchors.append({
                'left': Vector(shift, y_pos),
                'center': Vector(shift + width / 2, y_pos),
                'right': Vector(shift + width, y_pos),
            })

        if not any(glyphs):
            raise ValueError('The text contains no printable characters (only spaces and line breaks).')

        self._glyphs = tuple(glyphs)
        self._anchors = anchors
        self._line_of_glyph = [
            i_line for i_line, line in enumerate(text.split('\n')) for _ in line
        ]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._text!r}, font_size={self._font_size!r})'

    @property
    def text(self) -> str:
        """The text.

        Access:
            get
        """
        return self._text

    @property
    def font_size(self) -> float:
        """The font size (the height of upper case letters).

        Access:
            get
        """
        return self._font_size

    @property
    def stroke_width(self) -> t.Optional[float]:
        """The stroke width (None if the glyphs consist of polylines).

        Access:
            get
        """
        return self._stroke_width

    @property
    def glyphs(self) -> t.Tuple[Glyph, ...]:
        """The glyphs, one for each character (also for spaces, these glyphs are empty).

        Access:
            get
        """
        return self._glyphs

    def __iter__(self) -> t.Iterator[Glyph]:
        return iter(self._glyphs)

    def __len__(self) -> int:
        return len(self._glyphs)

    def line(self, i_line: int) -> t.Tuple[Glyph, ...]:
        """The glyphs of a text line.

        Args:
            i_line (int): index of the line (can be negative)

        Returns:
            Tuple[Glyph, ...]

        Raises:
            IndexError: Raised if there is no such line.
        """
        index = range(self.n_lines)[i_line]
        return tuple(glyph for glyph, i in zip(self._glyphs, self._line_of_glyph) if i == index)

    def shapes(self) -> t.Iterator[Shape]:
        """Iterate over the shapes of all glyphs.

        Yields:
            Shape
        """
        for glyph in self._glyphs:
            yield from glyph

    @property
    def n_lines(self) -> int:
        """Number of text lines.

        Access:
            get
        """
        return len(self._anchors)

    def baseline_anchor(self, pos: str, i_line: int = 0) -> Vector:
        """Return position of certain base line points.

        Args:
            pos (str): position on the base line. Can be "left", "center" or "right".
            i_line (int, optional):
                index of the base line to be used. Can be indexed like any array (0 is the first element, -1 the last
                etc.). Default to 0.

        Returns:
            Vector

        Raises:
            ValueError: Raised if `pos` is unknown.
            IndexError: Raised if there is no such line.
        """
        if pos not in _ANCHORS:
            raise ValueError(f'pos must be one of {_ANCHORS}, got {pos!r}.')
        return self._anchors[i_line][pos]

    @property
    def bounding_box(self) -> BoundingBox:
        """Bounding box of all glyphs.

        Access:
            get
        """
        glyphs = [glyph for glyph in self._glyphs if glyph]
        bbox = glyphs[0].bounding_box
        for glyph in glyphs[1:]:
            bbox = bbox.extended(glyph.bounding_box)
        return bbox

    @property
    def center(self) -> Vector:
        """Center of the bounding box of the text.

        Access:
            get
        """
        return self.bounding_box.center

    def _transform_anchors(self, func: t.Callable[[Vector], Vector]) -> None:
        self._anchors = [{name: func(anchor) for name, anchor in anchors.items()} for anchors in self._anchors]

    def _impl_translate(self, trans_vec: Vector) -> None:
        trans_vec = Vector(trans_vec)
        for glyph in self._glyphs:
            glyph._impl_translate(trans_vec)  # pylint: disable=protected-access
        self._transform_anchors(lambda anchor: anchor + trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        for glyph in self._glyphs:
            glyph._impl_rotate(theta)  # pylint: disable=protected-access
        self._transform_anchors(lambda anchor: anchor.rotated(theta))

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)
        for glyph in self._glyphs:
            glyph._impl_scale(fac)  # pylint: disable=protected-access
        self._transform_anchors(lambda anchor: anchor * fac)
        self._font_size *= abs(fac)
        if self._stroke_width:
            self._stroke_width *= abs(fac)

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        mirror_axis = Vector(mirror_axis)
        for glyph in self._glyphs:
            glyph._impl_mirror(mirror_axis)  # pylint: disable=protected-access
        self._transform_anchors(lambda anchor: anchor.mirrored(mirror_axis))

    def __mul__(self, other: t.Any) -> t.Any:
        """``text * unit('µm')`` creates a :class:`DimText`.

        Raises:
            DimensionError: Raised if `other` is not a unit of length.
        """
        from fibomat.units.unit_tag import _UnitTag  # pylint: disable=import-outside-toplevel,protected-access

        if not isinstance(other, _UnitTag):
            return NotImplemented
        return DimText(self, other)

    def __rmul__(self, other: t.Any) -> t.Any:
        return self.__mul__(other)


class DimText(DimComposite):
    """A :class:`Text` with a unit of length."""

    def __init__(self, text: Text, unit: LengthUnitLike, description: t.Optional[str] = None):
        """
        Args:
            text (Text): text, its coordinates are given in `unit`
            unit (str, pint.Unit, unit): unit of length
            description (str, optional): description

        Raises:
            TypeError: Raised if `text` is no :class:`Text` or `unit` is no unit.
            DimensionError: Raised if `unit` is no unit of length.
        """
        if not isinstance(text, Text):
            raise TypeError(f'text must be a Text, got {type(text).__name__}.')
        super().__init__(text, unit, description if description is not None else text.description)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._obj.text!r}, unit={self._unit!r})'

    @property
    def text_obj(self) -> Text:
        """The (unitless) text.

        Access:
            get
        """
        return self._obj

    @property
    def text(self) -> str:
        """The text.

        Access:
            get
        """
        return self._obj.text

    @property
    def glyphs(self) -> t.Tuple[DimGlyph, ...]:
        """The glyphs with the unit of the text.

        Access:
            get
        """
        return tuple(DimGlyph(glyph, self._unit) for glyph in self._obj.glyphs)

    def __iter__(self) -> t.Iterator[DimGlyph]:
        return iter(self.glyphs)

    def __len__(self) -> int:
        return len(self._obj)

    def shapes(self) -> t.Iterator[DimShape]:
        """Iterate over the shapes of all glyphs.

        Yields:
            DimShape
        """
        for shape in self._obj.shapes():
            yield DimShape(shape, self._unit)

    def layout_elements(self) -> t.Iterator[DimShape]:
        """The shapes of all glyphs (so that backends can export a text like a group of shapes).

        Yields:
            DimShape
        """
        yield from self.shapes()

    @property
    def n_lines(self) -> int:
        """Number of text lines.

        Access:
            get
        """
        return self._obj.n_lines

    def baseline_anchor(self, pos: str, i_line: int = 0) -> DimVector:
        """Return position of certain base line points.

        Args:
            pos (str): position on the base line. Can be "left", "center" or "right".
            i_line (int, optional): index of the base line to be used. Default to 0.

        Returns:
            DimVector
        """
        return DimVector.from_vector(self._obj.baseline_anchor(pos, i_line), self._unit)
