"""Provides the raster style :class:`LineByLineOutlined` and its enums.

The style fills an area line by line like :class:`~fibomat.raster_styles.two_d.LineByLine` and, in addition, exposes
the outline of the area. It is only supported by the :class:`StepAndRepeatBackend`, where NPVE does the work.

Example:
    >>> from fibomat.raster_styles import ScanSequence, one_d
    >>> from fibomat.units import unit
    >>> style = LineByLineOutlined(
    ...     0.02 * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, one_d.Curve(0.01 * unit('µm'), ScanSequence.CONSECUTIVE),
    ...     0.05 * unit('µm'), OutlineAlignement.INSET, OutlineScanStyle.INSIDE_OUT, OutlineNodeStyle.MITERED
    ... )
    >>> style.outline_alignement
    <OutlineAlignement.INSET: 1>
"""
from __future__ import annotations

import enum
import typing as t

from fibomat.raster_styles._helpers import check_length
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.raster_styles.scansequence import ScanSequence
from fibomat.raster_styles.two_d import LineByLine
from fibomat.units import DimFloat, LengthDimension


__all__ = ['OutlineAlignement', 'OutlineScanStyle', 'OutlineNodeStyle', 'LineByLineOutlined']


@enum.unique
class OutlineAlignement(enum.Enum):
    """Position of the outline with respect to the boundary of the shape (the values are the NPVE codes)."""
    INSET = 1
    OUTSET = 2
    CENTER = 0


@enum.unique
class OutlineScanStyle(enum.Enum):
    """Order in which the outlines are exposed (the values are the NPVE codes)."""
    INSIDE_OUT = 0
    OUTSIDE_IN = 1
    ALTERNATING = 2


@enum.unique
class OutlineNodeStyle(enum.Enum):
    """Style of the corners of the outline (the values are the NPVE codes)."""
    MITERED = 0
    ROUND = 2


def _check_enum(value: t.Any, enum_class: t.Type[enum.Enum], name: str) -> t.Any:
    """Check that `value` is a member of `enum_class`.

    Raises:
        TypeError: Raised otherwise.
    """
    if not isinstance(value, enum_class):
        raise TypeError(f'{name} must be a member of {enum_class.__name__}, got {value!r}.')
    return value


class LineByLineOutlined(LineByLine):
    """Fill an area line by line and expose its outline (see the module docstring)."""

    def __init__(
        self,
        line_pitch: DimFloat[LengthDimension],
        scan_sequence: ScanSequence,
        alpha: float,
        invert: bool,
        line_style: RasterStyle,
        outline_offset: DimFloat[LengthDimension],
        outline_alignement: OutlineAlignement,
        outline_scan_style: OutlineScanStyle,
        outline_node_style: OutlineNodeStyle,
    ):
        """
        Args:
            line_pitch (DimFloat): distance between the lines
            scan_sequence (ScanSequence): scan sequence of the lines
            alpha (float): angle of the lines
            invert (bool): direction in which the lines are created
            line_style (RasterStyle): raster style of the lines (dimension 1)
            outline_offset (DimFloat): width of the outline (the outline offset of NPVE)
            outline_alignement (OutlineAlignement): position of the outline
            outline_scan_style (OutlineScanStyle): order in which the outlines are exposed
            outline_node_style (OutlineNodeStyle): style of the corners of the outline

        Raises:
            TypeError: Raised if a pitch or the offset is no dimensioned value or an outline style is no enum member.
            ValueError: Raised if a pitch or the offset is no positive length or the arguments of
                :class:`~fibomat.raster_styles.two_d.LineByLine` are invalid.
        """
        super().__init__(line_pitch, scan_sequence, alpha, invert, line_style)

        self._outline_offset = check_length(outline_offset, 'outline_offset')
        self._outline_alignement = _check_enum(outline_alignement, OutlineAlignement, 'outline_alignement')
        self._outline_scan_style = _check_enum(outline_scan_style, OutlineScanStyle, 'outline_scan_style')
        self._outline_node_style = _check_enum(outline_node_style, OutlineNodeStyle, 'outline_node_style')

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(line_pitch={self._line_pitch!r}, alpha={self._alpha!r}, '
            f'invert={self._invert!r}, scan_sequence={self._scan_sequence!r}, line_style={self._line_style!r}, '
            f'outline_offset={self._outline_offset!r}, outline_alignement={self._outline_alignement!r}, '
            f'outline_scan_style={self._outline_scan_style!r}, outline_node_style={self._outline_node_style!r})'
        )

    @property
    def outline_offset(self) -> DimFloat[LengthDimension]:
        """Width of the outline.

        Access:
            get
        """
        return self._outline_offset

    @property
    def outline_alignement(self) -> OutlineAlignement:
        """Position of the outline.

        Access:
            get
        """
        return self._outline_alignement

    @property
    def outline_scan_style(self) -> OutlineScanStyle:
        """Order in which the outlines are exposed.

        Access:
            get
        """
        return self._outline_scan_style

    @property
    def outline_node_style(self) -> OutlineNodeStyle:
        """Style of the corners of the outline.

        Access:
            get
        """
        return self._outline_node_style
