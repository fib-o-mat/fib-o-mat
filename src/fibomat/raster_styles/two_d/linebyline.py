"""Provide the :class:`LineByLine` raster style.

Example:
    >>> from fibomat.raster_styles import ScanSequence
    >>> from fibomat.raster_styles.one_d import Curve
    >>> from fibomat.raster_styles.two_d import LineByLine
    >>> from fibomat.mill import Mill
    >>> from fibomat.shapes import Rect
    >>> from fibomat.units import unit
    >>> line_style = Curve(0.5 * unit('µm'), ScanSequence.CONSECUTIVE)
    >>> style = LineByLine(0.5 * unit('µm'), ScanSequence.SERPENTINE, alpha=0., invert=False, line_style=line_style)
    >>> pattern = style.rasterize(Rect(1, 1) * unit('µm'), Mill(1. * unit('ms'), 1), unit('µm'), unit('ms'))
    >>> pattern.dwell_points.shape  # two lines (y = 0 and y = -0.5) with three points each
    (6, 3)
"""
from __future__ import annotations

import math
import typing as t

from fibomat.curve_tools import fill_with_lines
from fibomat.mill import Mill
from fibomat.raster_styles._helpers import check_length
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.raster_styles.scansequence import ScanSequence, _apply_scan_sequence
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes import DimShape
from fibomat.shapes._line_non_continuous import LineNonContinuous
from fibomat.shapes.arc_spline import ArcSplineCompatible
from fibomat.units import DimFloat, LengthDimension, LengthUnit, TimeUnit, scale_to


if t.TYPE_CHECKING:  # pragma: no cover
    from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline
    from fibomat.shapes.arc_spline import ArcSpline


__all__ = ['LineByLine']


def _filling_shape(shape: t.Any) -> t.Union[ArcSpline, HollowArcSpline]:
    """Convert a shape to the closed curve (with holes) which can be filled.

    Args:
        shape (Shape): shape. It must be a HollowArcSpline, provide ``to_hollow_arc_spline()`` (e.g. a Ring) or be
            convertible to an ArcSpline.

    Returns:
        Union[ArcSpline, HollowArcSpline]

    Raises:
        TypeError: Raised if the shape cannot be filled.
    """
    from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline  # pylint: disable=import-outside-toplevel

    if isinstance(shape, HollowArcSpline):
        return shape

    to_hollow = getattr(shape, 'to_hollow_arc_spline', None)
    if callable(to_hollow):
        return to_hollow()

    if isinstance(shape, ArcSplineCompatible):
        return shape.to_arc_spline()

    raise TypeError(f'Cannot fill a shape of type {type(shape).__name__}.')


class LineByLine(RasterStyle):
    """Fill an area with parallel lines which are exposed according to a scan sequence.

    The lines (of the distance `line_pitch`, rotated by `alpha`) are created with
    :func:`~fibomat.curve_tools.fill_with_lines`, a raster style for curves (`line_style`) is applied to each of them.
    """

    def __init__(
        self,
        line_pitch: DimFloat[LengthDimension],
        scan_sequence: t.Union[ScanSequence, str],
        alpha: float,
        invert: bool,
        line_style: RasterStyle,
    ):
        """
        Args:
            line_pitch (DimFloat[LengthDimension]): distance between the lines, e.g. ``0.5 * unit('µm')``.
            scan_sequence (ScanSequence, str): order in which the lines are exposed
            alpha (float): angle between the lines and the x-axis in [-pi/2, pi/2]
            invert (bool): if False, the lines are created from the top to the bottom (with respect to the rotated
                frame), from the bottom to the top otherwise.
            line_style (RasterStyle): raster style of the lines (dimension 1), e.g. :class:`one_d.Curve`

        Raises:
            TypeError: Raised if line_pitch is not a dimensioned value.
            ValueError: Raised if line_pitch is not a positive length, alpha is not in [-pi/2, pi/2], or line_style
                is not a raster style for curves.
        """
        self._line_pitch = check_length(line_pitch, 'line_pitch')

        alpha = float(alpha)
        if not -math.pi / 2 <= alpha <= math.pi / 2:
            raise ValueError('alpha must be in [-pi/2, pi/2].')
        self._alpha = alpha

        self._invert = bool(invert)
        self._scan_sequence = ScanSequence.parse(scan_sequence)

        if not isinstance(line_style, RasterStyle) or line_style.dimension != 1:
            raise ValueError('line_style must be a raster style with dimension == 1.')
        self._line_style = line_style

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(line_pitch={self._line_pitch!r}, alpha={self._alpha!r}, '
            f'invert={self._invert!r}, scan_sequence={self._scan_sequence!r}, line_style={self._line_style!r})'
        )

    @property
    def dimension(self) -> int:
        return 2

    @property
    def line_pitch(self) -> DimFloat[LengthDimension]:
        """Distance between the lines.

        Access:
            get
        """
        return self._line_pitch

    @property
    def scan_sequence(self) -> ScanSequence:
        """Scan sequence.

        Access:
            get
        """
        return self._scan_sequence

    @property
    def alpha(self) -> float:
        """Angle between the lines and the x-axis.

        Access:
            get
        """
        return self._alpha

    @property
    def invert(self) -> bool:
        """Direction in which the lines are created.

        Access:
            get
        """
        return self._invert

    @property
    def line_style(self) -> RasterStyle:
        """Raster style which is applied to the lines.

        Access:
            get
        """
        return self._line_style

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit,
    ) -> RasterizedPattern:
        """Fill a shape with lines.

        Args:
            dim_shape (DimShape): closed shape with length unit (an ArcSpline compatible shape, a HollowArcSpline or a
                shape with ``to_hollow_arc_spline()``, e.g. a Ring).
            mill (Mill): mill
            out_length_unit (LengthUnit): length unit of the returned pattern
            out_time_unit (TimeUnit): time unit of the returned pattern

        Returns:
            RasterizedPattern: pattern (empty if the shape is smaller than the line pitch)

        Raises:
            TypeError: Raised if the shape cannot be filled.
        """
        filling_rows = fill_with_lines(
            _filling_shape(dim_shape.shape),
            scale_to(dim_shape.unit, self._line_pitch),
            self._alpha,
            self._invert,
        )

        return _apply_scan_sequence(
            filling_rows=[LineNonContinuous(row) for row in filling_rows],
            filling_rows_length_unit=dim_shape.unit,
            scan_sequence=self._scan_sequence,
            line_style=self._line_style,
            mill=mill,
            out_length_unit=out_length_unit,
            out_time_unit=out_time_unit,
        )
