"""Provide the :class:`Curve` raster style.

Example:
    >>> from fibomat.raster_styles.one_d import Curve
    >>> from fibomat.raster_styles import ScanSequence
    >>> from fibomat.mill import Mill
    >>> from fibomat.shapes import Line
    >>> from fibomat.units import unit
    >>> style = Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
    >>> pattern = style.rasterize(Line((0, 0), (1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1), unit('nm'), unit('ms'))
    >>> pattern.positions[:, 0].round(6).tolist()
    [0.0, 250.0, 500.0, 750.0, 1000.0]
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.curve_tools import rasterize
from fibomat.mill import Mill
from fibomat.raster_styles._helpers import apply_repeats, check_length, dwell_time_in, position_scale
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.raster_styles.scansequence import ScanSequence, _swap_pairs
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes import DimShape
from fibomat.units import DimFloat, LengthDimension, LengthUnit, TimeUnit, scale_to


__all__ = ['Curve']


_SUPPORTED_SEQUENCES = (ScanSequence.CONSECUTIVE, ScanSequence.BACKSTITCH, ScanSequence.BACK_AND_FORTH)


class Curve(RasterStyle):
    """Expose points with a constant distance (measured along the curve) on the outline of a shape.

    The first point is the start of the curve; the end of an open curve is only exposed if its distance to the start is
    a multiple of the pitch.
    """

    def __init__(self, pitch: DimFloat[LengthDimension], scan_sequence: t.Union[ScanSequence, str]):
        """
        Args:
            pitch (DimFloat[LengthDimension]): distance between the points, e.g. ``0.5 * unit('µm')``.
            scan_sequence (ScanSequence, str): ``CONSECUTIVE``, ``BACKSTITCH`` or ``BACK_AND_FORTH``.

        Raises:
            TypeError: Raised if pitch is not a dimensioned value.
            ValueError: Raised if pitch is not a positive length or the scan sequence is not supported.
        """
        self._pitch = check_length(pitch, 'pitch')

        scan_sequence = ScanSequence.parse(scan_sequence)
        if scan_sequence not in _SUPPORTED_SEQUENCES:
            raise ValueError(
                'scan_sequence must be "ScanSequence.CONSECUTIVE", "ScanSequence.BACKSTITCH" or '
                '"ScanSequence.BACK_AND_FORTH".'
            )

        self._scan_sequence = scan_sequence

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pitch={self._pitch!r}, scan_sequence={self._scan_sequence})'

    @property
    def dimension(self) -> int:
        return 1

    @property
    def pitch(self) -> DimFloat[LengthDimension]:
        """Distance between the points.

        Access:
            get
        """
        return self._pitch

    @property
    def scan_sequence(self) -> ScanSequence:
        """Scan sequence.

        Access:
            get
        """
        return self._scan_sequence

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:
        """Rasterize the outline of a shape.

        Args:
            dim_shape (DimShape): shape with length unit; it must be rasterizable by
                :func:`~fibomat.curve_tools.rasterize`.
            mill (Mill): mill
            out_length_unit (LengthUnit): length unit of the returned pattern
            out_time_unit (TimeUnit): time unit of the returned pattern

        Returns:
            RasterizedPattern

        Raises:
            TypeError: Raised if the shape cannot be rasterized or the mill is no :class:`~fibomat.mill.Mill`.
        """
        points = rasterize(dim_shape.shape, scale_to(dim_shape.unit, self._pitch))

        dwell_points = np.empty((points.n_points, 3))
        dwell_points[:, :2] = points.positions * position_scale(dim_shape.unit, out_length_unit)
        dwell_points[:, 2] = points.weights * dwell_time_in(mill, out_time_unit)

        if self._scan_sequence == ScanSequence.BACK_AND_FORTH:
            forwards, backwards = dwell_points, dwell_points[::-1]
            dwell_points = np.concatenate([forwards if i % 2 == 0 else backwards for i in range(mill.repeats)])
        else:
            if self._scan_sequence == ScanSequence.BACKSTITCH:
                # points are exposed in the order 1, 0, 3, 2, 5, 4, ...
                dwell_points = dwell_points[_swap_pairs(len(dwell_points))]
            dwell_points = apply_repeats(dwell_points, mill.repeats)

        return RasterizedPattern(dwell_points, out_length_unit, out_time_unit)
