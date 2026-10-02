"""Provide LineByLine raster style."""

from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.units import LengthUnit, TimeUnit, LengthQuantity, has_length_dim, scale_to
from fibomat.mill import Mill
from fibomat.shapes import DimShape
from fibomat.raster_styles.scansequence import (
    ScanSequence,
    _apply_scan_sequence,
)
from fibomat.curve_tools import fill_with_lines
from fibomat.shapes._line_non_continuous import LineNonContinuous


class LineByLine(RasterStyle):
    def __init__(
        self,
        line_pitch: LengthQuantity,
        scan_sequence: ScanSequence,
        alpha: float,
        invert: bool,
        line_style: RasterStyle,
    ):
        if not has_length_dim(line_pitch):
            raise ValueError("line_pitch must have dimension [length].")
        self._line_pitch = line_pitch

        self._alpha = alpha
        self._invert = invert

        self._scan_sequence = scan_sequence

        if line_style.dimension != 1:
            raise ValueError("line_style must have dimension == 1")

        # if not line_style.scan_sequence == ScanSequence.CONSECUTIVE:
        #     raise NotImplementedError

        self._line_style = line_style

    def __repr__(self):
        return "{}(line_pitch={!r}, alpha={!r}, invert={!r}, scan_sequence={!r}, line_style={!r})".format(
            self.__class__.__name__,
            self._line_pitch,
            self._alpha,
            self._invert,
            self._scan_sequence,
            self._line_style,
        )

    @property
    def dimension(self) -> int:
        return 2

    @property
    def line_pitch(self):
        return self._line_pitch

    @property
    def scan_sequence(self):
        return self._scan_sequence

    @property
    def alpha(self):
        return self._alpha

    @property
    def invert(self):
        return self._invert

    @property
    def line_style(self):
        return self._line_style

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit,
    ) -> RasterizedPattern:
        # dim_shape = DimObj.create(dim_shape)

        filling_rows = fill_with_lines(
            dim_shape.shape.to_arc_spline(),
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

        # rasterized_lines = []
        #
        # if self._scan_sequence == ScanSequence.CROSSECTION:
        #     line_mill = mill
        # else:
        #     line_mill = Mill(mill.dwell_time, repeats=1)
        #
        # #     line_mill = mill
        # # else:
        # # line_mill = Mill(mill.dwell_time, repeats=1)
        #
        # for row in filling_rows:
        #     # rasterized_row = []
        #
        #     row = LineNonContinuous(row)
        #
        #     #for line in row:
        #     rasterized_lines.append(
        #         self._line_style.rasterize(
        #             (row, dim_shape.unit),
        #             mill=line_mill,
        #             out_length_unit=out_length_unit,
        #             out_time_unit=out_time_unit
        #         ).dwell_points
        #     )
        #
        #     # rasterized_lines.append(rasterized_row))
        #
        # scan_lines = []
        #
        # for index, direction in zip(*_make_scan_index_sequence(len(rasterized_lines), mill.repeats, self._scan_sequence)):
        #     if not direction:
        #         scan_lines.append(rasterized_lines[index][::-1])
        #     else:
        #         scan_lines.append(rasterized_lines[index])
        #
        # # points = np.concatenate(scan_lines)
        #
        # # points._dwell_points[:, :2] *= scale_factor(out_length_unit, dim_shape.unit)
        # # points._dwell_points[:, 2] = scale_to(out_time_unit, mill.dwell_time)
        #
        # # scan_sequence_indices
        #
        # return RasterizedPattern(np.concatenate(scan_lines), out_length_unit, out_time_unit)
