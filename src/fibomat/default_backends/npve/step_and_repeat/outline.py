import enum

from fibomat.units import (
    LengthQuantity,
    has_length_dim,
)
from fibomat.raster_styles.rasterstyle import RasterStyle


from fibomat.raster_styles.two_d import LineByLine
from fibomat.raster_styles.scansequence import ScanSequence


@enum.unique
class OutlineAlignement(enum.Enum):
    INSET = 1
    OUTSET = 2
    CENTER = 0


@enum.unique
class OutlineScanStyle(enum.Enum):
    INSIDE_OUT = 0
    OUTSIDE_IN = 1
    ALTERNATING = 2


@enum.unique
class OutlineNodeStyle(enum.Enum):
    MITERED = 0
    ROUND = 2


class LineByLineOutlined(LineByLine):
    def __init__(
        self,
        line_pitch: LengthQuantity,
        scan_sequence: ScanSequence,
        alpha: float,
        invert: bool,
        line_style: RasterStyle,
        outline_offset: LengthQuantity,
        outline_alignement: OutlineAlignement,
        outline_scan_style: OutlineScanStyle,
        outline_node_style: OutlineNodeStyle,
    ):
        super().__init__(
            line_pitch,
            scan_sequence,
            alpha,
            invert,
            line_style,
        )

        if not has_length_dim(outline_offset):
            raise ValueError("outline_offset must have dimension [length].")

        self._outline_offset = outline_offset

        # if not isinstance(outline_alignement, OutlineAlignement):
        #     raise ValueError("outline_alignement must be type OutlineAlignement")

        self._outline_alignement = outline_alignement

        self._outline_scan_style = outline_scan_style
        self._outline_node_style = outline_node_style
