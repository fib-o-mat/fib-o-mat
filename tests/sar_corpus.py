"""Layouts for the golden master tests of the step and repeat backend (`tests/test_step_and_repeat_backend.py`).

The XML files in `tests/data/sar_golden` were created with the backend *before* its rework, so that the tests prove that
the file format did not change at all.
"""
import typing as t

import numpy as np

from fibomat.composite_shapes import HollowArcSpline
from fibomat.default_backends.npve import (
    LineByLineOutlined, NPVEMill, OutlineAlignement, OutlineNodeStyle, OutlineScanStyle
)
from fibomat.layout import Layout
from fibomat.linalg import DimVector
from fibomat.raster_styles import ScanSequence, one_d, two_d, zero_d
from fibomat.shapes import ArcSpline, Circle, Ellipse, Line, Polygon, Polyline, Rect, Spot
from fibomat.units import unit


def um(x: float, y: float) -> DimVector:
    return DimVector(x * unit('µm'), y * unit('µm'))


def repeats_mill(repeats: int = 2, dwell: float = 1., scan_direction: t.Optional[float] = None) -> NPVEMill:
    return NPVEMill(dwell_time=dwell * unit('µs'), repeats=repeats, scan_direction=scan_direction)


def dose_mill(dose: str, value: float = 2., dwell: float = 1., scan_direction: t.Optional[float] = None) -> NPVEMill:
    return NPVEMill(dwell_time=dwell * unit('µs'), dose=value * unit(dose), scan_direction=scan_direction)


def curve(pitch: float = 0.01) -> one_d.Curve:
    return one_d.Curve(pitch * unit('µm'), ScanSequence.CONSECUTIVE)


def lines(
    sequence: ScanSequence = ScanSequence.CONSECUTIVE, alpha: float = 0., invert: bool = False,
    line_pitch: float = 0.02, pitch: float = 0.01
) -> two_d.LineByLine:
    return two_d.LineByLine(line_pitch * unit('µm'), sequence, alpha, invert, curve(pitch))


def outlined(
    alignement: OutlineAlignement = OutlineAlignement.INSET,
    scan_style: OutlineScanStyle = OutlineScanStyle.INSIDE_OUT,
    node_style: OutlineNodeStyle = OutlineNodeStyle.MITERED, offset: float = 0.05
) -> LineByLineOutlined:
    return LineByLineOutlined(
        0.02 * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, curve(), offset * unit('µm'), alignement, scan_style,
        node_style
    )


def single_site(patterns: t.Sequence[t.Tuple[t.Any, t.Any, t.Any]], center=(1., 2.), fov=(10., 10.)) -> Layout:
    layout = Layout()
    site = layout.create_site(um(*center), um(*fov) if fov is not None else None)
    for dim_shape, mill, style in patterns:
        site.create_pattern(dim_shape, mill, style)
    return layout


def hollow() -> HollowArcSpline:
    return HollowArcSpline(
        ArcSpline([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)], True),
        [Circle(0.5, center=(0.5, 0.5)).to_arc_spline()]
    )


def cases() -> t.Dict[str, t.Tuple[Layout, t.Dict[str, t.Any]]]:
    """All cases: name -> (layout, keyword arguments of the backend)."""
    result: t.Dict[str, t.Tuple[Layout, t.Dict[str, t.Any]]] = {}

    result['spots_repeats'] = (single_site([
        (Spot((0.5, 0.25)) * unit('µm'), repeats_mill(3, 2.5), zero_d.SingleSpot()),
        (Spot((-1, 1)) * unit('µm'), repeats_mill(1, 0.1, scan_direction=0.5), zero_d.SingleSpot()),
    ]), {})

    result['spot_dose'] = (single_site([
        (Spot((1.5, -0.5)) * unit('µm'), dose_mill('ions', 100000.), zero_d.SingleSpot()),
    ]), {})

    result['curves'] = (single_site([
        (Line((0, 0), (3, 1)) * unit('µm'), repeats_mill(), curve()),
        (Polyline([(0, 0), (1, 1), (2, 0), (3, 1)]) * unit('µm'), repeats_mill(5, 0.5), curve(0.02)),
        (Rect(2, 1).translated((1, 1)) * unit('µm'), repeats_mill(), curve()),
        (Polygon([(0, 0), (1, 0), (1, 1)]) * unit('µm'), repeats_mill(1, 3.), curve(0.005)),
        (Line((0, 0), (3, 1)) * unit('µm'), dose_mill('nC/µm', 0.3), curve()),
    ]), {})

    result['areas'] = (single_site([
        (Rect(2, 1).translated((1, 1)) * unit('µm'), repeats_mill(), lines()),
        (Rect(2, 1, 0.3).translated((-3, 1)) * unit('µm'), repeats_mill(4), lines(ScanSequence.SERPENTINE, 0.3)),
        (Polygon([(0, 0), (2, 0), (1, 2)]) * unit('µm'), repeats_mill(),
         lines(ScanSequence.DOUBLE_SERPENTINE, -0.7, True, 0.05, 0.02)),
        (Circle(1.5, center=(2, -2)) * unit('µm'), dose_mill('nC/µm**2', 5., 0.5), lines()),
        (Ellipse(2, 1, center=(-2, -2)) * unit('µm'), repeats_mill(1, 1., 0.25), lines(alpha=np.pi / 2)),
        (hollow() * unit('µm'), repeats_mill(), lines(line_pitch=0.04)),
    ]), {})

    result['areas_nm'] = (single_site([
        (Rect(2000, 1000).translated((1000, 1000)) * unit('nm'), repeats_mill(), lines()),
        (Circle(1500, center=(2000, -2000)) * unit('nm'), repeats_mill(), lines()),
        (Line((0, 0), (3000, 1000)) * unit('nm'), repeats_mill(), curve()),
    ]), {})

    for index, (alignement, scan_style, node_style) in enumerate([
        (OutlineAlignement.INSET, OutlineScanStyle.INSIDE_OUT, OutlineNodeStyle.MITERED),
        (OutlineAlignement.OUTSET, OutlineScanStyle.OUTSIDE_IN, OutlineNodeStyle.ROUND),
        (OutlineAlignement.CENTER, OutlineScanStyle.ALTERNATING, OutlineNodeStyle.MITERED),
    ]):
        result[f'outlined_{index}'] = (single_site([
            (Rect(2, 1) * unit('µm'), repeats_mill(), outlined(alignement, scan_style, node_style, 0.05 * (index + 1))),
            (Circle(1) * unit('µm'), repeats_mill(), outlined(alignement, scan_style, node_style)),
        ]), {})

    layout = Layout()
    for index, center in enumerate([(0., 0.), (10., 0.), (10., 5.5), (-3., -4.)]):
        site = layout.create_site(um(*center), um(4., 4.))
        site.create_pattern(Rect(1, 1).translated((index, 0)) * unit('µm'), repeats_mill(index + 1), lines())
    result['several_sites'] = (layout, {})

    layout = Layout()
    layout.create_site(um(0, 0), um(4, 4))  # empty
    site = layout.create_site(um(5, 5), um(4, 4))
    site.create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
    layout.create_site(um(9, 9), um(4, 4))  # empty
    result['empty_sites_are_skipped'] = (layout, {})

    layout = Layout()
    site = layout.create_site(um(0, 0), um(4, 4))
    site.create_pattern(Rect(1, 1) * unit('µm'), repeats_mill(), lines())
    site.create_pattern(Circle(0.5, center=(1, 1)) * unit('µm'), repeats_mill(2, 2.), lines())
    layout.create_site(um(5, 5), um(4, 4))
    layout.create_site(um(5, 0), um(4, 4))
    result['shared_patterns'] = (layout, {'share_patterns': True})

    result['automatic_fov'] = (single_site([
        (Rect(2, 1).translated((3, 1)) * unit('µm'), repeats_mill(), lines()),
    ], fov=None), {})

    return result
