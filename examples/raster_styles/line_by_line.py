import numpy as np

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.units import unit
from fibomat import default_backends, shapes, curve_tools, raster_styles

sample = Layout()

mill = Mill(dwell_time=1 * unit('ms'), repeats=5)

site_consecutive = sample.create_site(
    dim_position=Vector(0, 0) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_consecutive.create_pattern(
    dim_shape=shapes.Circle(r=.4) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=50 * unit('nm'),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=50 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_cross_section = sample.create_site(
    dim_position=Vector(1, 0) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_cross_section.create_pattern(
    dim_shape=shapes.ArcSpline([(-.4, 0, 0), (.4, 0, 1)], is_closed=True) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=50 * unit('nm'),
        scan_sequence=raster_styles.ScanSequence.CROSSECTION,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=50 * unit('nm'), scan_sequence=raster_styles.ScanSequence.BACKSTITCH)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_serpentine = sample.create_site(
    dim_position=Vector(0, -1) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_serpentine.create_pattern(
    dim_shape=shapes.Rect(width=0.5, height=.25) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=50 * unit('nm'),
        scan_sequence=raster_styles.ScanSequence.SERPENTINE,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=50 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_double_serpentine = sample.create_site(
    dim_position=Vector(1, -1) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_double_serpentine.create_pattern(
    dim_shape=shapes.Polygon.regular_ngon(radius=.4, n=6, center=(0, 0)) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=50 * unit('nm'),
        scan_sequence=raster_styles.ScanSequence.DOUBLE_SERPENTINE,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=50 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_cross_section = sample.create_site(
    dim_position=Vector(0, -2) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_cross_section.create_pattern(
    dim_shape=shapes.Circle(r=.4) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=50 * unit('nm'),
        scan_sequence=raster_styles.ScanSequence.CROSSECTION,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=50 * unit('nm'), scan_sequence=raster_styles.ScanSequence.BACK_AND_FORTH)
    )
)

# ----------------------------------------------------------------------------------------------------------------------

site_double_serpentine_same_path = sample.create_site(
    dim_position=Vector(1, -2) * unit('µm'),
    dim_fov=Vector(1, 1) * unit('µm')
)

site_double_serpentine_same_path.create_pattern(
    dim_shape=shapes.Rect(width=.7, height=.35) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.two_d.LineByLine(
        line_pitch=50 * unit('nm'),
        scan_sequence=raster_styles.ScanSequence.DOUBLE_SERPENTINE_SAME_PATH,
        alpha=0, invert=False,
        line_style=raster_styles.one_d.Curve(pitch=50 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE)
    )
)


sample.export(default_backends.SpotListBackend).save('rasterized.txt')

