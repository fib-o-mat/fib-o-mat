from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.units import unit
from fibomat import shapes, raster_styles

from fibomat.default_backends.fei import FEIStreamFile


s = Layout()
site = s.create_site(
    dim_position=Vector(0, 0) * unit('µm'), dim_fov=Vector(5, 5) * unit('µm')
)

mill = Mill(dwell_time=1 * unit('ms'), repeats=4)

site.create_pattern(
    dim_shape=shapes.Line((-2, -2), (2, 2)) * unit('µm'),
    mill=mill,
    raster_style=raster_styles.one_d.Curve(
        pitch=1 * unit('nm'),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
    )
)

# see src/fibomat/default_backends/fei/__init__.py for all settings
exported = s.export(FEIStreamFile, n_rep=3, margin=0.76)
# exported = s.export(FEIStreamFile)
exported.save('file.str')
