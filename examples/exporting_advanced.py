from typing import Dict, Any

import numpy as np

from fibomat.layout import Layout
from fibomat.linalg import Vector
from fibomat.mill import Mill
from fibomat.units import unit
from fibomat import default_backends, shapes, raster_styles, utils, units


def custom_save_impl(filename: utils.PathLike, dwell_points: np.ndarray, parameters: Dict[str, Any]):
    # fov is in units of length_unit
    fov = max(parameters["fov"].width, parameters["fov"].height)

    base_dwell_time = units.scale_to(unit('µs'), parameters["base_dwell_time"])

    with open(filename, 'w') as fp:
        # first, write header data.
        fp.writelines([
            'CUSTOMFILEFORMAT\n',
            f'FOV={fov:.3f}\n',
            f'DWELL={base_dwell_time:.1f}\n',
            'BEGIN_DWELL_POINTS\n'
        ])

        # second, write dwell point data
        # dwell_points has shape (N, 3) where N is the numbre of dwell points. Each row in the array contains
        # (x, y, t_d) where x and y are the position of a spot and t_d the dwell time or dwell time multiplicand.
        # "%.5f %.5f %d" is the formatting string. See numpy doc for details on that.
        np.savetxt(fp, dwell_points, "%.5f %.5f %d")

        fp.write('END_DWELL_POINTS\n')


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

exported = s.export(
    default_backends.SpotListBackend,
    base_dwell_time=0.1 * unit('µs'),
    length_unit=unit('µm'),
    save_impl=custom_save_impl
)

exported.save('file.txt')

