from typing import Optional, Dict, Any, Callable
import collections
import datetime
import configparser

import numpy as np

from fibomat.backend import BackendBase
from fibomat import site, units, pattern, shapes, utils
from fibomat.linalg import BoundingBox
from fibomat.pattern import Pattern
from fibomat.rasterizedpattern import RasterizedPattern


def _default_save_impl(filename: utils.PathLike, dwell_points: np.ndarray, parameters: Dict[str, Any]):
    # if parameters['base_dwell_time']:
    #     dwell_points[:, 2] /= parameters['base_dwell_time']

    header = configparser.ConfigParser()

    header_dict = collections.OrderedDict()
    header_dict['length_unit'] = f'{parameters["length_unit"]:~P}'
    header_dict['time_unit'] = f'{parameters["time_unit"]:~P}'
    header_dict['fov'] = f'{parameters["fov"].width}, {parameters["fov"].height}'
    header_dict['center'] = f'{tuple(parameters["fov"].center)}'
    header_dict['base_dwell_time'] = f'{parameters["base_dwell_time"]}'
    header_dict['total_dwell_time'] = f'{parameters["total_dwell_time"]}'
    header_dict['number_of_points'] = f'{parameters["number_of_points"]}'
    header_dict['description'] = f'{parameters["description"]}'
    header_dict['time_stamp'] = f'{parameters["time_stamp"]}'

    header['Info'] = header_dict

    with open(filename, 'w') as fp:
        header.write(fp)
        fp.write('[Points]\n')
        np.savetxt(fp, dwell_points, "%.5f %.5f %d")


class SpotListBackend(BackendBase):
    name: str = 'spotlist'

    def __init__(
            self,
            #cformat_template: Optional[Tuple[str, jinja2.Template]] = None,
            save_impl: Callable = None,
            base_dwell_time: Optional[units.TimeQuantity] = None,
            length_unit: Optional[units.LengthUnit] = None,
            time_unit: Optional[units.TimeUnit] = None,
            description: Optional[str] = None,
            **kwargs
    ):
        """Spotlist backend.

        This backend rasterize all patterns and is able to output list of dwell points in the form of ::

            x1, y1, t1
            x2, y2, t2
            ...

        The exact formatting can be controlled by a custom formatting function.

        The formatting function (called save_impl in class parameters) must have the following form ::

            def func(filename: utils.PathLike, dwell_points: np.ndarray, parameters: Dict[str, Any]):
                # open the file `filename` and save dwell_points there
                ...

        The parameter dictionary holds further metadata. For all available keys, see  `_default_save_impl` in the
        sourcefile of the spotlist backend.

        Args:
            save_impl (Callable, optional): custom formatting function.
            base_dwell_time (TimeQuantity, optional):
                if given, the dwell times are autoamtically expressed as integer multiplicands of base_dwell_time.
            length_unit (LengthQuantity, optional): length unit of points.
            time_unit (TimeQuantity, optional): time unit of dwell times.
            description (str, optional): description.
            **kwargs:
        """
        super().__init__(**kwargs)

        # if format_template:
        #     if isinstance(format_template, str):
        #         self._format_template = jinja2.Template(format_template, undefined=jinja2.StrictUndefined)
        #     elif isinstance(format_template, jinja2.Template):
        #         self._format_template = format_template
        #         # self._format_template['undefined'] = jinja2.StrictUndefined
        #     else:
        #         raise TypeError('file_format must be `str` or `file_format`')
        # else:
        #     self._format_template = default_format_template

        if save_impl:
            self._save_impl = save_impl
        else:
            self._save_impl = _default_save_impl

        self._length_unit = length_unit if length_unit else units.U_('µm')
        self._time_unit = time_unit if time_unit else units.U_('µs')

        if base_dwell_time:
            self._base_dwell_time = base_dwell_time # units.scale_to(self._time_unit, base_dwell_time)
        else:
            self._base_dwell_time = None

        self._description = description

        self._fov: Optional = None
        self._site_count: int = 0
        self._site_pos: Optional = None
        self._dwell_points = []
        self._site_bounding_box: None  # for now save fov from site here if there is only one site

    def process_site(self, new_site: site.Site):
        self._site_pos = new_site.center.vector_as(self._length_unit)

        #raise Exception(new_site.fov_bounding_box)

        self._site_bounding_box = new_site.fov_bounding_box

        self._site_count += 1

        super().process_site(new_site)

    def save(self, filename: utils.PathLike) -> None:
        if self._site_count > 1:
            print('Warning: adding merging multiple sites.')

        dwell_points = np.concatenate(self._dwell_points)

        if self._base_dwell_time:
            dwell_points[:, 2] /= units.scale_to(self._time_unit, self._base_dwell_time)

        bbox = BoundingBox.from_points(dwell_points[:, :2])
        if bbox.height==0 or bbox.width==0:
            bbox= self._site_bounding_box

        self._save_impl(
            filename, dwell_points,
            {
                'length_unit': self._length_unit, 'time_unit': self._time_unit,
                'fov': bbox, 'base_dwell_time': self._base_dwell_time, 'total_dwell_time': np.sum(dwell_points[:, 2]),
                'number_of_points': len(dwell_points), 'description': self._description,
                'time_stamp': str(datetime.datetime.now()),
            }
        )

    def _rasterize_and_add(self, ptn: pattern.Pattern[shapes.Shape]):
        dwell_points = np.array(ptn.raster_style.rasterize(
            ptn.dim_shape, ptn.mill, self._length_unit, self._time_unit
        ).dwell_points)

        # dwell_points[:, 2] *= units.scale_to(self._time_unit, ptn.mill.dwell_time)
        dwell_points[:, :2] += self._site_pos

        self._dwell_points.append(dwell_points)

    def arc_spline(self, ptn: pattern.Pattern[shapes.ArcSpline]) -> None:
        self._rasterize_and_add(ptn)

    def rasterized_points(self, ptn: pattern.Pattern[shapes.RasterizedPoints]):
        self._rasterize_and_add(ptn)

    def rasterized_pattern(self, ptn: pattern.Pattern[RasterizedPattern]):
        self._rasterize_and_add(ptn)

    def spot(self, ptn: pattern.Pattern[shapes.Spot]) -> None:
        self._rasterize_and_add(ptn)

    def line(self, ptn: pattern.Pattern[shapes.Line]) -> None:
        self._rasterize_and_add(ptn)

    def polyline(self, ptn: pattern.Pattern[shapes.Polyline]) -> None:
        self._rasterize_and_add(ptn)

    def polygon(self, ptn: pattern.Pattern[shapes.Polygon]) -> None:
        self._rasterize_and_add(ptn)

    def arc(self, ptn: pattern.Pattern[shapes.Arc]) -> None:
        self._rasterize_and_add(ptn)

    def circle(self, ptn: pattern.Pattern[shapes.Circle]) -> None:
        self._rasterize_and_add(ptn)

    def ellipse(self, ptn: pattern.Pattern[shapes.Ellipse]) -> None:
        self._rasterize_and_add(ptn)

    def rect(self, ptn: pattern.Pattern[shapes.Rect]) -> None:
        self._rasterize_and_add(ptn)




