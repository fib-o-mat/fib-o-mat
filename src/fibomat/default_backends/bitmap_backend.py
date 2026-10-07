from fibomat.shapes.polyline import Polyline
from typing import Optional, Dict
import io


import numpy as np

try:
    import svgwrite
    import svgwrite.container
    import svgwrite.shapes
    import svgwrite.extensions
    
    from PIL import Image, ImageDraw

except ModuleNotFoundError as error:
    raise RuntimeError(
        "You need to install the package with 'exporting' as optional dependency to be able to use the svg backend."
    ) from error

from fibomat.backend import BackendBase
from fibomat.layout.site import Site
from fibomat.units import U_, Q_, scale_factor
from fibomat.utils import PathLike
from fibomat.layout.pattern import Pattern
from fibomat.shapes import Rect, Polygon, Circle
from fibomat.linalg import DimBoundingBox, scale, translate, Vector, DimVector


class BitmapBackend(BackendBase):
    def __init__(self, pixel_size: Q_ = Q_('2 nm'), description: Optional[str] = None):
        super().__init__()

        self._description = description
        self._pixel_size = pixel_size

        self._site = None
        # self._current_layer: Optional[Dict] = None
        self._center: Optional[Vector] = None

        self._image: Optional = None

        self._total_bounding_box = None

    def save(self, filename: str):
        if self._image:
            self._image.save(filename)
        else:
            raise RuntimeError('No site added.')

    def image(self):
        return self._image

    def process_site(self, new_site: Site) -> None:
        if self._image:
            raise RuntimeError('Only one site is allowed in bitmap backend.')

        self._site = new_site

        self._shift = DimVector(new_site.fov[0] / 2, new_site.fov[1] / 2)

        width = int(np.rint((new_site.fov[0] / self._pixel_size).to_base_units()))
        height = int(np.rint((new_site.fov[1] / self._pixel_size).to_base_units()))
        self._image = Image.new("L", (width, height), 'black')

        super().process_site(new_site)

    def _to_pixel(self, value):
        return int(np.rint((value / self._pixel_size).to_base_units()))

    # def _pixel_scale_factor(self, other_unit):
    #     return scale_factor(self._pixel_scale, other_unit) / self._pixel_scale.m

    # def _scale_and_shift_shape(self, ptn: Pattern):
    #     fak = self._pixel_scale_factor(ptn.dim_shape.unit)

    #     return ptn.dim_shape.shape.transformed(scale(fak) | translate(self._current_layer_center))

    def rect(self, ptn: Pattern[Rect]) -> None:
        # return super().rect(ptn)
        ptn = ptn.translated(self._shift)

        # print(self._image.size)

        points = ptn.dim_shape.shape.to_arc_spline().vertices[:, :2]
        pixel_points = [(self._to_pixel(x * ptn.dim_shape.unit), self._image.size[1] -self._to_pixel(y * ptn.dim_shape.unit))  for x, y in points]

        # print(pixel_points)

        draw = ImageDraw.Draw(self._image)
        draw.polygon(pixel_points, 'white')


    # def polygon(self, ptn: Pattern[Polygon]) -> None:
    #     return super().polygon(ptn)