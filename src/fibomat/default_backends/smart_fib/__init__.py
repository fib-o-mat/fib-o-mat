from typing import Optional, Tuple
import re
from fibomat.linalg.vectors import VectorLike

import numpy as np

import xmltodict

from fibomat.backend.backendbase import shape_type
from fibomat.linalg import BoundingBox, Vector
from fibomat.backend import BackendBase
from fibomat.layout import Site, Pattern
from fibomat.shapes import Polyline, Spot, Shape
from fibomat.mill import MillBase
from fibomat.raster_styles import two_d, one_d, zero_d
from fibomat.units import scale_factor, scale_to, U_, Q_
from fibomat import utils

from custom_backends.smart_fib_ely.schemas import ElyFileSchema
from custom_backends.smart_fib_ely.models import ElyFile, ElyLayer, ElyExposure, ElyProbe, ElyGIS, ElyArc, ElyPoint, ElyPolyline

class EArc(Shape):
    def __init__(self, radii: Tuple[float, float], angles: Tuple[float, float] = (0, 2*np.pi), center: Optional[VectorLike] = None, description: str = None):
        super().__init__(description)

        self._r_inner, self. _r_outer = radii
        self._alpha_1, self._alpha_2 = angles

        self._center = Vector(center) if center is not None else Vector()

    def __repr__(self):
        return f'EArc(radii=({self._r_inner}, {self. _r_outer}), angles=({self._alpha_1}, {self._alpha_2}, center={self._center}))'

    @property
    def center(self) -> Vector:
        return self._center

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self._center-(self._r_outer, self._r_outer), self._center + (self._r_outer, self._r_outer))

    @property
    def is_closed(self) -> bool:
        return True

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        self._center += Vector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        # TODO: rotate alphas
        self._center = self._center.rotated(theta)
        
    def _impl_scale(self, fac: float) -> None:
        self._center *= float(fac)
        self._r_outer *= float(fac)
        self._r_inner *= float(fac)

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        self._center = self._center.mirrored(mirror_axis)


class ElyBackendMeta(BackendBase):
    @shape_type(EArc)
    def arc(self, ptn: Pattern[EArc]):
        raise NotImplementedError


# class ElyOnlyGeometryBackend(ElyOnlyGeometryBackendMeta):
#     def __init__(self, description: Optional[str]):
#         super().__init__()

#         self._layers = []
#         self._layer_index = 0

#     # def print(self):
#     #     ely_file = ElyFile('Elyfile', self._layers)
#     #     print(xmltodict.unparse(ElyFileSchema().dump(ely_file), encoding='iso-8859-1', pretty=True))

#     def save(self, filename: utils.PathLike) -> None:

#         if len(self._layers) > 1:
#             raise RuntimeError('Only one layer supported currently. Consider using export_multi.')

#         ely_file = ElyFile('Elyfile', self._layers)

#         # import json
#         # print(json.dumps(ElyFileSchema().dump(ely_file), indent=4))

#         raw_xml = xmltodict.unparse(ElyFileSchema().dump(ely_file), encoding='iso-8859-1', pretty=True)

#         xml = raw_xml.replace('C/m&amp;#178;', 'C/m&#178;')
#         xml = re.sub(r'#-#\d+#-#', '', xml)
#         # xml = ElementTree.parse(raw_xml)

#         with open(filename, 'w') as fp:
#             fp.write(xml)

#     def process_site(self, new_site: Site) -> None:
#         if not np.allclose(new_site.center.vector, 0):
#             raise RuntimeError('Site must be located at (0, 0)')

#         fov = scale_to(U_('µm'), new_site.square_fov.vector[0] * new_site.square_fov.unit)

#         self._layers.append(ElyLayer(f'Layer_{self._layer_index}', fov))
#         self._layer_index += 1

#         super().process_site(new_site)

    


class ElyMill(MillBase):
    def __init__(self, area_dose: Q_, dwell_time, probe: ElyProbe):
        if not area_dose.check('[current] / [length]**2 * [time]'):
            raise ValueError('"area_dose" must be an area dose (e.g. coulomb/m**2')
        if not dwell_time.check('[time]'):
            raise ValueError('"time" must be given in time units')

        super().__init__(area_dose=area_dose, dwell_time=dwell_time, probe=probe)


class ElyBackend(ElyBackendMeta):
    def __init__(self, only_geometry: Optional[bool] = False, description: Optional[str] = None):
        super().__init__()

        self._only_geometry = only_geometry

        self._layers = []
        self._layer_index = 0

    def save(self, filename: utils.PathLike) -> None:

        if len(self._layers) > 1:
            raise RuntimeError('Only one layer supported currently. Consider using export_multi.')

        ely_file = ElyFile('Elyfile', self._layers)

        # import json
        # print(json.dumps(ElyFileSchema().dump(ely_file), indent=4))

        raw_xml = xmltodict.unparse(ElyFileSchema().dump(ely_file), encoding='iso-8859-1', pretty=True)

        xml = raw_xml.replace('C/m&amp;#178;', 'C/m&#178;')
        xml = re.sub(r'#-#\d+#-#', '', xml)
        # xml = ElementTree.parse(raw_xml)

        with open(filename, 'w') as fp:
            fp.write(xml)

    def process_site(self, new_site: Site) -> None:
        if not np.allclose(new_site.center.vector, 0):
            raise RuntimeError('Site must be located at (0, 0)')

        fov = scale_to(U_('µm'), new_site.square_fov.vector[0] * new_site.square_fov.unit)

        self._layers.append(ElyLayer(f'Layer_{self._layer_index}', fov))
        self._layer_index += 1

        super().process_site(new_site)

    def _make_exposure(self, ptn: Pattern):
        if self._only_geometry:
            return None
        else:

            if not isinstance(ptn.raster_style, two_d.LineByLine):
                raise RuntimeError('only two-dim LineByLine raster style is currently allowed')
            
            probe = ptn.mill['probe']
            gis = ElyGIS()

            pixel_spacing = ptn.raster_style.line_style.pitch
            track_spacing = ptn.raster_style.line_pitch

            return ElyExposure(ptn.mill['dwell_time'], ptn.mill['area_dose'], pixel_spacing, track_spacing, probe, gis)

    # def polyline(self, ptn: Pattern[Polyline]) -> None:
    #     raise NotImplemented
        
    # def spot(self, ptn: Pattern[Spot]) -> None:
    #     if not isinstance(ptn.raster_style, SingleSpot):
    #         raise ValueError('raster_style must be SingleSpot for a spot.')

    #     scale_fac = scale_factor(U_('µm'), ptn.dim_shape.unit)
    #     spot = scale_fac * ptn.dim_shape.shape.position

    #     probe = ptn.mill['probe']
    #     gis = ElyGIS()

    #     exposure = ElyExposure(ptn.mill['dwell_time'], ptn.mill['area_dose'], ptn.mill['spacing'], probe, gis)


    #     self._layers[-1].add_point(spot, exposure)

    # def arc(self, ptn: Pattern[EArc]):

    def polyline(self, ptn: Pattern[Polyline]) -> None:
        scale_fac = scale_factor(U_('µm'), ptn.dim_shape.unit)
        polyline = scale_fac * ptn.dim_shape.shape.points

        if self._only_geometry:
            self._layers[-1].geoms.append(ElyPolyline(polyline))
        else:
            raise NotImplementedError()

    def spot(self, ptn: Pattern[Spot]) -> None:
        scale_fac = scale_factor(U_('µm'), ptn.dim_shape.unit)
        spot = scale_fac * ptn.dim_shape.shape.position
        
        if self._only_geometry:
            self._layers[-1].geoms.append(ElyPoint(spot))
        else:
            raise NotImplementedError()

    def arc(self, ptn: Pattern[EArc]) -> None:
        scale_fac = scale_factor(U_('µm'), ptn.dim_shape.unit)
        pos = scale_fac * ptn.dim_shape.shape._center
        r1 = scale_fac * ptn.dim_shape.shape._r_inner
        r2 = scale_fac * ptn.dim_shape.shape._r_outer

        exposure = self._make_exposure(ptn)

        self._layers[-1].geoms.append(
                ElyArc(
                    pos.x, pos.y,
                    r1, r2,
                    np.rad2deg(ptn.dim_shape.shape._alpha_1), np.rad2deg(ptn.dim_shape.shape._alpha_2),
                    exposure=exposure
                )
            )
        

class ElyOnlyGeometryBackend(ElyBackend):
    def __init__(self, description: Optional[str] = None):
        super().__init__(only_geometry=True, description=description)

