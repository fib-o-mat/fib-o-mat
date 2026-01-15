from typing import Tuple, List, Optional

import numpy as np

from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.units import LengthUnit, TimeUnit, scale_to, has_length_dim, QuantityType, scale_factor, LengthQuantity
from fibomat.shapes.arc_spline import ArcSplineCompatible, ArcSpline
from fibomat.shapes import DimShape
from fibomat.mill import Mill
from fibomat.curve_tools import inflate, deflate, rasterize, smooth
from fibomat.curve_tools.smooth import NonSmoothableError
from fibomat.optimize import optimize_curve
from fibomat.raster_styles.scansequence import ScanSequence, _apply_scan_sequence
from fibomat.raster_styles.one_d.curve import Curve
from fibomat.raster_styles.zero_d.prerasterized import PreRasterized


def _print_optimize_result(alpha, flux_matrix, nominal_flux):
    print('mean_alpha =', np.mean(alpha))
    print('std_alpha =', np.std(alpha))
    print('min_alpha = ', np.min(alpha))
    print('max_alpha = ', np.max(alpha))

    flux = flux_matrix @ alpha
    print('mean_flux =', np.mean(flux))
    print('std_flux =', np.std(flux))
    print('min_flux = ', np.min(flux))
    print('max_flux = ', np.max(flux))
    print('nominal_flux =', nominal_flux)


class ContourParallel(RasterStyle):

    _inwards = 'inwards'
    _outwards = 'outwards'

    def __init__(
        self,
        offset_pitch: LengthQuantity,
        offset_direction: str,
        start_direction: str,
        scan_sequence: ScanSequence,
        line_style: RasterStyle,
        include_original_curve: bool,
        optimize: bool,
        *,
        offset_steps: Optional[int] = None,
        offset_distance: Optional[LengthQuantity] = None,
        smooth_radius: Optional[LengthQuantity] = None
    ):
        self._scan_sequence = scan_sequence

        self._line_style = line_style

        if not has_length_dim(offset_pitch):
            raise ValueError('offset_pitch must have [length] as dimension')

        if offset_pitch <= 0.:
            raise ValueError('offset_pitch <= 0.')
        self._offset_pitch = offset_pitch

        if offset_direction not in [self._inwards, self._outwards]:
            raise ValueError(f'offset_direction must be "{self._inwards}" or "{self._outwards}".')

        self._offset_direction = offset_direction
        self._offset_steps = offset_steps
        self._offset_distance = offset_distance

        if start_direction not in [self._inwards, self._outwards]:
            raise ValueError(f'start_direction must be "{self._inwards}" or "{self._outwards}".')
        self._start_direction = start_direction

        self._include_original_curve = include_original_curve

        self._smooth_radius = smooth_radius
        self._optimize = optimize

    @property
    def dimension(self) -> int:
        return 2

    def _do_smooth(self, spline: ArcSpline, curve_unit: LengthUnit):
        if self._smooth_radius:
            try:
                return smooth(spline, self._smooth_radius.to(curve_unit).m)
            except NonSmoothableError as e:
                print('Warning: could not smooth spline.')
                return spline
        else:
            return spline

    def _gen_offset_curves(self, base_curve: ArcSpline, curve_unit: LengthUnit) -> List[ArcSpline]:

        curves = []
        if self._include_original_curve:
            curves.append(base_curve)

        if self._offset_direction == self._inwards:
            curves.extend(deflate(
                base_curve,
                self._offset_pitch.to(curve_unit).m,
                n_steps=self._offset_steps,
                distance=self._offset_distance.to(curve_unit).m if self._offset_distance else None
            ))

            if self._start_direction == self._outwards:
                curves = list(reversed(curves))

        else:
            curves.extend(inflate(
                base_curve,
                self._offset_pitch.to(curve_unit).m,
                n_steps=self._offset_steps,
                distance=self._offset_distance.to(curve_unit).m if self._offset_distance else None
            ))

            if self._start_direction == self._inwards:
                curves = list(reversed(curves))

        return [self._do_smooth(curve, curve_unit) for curve in curves]

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:

        assert isinstance(dim_shape._shape, ArcSplineCompatible), f"{dim_shape._shape} is not ArcSplineCompatible and thus cannot be rasterized countour parallel."
        
        offsetted_curves = self._gen_offset_curves(dim_shape._shape.to_arc_spline(), dim_shape.unit)

        if not self._optimize:
            return _apply_scan_sequence(
                filling_rows=offsetted_curves,
                filling_rows_length_unit=dim_shape.unit,
                scan_sequence=self._scan_sequence,
                line_style=self._line_style,
                mill=mill,
                out_length_unit=out_length_unit,
                out_time_unit=out_time_unit
            )
        else:
            if not (
                isinstance(self._line_style, Curve) and self._line_style.scan_sequence == ScanSequence.CONSECUTIVE
            ):
                raise RuntimeError(
                    'Only one_d.Curve with ScanSequence.CONSECUTIVE scan sequence is supported if optimize == True.'
                )

            nominal_flux = mill.beam.nominal_flux_per_spot_on_line(self._line_style.pitch).to('ions / nm**2 / µs')
            hint = None

            rasterized_curves = []

            for i_curve, curve in enumerate(offsetted_curves):
                print(i_curve, '/', len(offsetted_curves) - 1)

                dwell_points, hint, flux_matrix = optimize_curve(
                    (curve, dim_shape.unit), self._line_style.pitch, mill.beam, nominal_flux, hint, info=True
                )
                _print_optimize_result(dwell_points[:, 2], flux_matrix, nominal_flux)

                rasterized_curves.append(RasterizedPoints(dwell_points, dim_shape.obj.is_closed))

            return _apply_scan_sequence(
                filling_rows=rasterized_curves,
                filling_rows_length_unit=dim_shape.unit,
                scan_sequence=self._scan_sequence,
                line_style=PreRasterized(),
                mill=mill,
                out_length_unit=out_length_unit,
                out_time_unit=out_time_unit
            )
