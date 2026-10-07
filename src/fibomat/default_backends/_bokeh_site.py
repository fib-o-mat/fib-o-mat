from typing import Dict, List, Any, Optional, Union
import copy
import enum

import numpy as np

import bokeh.palettes as bp

from fibomat.layout.pattern import Pattern
from fibomat import shapes, composite_shapes
from fibomat.curve_tools import rasterize

from fibomat.linalg import Vector, DimVector, DimVectorLike
from fibomat.units import LengthUnit, scale_factor, LengthQuantity, scale_to


@enum.unique
class ShapeType(enum.Enum):
    SPOT = 'spot'
    NON_FILLED_CURVE = 'non_filled_curve'
    FILLED_CURVE = 'filled_curve'
    SITE = 'site'


def _css_to_rgb(css: str):
    css = css[1:]
    return tuple(int(css[2*i:2*(i+1)], 16) for i in range(3))


_colors = {
    'shape': bp.all_palettes['Colorblind'][4][1],
    'shape_alpha': .5,
    'site': bp.all_palettes['Colorblind'][4][0],
    'site_alpha': 0.25,
    'other': bp.all_palettes['Colorblind'][4][2],
    'other_alpha': .25
}


class BokehSite:
    plot_data_keys: List[str] = [
        'x', 'y', 'x_hole', 'y_hole',
        'site_id', 'shape_type', 'shape_prop', 'raster_style', 'mill',
        'color', 'fill_alpha', 'hatch_pattern', 'description'
    ]

    def __init__(
        self,
        site_index: int,
        plot_unit: LengthUnit,
        dim_center: DimVectorLike,
        theta: float,
        rasterize_pitch: LengthQuantity,
        cycle_colors: bool = False,
        dim_fov: Optional[DimVector] = None,
        description: Optional[str] = None,
    ):
        shape_types = [ShapeType.SPOT, ShapeType.NON_FILLED_CURVE, ShapeType.FILLED_CURVE, ShapeType.SITE]
        plot_data_proto = {key: [] for key in self.plot_data_keys}
        self.plot_data = {shape_type: copy.deepcopy(plot_data_proto) for shape_type in shape_types}

        self._plot_unit = plot_unit

        dim_center = DimVector(dim_center)
        self._center: Vector = scale_factor(plot_unit, dim_center.unit) * dim_center.vector

        self._theta = theta

        if dim_fov:
            dim_fov = DimVector(dim_fov)
            self._fov: Vector = scale_factor(plot_unit, dim_fov.unit) * dim_fov.vector
        else:
            self._fov = None

        self._rasterize_pitch = rasterize_pitch

        self._colors = copy.deepcopy(_colors)
        if cycle_colors:
            # color = RGB(*_css_to_rgb(bp.YlGnBu9[site_index % len(bp.YlGnBu9)]))
            color = bp.Viridis11[site_index % len(bp.YlGnBu9)]
            self._colors['site'] = color
            self._colors['shape'] = color

        if self._fov is not None:
            self._site_description = f'Site {site_index}, fov=({self._fov.x}, {self._fov.y}, center=({self._center.x}, {self._center.y})) {self._plot_unit:~P}'
            self._site_description += (', ' + str(description)) if description else ''
            self._label = self._site_description

            points = np.array(shapes.Rect(width=self._fov.x, height=self._fov.y, theta=self._theta, center=self._center).corners)

            self._add_plot_data(
                shape_type=ShapeType.SITE,
                x=[[list(points[:, 0])]], y=[[list(points[:, 1])]],
                site_id=str(self._label), shape_prop='Site',
                raster_style='', mill='',
                color=self._colors['site'], fill_alpha=self._colors['site_alpha'],
                description=self._site_description
            )
        else:
            # it's a annotation layer
            self._site_description = 'Annotations'
            self._label: str = self._site_description
            self._colors['shape'] = 'black'  # bp.all_palettes['Colorblind'][4][3]

    def _add_plot_data(self, shape_type: ShapeType, **kwargs):
        for key in self.plot_data_keys:
            try:
                self.plot_data[shape_type][key].append(kwargs[key])
            except KeyError:
                self.plot_data[shape_type][key].append(None)

    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        s = scale_factor(self._plot_unit, ptn.dim_shape.unit)
        shape: shapes.Spot = ptn.dim_shape.shape

        color = ptn.kwargs.get('_color', self._colors['shape'])
        if not color:
            color = self._colors['shape']

        self._add_plot_data(
            shape_type=ShapeType.SPOT,
            x=self._center.x + s * shape.center.x, y=self._center.y + s * shape.center.y,
            site_id=str(self._label), shape_prop=str(shape),
            raster_style=str(ptn.raster_style), mill=str(ptn.mill),
            color=color,
            description=f'Pattern: {ptn.description} | Shape: {ptn.dim_shape.shape.description}',
        )

    def _segmentize_pattern(self, ptn: Pattern) -> np.ndarray:
        curve = shapes.ArcSpline.from_shape(ptn.dim_shape.shape)
        s = scale_factor(self._plot_unit, ptn.dim_shape.unit)

        points = []

        for segment in curve.segments:
            if isinstance(segment, shapes.Line):
                segment: shapes.Line
                points.append([np.asarray(segment.start)])
            elif isinstance(segment, shapes.Arc):
                segment: shapes.Arc
                # set the rasterizing pitch so that the maximum distance between rasterized polygon and real arc is
                # self._rasterize_pitch
                delta = scale_to(ptn.dim_shape.unit, self._rasterize_pitch)
                pitch_squared = 4 * (segment.radius**2 - (segment.radius - delta)**2)
                if pitch_squared > 0:
                    arc_points = rasterize(
                        shapes.ArcSpline.from_shape(segment), np.sqrt(pitch_squared)
                    ).dwell_points[:, :2]
                    points.append(arc_points)
                else:
                    raise RuntimeError(
                        'Could not rasterize curve with given rasterize_pitch. Try to decrease rasterize_pitch.'
                    )
            else:
                raise RuntimeError

        if curve.segments:
            points.append([np.asarray(curve.segments[-1].end)])
        # print(points)
        points = np.concatenate(points).reshape(-1, 2)
        points *= s
        points += self._center

        return points

    def non_filled_curve(self, ptn: Pattern):
        points = self._segmentize_pattern(ptn)

        color = ptn.kwargs.get('_color', self._colors['shape'])
        if not color:
            color = self._colors['shape']

        self._add_plot_data(
            shape_type=ShapeType.NON_FILLED_CURVE,
            x=list(points[:, 0]), y=list(points[:, 1]),
            site_id=str(self._label), shape_prop=str(ptn.dim_shape.shape),
            raster_style=str(ptn.raster_style), mill=str(ptn.mill),
            color=color,
            description=f'Pattern: {ptn.description} | Shape: {ptn.dim_shape.shape.description}',
        )

    def filled_curve(self, ptn: Pattern, hatch_pattern: Optional[str] = None):
        points = self._segmentize_pattern(ptn)

        color = ptn.kwargs.get('_color', self._colors['shape'])
        if not color:
            color = self._colors['shape']

        self._add_plot_data(
            shape_type=ShapeType.FILLED_CURVE,
            x=[[list(points[:, 0])]], y=[[list(points[:, 1])]],
            site_id=str(self._label), shape_prop=str(ptn.dim_shape.shape),
            raster_style=str(ptn.raster_style), mill=str(ptn.mill),
            color=color, fill_alpha=self._colors['shape_alpha'],
            hatch_pattern=hatch_pattern,
            description=f'Pattern: {ptn.description} | Shape: {ptn.dim_shape.shape.description}',
        )

    def filled_curve_with_holes(self, ptn: Pattern[composite_shapes.HollowArcSpline], hatch_pattern: Optional[str] = None):
        boundary_points = self._segmentize_pattern(Pattern(
            dim_shape=ptn.dim_shape.shape.boundary * ptn.dim_shape.unit, mill=None, raster_style=None
        ))

        hole_points_x = []
        hole_points_y = []

        for hole in ptn.dim_shape.shape.holes:
            points = self._segmentize_pattern(Pattern(
                dim_shape=hole * ptn.dim_shape.unit, mill=None, raster_style=None
            ))
            hole_points_x.append(list(points[:, 0]))
            hole_points_y.append(list(points[:, 1]))

        color = ptn.kwargs.get('_color', self._colors['shape'])
        if not color:
            color = self._colors['shape']

        self._add_plot_data(
            shape_type=ShapeType.FILLED_CURVE,
            x=[[list(boundary_points[:, 0]), *hole_points_x]], y=[[list(boundary_points[:, 1]), *hole_points_y]],
            site_id=str(self._label), shape_prop=str(ptn.dim_shape.shape),
            raster_style=str(ptn.raster_style), mill=str(ptn.mill),
            color=color, fill_alpha=self._colors['shape_alpha'],
            hatch_pattern=hatch_pattern,
            description=f'Pattern: {ptn.description} | Shape: {ptn.dim_shape.shape.description}',
        )
