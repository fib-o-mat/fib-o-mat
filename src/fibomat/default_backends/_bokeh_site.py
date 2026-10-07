"""Provides the :class:`BokehSite` class, which collects the plot data of the patterns of one site."""
from __future__ import annotations

import copy
import enum
import typing as t

import bokeh.palettes as bp
import numpy as np

from fibomat import composite_shapes, shapes
from fibomat.curve_tools import rasterize_with_const_error
from fibomat.layout.pattern import Pattern
from fibomat.linalg import DimBoundingBox, DimVector, DimVectorLike, Vector
from fibomat.mill import Mill
from fibomat.units import DimFloat, LengthUnit, scale_factor, scale_to


__all__ = ['BokehSite', 'ShapeType']


@enum.unique
class ShapeType(enum.Enum):
    """Kinds of plotted graphical elements (each has its own bokeh data source)."""
    SPOT = 'spot'
    NON_FILLED_CURVE = 'non_filled_curve'
    FILLED_CURVE = 'filled_curve'
    SITE = 'site'


_COLORS = {
    'shape': bp.all_palettes['Colorblind'][4][1],
    'shape_alpha': .5,
    'site': bp.all_palettes['Colorblind'][4][0],
    'site_alpha': 0.25,
}

_ANNOTATION_COLOR = 'black'

NO_HATCH = ' '
"""Hatch pattern of areas which are not hatched (bokeh does not accept None)."""


def describe_mill(mill: t.Any) -> str:
    """Short text which describes a mill (used in the tooltips).

    Args:
        mill (MillBase, optional): mill

    Returns:
        str
    """
    if isinstance(mill, Mill):
        return f'dwell time {mill.dwell_time.quantity:~P}, repeats {mill.repeats}'
    return '' if mill is None else str(mill)


class BokehSite:
    """Plot data of one site (or of the annotation layer).

    All coordinates are converted to the unit of the plot and shifted by the center of the site. Rotated sites do not
    need special treatment: the patterns of a site are rotated together with it.
    """

    plot_data_keys: t.List[str] = [
        'x', 'y', 'x_hole', 'y_hole',
        'site_id', 'shape_type', 'shape_prop', 'raster_style', 'mill',
        'color', 'fill_alpha', 'hatch_pattern', 'description'
    ]
    """Columns of the bokeh data sources."""

    def __init__(
        self,
        site_index: int,
        plot_unit: LengthUnit,
        dim_center: DimVectorLike,
        rasterize_pitch: DimFloat[t.Any],
        *,
        cycle_colors: bool = False,
        fov_bounding_box: t.Optional[DimBoundingBox] = None,
        description: t.Optional[str] = None,
        annotation: bool = False,
    ):
        """
        Args:
            site_index (int): index of the site (used for the label and the color)
            plot_unit (LengthUnit): length unit of the plot
            dim_center (DimVectorLike): center of the site
            rasterize_pitch (DimFloat): maximum distance between an arc and its polygonal approximation in the plot
            cycle_colors (bool): if True, every site gets its own color
            fov_bounding_box (DimBoundingBox, optional): field of view of the site; the outline of the site is only
                plotted if it is given.
            description (str, optional): description of the site
            annotation (bool): if True, this is the annotation layer (black shapes, no site outline).
        """
        plot_data_proto: t.Dict[str, t.List[t.Any]] = {key: [] for key in self.plot_data_keys}
        self.plot_data: t.Dict[ShapeType, t.Dict[str, t.List[t.Any]]] = {
            shape_type: copy.deepcopy(plot_data_proto) for shape_type in ShapeType
        }

        self._plot_unit = plot_unit
        self._site_index = site_index
        self._rasterize_pitch = rasterize_pitch

        dim_center = DimVector(dim_center)
        self._center: Vector = dim_center.vector_as(plot_unit)

        self._colors = copy.deepcopy(_COLORS)
        if cycle_colors:
            color = bp.Viridis11[site_index % len(bp.Viridis11)]
            self._colors['site'] = color
            self._colors['shape'] = color

        self.is_annotation = annotation

        if annotation:
            self._label = 'Annotations'
            self._colors['shape'] = _ANNOTATION_COLOR
        else:
            self._label = f'Site {site_index}'
            site_description = f'{self._label}, center=({self._center.x:.6g}, {self._center.y:.6g}) {plot_unit:~P}'

            if fov_bounding_box is not None:
                corners = np.array([
                    np.asarray(corner.vector_as(plot_unit)) for corner in fov_bounding_box.corners
                ])
                fov = fov_bounding_box.upper_right.vector_as(plot_unit) - fov_bounding_box.lower_left.vector_as(
                    plot_unit
                )
                site_description += f', fov=({fov.x:.6g}, {fov.y:.6g}) {plot_unit:~P}'
                if description:
                    site_description += f', {description}'

                self._label = site_description
                self._add_plot_data(
                    shape_type=ShapeType.SITE,
                    x=[[list(corners[:, 0])]], y=[[list(corners[:, 1])]],
                    site_id=self._label, shape_prop='Site',
                    raster_style='', mill='',
                    color=self._colors['site'], fill_alpha=self._colors['site_alpha'],
                    description=site_description
                )

    @property
    def label(self) -> str:
        """Label of the site in the legend."""
        return self._label

    @property
    def site_index(self) -> int:
        """Index of the site (-1 for the annotation layer)."""
        return self._site_index

    @property
    def center(self) -> Vector:
        """Center of the site in the unit of the plot."""
        return self._center

    def _add_plot_data(self, shape_type: ShapeType, **kwargs: t.Any) -> None:
        for key in self.plot_data_keys:
            self.plot_data[shape_type][key].append(kwargs.get(key))

    def _color(self, ptn: Pattern) -> str:
        return ptn.kwargs.get('_color') or self._colors['shape']

    @staticmethod
    def _description(ptn: Pattern) -> str:
        return f'Pattern: {ptn.description} | Shape: {ptn.dim_shape.shape.description}'

    def _pattern_info(self, ptn: Pattern) -> t.Dict[str, str]:
        return dict(
            site_id=self._label, shape_prop=str(ptn.dim_shape.shape),
            raster_style=str(ptn.raster_style), mill=describe_mill(ptn.mill),
            description=self._description(ptn),
        )

    def _outline(self, curve: shapes.ArcSpline, unit: t.Any, close: bool = False) -> np.ndarray:
        """Points of a polygon/polyline which approximates a curve, in plot coordinates.

        Args:
            curve (ArcSpline): curve in the length unit `unit`
            unit (LengthUnit): length unit of the curve
            close (bool): if True, the first point is repeated at the end of a closed curve (needed to draw it as a
                line; polygons are closed by bokeh).

        Returns:
            np.ndarray: points with shape (n, 2)
        """
        max_distance = scale_to(unit, self._rasterize_pitch)
        points = np.array(rasterize_with_const_error(curve, max_distance).points, dtype=float)
        if close and curve.is_closed:
            points = np.vstack((points, points[:1]))
        return points * scale_factor(self._plot_unit, unit) + np.asarray(self._center)

    def _segmentize_pattern(self, ptn: Pattern, close: bool = False) -> np.ndarray:
        curve = shapes.ArcSpline.from_shape(ptn.dim_shape.shape)
        return self._outline(curve, ptn.dim_shape.unit, close)

    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        """Add a spot."""
        scale = scale_factor(self._plot_unit, ptn.dim_shape.unit)
        position = ptn.dim_shape.shape.center

        self._add_plot_data(
            shape_type=ShapeType.SPOT,
            x=self._center.x + scale * position.x, y=self._center.y + scale * position.y,
            color=self._color(ptn),
            **self._pattern_info(ptn),
        )

    def non_filled_curve(self, ptn: Pattern) -> None:
        """Add a curve which is drawn as a line."""
        points = self._segmentize_pattern(ptn, close=True)

        self._add_plot_data(
            shape_type=ShapeType.NON_FILLED_CURVE,
            x=list(points[:, 0]), y=list(points[:, 1]),
            color=self._color(ptn),
            **self._pattern_info(ptn),
        )

    def filled_curve(self, ptn: Pattern, hatch_pattern: str = NO_HATCH) -> None:
        """Add a closed curve which is drawn filled."""
        points = self._segmentize_pattern(ptn)

        self._add_plot_data(
            shape_type=ShapeType.FILLED_CURVE,
            x=[[list(points[:, 0])]], y=[[list(points[:, 1])]],
            color=self._color(ptn), fill_alpha=self._colors['shape_alpha'],
            hatch_pattern=hatch_pattern,
            **self._pattern_info(ptn),
        )

    def filled_curve_with_holes(
        self, ptn: Pattern[composite_shapes.HollowArcSpline], hatch_pattern: str = NO_HATCH
    ) -> None:
        """Add a filled shape with holes."""
        unit = ptn.dim_shape.unit
        hollow = ptn.dim_shape.shape

        boundary_points = self._outline(hollow.boundary, unit)
        hole_points = [self._outline(hole, unit) for hole in hollow.holes]

        self._add_plot_data(
            shape_type=ShapeType.FILLED_CURVE,
            x=[[list(boundary_points[:, 0]), *(list(points[:, 0]) for points in hole_points)]],
            y=[[list(boundary_points[:, 1]), *(list(points[:, 1]) for points in hole_points)]],
            color=self._color(ptn), fill_alpha=self._colors['shape_alpha'],
            hatch_pattern=hatch_pattern,
            **self._pattern_info(ptn),
        )
