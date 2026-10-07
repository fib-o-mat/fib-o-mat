"""Provides the :class:`StepAndRepeatBackend`, which exports a layout as NPVE step and repeat file (XML).

Example::

    mill = NPVEMill(dwell_time=1. * unit('µs'), repeats=2)
    exported = layout.export(StepAndRepeatBackend)
    exported.save('patterns.xml')

.. note:: The file format is defined by the schemas in ``common_schemas.py`` and ``sar_schemas.py``; the tests compare
    the generated files with reference files, so that the format does not change by accident.
"""
from __future__ import annotations

import typing as t
import warnings

import numpy as np
import xmltodict

from fibomat.backend import BackendBase
from fibomat.composite_shapes import HollowArcSpline, Ring
from fibomat.curve_tools.rasterize import rasterize_with_const_error
from fibomat.default_backends.npve.step_and_repeat.common_models import FIBShape, ShapeTexture
from fibomat.default_backends.npve.step_and_repeat.npve_mill import NPVEMill
from fibomat.default_backends.npve.step_and_repeat.outline import (
    LineByLineOutlined, OutlineAlignement, OutlineNodeStyle, OutlineScanStyle
)
from fibomat.default_backends.npve.step_and_repeat.sar_models import SaRFile, SaRSharedShapes, SaRSite
from fibomat.default_backends.npve.step_and_repeat.sar_schemas import SaRFileSchema
from fibomat.layout.pattern import Pattern
from fibomat.layout.site import Site
from fibomat.linalg import Vector
from fibomat.shapes import Circle, DimShape, Ellipse, Line, Polygon, Polyline, Rect, Spot
from fibomat.units import DimFloat, has_length_dim, scale_factor, unit
from fibomat.utils import PathLike


__all__ = [
    'StepAndRepeatBackend', 'NPVEMill', 'LineByLineOutlined', 'OutlineAlignement', 'OutlineScanStyle',
    'OutlineNodeStyle'
]


_MICRON = unit('µm')

_NODE_FIRST = 0
_NODE_NEXT = 1
_NODE_LAST = 129
"""NPVE node types: first node, further nodes and last node (which closes the shape)."""


class StepAndRepeatBackend(BackendBase):
    """Exports a layout as NPVE step and repeat file.

    All patterns must use an :class:`NPVEMill` and the raster styles
    :class:`~fibomat.raster_styles.zero_d.SingleSpot`, :class:`~fibomat.raster_styles.one_d.Curve` (scan sequence
    consecutive) or :class:`~fibomat.raster_styles.two_d.LineByLine` (also :class:`LineByLineOutlined`). Supported
    shapes are spots, lines, polylines, polygons, rectangles, circles, ellipses (not rotated), rings and hollow arc
    splines (which are approximated by polygons).

    Patterns with the keyword argument ``use_bitmap`` are exported as bitmap texture (this needs the bitmap backend).
    """

    def __init__(
        self,
        share_patterns: bool = False,
        skip_empty_sites: bool = True,
        description: t.Optional[str] = None,
        approximation_error: t.Optional[DimFloat[t.Any]] = None,
    ):
        """
        Args:
            share_patterns (bool): if True, only the first site contains patterns, which are shared by all sites (the
                other sites must be empty).
            skip_empty_sites (bool): if True, empty sites are ignored, otherwise they raise an error (NPVE does not do
                what everybody expects if an empty site is in a step and repeat list). Ignored (False) if
                `share_patterns` is True.
            description (str, optional): description
            approximation_error (DimFloat, optional): maximum distance between the arcs of hollow arc splines and
                their polygonal approximation, default ``0.0025 * unit('µm')``.

        Raises:
            TypeError: Raised if approximation_error is no dimensioned value.
            ValueError: Raised if approximation_error is no positive length.
        """
        super().__init__(description)

        if approximation_error is None:
            approximation_error = 0.0025 * _MICRON
        if not isinstance(approximation_error, DimFloat):
            raise TypeError('approximation_error must be a dimensioned value like 0.0025 * unit("µm").')
        if not has_length_dim(approximation_error) or not approximation_error.magnitude > 0.:
            raise ValueError('approximation_error must be a positive length.')
        self._approximation_error = approximation_error.m_as('µm')

        self._warned_about_spot_dose = False

        self._share_patterns = bool(share_patterns)
        self._shared_patterns: t.Union[SaRSharedShapes, t.List[t.Any]] = []

        self._skip_empty_sites = False if self._share_patterns else bool(skip_empty_sites)

        self._site_index = 0
        self._shape_index = 10000

        self._current_site: t.Optional[SaRSite] = None
        self._sites: t.List[SaRSite] = []

    # sites

    def process_site(self, new_site: Site) -> None:
        if self._share_patterns:
            if self._site_index != 0 and not new_site.empty:
                raise ValueError("Only first site should contain patterns.")
        elif new_site.empty:
            if self._skip_empty_sites:
                return
            raise ValueError(
                "Site may not be empty (NPVE is not doing what everybody would expect "
                "if an empty site is encountered in an step and repeat list)."
            )

        if self._sites:
            last_site_pos = self._sites[-1].center
        else:
            last_site_pos = new_site.center.vector_as(_MICRON)

        self._current_site = SaRSite(self._site_index, new_site, last_site_pos)
        self._sites.append(self._current_site)
        self._site_index += 1

        if self._site_index == 1 or not self._share_patterns:
            super().process_site(new_site)

            if self._share_patterns:
                self._shared_patterns = SaRSharedShapes(self._current_site.shapes["shapes_list"])
                self._current_site.shapes["shapes_list"] = []

    def process_pattern(self, ptn: Pattern) -> None:
        if "use_bitmap" in ptn.kwargs:
            self._add_bitmap(ptn)
        else:
            super().process_pattern(ptn)

    def _add_bitmap(self, ptn: Pattern) -> None:
        """Add a pattern as bitmap texture of a rectangle (this needs the bitmap backend)."""
        from fibomat.default_backends.bitmap_backend import BitmapBackend  # pylint: disable=import-outside-toplevel
        from fibomat.layout import Layout  # pylint: disable=import-outside-toplevel

        bbox = ptn.bounding_box
        layout = Layout()
        site = layout.create_site(dim_position=bbox.center, dim_fov=(bbox.width, bbox.height))
        site += ptn
        bitmap = layout.export(BitmapBackend).image()

        texture = ShapeTexture(bitmap, (bbox.width, bbox.height), ptn.raster_style)

        half_width = bbox.width.m_as("µm") / 2
        half_height = bbox.height.m_as("µm") / 2
        center = Vector(bbox.center[0].m_as("µm"), bbox.center[1].m_as("µm"))

        self._add_shape(
            "TRectangle", center,
            [
                (center + (-half_width, -half_height), _NODE_FIRST),
                (center + (half_width, -half_height), _NODE_NEXT),
                (center + (half_width, half_height), _NODE_NEXT),
                (center + (-half_width, half_height), _NODE_LAST),
            ],
            ptn, shape_texture=texture
        )

    # file

    def _to_str(self) -> str:
        """The step and repeat file as XML string.

        Raises:
            RuntimeError: Raised if there are no sites.
        """
        if not self._sites:
            raise RuntimeError("Step and repeat list does not contain any sites.")

        if self._share_patterns:
            sar_file = SaRFile(self._sites, self._shared_patterns)
        else:
            sar_file = SaRFile(self._sites)
        return xmltodict.unparse(SaRFileSchema().dump(sar_file), encoding="iso-8859-1", pretty=True)

    def print(self) -> None:
        """Print the XML of the step and repeat file."""
        print(self._to_str())

    def save(self, filename: PathLike) -> None:
        """Save the step and repeat file.

        Args:
            filename (PathLike): filename

        Raises:
            RuntimeError: Raised if there are no sites.
        """
        text = self._to_str()
        with open(filename, "w", encoding="iso-8859-1") as file:
            file.write(text)

    # shapes

    def _add_shape(
        self, class_: str, rotation_center: t.Any, nodes: t.List[t.Tuple[t.Any, int]], ptn: Pattern, **kwargs: t.Any
    ) -> None:
        """Add a shape (coordinates in µm) to the current site."""
        if self._current_site is None:
            raise RuntimeError('Patterns must be added to a site.')

        self._current_site.add_fib_shape(
            FIBShape(
                class_=class_, id=self._shape_index, rotation_center=rotation_center, nodes=nodes, mill=ptn.mill,
                raster_style=ptn.raster_style, **kwargs
            )
        )
        self._shape_index += 1

    @staticmethod
    def _scaled(ptn: Pattern) -> t.Any:
        """The shape of a pattern in µm."""
        return ptn.dim_shape.shape.scaled(scale_factor(_MICRON, ptn.dim_shape.unit))

    @staticmethod
    def _derived(ptn: Pattern, shape: t.Any, **kwargs: t.Any) -> Pattern:
        """A pattern with another shape (in the unit of the pattern) and the mill and raster style of `ptn`."""
        return Pattern(DimShape(shape, ptn.dim_shape.unit), ptn.mill, ptn.raster_style, **kwargs)

    def spot(self, ptn: Pattern[Spot]) -> None:
        if not self._warned_about_spot_dose:
            warnings.warn("Check the spot dose in NPVE as the calculation includes some magic numbers.", stacklevel=2)
            self._warned_about_spot_dose = True

        scaled_spot = self._scaled(ptn)
        self._add_shape("TFIBSpot", scaled_spot.center, [(scaled_spot.center, _NODE_FIRST)], ptn)

    def rect(self, ptn: Pattern[Rect]) -> None:
        rect: Rect = ptn.dim_shape.shape

        if ptn.raster_style.dimension == 1:
            self.polyline(self._derived(ptn, Polyline([*rect.corners, rect.corners[0]])))
        else:
            self.polygon(self._derived(
                ptn, Polygon(rect.corners, description=rect.description), description=ptn.description, **ptn.kwargs
            ))

    def polygon(self, ptn: Pattern[Polygon]) -> None:
        if ptn.raster_style.dimension == 1:
            points = ptn.dim_shape.shape.points
            self.polyline(self._derived(ptn, Polyline([*points, points[0]])))
        else:
            scaled_poly: Polygon = self._scaled(ptn)

            nodes = [(scaled_poly.points[0], _NODE_FIRST)]
            nodes.extend((point, _NODE_NEXT) for point in scaled_poly.points[1:-1])
            nodes.append((scaled_poly.points[-1], _NODE_LAST))

            self._add_shape("TPolygon", scaled_poly.center, nodes, ptn)

    def polyline(self, ptn: Pattern[Polyline]) -> None:
        is_line = len(ptn.dim_shape.shape.points) == 2
        scaled_poly: Polyline = self._scaled(ptn)

        nodes = [(scaled_poly.points[0], _NODE_FIRST)]

        if is_line:
            nodes.append((scaled_poly.points[1], _NODE_LAST))
        else:
            nodes.extend((point, _NODE_NEXT) for point in scaled_poly.points[1:])

        self._add_shape("TLine" if is_line else "TPolyline", scaled_poly.center, nodes, ptn)

    def line(self, ptn: Pattern[Line]) -> None:
        line: Line = ptn.dim_shape.shape
        self.polyline(self._derived(ptn, Polyline([line.start, line.end])))

    def circle(self, ptn: Pattern[Circle]) -> None:
        scaled_circle: Circle = self._scaled(ptn)
        r = scaled_circle.r  # pylint: disable=invalid-name
        center = scaled_circle.center

        self._add_shape(
            "TEllipse", center,
            [
                (center + (-r, -r), _NODE_FIRST),
                (center + (r, -r), _NODE_NEXT),
                (center + (r, r), _NODE_NEXT),
                (center + (-r, r), _NODE_LAST),
            ],
            ptn
        )

    def ellipse(self, ptn: Pattern[Ellipse]) -> None:
        scaled_ellipse: Ellipse = self._scaled(ptn)

        if abs(scaled_ellipse.theta) > 0.0001:
            raise NotImplementedError("Rotated ellipses are not supported by the step and repeat backend.")

        a, b = scaled_ellipse.a, scaled_ellipse.b  # pylint: disable=invalid-name
        center = scaled_ellipse.center

        self._add_shape(
            "TEllipse", center,
            [
                (center + (-a, -b), _NODE_FIRST),
                (center + (a, -b), _NODE_NEXT),
                (center + (a, b), _NODE_NEXT),
                (center + (-a, b), _NODE_LAST),
            ],
            ptn
        )

    def ring(self, ptn: Pattern[Ring]) -> None:
        scaled_ring: Ring = self._scaled(ptn)
        r_outer = scaled_ring.r_outer
        center = scaled_ring.center

        scan_direction = ptn.mill.scan_direction
        self._add_shape(
            "TRing", center,
            [
                (center + (-r_outer, r_outer), _NODE_FIRST),
                (center + (r_outer, r_outer), _NODE_NEXT),
                (center + (r_outer, -r_outer), _NODE_NEXT),
                (center + (-r_outer, -r_outer), _NODE_LAST),
            ],
            ptn, outline_thickness=scaled_ring.thickness, angle=0.0 if scan_direction is None else scan_direction
        )

    def hollow_arc_spline(self, ptn: Pattern[HollowArcSpline]) -> None:
        def make_node_list(points: np.ndarray) -> t.List[t.Tuple[t.Any, int]]:
            if len(points) < 3:
                raise ValueError("A polygon needs at least three points.")

            nodes_ = [(points[0], _NODE_FIRST)]
            nodes_.extend((point, _NODE_NEXT) for point in points[1:-1])
            nodes_.append((points[-1], _NODE_LAST))
            return nodes_

        scaled_hollow_arc_spline: HollowArcSpline = self._scaled(ptn)

        nodes: t.List[t.Tuple[t.Any, int]] = []
        for arc_spline in [scaled_hollow_arc_spline.boundary] + list(scaled_hollow_arc_spline.holes):
            nodes.extend(make_node_list(rasterize_with_const_error(arc_spline, self._approximation_error).points))

        self._add_shape("TPolygon", scaled_hollow_arc_spline.center, nodes, ptn)
