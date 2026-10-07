"""Provides the :class:`BokehBackend`, which plots a layout with bokeh.

The plots are interactive (pan, zoom, hover tooltips, a tool to measure distances and angles) and can be saved as
HTML files which are completely self-contained: all JavaScript and CSS is embedded, so that no internet connection is
needed to view them.

Example::

    layout.plot()  # opens the plot in the browser
    layout.plot(show=False, filename='design.html')  # self-contained HTML file
"""
from __future__ import annotations

import pathlib
import tempfile
import typing as t
import warnings
import webbrowser

import bokeh.models as bm
import bokeh.plotting as bp
import bokeh.resources as br
import numpy as np
from bokeh.embed import file_html
from bokeh.palettes import Viridis256
from bokeh.transform import linear_cmap

from fibomat import arrangements, composite_shapes, shapes
from fibomat.backend import BackendBase, shape_type
from fibomat.default_backends._unit_helpers import to_length_unit
from fibomat.default_backends._bokeh_site import BokehSite, ShapeType
from fibomat.default_backends.bokeh_image import BokehImage
from fibomat.default_backends.measuretool import MeasureTool
from fibomat.default_backends.stub_raster_style import StubRasterStyle
from fibomat.layout.pattern import Pattern
from fibomat.layout.site import Site
from fibomat.shapes import DimShape
from fibomat.linalg import DimVector
from fibomat.units import DimFloat, LengthUnit, has_length_dim, scale_factor
from fibomat.units import unit as make_unit
from fibomat.utils import PathLike


__all__ = ['BokehBackend', 'BokehImage', 'StubRasterStyle']


_DEFAULT_RASTERIZE_PITCH = 0.001 * make_unit('µm')

_TOOLTIPS = [
    ('shape', '@shape_prop'),
    ('mill', '@mill'),
    ('raster style', '@raster_style'),
    ('site', '@site_id'),
    ('description', '@description'),
]


class BokehBackend(BackendBase):
    """The default backend for plotting layouts, based on the bokeh library.

    All shapes of the library are supported. Curves are approximated by polygons for plotting (see `rasterize_pitch`).

    The plot is created by :meth:`plot` (called automatically by :meth:`save`, :meth:`html` and :meth:`show`) and can
    be accessed with :attr:`fig`.
    """

    def __init__(
        self,
        *,
        unit: t.Optional[LengthUnit] = None,  # pylint: disable=redefined-outer-name
        title: t.Optional[str] = None,
        hide_sites: bool = False,
        rasterize_pitch: t.Optional[DimFloat[t.Any]] = None,
        fullscreen: bool = True,
        legend: bool = True,
        cycle_colors: bool = True,
        image_alpha: float = 0.75,
        plot_reduced_lattices: bool = False,
        only_sites: bool = False,
        plot_rasterized: bool = False,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            unit (LengthUnit, optional): length unit of the plot, default ``unit('µm')``
            title (str, optional): title of the plot
            hide_sites (bool): if True, the outlines of the sites are not shown
            rasterize_pitch (DimFloat, optional): maximum distance between a curve and its polygonal approximation in
                the plot, default ``0.001 * unit('µm')``.
            fullscreen (bool): if True, the plot uses the whole page
            legend (bool): if True, a legend is shown
            cycle_colors (bool): if True, different sites get different colors
            image_alpha (float): alpha value (opacity) of images
            plot_reduced_lattices (bool): if True, only four elements of each lattice are plotted, together with a
                hatched outline of the whole lattice
            only_sites (bool): if True, only the sites are plotted
            plot_rasterized (bool): if True, the patterns are rasterized and the dwell points are plotted (colored by
                the dwell time)
            description (str, optional): description of the layout

        Raises:
            ValueError: Raised if unit or rasterize_pitch are no lengths or rasterize_pitch is not positive.
            TypeError: Raised if rasterize_pitch is no dimensioned value.
        """
        super().__init__(description)

        self._unit = to_length_unit(unit if unit is not None else 'µm')
        self._title = str(title) if title else ''

        if rasterize_pitch is None:
            rasterize_pitch = _DEFAULT_RASTERIZE_PITCH
        if not isinstance(rasterize_pitch, DimFloat):
            raise TypeError('rasterize_pitch must be a dimensioned value like 0.001 * unit("µm").')
        if not has_length_dim(rasterize_pitch):
            raise ValueError("rasterize_pitch's dimension must be [length].")
        if not rasterize_pitch.magnitude > 0.:
            raise ValueError('rasterize_pitch must be positive.')
        self._rasterize_pitch = rasterize_pitch

        self._hide_sites = bool(hide_sites)
        self._fullscreen = bool(fullscreen)
        self._legend = bool(legend)
        self._cycle_colors = bool(cycle_colors)
        self._image_alpha = float(image_alpha)
        self._plot_reduced_lattices = bool(plot_reduced_lattices)
        self._only_sites = bool(only_sites)
        self._plot_rasterized = bool(plot_rasterized)

        self._bokeh_sites: t.List[BokehSite] = []
        self._annotation_site = BokehSite(
            site_index=-1, plot_unit=self._unit, dim_center=DimVector(0 * make_unit('µm'), 0 * make_unit('µm')),
            rasterize_pitch=self._rasterize_pitch, annotation=True,
        )
        self._images: t.List[t.Any] = []
        self._point_cloud: t.List[np.ndarray] = []

        self._fig: t.Optional[bp.figure] = None

    # ------------------------------------------------------------------------------------------------------------------
    # collecting the data
    # ------------------------------------------------------------------------------------------------------------------

    def process_site(self, new_site: Site) -> None:
        fov_bounding_box = None
        try:
            fov_bounding_box = new_site.fov_bounding_box
        except ValueError:
            pass  # an empty site without fov has no outline

        self._bokeh_sites.append(
            BokehSite(
                site_index=len(self._bokeh_sites),
                plot_unit=self._unit,
                dim_center=new_site.center,
                rasterize_pitch=self._rasterize_pitch,
                cycle_colors=self._cycle_colors,
                fov_bounding_box=fov_bounding_box,
                description=new_site.description,
            )
        )

        if not self._only_sites:
            super().process_site(new_site)

    def process_pattern(self, ptn: Pattern) -> None:
        if self._plot_rasterized and '_annotation' not in ptn.kwargs:
            try:
                self._process_rasterized(ptn)
                return
            except Exception as error:  # pylint: disable=broad-except
                warnings.warn(f'Rasterization failed for {ptn!r} ({error}), plotting the shape instead.', stacklevel=2)

        if self._plot_reduced_lattices and isinstance(ptn.dim_shape, arrangements.DimLattice):
            self._process_reduced_lattice(ptn)
            return

        super().process_pattern(ptn)

    def _process_rasterized(self, ptn: Pattern) -> None:
        """Rasterize a pattern and collect its dwell points (in plot coordinates, with the dwell time in ms)."""
        rasterized = ptn.raster_style.rasterize(
            dim_shape=ptn.dim_shape, mill=ptn.mill, out_length_unit=self._unit, out_time_unit=make_unit('ms')
        )
        points = np.array(rasterized.dwell_points)
        points[:, :2] += np.asarray(self._bokeh_sites[-1].center)
        self._point_cloud.append(points)

    def _process_reduced_lattice(self, ptn: Pattern) -> None:
        """Plot only the first elements of a lattice and a hatched outline of the whole lattice."""
        lattice = ptn.dim_shape
        elements = lattice.elements_by_uv

        for i_v in range(min(2, elements.shape[0])):
            for i_u in range(min(2, elements.shape[1])):
                element = elements[i_v, i_u]
                if element is not None:
                    super().process_pattern(
                        Pattern(element, ptn.mill, ptn.raster_style, description=ptn.description, **ptn.kwargs)
                    )

        bbox = ptn.bounding_box
        rect = shapes.Rect(
            bbox.width.m_as('µm'), bbox.height.m_as('µm'), center=bbox.center.vector_as(make_unit('µm'))
        ) * make_unit('µm')
        self._site_of(ptn).filled_curve(
            Pattern(rect, ptn.mill, ptn.raster_style, description=ptn.description, **ptn.kwargs), hatch_pattern='x'
        )

    def _site_of(self, ptn: Pattern) -> BokehSite:
        """The bokeh site to which a pattern belongs (the annotation layer for annotations)."""
        if '_annotation' in ptn.kwargs:
            return self._annotation_site
        if not self._bokeh_sites:
            raise RuntimeError('Patterns must be added to a site.')
        return self._bokeh_sites[-1]

    def process_unknown(self, ptn: Pattern) -> None:
        shape = ptn.dim_shape.shape
        converted = None

        if callable(getattr(shape, 'to_hollow_arc_spline', None)):
            converted = shape.to_hollow_arc_spline()
        elif callable(getattr(shape, 'to_arc_spline', None)):
            converted = shape.to_arc_spline()

        if converted is not None:
            self.process_pattern(
                Pattern(
                    DimShape(converted, ptn.dim_shape.unit), ptn.mill, ptn.raster_style,
                    description=ptn.description, **ptn.kwargs
                )
            )
        else:
            # plot the bounding box of shapes which cannot be plotted
            bbox = shape.bounding_box
            self._filled_curve(Pattern(
                DimShape(shapes.Rect(bbox.width, bbox.height, center=bbox.center), ptn.dim_shape.unit),
                ptn.mill, ptn.raster_style, description=ptn.description, **ptn.kwargs
            ))

    # shape methods

    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        self._site_of(ptn).spot(ptn)

    def _non_filled_curve(self, ptn: Pattern) -> None:
        self._site_of(ptn).non_filled_curve(ptn)

    def _filled_curve(self, ptn: Pattern) -> None:
        self._site_of(ptn).filled_curve(ptn)

    def _filled_curve_with_holes(self, ptn: Pattern) -> None:
        self._site_of(ptn).filled_curve_with_holes(ptn)

    def _plot_pattern(self, ptn: Pattern) -> None:
        """Plot a curve or an area, depending on the shape and the raster style."""
        if isinstance(ptn.dim_shape.shape, composite_shapes.HollowArcSpline):
            self._filled_curve_with_holes(ptn)
        elif not ptn.dim_shape.shape.is_closed or ptn.raster_style.dimension < 2:
            self._non_filled_curve(ptn)
        else:
            self._filled_curve(ptn)

    def line(self, ptn: Pattern[shapes.Line]) -> None:
        self._plot_pattern(ptn)

    def polyline(self, ptn: Pattern[shapes.Polyline]) -> None:
        self._plot_pattern(ptn)

    def arc(self, ptn: Pattern[shapes.Arc]) -> None:
        self._plot_pattern(ptn)

    def arc_spline(self, ptn: Pattern[shapes.ArcSpline]) -> None:
        self._plot_pattern(ptn)

    def parametric_curve(self, ptn: Pattern[shapes.ParametricCurve]) -> None:
        self.process_unknown(ptn)

    def polygon(self, ptn: Pattern[shapes.Polygon]) -> None:
        self._plot_pattern(ptn)

    def rect(self, ptn: Pattern[shapes.Rect]) -> None:
        self._plot_pattern(ptn)

    def ellipse(self, ptn: Pattern[shapes.Ellipse]) -> None:
        self._plot_pattern(ptn)

    def circle(self, ptn: Pattern[shapes.Circle]) -> None:
        self._plot_pattern(ptn)

    def ring(self, ptn: Pattern[composite_shapes.Ring]) -> None:
        self.process_unknown(ptn)

    def hollow_arc_spline(self, ptn: Pattern[composite_shapes.HollowArcSpline]) -> None:
        self._plot_pattern(ptn)

    def rasterized_points(self, ptn: Pattern[shapes.RasterizedPoints]) -> None:
        # the points are not plotted, but their bounding box (hatched)
        rect = shapes.Rect.from_bounding_box(ptn.dim_shape.shape.bounding_box)

        self._site_of(ptn).filled_curve(
            Pattern(
                DimShape(rect, ptn.dim_shape.unit), ptn.mill, ptn.raster_style, description=ptn.description,
                **ptn.kwargs
            ),
            hatch_pattern='/'
        )

    @shape_type(BokehImage)
    def bokeh_image(self, ptn: Pattern[BokehImage]) -> None:
        """Add an image. Images can only be added as annotations (see :meth:`Layout.add_annotation`)."""
        if '_annotation' not in ptn.kwargs:
            raise RuntimeError('A BokehImage can only be added as annotation.')
        self._images.append(ptn.dim_shape)

    # ------------------------------------------------------------------------------------------------------------------
    # the plot
    # ------------------------------------------------------------------------------------------------------------------

    def _sites(self) -> t.List[BokehSite]:
        return self._bokeh_sites + [self._annotation_site]

    def _collect_plot_data(self, shape_type_: ShapeType) -> t.Dict[str, t.List[t.Any]]:
        """Concatenate the plot data of all sites (and the annotations)."""
        return {
            key: [value for site in self._sites() for value in site.plot_data[shape_type_][key]]
            for key in BokehSite.plot_data_keys
        }

    def _plot_rasterized_points(self, fig: bp.figure) -> None:
        points = np.concatenate(self._point_cloud, axis=0)
        times = points[:, 2]
        low, high = float(np.min(times)), float(np.max(times))
        if low == high:
            high = low + 1.

        palette = Viridis256[::-1]
        source = bm.ColumnDataSource(data=dict(x=points[:, 0], y=points[:, 1], t=times))
        fig.scatter(x='x', y='y', size=5, color=linear_cmap('t', palette, low, high), source=source)
        fig.add_layout(
            bm.ColorBar(
                color_mapper=bm.LinearColorMapper(palette=palette, low=low, high=high),
                label_standoff=12, location=(0, 0), title='dwell time (ms)'
            ),
            'right'
        )

    def _add_images(self, fig: bp.figure) -> None:
        for image in self._images:
            center = image.center.vector_as(self._unit)
            scale = scale_factor(self._unit, image.unit)
            width = image.shape.width * scale
            height = image.shape.height * scale

            rgba = np.ascontiguousarray(image.shape.data)
            fig.image_rgba(
                image=[rgba.view(dtype=np.uint32).reshape(rgba.shape[:2])],
                x=center.x - width / 2, y=center.y - height / 2, dw=width, dh=height, global_alpha=self._image_alpha
            )

    def _sorted_legend_items(self, fig: bp.figure) -> None:
        """Order the legend items: annotations first, then the sites by their index; hide the legend if wanted."""
        if not fig.legend:
            return

        order = {site.label: site.site_index for site in self._sites()}

        unique = {item.label['value']: item for item in fig.legend.items}
        items = sorted(unique.values(), key=lambda item: order.get(item.label['value'], len(order)))

        fig.legend.items = items
        fig.legend.visible = self._legend

    def plot(self) -> bp.figure:
        """Create the bokeh figure from the collected data. The figure is available as :attr:`fig`, too.

        Returns:
            bokeh.plotting.figure
        """
        fig = bp.figure(
            title=self._title,
            x_axis_label=f'x / {self._unit:~P}',
            y_axis_label=f'y / {self._unit:~P}',
            match_aspect=True,
            sizing_mode='stretch_both' if self._fullscreen else 'stretch_width',
            tools='pan,wheel_zoom,reset,save',
        )
        fig.add_tools(bm.BoxZoomTool(match_aspect=True))

        if self._plot_rasterized and self._point_cloud:
            self._plot_rasterized_points(fig)

        self._add_images(fig)

        # sites (below the shapes)
        site_data = self._collect_plot_data(ShapeType.SITE)
        if not self._hide_sites and site_data['x']:
            site_glyphs = fig.multi_polygons(
                xs='x', ys='y', line_width=2, fill_color='color', line_color='color', fill_alpha='fill_alpha',
                line_alpha='fill_alpha', legend_group='site_id', source=bm.ColumnDataSource(site_data),
            )
            fig.add_tools(bm.HoverTool(
                renderers=[site_glyphs], tooltips=[('description', '@description')], point_policy='follow_mouse'
            ))

        # shapes
        renderers = [
            fig.scatter(
                x='x', y='y', fill_color='color', line_color='color', fill_alpha=0.25, legend_group='site_id',
                size=10, source=bm.ColumnDataSource(self._collect_plot_data(ShapeType.SPOT)),
            ),
            fig.multi_line(
                xs='x', ys='y', line_color='color', line_width=2, legend_group='site_id',
                source=bm.ColumnDataSource(self._collect_plot_data(ShapeType.NON_FILLED_CURVE)),
            ),
            fig.multi_polygons(
                xs='x', ys='y', line_width=2, fill_color='color', line_color='color', fill_alpha='fill_alpha',
                hatch_pattern='hatch_pattern', legend_group='site_id',
                source=bm.ColumnDataSource(self._collect_plot_data(ShapeType.FILLED_CURVE)),
            ),
        ]
        # the hover tool of the shapes is added after the one of the sites, so its tooltip is shown on top
        fig.add_tools(bm.HoverTool(renderers=renderers, tooltips=_TOOLTIPS, point_policy='follow_mouse'))

        self._sorted_legend_items(fig)

        # tool to measure distances and angles
        measure_source = bm.ColumnDataSource(data=dict(x=[], y=[]))
        measure_label = bm.Label(
            x=bm.Node(target='frame', symbol='left', offset=5), y=bm.Node(target='frame', symbol='bottom', offset=-5),
            anchor='bottom_left', text='', padding=10,
        )
        fig.add_tools(MeasureTool(source=measure_source, label=measure_label, measure_unit=f'{self._unit:~P}'))
        fig.add_layout(measure_label)
        fig.line('x', 'y', line_width=3, source=measure_source)

        self._fig = fig
        return fig

    @property
    def fig(self) -> bp.figure:
        """The bokeh figure (created on first access).

        Access:
            get
        """
        return self._fig if self._fig is not None else self.plot()

    def html(self, use_cdn: bool = False) -> str:
        """The plot as HTML document.

        Args:
            use_cdn (bool): if True, bokeh's JavaScript is loaded from the internet (small file). Otherwise
                (default) all JavaScript and CSS is embedded and the document is completely self-contained.

        Returns:
            str
        """
        resources = br.CDN if use_cdn else br.INLINE
        return file_html(self.fig, resources=resources, title=self._title or 'fib-o-mat')

    def save(self, filename: PathLike, use_cdn: bool = False) -> None:
        """Save the plot as HTML file.

        Args:
            filename (PathLike): filename
            use_cdn (bool): if True, bokeh's JavaScript is loaded from the internet. Otherwise (default), all
                JavaScript and CSS is embedded and the file is completely self-contained.
        """
        pathlib.Path(filename).write_text(self.html(use_cdn), encoding='utf-8')

    def show(self) -> None:
        """Open the plot in the browser (as self-contained HTML file in the temporary directory)."""
        with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8') as file:
            file.write(self.html())

        webbrowser.open(pathlib.Path(file.name).as_uri())

