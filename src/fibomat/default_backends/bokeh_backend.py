import inspect
import itertools
import pathlib
from typing import Any, Dict, List, Optional, Union

import bokeh.models as bm
import bokeh.plotting as bp
import bokeh.resources as br
import numpy as np
import PIL.Image
from bokeh.util import compiler
from bokeh.util.compiler import AttrDict, set_cache_hook
from fibomat import layout, shapes
from fibomat.linalg.vectors import DimVector, DimVectorLike
from fibomat.backend import BackendBase
from fibomat.backend.backendbase import ShapeNotSupportedError, shape_type
from fibomat.default_backends._bokeh_site import BokehSite, ShapeType
from fibomat.default_backends.measuretool import MeasureTool
from fibomat.mill import Mill
from fibomat.pattern import Pattern
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.shapes import DimShape
from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.site import Site
from fibomat.units import (
    Q_,
    U_,
    LengthQuantity,
    LengthUnit,
    TimeUnit,
    has_length_dim,
    scale_factor,
)
from fibomat.utils import PathLike

_orig_bundle_models = compiler._bundle_models


def _ugly_patched_bundle_models(*args, **kwargs):
    js = _orig_bundle_models(*args, **kwargs)
    js = js.replace(
        'factory(root["Bokeh"], undefined);', 'factory(this["Bokeh"], undefined);'
    )
    return js


compiler._bundle_models = _ugly_patched_bundle_models


def load_measuretool_hook(custom_model, foo):
    if custom_model.cls == MeasureTool:
        with open(
            pathlib.Path(__file__).parent.resolve() / "bokeh-measuretool.min.js",
            "r",
        ) as fp:
            return AttrDict({"deps": [], "code": fp.read()})


set_cache_hook(load_measuretool_hook)


class StubRasterStyle(RasterStyle):
    def __init__(self, dimension: int):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit,
    ) -> RasterizedPoints:
        pass


from bokeh.core.properties import Instance
from bokeh.models import ColumnDataSource, CustomJS, Tool, Label, Node
from bokeh.plotting import figure, show
from bokeh.util.compiler import TypeScript

CODE = """
import {GestureTool, GestureToolView} from "models/tools/gestures/gesture_tool"
import {ColumnDataSource} from "models/sources/column_data_source"
import {Label} from "models/annotations/label"
import {PanEvent} from "core/ui_events"
import * as p from "core/properties"

export class DrawToolView extends GestureToolView {
  declare model: DrawTool

  // this is executed when the pan/drag event starts
  _pan_start(e: PanEvent): void {
        const {frame} = this.plot_view
        const {sx, sy} = e

        if (!frame.bbox.contains(sx, sy))
            return
        const x = frame.x_scale.invert(sx)
        const y = frame.y_scale.invert(sy)
        this.model.source.data = {x: [x, x], y: [y, y]}
  }

  // this is executed on subsequent mouse/touch moves
  _pan(e: PanEvent): void {
    const {frame} = this.plot_view
    const {sx, sy} = e

    if (!frame.bbox.contains(sx, sy))
      return

    const x = frame.x_scale.invert(sx)
    const y = frame.y_scale.invert(sy)

    const {source} = this.model

    source.get_array("x").pop()
    source.get_array("y").pop()

    source.get_array("x").push(x)
    source.get_array("y").push(y)


    const x_start = source.get_array("x")[0] as number
    const x_end = source.get_array("x")[1] as number

    const y_start = source.get_array("y")[0] as number
    const y_end = source.get_array("y")[1] as number


    const dx = x_end - x_start
    const dy = y_end - y_start
    const distance = Math.sqrt(dx * dx + dy * dy)

    const angle_rad = Math.atan2(dy, dx)
    const angle = angle_rad * 180 / Math.PI

    this.model.label.text = "distance=" + Number(distance).toFixed(3) + ", angle= " + Number(angle).toFixed(2) + "° (" + Number(angle_rad).toFixed(2) + " rad)"

    source.change.emit()



  }

  // this is executed then the pan/drag ends
  _pan_end(_e: PanEvent): void {}
}

export namespace DrawTool {
  export type Attrs = p.AttrsOf<Props>

  export type Props = GestureTool.Props & {
    source: p.Property<ColumnDataSource>,
    label: p.Property<Label>,
    better_label: p.Property<Label>,
  }
}

export interface DrawTool extends DrawTool.Attrs {}

export class DrawTool extends GestureTool {
  declare properties: DrawTool.Props
  declare __view_type__: DrawToolView

  constructor(attrs?: Partial<DrawTool.Attrs>) {
    super(attrs)
  }

  tool_name = "Draw Tool"
  tool_icon = "bk-tool-icon-lasso-select"
  event_type = "pan" as "pan"
  default_order = 12

  static {
    this.prototype.default_view = DrawToolView

    this.define<DrawTool.Props>(({Ref}) => ({
      source: [ Ref(ColumnDataSource) ],
      label: [ Ref(Label) ],
    }))
  }

  override initialize(): void {
      super.initialize()

    }

}
"""


class DrawTool(Tool):
    __implementation__ = TypeScript(CODE)
    source = Instance(ColumnDataSource)
    label = Instance(Label)


# if file = np.array, it should be dtype==uint32! (or16?)
class BokehImage(DimShape):
    def __init__(
        self,
        file: Union[PathLike, PIL.Image.Image],
        image_pixel_scale: LengthQuantity,
        center: Optional[DimVector] = None,
    ):
        self._file = file

        if not has_length_dim(image_pixel_scale):
            raise ValueError("image_pixel_scale must have length dimension")

        self._image_pixel_scale = image_pixel_scale

        if isinstance(file, PIL.Image.Image):
            image = file.convert("RGBA")

            # self._height, self._width = image.shape[:2]
            # self._image_data = np.asarray(image)
            # print(self._image_data.shape)
        else:
            image = PIL.Image.open(file).convert("RGBA")

        self._width, self._height = image.size
        self._image_data = np.ascontiguousarray(image, dtype=np.uint32)[::-1, :]

        self._width *= image_pixel_scale.m
        self._height *= image_pixel_scale.m

        if center is not None:
            self._center = DimVector(center)
        else:
            self._center = DimVector()

        super().__init__(self, image_pixel_scale.units)

    @property
    def shape(self):
        return self

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def center(self):
        return self._center

    @property
    def data(self):
        return self._image_data

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        raise NotImplementedError

    def _impl_rotate(self, theta: float) -> None:
        raise NotImplementedError

    def _impl_scale(self, trans_vec: DimVectorLike) -> None:
        raise NotImplementedError

    def _impl_mirror(self, trans_vec: DimVectorLike) -> None:
        raise NotImplementedError


class BokehBackendBase(BackendBase):
    @shape_type(BokehImage)
    def bokeh_image(self, ptn: Pattern[BokehImage]):
        raise ShapeNotSupportedError


class BokehBackend(BokehBackendBase):
    """
    The default backend for plotting projects, based on the bokeh library.
    All shapes defined in fibomat library are supported.

    .. note::
        :class:`~fibomat.shapes.arc.shapes.Arc` and
        :class:`~fibomat.shapes.curve.shapes.Curve` are rasterized during plotting
        due to lack of supported of the HoverTool for this shapes in the bokeh library.
        The pitch can be defined via the `rasterize_pitch` parameter.
    """

    name = "bokeh"

    def __init__(
        self,
        *,
        unit: Optional[LengthUnit] = None,
        title: Optional[str] = None,
        hide_sites: bool = False,
        rasterize_pitch: Optional[LengthQuantity] = None,
        fullscreen: bool = True,
        legend: bool = True,
        cycle_colors: bool = True,
        image_alpha: float = 0.75,
        plot_reduced_lattices: bool = False,
        only_sites: bool = False,
        **kwargs,
    ):
        """
        Args:
            unit (units.UnitType, optional): used unit for plotting, default to units.U_('µm')
            title (str, optional): title of plot, default to ''
            hide_sites (bool, optional): if true, sides' outlines are not shown, default to false
            rasterize_pitch (units.QuantityType. optional):
                curve_tools.rasterize pitch for shapes.Arc, ... and shapes.Curve, default to
                units.Q_('0.01 µm')
            fullscreen (bool, optional): if true, plot will be take the whole page, default to True
            cycle_colors (bool): if True, different sites get different colors.
            image_alpha (float): alpha value (transparency) of images, default to 0.75
        """
        super().__init__(**kwargs)

        if unit:
            if not has_length_dim(unit):
                raise ValueError("unit's dimension must by [length].")
            self._unit = unit
        else:
            self._unit = U_("µm")

        if title:
            self._title = str(title)
        else:
            self._title = ""

        self._hide_sites = bool(hide_sites)

        if rasterize_pitch:
            if not has_length_dim(rasterize_pitch):
                raise ValueError("rasterize_pitch's dimension must by [length].")
            self._rasterize_pitch = rasterize_pitch
        else:
            self._rasterize_pitch = Q_("0.001 µm")

        self._fullscreen = bool(fullscreen)
        self._legend = bool(legend)

        self._only_sites = only_sites

        self._cycle_colors = cycle_colors

        self._plot_reduced_latties = plot_reduced_lattices

        self._bokeh_sites: List[BokehSite] = []
        self._annotation_site = BokehSite(
            site_index=-1,
            plot_unit=self._unit,
            dim_center=DimVector(),
            theta=0,
            rasterize_pitch=self._rasterize_pitch,
            description="Annotations",
        )
        self._image_annotation: List[BokehImage] = []
        self._image_alpha = float(image_alpha)
        self.fig: Optional[bp.Figure] = None

    def process_site(self, site: Site):
        self._bokeh_sites.append(
            BokehSite(
                site_index=len(self._bokeh_sites),
                plot_unit=self._unit,
                dim_center=site.center,
                theta=site._theta_vec.angle_about_x_axis,
                rasterize_pitch=self._rasterize_pitch,
                cycle_colors=self._cycle_colors,
                dim_fov=site.fov,
                description=site.description,
            )
        )

        if not self._only_sites:
            super().process_site(site)

    def process_pattern(self, ptn: Pattern) -> None:
        # super().process_pattern(ptn)
        def dispatch(extracted_ptn):
            try:
                method = self.implemented_shape_methods[
                    type(extracted_ptn.dim_shape.shape)
                ]
                return method(self, extracted_ptn)
            except KeyError:
                # try bases classes
                for base in inspect.getmro(extracted_ptn.dim_shape.shape.__class__):
                    try:
                        method = self.implemented_shape_methods[base]
                        return method(self, extracted_ptn)
                    except KeyError:
                        pass
                self.process_unknown(extracted_ptn)

        if isinstance(ptn.dim_shape, layout.LayoutBase):
            if (
                isinstance(ptn.dim_shape, layout.DimLattice)
                and self._plot_reduced_latties
            ):
                # plot only first 4 elements and dashed boundary
                lattice_elements = ptn.dim_shape.elements_by_uv

                v_max = min(2, lattice_elements.shape[0])
                u_max = min(2, lattice_elements.shape[0])

                for v in range(v_max):
                    for u in range(u_max):
                        # first_four.append(lattice.elements_by_uv[v, u])
                        extracted_shape = layout.Group(lattice_elements[v, u])
                        # print(extracted_shape)
                        if extracted_shape:
                            if isinstance(extracted_shape, layout.LayoutBase):
                                self.process_pattern(
                                    Pattern(
                                        extracted_shape,
                                        ptn.mill,
                                        ptn.raster_style,
                                        **ptn.kwargs,
                                        description=ptn.description,
                                    )
                                )
                            else:
                                dispatch(
                                    Pattern(
                                        extracted_shape,
                                        ptn.mill,
                                        ptn.raster_style,
                                        **ptn.kwargs,
                                        description=ptn.description,
                                    )
                                )

                bounding_box = ptn.bounding_box

                rect = shapes.Rect(
                    bounding_box.width.m_as("µm"),
                    bounding_box.height.m_as("µm"),
                    center=bounding_box.center.vector_as(U_("µm")),
                ) * U_("µm")

                self._bokeh_sites[-1].filled_curve(
                    Pattern(
                        rect,
                        ptn.mill,
                        ptn.raster_style,
                        **ptn.kwargs,
                        description=ptn.description,
                    ),
                    hatch_pattern="x",
                )
            else:
                # ptn.dim_shape[0]: layout.LayoutBase
                for extracted_shape in ptn.dim_shape.layout_elements():
                    if isinstance(extracted_shape, layout.LayoutBase):
                        self.process_pattern(
                            Pattern(
                                extracted_shape,
                                ptn.mill,
                                ptn.raster_style,
                                **ptn.kwargs,
                                description=ptn.description,
                            )
                        )
                    else:
                        dispatch(
                            Pattern(
                                extracted_shape,
                                ptn.mill,
                                ptn.raster_style,
                                **ptn.kwargs,
                                description=ptn.description,
                            )
                        )
        else:
            dispatch(ptn)

    def process_unknown(self, ptn: Pattern) -> None:
        if hasattr(ptn.dim_shape, "shape") and hasattr(
            ptn.dim_shape.shape, "to_arc_spline"
        ):
            new_pattern = Pattern(
                ptn.dim_shape.shape.to_arc_spline() * ptn.dim_shape.unit,
                ptn.mill,
                ptn.raster_style,
                **ptn.kwargs,
                description=ptn.description,
            )

            self.process_pattern(new_pattern)
        else:
            bbox = ptn.dim_shape.shape.bounding_box
            bbox_ptn = Pattern(
                dim_shape=shapes.Rect(bbox.width, bbox.height, 0, bbox.center)
                * ptn.dim_shape.unit,
                mill=ptn.mill,
                raster_style=ptn.raster_style,
                **ptn.kwargs,  # True if 'annotation' in ptn.kwargs else False
            )
            self._filled_curve(bbox_ptn)

    def _collect_plot_data(self, shape_type: ShapeType) -> Dict[str, Any]:
        # https://stackoverflow.com/a/40826547
        keys = BokehSite.plot_data_keys
        data_dicts = [site.plot_data[shape_type] for site in self._bokeh_sites]
        data_dicts.append(self._annotation_site.plot_data[shape_type])
        return {
            key: list(itertools.chain(*[data_dict[key] for data_dict in data_dicts]))
            for key in keys
        }

    @staticmethod
    def _create_datasources(extra_keys: Optional[List[str]] = None):
        keys = (
            BokehSite.plot_data_keys
            if not extra_keys
            else BokehSite.plot_data_keys + extra_keys
        )

        def _create():
            return bm.ColumnDataSource({key: [] for key in keys})

        return {"spots": _create(), "non_filled": _create(), "filled": _create()}

    @staticmethod
    def _create_renderers(fig, data_sources):
        spot_glyphs = fig.circle_x(
            x="x",
            y="y",
            fill_color="color",
            line_color="color",
            fill_alpha=0.25,
            legend_group="site_id",
            size=10,
            source=data_sources["spots"],
        )

        non_filled_curve_glyphs = fig.multi_line(
            xs="x",
            ys="y",
            line_color="color",
            line_width=2,
            legend_group="site_id",
            source=data_sources["non_filled"],
        )

        filled_curve_glyphs = fig.multi_polygons(
            xs="x",
            ys="y",
            line_width=2,
            fill_color="color",
            line_color="color",
            fill_alpha="fill_alpha",
            hatch_pattern="hatch_pattern",
            legend_group="site_id",
            source=data_sources["filled"],
        )

        return {
            "spots": spot_glyphs,
            "non_filled": non_filled_curve_glyphs,
            "filled": filled_curve_glyphs,
        }

    def _plot_impl(self, data_sources):
        spot_data = self._collect_plot_data(ShapeType.SPOT)
        non_filled = self._collect_plot_data(ShapeType.NON_FILLED_CURVE)
        filled = self._collect_plot_data(ShapeType.FILLED_CURVE)

        data_sources["spots"].stream(spot_data)
        data_sources["non_filled"].stream(non_filled)
        data_sources["filled"].stream(filled)

        return {
            "spots": len(spot_data["x"]),
            "non_filled": len(non_filled["x"]),
            "filled": len(filled["x"]),
        }

    def plot(self):
        tooltips = [
            # ('type', 'shape'),
            ("shape", "@shape_prop"),
            # ('collection_index', '@collection_index'),
            ("mill", "@mill"),
            ("raster style", "@raster_style"),
            ("site", "@site_id"),
            ("description", "@description"),
            # ('mill_settings', '@mill_settings'),
        ]

        site_tooltips = [
            # ('site', '@site'),
            ("description", "@description")
        ]

        fig = bp.figure(
            title=self._title,
            x_axis_label=f"x / {self._unit:~P}",
            y_axis_label=f"y / {self._unit:~P}",
            match_aspect=True,
            sizing_mode="stretch_both" if self._fullscreen else "stretch_width",
            tools="pan,wheel_zoom,reset,save",
        )

        # line_color=bc.groups.red.Crimson, line_width=3  # bpal.all_palettes['Colorblind'][4][3]

        # fig.add_tools(
        #     MeasureTool(
        #         measure_unit=f"{self._unit:~P}",
        #     )
        # )

        fig.add_tools(bm.BoxZoomTool(match_aspect=True))

        data_sources = self._create_datasources()
        renderers = self._create_renderers(fig, data_sources)

        self._plot_impl(data_sources)

        # spot_glyphs = fig.circle_x(
        #     x='x', y='y',
        #     fill_color='color', line_color='color', fill_alpha=.25,
        #     legend_group='site_id',
        #     size=10,
        #     source=bm.ColumnDataSource(self._collect_plot_data(ShapeType.SPOT))
        # )

        # non_filled_curve_glyphs = fig.multi_line(
        #     xs='x', ys='y',
        #     line_color='color', line_width=2,
        #     legend_group='site_id',
        #     source=bm.ColumnDataSource(self._collect_plot_data(ShapeType.NON_FILLED_CURVE))
        # )

        # filled_curve_glyphs = fig.multi_polygons(
        #     xs='x', ys='y',
        #     line_width=2,
        #     fill_color='color', line_color='color', fill_alpha='fill_alpha',
        #     hatch_pattern='hatch_pattern',
        #     legend_group='site_id',
        #     source=bm.ColumnDataSource(self._collect_plot_data(ShapeType.FILLED_CURVE))
        # )

        # images
        if images := self._image_annotation:
            for image in images:
                # image = self._image_annotation

                # center = image.center.vector_as(self._unit)

                # input_offset_x = bm.Spinner(
                #   title="Image x", low=-100000, high=100000, step=0.01, value=center.x, width=80)
                # input_offset_y = bm.Spinner(
                #   title="Image y", low=-100000, high=100000, step=0.01, value=center.y, width=80)

                image_center = image.center.vector_as(self._unit)
                image_scale = scale_factor(self._unit, image.unit)
                width = image.width * image_scale
                height = image.height * image_scale

                print(image.data.dtype, image.data.shape)

                # https://stackoverflow.com/questions/52433129/python-bokeh-get-image-from-webcam-and-show-it-in-dashboard
                image_data = (
                    image.data
                )  # .view(dtype=np.uint32).reshape(image.data.shape)

                img = np.empty(image_data.shape[:2], dtype=np.uint32)
                view = img.view(dtype=np.uint8).reshape(image_data.shape)
                view[:] = image_data[:]

                # image.data.view(dtype=np.uint32).reshape(image.data.shape)

                rendered_image = fig.image_rgba(
                    image=[img],
                    x=image_center.x - width / 2,
                    y=image_center.y - height / 2,
                    dw=width,
                    dh=height,
                    # anchor="center",
                    global_alpha=self._image_alpha,
                )

                # input_offset_x.js_link('value', rendered_image.glyph, 'x')
                # input_offset_y.js_link('value', rendered_image.glyph, 'y')

        # layers
        # https://github.com/bokeh/bokeh/issues/9087
        if not self._hide_sites:
            site_glyphs = fig.multi_polygons(
                xs="x",
                ys="y",
                line_width=2,
                fill_color="color",
                line_color="color",
                fill_alpha="fill_alpha",
                line_alpha="fill_alpha",
                legend_group="site_id",
                source=bm.ColumnDataSource(self._collect_plot_data(ShapeType.SITE)),
            )

            site_glyphs_hover = bm.HoverTool(
                renderers=[site_glyphs],
                tooltips=site_tooltips,
                point_policy="follow_mouse",
            )
            fig.add_tools(site_glyphs_hover)

        # hover tool for shapes
        # add shape hover tool after site hovertool so it is rendered on top of the site tooltip
        shape_glyphs_hover = bm.HoverTool(
            # renderers=[
            #     spot_glyphs, non_filled_curve_glyphs,  filled_curve_glyphs
            # ],
            renderers=list(renderers.values()),
            tooltips=tooltips,
            point_policy="follow_mouse",
        )
        fig.add_tools(shape_glyphs_hover)

        def sorter(item):
            value = item.label["value"]
            if value == "Annotations":
                return -1
            else:
                return int(value.split(",")[0].split(" ")[1])

        legend_tmp = {x.label["value"]: x for x in fig.legend.items}.values()
        fig.legend.items.clear()
        fig.legend.items.extend(sorted(legend_tmp, key=sorter))

        fig.legend.visible = self._legend

        measure_source = ColumnDataSource(data=dict(x=[], y=[]))

        frame_left = Node(target="frame", symbol="left", offset=5)
        frame_bottom = Node(target="frame", symbol="bottom", offset=-5)
        label = Label(
            x=frame_left,
            y=frame_bottom,
            anchor="bottom_left",
            text="",
            padding=10,
            # border_radius=5,
            # border_line_color="black",
            # background_fill_color="white",
        )

        fig.add_tools(DrawTool(source=measure_source, label=label))
        fig.add_layout(label)

        fig.line("x", "y", line_width=3, source=measure_source)

        # if self._image_annotation:
        #     self.fig = bl.column([bl.row([input_offset_x, input_offset_y]), fig], width_policy='max')
        # else:
        #     self.fig = fig
        self.fig = fig

    # def _gen_html(self, use_cdn: bool):
    #     resources = br.CDN if use_cdn else br.INLINE
    #     return be.file_html(models=self.fig, resources=resources, title=self._title, template=_RenderWrapper())

    def show(self):
        # with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as fp:
        #     fp.write(self._gen_html(True))
        #     fp.flush()
        #     webbrowser.open(fp.name)
        bp.show(self.fig)

    def save(self, filename: PathLike, use_cdn: bool = False):
        # with open(filename, 'w') as fp:
        #     fp.write(self._gen_html(use_cdn))
        resources = br.CDN if use_cdn else br.INLINE
        bp.save(self.fig, filename, resources=resources)

    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        if "_annotation" in ptn.kwargs:
            self._annotation_site.spot(ptn)
        else:
            self._bokeh_sites[-1].spot(ptn)

    def _non_filled_curve(self, ptn):
        if "_annotation" in ptn.kwargs:
            self._annotation_site.non_filled_curve(ptn)
        else:
            self._bokeh_sites[-1].non_filled_curve(ptn)

    def _filled_curve(self, ptn):
        if "_annotation" in ptn.kwargs:
            self._annotation_site.filled_curve(ptn)
        else:
            self._bokeh_sites[-1].filled_curve(ptn)

    def _filled_curve_with_holes(self, ptn):
        if "_annotation" in ptn.kwargs:
            self._annotation_site.filled_curve_with_holes(ptn)
        else:
            self._bokeh_sites[-1].filled_curve_with_holes(ptn)

    def _dispatch_pattern(self, ptn):
        if not ptn.dim_shape.shape.is_closed:
            self._non_filled_curve(ptn)
        elif isinstance(ptn.dim_shape.shape, shapes.HollowArcSpline):
            self._filled_curve_with_holes(ptn)
        elif ptn.raster_style.dimension < 2:
            self._non_filled_curve(ptn)
        else:
            self._filled_curve(ptn)

    def line(self, ptn: Pattern[shapes.Line]) -> None:
        self._dispatch_pattern(ptn)

    def polyline(self, ptn: Pattern[shapes.Polyline]) -> None:
        self._dispatch_pattern(ptn)

    def arc(self, ptn: Pattern[shapes.Arc]) -> None:
        self._dispatch_pattern(ptn)

    def arc_spline(self, ptn: Pattern[shapes.ArcSpline]) -> None:
        self._dispatch_pattern(ptn)

    def polygon(self, ptn: Pattern[shapes.Polygon]) -> None:
        self._dispatch_pattern(ptn)

    def rect(self, ptn: Pattern[shapes.Rect]) -> None:
        self._dispatch_pattern(ptn)

    def ellipse(self, ptn: Pattern[shapes.Ellipse]) -> None:
        self._dispatch_pattern(ptn)

    def circle(self, ptn: Pattern[shapes.Circle]) -> None:
        self._dispatch_pattern(ptn)

    def rasterized_points(self, ptn: Pattern[shapes.RasterizedPoints]):
        rect = shapes.Rect.from_bounding_box(ptn.dim_shape.shape.bounding_box)

        new_pattern = Pattern(
            dim_shape=rect * ptn.dim_shape.unit,
            mill=ptn.mill,
            raster_style=ptn.raster_style,
            description=ptn.description,
            **ptn.kwargs,
        )

        if "_annotation" in ptn.kwargs:
            self._annotation_site.filled_curve(new_pattern, hatch_pattern="/")
        else:
            self._bokeh_sites[-1].filled_curve(new_pattern, hatch_pattern="/")

    def hollow_arc_spline(self, ptn: Pattern[shapes.HollowArcSpline]) -> None:
        self._dispatch_pattern(ptn)

    def bokeh_image(self, ptn: Pattern[BokehImage]):
        if "_annotation" in ptn.kwargs:
            # if not self._image_annotation:
            #     self._image_annotation = ptn.dim_shape
            # else:
            #     raise RuntimeError('currently, only one image is supported.')
            self._image_annotation.append(ptn.dim_shape)
        else:
            raise RuntimeError("BokehImage can only added to annotation site")
