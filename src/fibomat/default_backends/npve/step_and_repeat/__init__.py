from typing import Optional, List
import warnings


import xmltodict
import numpy as np
# from colored import Fore, Style, Back

from fibomat.utils import PathLike
from fibomat.backend import BackendBase

# from fibomat.default_backends import BitmapBackend
from fibomat.sample import Sample
from fibomat.site import Site
from fibomat.pattern import Pattern
from fibomat.shapes import (
    Rect,
    Polygon,
    Polyline,
    Circle,
    HollowArcSpline,
    Spot,
    Line,
    Ellipse,
    Ring,
)
from fibomat.units import (
    U_,
    scale_factor,
    # LengthUnit,
    # TimeUnit,
    # LengthQuantity,
    # has_length_dim,
    # scale_to,
)
from fibomat.curve_tools.rasterize import rasterize_with_const_error
from fibomat.linalg import Vector
from fibomat.default_backends.npve.step_and_repeat.sar_models import (
    SaRFile,
    SaRSite,
    SaRSharedShapes,
)
from fibomat.default_backends.npve.step_and_repeat.sar_schemas import SaRFileSchema
from fibomat.default_backends.npve.step_and_repeat.common_models import (
    FIBShape,
    ShapeTexture,
)


class StepAndRepeatBackend(BackendBase):
    name = "StepAndRepeatBackend"

    def __init__(
        self,
        share_patterns: Optional[bool] = False,
        skip_empty_sites: Optional[bool] = True,
        description: Optional[str] = None,
    ):
        super().__init__()

        self._warned_about_spot_dose = False

        self._share_patterns = share_patterns
        self._shared_patterns = []

        if self._share_patterns:
            self._skip_empty_sites = False
        else:
            self._skip_empty_sites = skip_empty_sites

        self._site_index = 0

        self._shape_index = 10000

        self._current_site: Optional[SaRSite] = None

        self._sites: List[SaRSite] = []

    def process_site(self, new_site: Site) -> None:
        # if not new_site.empty:
        if self._share_patterns:
            if self._site_index == 0 and new_site.empty:
                # raise ValueError('First site must contain any patterns.')
                pass
            elif self._site_index != 0 and not new_site.empty:
                raise ValueError("Only first site should contain patterns.")
        else:
            if new_site.empty:
                if self._skip_empty_sites:
                    return
                else:
                    raise ValueError(
                        "Site may not be empty (NPVE is not doing what everybody would expect "
                        "if an empty site is encountered in an step and repeat list)."
                    )

        if self._sites:
            last_site_pos = self._sites[-1].center
        else:
            last_site_pos = new_site.center.vector_as(U_("µm"))

        self._current_site = SaRSite(self._site_index, new_site, last_site_pos)
        self._sites.append(self._current_site)
        self._site_index += 1

        if self._site_index == 1 or not self._share_patterns:
            super().process_site(new_site)

            if self._share_patterns:
                self._shared_patterns = SaRSharedShapes(
                    self._current_site.shapes["shapes_list"]
                )

                self._current_site.shapes["shapes_list"] = []

        # print(self._shared_patterns.fib_shapes)

    def process_pattern(self, ptn: Pattern) -> None:
        if "use_bitmap" in ptn.kwargs:
            print(ptn.mill, ptn)
            s = Sample()
            bbox = ptn.bounding_box
            site = s.create_site(
                dim_position=bbox.center, dim_fov=(bbox.width, bbox.height)
            )
            site += ptn
            exported = s.export(BitmapBackend)
            exported.save(f"bitmap_{self._shape_index}.png")
            bitmap = exported.image()

            texture = ShapeTexture(bitmap, (bbox.width, bbox.height), ptn.raster_style)

            class_ = "TRectangle"

            # scaled_rect: Rect = bbox.shape.scaled(scale_factor(U_('µm'), ptn.dim_shape.unit))
            # # , origin='center'
            # scaled_poly = Polygon(scaled_rect.corners, description=rect.description)

            width2 = bbox.width.m_as("µm") / 2
            hight2 = bbox.height.m_as("µm") / 2

            center = Vector(bbox.center[0].m_as("µm"), bbox.center[1].m_as("µm"))

            nodes = [
                ((center + (-width2, -hight2)), 0),
                ((center + (width2, -hight2)), 1),
                ((center + (width2, hight2)), 1),
                ((center + (-width2, hight2)), 129),
            ]

            self._current_site.add_fib_shape(
                FIBShape(
                    class_=class_,
                    id=self._shape_index,
                    rotation_center=center,
                    nodes=nodes,
                    mill=ptn.mill,
                    raster_style=ptn.raster_style,
                    shape_texture=texture,
                )
            )

            self._shape_index += 1

        else:
            return super().process_pattern(ptn)

    def _to_str(self):
        if not self._sites:
            raise RuntimeError("Step and repeat list does not contain any sites.")

        if self._share_patterns:
            sar_file = SaRFile(self._sites, self._shared_patterns)
        else:
            sar_file = SaRFile(self._sites)
        return xmltodict.unparse(
            SaRFileSchema().dump(sar_file), encoding="iso-8859-1", pretty=True
        )

    def print(self):
        # sar_file = SaRFile(self.sites)
        # print(SaRFileSchema().dump(sar_file))
        print(self._to_str())

    def save(self, filename: PathLike) -> None:
        with open(filename, "w") as fp:
            fp.write(self._to_str())

    def spot(self, ptn: Pattern[Spot]):
        class_ = "TFIBSpot"

        if not self._warned_about_spot_dose:
            warnings.warn(
                f"{Fore.white}{Back.red}"
                "Check the spot dose in NPVE "
                "as the calculation includes some magic numbers."
                f"{Style.reset}"
            )
            self._warned_about_spot_dose = True

        # spot: Spot = ptn.dim_shape.shape

        scaled_spot: Spot = ptn.dim_shape.shape.scaled(
            scale_factor(U_("µm"), ptn.dim_shape.unit)
        )  # , origin='center'

        nodes = [(scaled_spot.center, 0)]

        self._current_site.add_fib_shape(
            FIBShape(
                class_=class_,
                id=self._shape_index,
                rotation_center=scaled_spot.center,
                nodes=nodes,
                mill=ptn.mill,
                raster_style=ptn.raster_style,
            )
        )

        self._shape_index += 1

    def rect(self, ptn: Pattern[Rect]) -> None:
        if ptn.raster_style.dimension == 1:
            self.polyline(
                Pattern(
                    dim_shape=Polyline(
                        [*ptn.dim_shape.shape.corners, ptn.dim_shape.shape.corners[0]]
                    )
                    * ptn.dim_shape.unit,
                    raster_style=ptn.raster_style,
                    mill=ptn.mill,
                )
            )
        else:
            rect: Rect = ptn.dim_shape.shape

            polygon = Polygon(rect.corners, description=rect.description)
            self.polygon(
                Pattern(
                    dim_shape=polygon * ptn.dim_shape.unit,
                    mill=ptn.mill,
                    raster_style=ptn.raster_style,
                    description=ptn.description,
                    **ptn.kwargs,
                )
            )

    def polygon(self, ptn: Pattern[Polygon]) -> None:
        if ptn.raster_style.dimension == 1:
            self.polyline(
                Pattern(
                    dim_shape=Polyline(
                        [*ptn.dim_shape.shape.points, ptn.dim_shape.shape.points[0]]
                    )
                    * ptn.dim_shape.unit,
                    raster_style=ptn.raster_style,
                    mill=ptn.mill,
                )
            )
        else:
            class_ = "TPolygon"

            scaled_poly: Polygon = ptn.dim_shape.shape.scaled(
                scale_factor(U_("µm"), ptn.dim_shape.unit)
            )  # , origin='center'

            nodes = [(scaled_poly.points[0], 0)]

            for point in scaled_poly.points[1:-1]:
                nodes.append((point, 1))

            nodes.append((scaled_poly.points[-1], 129))

            self._current_site.add_fib_shape(
                FIBShape(
                    class_=class_,
                    id=self._shape_index,
                    rotation_center=scaled_poly.center,
                    nodes=nodes,
                    mill=ptn.mill,
                    raster_style=ptn.raster_style,
                )
            )

            self._shape_index += 1

    def polyline(self, ptn: Pattern[Polyline]) -> None:
        is_line = len(ptn.dim_shape.shape.points) == 2
        class_ = "TLine" if is_line else "TPolyline"

        scaled_poly: Polyline = ptn.dim_shape.shape.scaled(
            scale_factor(U_("µm"), ptn.dim_shape.unit)
        )  # , origin='center'

        nodes = [(scaled_poly.points[0], 0)]

        if is_line:
            nodes.append((scaled_poly.points[1], 129))
        else:
            for point in scaled_poly.points[1:]:
                nodes.append((point, 1))

        # nodes.append((scaled_poly.points[-1], 1))

        self._current_site.add_fib_shape(
            FIBShape(
                class_=class_,
                id=self._shape_index,
                rotation_center=scaled_poly.center,
                nodes=nodes,
                mill=ptn.mill,
                raster_style=ptn.raster_style,
            )
        )

        self._shape_index += 1

    def line(self, ptn: Pattern[Line]) -> None:
        # ptn.dim_shape =

        self.polyline(
            Pattern(
                dim_shape=Polyline([ptn.dim_shape.shape.start, ptn.dim_shape.shape.end])
                * ptn.dim_shape.unit,
                raster_style=ptn.raster_style,
                mill=ptn.mill,
            )
        )

    def circle(self, ptn: Pattern[Circle]) -> None:
        class_ = "TEllipse"

        scaled_circle: Circle = ptn.dim_shape.shape.scaled(
            scale_factor(U_("µm"), ptn.dim_shape.unit)
        )  # , origin='center'
        r = scaled_circle.r
        nodes = [
            (scaled_circle.center + (-r, -r), 0),
            (scaled_circle.center + (r, -r), 1),
            (scaled_circle.center + (r, r), 1),
            (scaled_circle.center + (-r, r), 129),
        ]

        self._current_site.add_fib_shape(
            FIBShape(
                class_=class_,
                id=self._shape_index,
                rotation_center=scaled_circle.center,
                nodes=nodes,
                mill=ptn.mill,
                raster_style=ptn.raster_style,
            )
        )

        self._shape_index += 1

    def ellipse(self, ptn: Pattern[Ellipse]):
        class_ = "TEllipse"

        scaled_ellipse: Ellipse = ptn.dim_shape.shape.scaled(
            scale_factor(U_("µm"), ptn.dim_shape.unit)
        )

        if abs(scaled_ellipse.theta) > 0.0001:
            raise NotImplementedError("Rotation not allowed here!")

        a = scaled_ellipse.a
        b = scaled_ellipse.b

        nodes = [
            (scaled_ellipse.center + (-a, -b), 0),
            (scaled_ellipse.center + (a, -b), 1),
            (scaled_ellipse.center + (a, b), 1),
            (scaled_ellipse.center + (-a, b), 129),
        ]

        self._current_site.add_fib_shape(
            FIBShape(
                class_=class_,
                id=self._shape_index,
                rotation_center=scaled_ellipse.center,
                nodes=nodes,
                mill=ptn.mill,
                raster_style=ptn.raster_style,
            )
        )

        self._shape_index += 1

    def ring(self, ptn: Pattern[Ring]):
        class_ = "TRing"

        scaled_ring: Ring = ptn.dim_shape.shape.scaled(
            scale_factor(U_("µm"), ptn.dim_shape.unit)
        )  # , origin='center'
        r_outer = scaled_ring._r_outer

        # print(scaled_ring.center, r_outer)
        nodes = [
            (scaled_ring.center + (-r_outer, r_outer), 0),
            (scaled_ring.center + (r_outer, r_outer), 1),
            (scaled_ring.center + (r_outer, -r_outer), 1),
            (scaled_ring.center + (-r_outer, -r_outer), 129),
        ]

        self._current_site.add_fib_shape(
            FIBShape(
                class_=class_,
                id=self._shape_index,
                rotation_center=scaled_ring.center,
                nodes=nodes,
                mill=ptn.mill,
                raster_style=ptn.raster_style,
                outline=scaled_ring._thickness,
                angle=scaled_ring._theta,
            )
        )

        self._shape_index += 1

    def hollow_arc_spline(self, ptn: Pattern[HollowArcSpline]) -> None:
        def _make_node_list(points: np.ndarray):
            if len(points) < 3:
                raise RuntimeError("len(points) < 3")

            nodes_ = [(points[0], 0)]

            for point in points[1:-1]:
                nodes_.append((point, 1))

            nodes_.append((points[-1], 129))

            return nodes_

        class_ = "TPolygon"

        scaled_hollow_arc_spline: HollowArcSpline = ptn.dim_shape.shape.scaled(
            scale_factor(U_("µm"), ptn.dim_shape.unit)
        )  # , origin='center'

        nodes = []

        # TODO: make error user definable!

        for arc_spline in [scaled_hollow_arc_spline.boundary] + list(
            scaled_hollow_arc_spline.holes
        ):
            nodes.extend(
                _make_node_list(rasterize_with_const_error(arc_spline, 0.0025).points)
            )

        self._current_site.add_fib_shape(
            FIBShape(
                class_=class_,
                id=self._shape_index,
                rotation_center=scaled_hollow_arc_spline.center,
                nodes=nodes,
                mill=ptn.mill,
                raster_style=ptn.raster_style,
            )
        )

        self._shape_index += 1
