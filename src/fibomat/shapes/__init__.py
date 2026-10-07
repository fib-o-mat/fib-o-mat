from fibomat.shapes.shape import Shape
from fibomat.shapes.dim_shape import DimShape
from fibomat.shapes.arc_spline import ArcSpline, ArcSplineCompatible
from fibomat.shapes.line import Line
from fibomat.shapes.polyline import Polyline
from fibomat.shapes.polygon import Polygon
from fibomat.shapes.spot import Spot
from fibomat.shapes.rect import Rect
from fibomat.shapes.circle import Circle
from fibomat.shapes.ellipse import Ellipse
from fibomat.shapes.parametric_curve import ParametricCurve
from fibomat.shapes.arc import Arc
from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.shapes.biarc import Biarc


__all__ = [
    "Shape",
    "ArcSpline",
    "Line",
    "Polyline",
    "Polygon",
    "Spot",
    "Rect",
    "Circle",
    "Ellipse",
    "ParametricCurve",
    "Arc",
    "RasterizedPoints",
    "DimShape",
    "ArcSplineCompatible",
    "Biarc",
]


# The composite shapes live in `fibomat.composite_shapes` (they depend on `curve_tools` and `arrangements`, which depend on this
# package). For backward compatibility, they are still available as attributes of this package; they are imported
# lazily on first access.
_COMPOSITE_SHAPES = ("HollowArcSpline", "Ring", "Text", "DimText")


def __getattr__(name: str):
    if name in _COMPOSITE_SHAPES:
        import importlib  # pylint: disable=import-outside-toplevel

        return getattr(importlib.import_module("fibomat.composite_shapes"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
