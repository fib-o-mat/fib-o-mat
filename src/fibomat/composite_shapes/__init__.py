"""Composite shapes: shapes which are composed of other shapes.

In contrast to the primitives of :mod:`fibomat.shapes` (lines, arcs, rects, arc splines, ...), these shapes are built from
several other shapes and depend on higher level packages (:mod:`fibomat.curve_tools`). Hence, they live in their own
package which is imported after the primitives:

* :class:`HollowArcSpline`: a closed arc spline with holes,
* :class:`Ring`: an annulus (a circle with a circular hole),
* :class:`Text` and :class:`DimText`: text, a sequence of glyphs (:class:`Glyph`, :class:`DimGlyph`), each consisting
  of polylines or polygons.

Example::

    from fibomat.composite_shapes import HollowArcSpline, Ring, Text

    ring = Ring(r_outer=2, thickness=0.5)
    text = Text('fib-o-mat', font_size=5)
"""
from fibomat.composite_shapes.hollow_arc_spline import HollowArcSpline
from fibomat.composite_shapes.ring import Ring
from fibomat.composite_shapes.glyph import Glyph, DimGlyph
from fibomat.composite_shapes.text import Text, DimText


__all__ = ['HollowArcSpline', 'Ring', 'Glyph', 'DimGlyph', 'Text', 'DimText']
