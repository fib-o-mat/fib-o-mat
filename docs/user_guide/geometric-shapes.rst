Geometric shapes
================

fib-o-mat supports a decent number of different geometric shapes. These shapes eventually define the patterning geometry.

Zero-dimensional shapes:

    - :class:`~fibomat.shapes.Spot`
    - :class:`~fibomat.shapes.RasterizedPoints`

One/two-dimensional shapes:

    - :class:`~fibomat.shapes.Line`
    - :class:`~fibomat.shapes.Polyline`
    - :class:`~fibomat.shapes.Polygon`
    - :class:`~fibomat.shapes.Arc`
    - :class:`~fibomat.shapes.ArcSpline`
    - :class:`~fibomat.shapes.ParametricCurve`
    - :class:`~fibomat.shapes.Rect`
    - :class:`~fibomat.shapes.Circle`
    - :class:`~fibomat.shapes.Ellipse`

Composite shapes (they are composed of other shapes, see :mod:`fibomat.composite_shapes`):

    - :class:`~fibomat.composite_shapes.HollowArcSpline`
    - :class:`~fibomat.composite_shapes.Ring`
    - :class:`~fibomat.composite_shapes.Text`

See the source file of the plot below for usage examples of all shapes.

.. fibomat-plot:: ../examples/all_shapes.py


Rigid transformations and isotropic scaling
-------------------------------------------

All shapes can be translated, rotated, mirrored and scaled.

These transformations are invoked by calling the :meth:`~fibomat.shapes.Shape.translated`, :meth:`~fibomat.shapes.Shape.rotated`, :meth:`~fibomat.shapes.Shape.mirrored` and :meth:`~fibomat.shapes.Shape.scaled` methods, respectively.
Due to the immutability of the objects, the transformation methods return a new object with the transformation applied.

Further, all shapes have a :attr:`~fibomat.shapes.Shape.center` property. For most shapes, it is the center of the bounding box of the shape, see the :doc:`API reference <../api/shapes>` for details.

Below, all transformation methods are shown, exemplarily with the line shape.

Translating
+++++++++++

Translating a shape with a translation vector: ::

    line = Line(start=(1, 0), end=(2, 1))

    translated_line = line.translated((2, 4.5))
    # translated_line.start == (3, 4.5)
    # translated_line.end == (4, 5.5)

Rotating
++++++++

Rotating a shape about the coordinate origin (0, 0): ::

    line = Line(start=(1, 0), end=(1, 1))

    # rotation about (0, 0) with the angle pi/2 in the mathematically positive direction (counterclockwise)
    rotated_line = line.rotated(np.pi / 2)
    # rotated_line.start == (0, 1)
    # rotated_line.end == (-1, 1)

Rotating a shape about a custom origin: ::

    line = Line(start=(1, 0), end=(1, 1))

    # rotation about (3.5, 4.5) with the angle pi/2 in the mathematically positive direction (counterclockwise)
    rotated_line = line.rotated(np.pi / 2, origin=(3.5, 4.5))
    # rotated_line.start == (8, 2)
    # rotated_line.end == (7, 2)

Rotating a shape about its center: ::

    line = Line(start=(1, 0), end=(1, 1))

    # rotation about the center (1, 0.5) of the line with the angle pi/2
    rotated_line = line.rotated(np.pi / 2, origin='center')
    # rotated_line.start == (1.5, 0.5)
    # rotated_line.end == (0.5, 0.5)

(All values in the comments are rounded, the calculation is done in floating point arithmetic.)

Mirroring
+++++++++

Mirroring at an axis through the origin: ::

    line = Line(start=(1, 0), end=(1, 1))

    # mirror the line at the x-axis
    mirrored_line = line.mirrored((1, 0))
    # mirrored_line.start == (1, 0)
    # mirrored_line.end == (1, -1)

Scaling
+++++++

Only isotropic scaling is supported in fib-o-mat (the x- and y-axis are always scaled equally).
Scaling supports the ``origin`` parameter like the rotation does. ::

    line = Line(start=(1, 0), end=(1, 1))

    scaled_line_1 = line.scaled(2)
    scaled_line_2 = line.scaled(2, origin=(3, 4))  # custom origin
    scaled_line_3 = line.scaled(2, origin='center')  # the center of the line is the origin

Chaining of transformations
+++++++++++++++++++++++++++

Multiple transformations can be applied in one step with the :meth:`~fibomat.shapes.Shape.transformed` method.

This method expects chained transformation stubs which are defined in the :mod:`~fibomat.linalg` submodule and applies them all at once ::

    # import the stubs
    from fibomat.linalg import translate, rotate, mirror, scale

    line = Line(start=(1, 0), end=(1, 1))

    # translate, rotate, ... expect the same parameters as shown above.
    # the stubs are chained with "|" and are applied from left to right.
    # the 'center' which is used in the rotate transformation is the center after the translation
    transformed_line = line.transformed(
        translate((1, 1)) | rotate(np.pi / 2, origin='center') | mirror((1, 0)) | scale(2)
    )

This is more convenient than calling the transformation methods individually.

Defining a custom pivot position
++++++++++++++++++++++++++++++++

Every shape has a :attr:`~fibomat.shapes.Shape.pivot` property. By default, it is equal to the :attr:`~fibomat.shapes.Shape.center` property. In contrast to the center, the pivot can be customized by assigning a function to the property. This function takes the element itself as parameter ::

    line = Line(start=(1, 0), end=(1, 1))
    # line.pivot == line.center == (1, 0.5)

    line.pivot = lambda self: self.center + (1, 1)
    # line.pivot == (2, 1.5) != line.center == (1, 0.5)

    translated_line = line.translated((2, 3))
    # translated_line.pivot == (4, 4.5) != translated_line.center == (3, 3.5)

Additionally, the pivot can be used as origin of rotations and scalings ::

    rotated_line = line.rotated(np.pi / 2, origin='pivot')
    scaled_line = line.scaled(2, origin='pivot')

The pivot is also the point which is moved to a lattice point if a shape is placed in a lattice, which makes it useful when dealing with :doc:`groups and lattices <grouping_arranging>`.


Parametric curves
-----------------

:class:`~fibomat.shapes.ParametricCurve` is a shape which is defined by a parametric function. It can be transformed like all other shapes. Additionally, it can be approximated by an arc spline, which is the representation that is used for most operations.

The shape represents parametric functions of the type ``f: [a, b] -> R^2``. This function must be an element of ``C^3`` (three times continuously differentiable) in the interior of ``[a, b]``. At ``a`` and ``b``, the curve must be only ``C^0``. This means that the curve must not have any kinks, etc. ``[a, b]`` is the domain of the curve.

The easiest way to define a parametric curve is the :meth:`~fibomat.shapes.ParametricCurve.from_sympy_curve` class method.
This method takes a `sympy <https://sympy.org>`__ curve (:class:`sympy.geometry.curve.Curve`) which defines ``f``; all required derivatives and the other functions are calculated automatically. ::

    from fibomat.shapes import ParametricCurve

    # sympy example

    import sympy
    from sympy.abc import t
    from sympy import sin, cos

    curve = sympy.Curve(
        [2 * cos(2 * t) + 3 * cos(t), 2 * sin(2 * t) - 3 * sin(t)],
        (t, 0, 2 * np.pi)
    )

    parametric_curve = ParametricCurve.from_sympy_curve(curve)
    spline_from_sympy = parametric_curve.to_arc_spline(epsilon=0.1)

The arc length of the curve is integrated numerically.

A more involved method is to define the function ``f`` and its first two derivatives by hand.

These functions (``f``, ``df/du``, ``d^2f/du^2``) must take a numpy array ``u`` and return a numpy array with the shape ``(N, 2)``, which contains the calculated points, where ``N`` is the length of ``u``. For a scalar ``u`` the shape of the result is ``(2,)``. Optionally, the third derivative, the `curvature <https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization>`__ and the length of the curve can be passed. This speeds up things significantly.
The curvature function must take a numpy array ``u`` and return a numpy array with the curvatures (with the same shape as ``u``).
The length function must take two floats ``u_0`` and ``u_1`` and return the arc length of ``f`` between ``u_0`` and ``u_1``.

As an example, we take the function ``f(u) = (u, u**2)``. The derivatives are ``f'(u) = (1, 2*u)`` and ``f''(u) = (0, 2)``.
The curvature is given by ``k(u) = 2 / (1 + 4 u**2)**1.5`` and the length by ``L(u_0, u_1) = l(u_1) - l(u_0)`` with ``l(u) = u*sqrt(4*u**2 + 1)/2 + asinh(2*u)/4``.

The required functions can easily be defined with numpy. Special care must be taken if any of the derivatives is constant. ::

    import numpy as np

    def f(u):
        u = np.asarray(u)
        return np.array((u, u**2)).T


    def df(u):
        u = np.asarray(u)
        return np.array((np.full_like(u, 1), 2 * u)).T


    def d2f(u):
        u = np.asarray(u)
        return np.array((np.full_like(u, 0), np.full_like(u, 2))).T


    def curvature(u):
        u = np.asarray(u)
        return 2 / (1 + 4 * u**2)**1.5


    def length_impl(u):
        return u * np.sqrt(4 * u**2 + 1) / 2 + np.arcsinh(2 * u) / 4


    def length(u_0, u_1):
        return length_impl(u_1) - length_impl(u_0)

    parametric_curve = ParametricCurve(
        f, df, d2f, domain=(0, 1), curvature=curvature, length=length
    )

All parametric curves can be rasterized with equidistant points. The method :meth:`~fibomat.shapes.ParametricCurve.rasterize` returns the values in the parametric domain where the points are located and :meth:`~fibomat.shapes.ParametricCurve.rasterize_at` returns the actual points in ``R^2``.

Parametric curves are approximated by arc splines with :meth:`~fibomat.shapes.ParametricCurve.to_arc_spline`. The parameter ``epsilon`` is the maximum distance between the parametric curve and the arc spline; the distance is guaranteed.

Both examples can be found in `examples/parametric_curve.py <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/parametric_curve.py>`__.

If a curve has kinks, it can be split at them and both parts can be treated individually.
Finally, the approximations are stitched together with the :meth:`~fibomat.shapes.ArcSpline.from_segments` class method of the arc spline ::

    parametric_curve_part_1 = ParametricCurve(...)
    parametric_curve_part_2 = ParametricCurve(...)

    arc_spline = ArcSpline.from_segments([
        parametric_curve_part_1.to_arc_spline(epsilon=0.1),
        parametric_curve_part_2.to_arc_spline(epsilon=0.1),
    ])

Operations on :class:`~fibomat.shapes.ArcSpline`\ s
----------------------------------------------------

All shapes except spots can be converted to arc splines. For this, shapes provide the method ``to_arc_spline``, e.g. ::

    line = shapes.Line(start=(0, 0), end=(1, 1))

    line_as_arc_spline = line.to_arc_spline()

All curve tools are implemented in the :mod:`~fibomat.curve_tools` submodule. The geometry calculations are done by a compiled extension (written in Rust), hence they are fast.

Intersections
+++++++++++++

With the :func:`~fibomat.curve_tools.curve_intersections` function, the intersections of two curves can be calculated.
The function returns a :class:`~fibomat.curve_tools.CurveIntersections` object with the lists ``points`` and ``overlaps``. The first one contains the intersection points (:class:`~fibomat.curve_tools.Intersection`), the latter one contains the intervals where the two curves are identical (:class:`~fibomat.curve_tools.Overlap`).
Each entry holds the position and the indices of the segments of the curves. ::

    result = curve_tools.curve_intersections(spline, line)

    for intersection in result.points:
        print(intersection.position, intersection.index_1, intersection.index_2)

Additionally, self-intersections can be calculated with the :func:`~fibomat.curve_tools.self_intersections` function. It returns a list of :class:`~fibomat.curve_tools.Intersection` objects.

.. fibomat-plot:: ../examples/curve_intersections.py


Boolean operations
++++++++++++++++++

fib-o-mat supports the modes union, xor, exclude and intersect (see :class:`~fibomat.curve_tools.CombineMode`) of the :func:`~fibomat.curve_tools.combine_curves` function. The function returns a :class:`~fibomat.curve_tools.CombineResult` with the lists ``boundaries`` and ``holes``. They contain the outer outlines of the regions of the result and the outlines of the holes which are enclosed by the regions, respectively. All entries are arc splines. ::

    from fibomat.curve_tools import CombineMode

    result = curve_tools.combine_curves(circle_1, circle_2, CombineMode.UNION)  # or simply 'union'
    result.boundaries  # [ArcSpline]
    result.holes  # []

In the example below, these operations are applied to two overlapping circles.

.. warning:: Boolean operations can only be applied to closed arc splines.

.. fibomat-plot:: ../examples/boolean_operations.py

Offsetting
++++++++++

Offsetting can be done with the :func:`~fibomat.curve_tools.inflate` and :func:`~fibomat.curve_tools.deflate` functions.
They offset a given closed arc spline outwards and inwards, respectively.
For this, the offset pitch must be defined and the number of offset steps or the total offset distance. If an arc spline is deflated, the number of steps or the distance can be left out. In this case, the spline is deflated until nothing is left.

The function :func:`~fibomat.curve_tools.offset` moves a single curve by a distance; closed curves are moved inwards or outwards, open ones to the left or right of their direction (see :class:`~fibomat.curve_tools.OffsetDirection`). :func:`~fibomat.curve_tools.offset_with_islands` offsets regions with islands.

.. fibomat-plot:: ../examples/offsetting.py


Rasterizing
+++++++++++

If needed, shape outlines can be rasterized manually with the :func:`~fibomat.curve_tools.rasterize` function. The distance between two consecutive points, measured along the curve, is the pitch.

For visualization purposes, the rasterized points are displayed with :class:`~fibomat.shapes.Spot`\ s here.

.. fibomat-plot:: ../examples/rasterize.py

Fill with lines
+++++++++++++++

Closed arc splines (also arc splines with holes) can be filled with parallel lines with the :func:`~fibomat.curve_tools.fill_with_lines` function.

The function sweeps a line over the spline. For every step, the sweeping line is trimmed at the outline of the arc spline. All remaining parts inside the spline are collected in a list (this list is called a row in the following). The sweeping is repeated until the complete spline is swept.

Besides the enclosing spline and a pitch, the function requires the angle of the filling lines to the x-axis and the boolean argument ``invert``, which defines the sweeping direction. See the figure below for further explanation. The function returns a list of lists where each entry of the outer list contains all lines of a row. One of the (infinite) lines passes through the optional point ``seed`` (by default the center of the shape), so the lines are always aligned to the same grid.

.. list-table:: Influence of the ``alpha`` and ``invert`` parameters on the filling lines. The big arrows indicate the sweeping direction which is used to determine the filling lines. The returned rows are sorted ascending in the sweeping direction.

    * - .. figure:: /_static/alpha_ge.png
            :height: 250px

      - .. figure:: /_static/alpha_l.png
            :height: 250px

A line with the height ``y`` (measured perpendicular to the lines) is part of the fill if ``y_min <= y < y_max``; a line which lies exactly on the upper boundary of a shape is not part of the fill, while a line on the lower boundary is.

The example below fills a rectangle and an arc spline with holes.

.. fibomat-plot:: ../examples/fill_with_lines.py

Smoothing
+++++++++

Arc splines with kinks can be smoothed with the :func:`~fibomat.curve_tools.smooth` function. Corners are replaced with arcs which have a user-defined radius and touch the adjacent segments tangentially. If this is not possible (e.g. because the radius is too large for the adjacent segments or the curve has a cusp), a :class:`~fibomat.curve_tools.NonSmoothableError` is raised.

.. fibomat-plot:: ../examples/smoothing.py


Shapes with holes and rings
---------------------------

A :class:`~fibomat.composite_shapes.HollowArcSpline` is a closed arc spline (the boundary) with holes. Holes which overlap with the boundary or with each other are combined with Boolean operations automatically. The result must be simply connected: holes may not separate the shape into several parts.

A :class:`~fibomat.composite_shapes.Ring` is a circular ring. It is converted to a hollow arc spline with :meth:`~fibomat.composite_shapes.Ring.to_hollow_arc_spline`. ::

    from fibomat.composite_shapes import HollowArcSpline, Ring

    hollow = HollowArcSpline(boundary=Rect(4, 4).to_arc_spline(), holes=[Circle(r=1).to_arc_spline()])
    ring = Ring(r_outer=2, thickness=0.5)

Text
----

fib-o-mat supports simple text blocks composed of glyphs of the Hershey font family. By default, the glyphs are composed of polylines. Consequently, the text is one-dimensional and can be used with any backend supporting polylines. If a stroke width is given, the strokes are united and the glyphs become polygons (or shapes with holes), which can be filled.

To create a text, use the :class:`~fibomat.composite_shapes.Text` class. Use the ``font_size`` parameter to scale the text. By default, the height of upper-case letters is 1.
The text is placed at ``(0, 0)`` by default. A text can be transformed like a shape (e.g. it can be translated). Its glyphs are available as :class:`~fibomat.composite_shapes.Glyph`\ s (``text.glyphs``). If a text is multiplied with a unit, it can be used as the shape of a pattern and plotted.

.. fibomat-plot:: ../examples/text.py

To get further control of the positioning, the :class:`~fibomat.composite_shapes.Text` class has the method :meth:`~fibomat.composite_shapes.Text.baseline_anchor`. With it, the left, right and center positions of the baseline can be accessed.

See the example below.

.. fibomat-plot:: ../examples/text_positioning.py


Importing shapes from vector graphic files
------------------------------------------

The submodule ``fibomat.from_file`` can import shapes from vector graphic files (SVG and DXF; it requires the extra ``io``, see :doc:`installation`). The documentation of this feature will be added after the module has been reworked.
