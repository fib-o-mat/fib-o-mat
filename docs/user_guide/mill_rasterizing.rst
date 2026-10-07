Mill & rasterization settings
=============================

To create a complete :class:`~fibomat.layout.Pattern`, a :class:`~fibomat.mill.Mill` and a :class:`~fibomat.raster_styles.RasterStyle` must be defined along with a patterning shape.

In a pattern, the following pieces of information are collected:

    1. what should be rasterized (geometric shape)
    2. in which way the shape should be rasterized (rasterization style, pitches)
    3. how the rasterized shape should be milled (dwell time and number of repeats)

Defining a mill
---------------

In the most simple case, the :class:`~fibomat.mill.Mill` takes the dwell time per spot and the number of repeats of a shape as attributes. ::

    from fibomat.mill import Mill
    from fibomat.units import unit

    mill = Mill(dwell_time=1 * unit('ms'), repeats=5)

The dwell time must be a dimensioned value with the dimension time and the number of repeats an integer which is at least 1. Both are available as properties (``mill.dwell_time``, ``mill.repeats``).

|:test_tube:| Providing custom parameters to a mill object
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Backends and custom raster styles may need parameters which are not part of the :class:`~fibomat.mill.Mill`, e.g. the dose or the scan direction. For this, :class:`~fibomat.mill.MillBase` is the base class of all mills: it stores named settings which are read with ``mill['name']``. :class:`~fibomat.mill.SpecialMill` can be used to store arbitrary settings; own mill classes can be derived from :class:`~fibomat.mill.MillBase`. ::

    from fibomat.mill import SpecialMill

    special_mill = SpecialMill(dwell_time=1 * unit('ms'), repeats=5, use_flood_gun=True, defocus=20)

    # ...

    # access the extra parameters at a later point
    print(special_mill['use_flood_gun'])
    print(special_mill['defocus'])

Reading a setting which does not exist (or is ``None``) raises a ``KeyError`` with a hint. In the section about extending fib-o-mat, an example is given.

.. note:: The raster styles and backends which are part of fib-o-mat need a :class:`~fibomat.mill.Mill` (a constant dwell time) or a mill which is derived from it. Mills with a position dependent dwell time (:class:`~fibomat.mill.DDDMill`, :class:`~fibomat.mill.SILMill`) are not supported yet.

|:test_tube:| Defining an ion beam shape
++++++++++++++++++++++++++++++++++++++++

For dose calculations or optimization routines, for example, the beam shape must be known.
For this, the fib-o-mat package provides the :class:`~fibomat.mill.GaussBeam` class. As the name indicates, this class describes the ion beam with a Gaussian shape. A custom beam profile is defined by deriving from :class:`~fibomat.mill.IonBeam`.

A :class:`~fibomat.mill.GaussBeam` is defined by the full width at half maximum of the beam and the total beam current. ::

    from fibomat.mill import GaussBeam

    beam = GaussBeam(fwhm=3 * unit('nm'), current=1 * unit('pA'))

The class provides some utility methods, which are explained in the code snippet below ::

    # returns the standard deviation of the distribution
    print(beam.std)

    # Calculate the ion flux at the position (0, 0) µm for spots at (-1, 0), (0, 0), (1, 0) µm,
    # hence, the influence of the surrounding spots of the spot at (0, 0) is calculated.
    fluxes, flux_unit = beam.flux_at((0, 0), [(-1, 0), (0, 0), (1, 0)], unit('µm'))

    # Calculate the ion flux of a single, isolated spot.
    # This is the same as calling beam.flux_at((0, 0), (0, 0), unit('µm'))
    print(beam.nominal_flux_per_spot())

    # Calculate the ion flux of a spot on a line with the pitch 1 nm
    print(beam.nominal_flux_per_spot_on_line(1 * unit('nm')))

    # Calculate the ion flux of a spot on a rectangular grid with the pitches 1 nm and 1 nm in x and y direction, respectively.
    print(beam.nominal_flux_per_spot_in_rect(1 * unit('nm'), 1 * unit('nm')))

The ``nominal_flux_*`` methods can be used to calculate a nominal flux for optimization routines or to calculate the ion dose on a regularly rasterized line or grid.


Specifying the rasterization style
----------------------------------

The rasterization styles define how a shape is rasterized. Different raster styles are predefined for different dimensions. The creation of custom raster styles is explained in :doc:`extending/custom_raster_style`.

The raster styles which are part of the fib-o-mat package are introduced in the following. The pitches of all raster styles are dimensioned values (e.g. ``50 * unit('nm')``), and ``scan_sequence`` is a member of :class:`~fibomat.raster_styles.ScanSequence` (or its string value, e.g. ``'serpentine'``).

The subsections refer to examples in the git repository at `<https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/raster_styles>`__. The examples can be executed with

.. code-block:: bash

    $ python examples/raster_styles/spot.py && beam_simulation rasterized.txt

if the current directory is the root of the fib-o-mat repository. ``spot.py`` can be replaced by all other scripts in the ``examples/raster_styles`` directory. See also :ref:`ion beam simulation <user_guide/exporting_visualization:ion beam simulation>`.

Zero-dimensional
++++++++++++++++

    - :class:`~fibomat.raster_styles.zero_d.SingleSpot`
    - :class:`~fibomat.raster_styles.zero_d.PreRasterized`

Both zero-dimensional raster styles do not take any parameters. These raster styles can only be used for :class:`~fibomat.shapes.Spot`\ s and pre-rasterized objects (:class:`~fibomat.shapes.RasterizedPoints`), respectively. A spot is exposed ``repeats`` times. For pre-rasterized points, the weight of each point is multiplied with the dwell time and the whole sequence is repeated ``repeats`` times.

Examples:
    * `single spots <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/raster_styles/spot.py>`__
    * |:test_tube:| `manual rasterization <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/raster_styles/pre_rasterized.py>`__

One-dimensional
+++++++++++++++

The only raster style for one-dimensional shapes is the :class:`~fibomat.raster_styles.one_d.Curve` style.
This style expects a pitch (the distance between neighboring spots, measured along the curve) and a scan sequence. The first spot is the start of the curve. All three possible scan sequences are visualized below.

.. list-table:: Available scan sequences for one-dimensional shapes.

    * - .. figure:: /_static/consecutive_1d.png
            :height: 250px

      - .. figure:: /_static/back_stitch_1d.png
            :height: 250px

      - .. figure:: /_static/back_and_forth_1d.png
            :height: 250px

Examples:
    * `all one-dimensional styles <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/raster_styles/one_dim.py>`__

Two-dimensional
+++++++++++++++

fib-o-mat includes two different rasterization methods for two-dimensional shapes (:class:`~fibomat.raster_styles.two_d.LineByLine` and :class:`~fibomat.raster_styles.two_d.ContourParallel`).
Both rasterization styles fill a given shape with lines or curves. The order of these lines and curves is defined by a scan sequence.
All available scan sequences are shown below.


:class:`~fibomat.raster_styles.two_d.LineByLine` rasterization
****************************************************************

The line-by-line rasterization style rasterizes a closed shape by sweeping a line over it. This style is commonly supported by other (proprietary) patterning software.

Details on the method can be found in the description of the :func:`~fibomat.curve_tools.fill_with_lines` function, see :ref:`here <user_guide/geometric-shapes:fill with lines>`. Shapes with holes (:class:`~fibomat.composite_shapes.HollowArcSpline` and :class:`~fibomat.composite_shapes.Ring`) are supported.

.. list-table:: Available scan sequences for two-dimensional shapes.

    * - .. figure:: /_static/consecutive_2d.png
            :height: 250px

      - .. figure:: /_static/cross_section_2d.png
            :height: 250px

    * - .. figure:: /_static/serpentine_2d.png
            :height: 250px

      - .. figure:: /_static/double_serpentine_2d.png
            :height: 250px

    * - .. figure:: /_static/double_serpentine_same_path_2d.png
            :height: 250px

      - .. figure:: /_static/back_stitch_2d.png
            :height: 250px

The scan sequences in the figure above only define the order of the individual one-dimensional shapes which fill the two-dimensional shape.
In the figure, the two-dimensional shape is a rectangle which is filled with lines.
Hence, the two-dimensional rasterization styles also require a one-dimensional rasterization style as parameter (among others), which is used for the filling lines.

For a double serpentine, the directions of the lines alternate over the whole sequence, so the beam never has to jump between the end of a pass and the start of the next one.


:class:`~fibomat.raster_styles.two_d.ContourParallel` offset rasterization
***************************************************************************

This style generates contour-parallel offset curves of the passed shape to rasterize it.

.. warning:: ``ContourParallel`` is an experimental feature, which is not reworked yet. It needs the optional dependency numba (``pip install "fibomat[experimental]"``); without it, a placeholder is used which raises an ``ImportError`` when it is instantiated.

.. To decrease the influence of artifacts due to offsetting, this rasterization style supports optimizing the rasterized dwell points. See the use case :ref:`use_cases/plasmonic_antennas:plasmonic tetramer antennas based on single-crystalline gold flakes` for a usage example of the optimization process.


Examples:
    * `various LineByLine styles <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/raster_styles/line_by_line.py>`__
    * `various ContourParallel styles <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/raster_styles/contour_parallel.py>`__

Spirals
*******

The :class:`~fibomat.raster_styles.two_d.Spiral` style fills a shape with an Archimedean spiral (``Spiral(pitch, spiral_pitch, scan_sequence, direction)``): ``pitch`` is the distance between the spots on the spiral and ``spiral_pitch`` the distance between the arms. The ``direction`` is ``'outwards'``, ``'inwards'`` or ``'out-in'``. This style is not reworked yet.
