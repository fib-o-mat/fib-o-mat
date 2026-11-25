Mill & rasterization settings
=============================

To create a complete :class:`~fibomat.layout.Pattern`, a :class:`~fibomat.mill.Mill` and a :class:`~fibomat.raster_styles.RasterStyle` must be defined along with a patterning shape.

In a pattern, the following pieces of information are collected:

    1. what should be rasterized (geometric shape)
    2. in which way the shape should be rasterized (rasterization style, pitches)
    3. how the rasterized shape should be milled (dwell time and number of repeats)

Defining a mill
---------------
The milling behaviour for a pattern is specified by a ``Mill`` object. The current implementation provides several related classes in
``fibomat.mill.mill``:

- ``Mill`` — simple, constant dwell time per spot and number of repeats
- ``DDDMill`` — base for a dwell-time-per-point function and repeats (used for position-dependent dwell)
- ``MatrixMill`` — use a 2D dwell-time matrix (image) as a rasterized dwell map
- ``SpecialMill`` — lightweight container for custom/back-end specific parameters
-  ``SILMill`` — example customized mill for fabrication of solid immersion lenses

Simple constant dwell time
~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a fixed dwell time per spot, create a ``Mill`` with a quantity and an integer number of repeats::

    from fibomat.mill import Mill
    from fibomat.units import unit

    mill = Mill(dwell_time=Q_('1 microsecond'), repeats=1)

Note that ``Mill`` stores the dwell-time as a (constant) function internally (it subclasses ``DDDMill``). If only milling one pattern, normally ``repeats`` will be 1 and multiple repeats of the milling process will instead be marked in the exported file in the end.
The dwell time must be a dimensioned value with the dimension time and the number of repeats an integer which is at least 1. Both are available as properties (``mill.dwell_time``, ``mill.repeats``).


Position-dependent dwell (DDDMill)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the dwell time depends on the physical location, use a ``DDDMill``-style object which holds a function
``dwell_time(point: np.ndarray) -> Quantity`` and an integer ``repeats``. The provided ``Mill`` above is a convenience
wrapper that builds a constant function for you. You can also subclass or construct a ``DDDMill`` directly.

Example of a custom DDD mill (callable returns a pint quantity)::

    def dwell_func(point):
        x, y = point[0], point[1]
        return Q_(max(0.1, 1.0 - (x**2 + y**2)), 'microsecond')

    from fibomat.mill.mill import DDDMill
    custom = DDDMill(dwell_time=dwell_func, repeats=1)


Matrix-based dwell map: ``MatrixMill``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``MatrixMill`` allows supplying a 2D numpy array of dwell values (e.g. from an image) together with a pixel size ``dx``
and an ``origin``. It converts a point (x, y) to matrix indices and returns the corresponding dwell as a pint quantity.

.. warning:: ``MatrixMill`` is still under development and not yet fully supported.

Extra/back-end parameters (SpecialMill)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you need to bundle arbitrary extra parameters for a back-end, use ``SpecialMill`` which is a small container built on
``MillBase``. Customized ack-ends may choose to read these attributes but note that built-in back-ends probably won't work with this mill.

Use-case specific: ``SILMill``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``SILMill`` is a specialised DDDMill that implements a radial dwell profile for a SIL geometry. Find documentation on usage in :doc:`../use_cases/sil_mill_guide` .


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

.. note:: The raster styles which are part of fib-o-mat need a :class:`~fibomat.mill.Mill` (a constant dwell time) or a mill which is derived from it. Mills with a position dependent dwell time (:class:`~fibomat.mill.DDDMill`, :class:`~fibomat.mill.SILMill`) are not supported yet.

Defining a mill by its dose
+++++++++++++++++++++++++++

Some patterning systems calculate the number of repeats from a dose (e.g. NPVE). For these, the :class:`~fibomat.mill.DoseMill` is defined by exactly two of the three settings dwell time, number of repeats and dose ::

    from fibomat.mill import DoseMill

    mill = DoseMill(dwell_time=1 * unit('µs'), dose=2 * unit('ions / nm**2'))
    mill = DoseMill(repeats=10, dose=100 * unit('ions'))
    mill = DoseMill(dwell_time=1 * unit('µs'), repeats=10)

The setting which is not given is ``None`` (and ``mill['dose']`` etc. raises a ``KeyError``); it is not calculated by the mill, because this needs the beam current and the pitches of the raster style. The dose is a charge per spot (e.g. ``unit('ions')``), per length (curves, e.g. ``unit('nC / µm')``) or per area (areas, e.g. ``unit('ions / nm**2')``).

A :class:`~fibomat.mill.DoseMill` is meant for backends which pass the settings to the patterning software; it cannot be used with the raster styles of fib-o-mat, which need a :class:`~fibomat.mill.Mill`. The ``NPVEMill`` of the NPVE step and repeat backend is a :class:`~fibomat.mill.DoseMill` which always needs a dwell time. Its parameter ``scan_direction`` is deprecated.


Defining an ion beam shape
--------------------------
For dose calculations or optimization routines the beam shape is required. The package provides ion beam descriptions
under ``fibomat.mill.ionbeam`` (for example ``GaussBeam``). A Gaussian beam is constructed from a full-width-half-maximum
and a current::

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

:class:`~fibomat.raster_styles.two_d.spiral.Spiral` spiral rasterization
**************************************************************************
The spiral rasterstyle rasterizes a shape by taking the circumcircle of the shape's bounding box, then filling this circle with an archimedian spiral and rasterizing this spiral as a 1-dim shape. This style expects two pitch-parameters: The pitch determines with what pitch the spiral should be rasterized, the spiral pitch determines how dense the spiral arms are supposed to be. Furthermore the user can decide in which direction the spiral shall be rasterized: From the inside to the outside, in reverse direction, or from inside to outside and then back in the reversed sense. See :doc:`../use_cases/sil_mill_guide` for an example. 
.. note:: The spiral rasterstyle is still under development and so far, no matter which scan style the user passes over, the consecutive scan is used. Also there might be problems with complex shapes, eg. shapes with holes. Please report errors via git issue.





Examples:
    * `various LineByLine styles <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/raster_styles/line_by_line.py>`__
    * `various ContourParallel styles <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/raster_styles/contour_parallel.py>`__

Spirals
*******

The :class:`~fibomat.raster_styles.two_d.Spiral` style fills a shape with an Archimedean spiral (``Spiral(pitch, spiral_pitch, scan_sequence, direction)``): ``pitch`` is the distance between the spots on the spiral and ``spiral_pitch`` the distance between the arms. The ``direction`` is ``'outwards'``, ``'inwards'`` or ``'out-in'``. This style is not reworked yet.
