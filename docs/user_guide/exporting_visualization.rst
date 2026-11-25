Exporting & visualization
=========================

Exporting is handled by special exporting classes, the *backends*. These classes implement all the functionality to map the internal representation of a pattern design in fib-o-mat to any kind of output format.

Exporting is invoked via the :meth:`~fibomat.layout.Layout.export` method of the :class:`~fibomat.layout.Layout` class. The backend is passed as a class (backends are not registered by name).

Internally, the following happens on export:

    1. Loop over every site in the layout
    2. Add every pattern to the exporting backend

To get started with exporting, see the examples below. Further details on exporting can be found in the extending section.


Exporting microscope readable output
------------------------------------
Currently, the fib-o-mat package includes one generic backend to generate microscope readable output: the :class:`~fibomat.default_backends.SpotListBackend`. It can be used to export rasterized patterns and is highly adaptable, as demonstrated below. For specific instruments, the packages ``fibomat.default_backends.npve`` (step-and-repeat and deflection lists) and ``fibomat.default_backends.fei`` (stream files) provide derived backends.

See :doc:`extending/custom_backends` for an example of how to write a custom exporting backend from scratch.

The :class:`~fibomat.default_backends.SpotListBackend` rasterizes all patterns. In doing so, all added sites are merged. The patterning order of the individual patterns is given by the order of adding the patterns to sites and the sites to the layout.

To illustrate the different settings of the backend, we start with a simple pattern design ::

    from fibomat.layout import Layout
    from fibomat.linalg import Vector
    from fibomat.mill import Mill
    from fibomat.units import unit
    from fibomat import default_backends, shapes, raster_styles


    layout = Layout()
    site = layout.create_site(
        dim_position=Vector(0, 0) * unit('µm'), dim_fov=Vector(5, 5) * unit('µm')
    )

    mill = Mill(dwell_time=1 * unit('ms'), repeats=4)

    site.create_pattern(
        dim_shape=shapes.Line((-2, -2), (2, 2)) * unit('µm'),
        mill=mill,
        raster_style=raster_styles.one_d.Curve(
            pitch=1 * unit('nm'),
            scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
        )
    )

    exported = layout.export(default_backends.SpotListBackend)
    exported.save('file.txt')

Source: `https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/exporting_simple.py <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/exporting_simple.py>`__

This will generate a file with the following content

.. code-block:: none

    [Info]
    length_unit = µm
    time_unit = µs
    fov = 3.999395954391112, 3.999395954391112
    center = (-0.0003020228044439133, -0.0003020228044439133)
    base_dwell_time = None
    total_dwell_time = 22628000.0
    number_of_points = 22628
    description = None
    time_stamp = 2026-10-07 21:02:31.741799

    [Points]
    -2.00000 -2.00000 1000
    -1.99929 -1.99929 1000
    -1.99859 -1.99859 1000
    -1.99788 -1.99788 1000
    -1.99717 -1.99717 1000
    -1.99646 -1.99646 1000
    -1.99576 -1.99576 1000
    -1.99505 -1.99505 1000
    ...

This output is most likely not tremendously useful, besides using it to visualize the rasterized data in the
:ref:`ion beam simulation tool <user_guide/exporting_visualization:ion beam simulation>`.

The second backend :class:`~fibomat.default_backends.fei.FEIStreamFile` produces streamfiles tailored to a specific microscope with the following format:

.. code-block:: none

    s16,25ns
    Number of Points in File
    Number of times the microscope shall repeat the file
    t1 x1 y1
    ...
That is, the dwell times are automatically converted to units of 25 ns. The field of view gets included in the filename. The number of repeats as well as a margin (the ratio of the field of view that should be filled by the pattern) have to be handed over as a parameter.

::
    # Export to FEI stream file
    exported = site.export(FEIStreamFile, n_rep=684, margin=0.76)
    exported.save('sil_pattern.str')

.. note:: The FEIStream backend is still under development and has some bugs left, for example it causes errors if the bounding box of the shape is not twodimensional.

Customize output format
+++++++++++++++++++++++

The :class:`~fibomat.default_backends.SpotListBackend` is highly customizable. In the following, all customization options are introduced.
All settings are passed to the backend via the :meth:`~fibomat.layout.Layout.export` call of the :class:`~fibomat.layout.Layout` class.

The following parameters can be set

    * ``save_impl``: allows to pass a custom saving function which handles the writing to a file. Thereby, the file format can be set.
    * ``base_dwell_time``: if set, all dwell times are divided by the base dwell time. Hence, the dwell times are expressed as integer multiples of the base dwell time.
    * ``length_unit``: dwell points are converted to this unit before saving (default: µm)
    * ``time_unit``: dwell times are converted to this unit before saving (default: µs, not used if ``base_dwell_time`` is given)
    * ``description``: a description which is available to the ``save_impl`` function. By default, the description of the layout is used.

Now, assume that we would like to export to a file with the layout

.. code-block:: none

    CUSTOMFILEFORMAT
    FOV=...
    DWELL=...
    BEGIN_DWELL_POINTS
    x1, y1, t1
    ...
    END_DWELL_POINTS

where FOV is the needed field of view and DWELL the base dwell time. ``x_n`` and ``y_n`` are the dwell point positions given in µm and ``t_n`` are the dwell times, given as multiples of the base dwell time (= 0.1 µs).

To use the correct lengths and time multiples in the exported file, we only need to add these parameters to the exporting function ::

    exported = layout.export(
        default_backends.SpotListBackend,
        base_dwell_time=0.1 * unit('µs'),
        length_unit=unit('µm')
    )


Finally, a custom ``save_impl`` is missing. The ``save_impl`` function expects three parameters:
a filename, a numpy array with the dwell points and, last, a dictionary which contains some useful information about the rasterized pattern and the backend's settings. See the :class:`~fibomat.default_backends.SpotListBackend` documentation for the available keys. ::

    from typing import Any, Dict

    import numpy as np

    from fibomat import units, utils


    def custom_save_impl(filename: utils.PathLike, dwell_points: np.ndarray, parameters: Dict[str, Any]):
        # fov is in units of length_unit
        fov = max(parameters["fov"].width, parameters["fov"].height)

        base_dwell_time = units.scale_to(unit('µs'), parameters["base_dwell_time"])

        with open(filename, 'w') as fp:
            # first, write header data.
            fp.writelines([
                'CUSTOMFILEFORMAT\n',
                f'FOV={fov:.3f}\n',
                f'DWELL={base_dwell_time:.1f}\n',
                'BEGIN_DWELL_POINTS\n'
            ])

            # second, write dwell point data
            # dwell_points has shape (N, 3) where N is the number of dwell points. Each row in the array contains
            # (x, y, t_d) where x and y are the position of a spot and t_d the dwell time or dwell time multiple.
            # "%.5f %.5f %d" is the formatting string. See the numpy docs for details.
            np.savetxt(fp, dwell_points, "%.5f %.5f %d")

            fp.write('END_DWELL_POINTS\n')


If added to the export function ::

    exported = layout.export(
        default_backends.SpotListBackend,
        base_dwell_time=0.1 * unit('µs'),
        length_unit=unit('µm'),
        save_impl=custom_save_impl
    )

    exported.save('file.txt')

the export yields a file with the content

.. code-block:: none

    CUSTOMFILEFORMAT
    FOV=3.999
    DWELL=0.1
    BEGIN_DWELL_POINTS
    -2.00000 -2.00000 10000
    -1.99929 -1.99929 10000
    -1.99859 -1.99859 10000
    -1.99788 -1.99788 10000
    -1.99717 -1.99717 10000
    -1.99646 -1.99646 10000
    ...
    END_DWELL_POINTS

Source: `https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/exporting_advanced.py <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/exporting_advanced.py>`__


NPVE shape files and step and repeat lists
++++++++++++++++++++++++++++++++++++++++++

.. danger:: **The doses of spots and lines were wrong in fib-o-mat 0.7.1 and older.**

    NPVE converts the doses of spots and lines with the spot size :math:`d` of its active beam. Up to version 0.7.1, fib-o-mat converted them for a fixed spot size of 100 nm, so NPVE applied

    * to lines: :math:`\lambda_\text{NPVE} = \lambda_\text{script} \cdot \dfrac{d}{100\,\text{nm}}`
    * to spots: :math:`Q_\text{NPVE} = Q_\text{script} \cdot \left(\dfrac{d}{100\,\text{nm}}\right)^2`

    With a spot size of 0.5 nm, a line got 0.5 % and a spot 0.0025 % of the dose in the script. Area doses and patterns which are defined by the number of repeats are not affected. Files with spots or lines with a dose which were exported with these versions must be exported again.


Two backends in :mod:`fibomat.default_backends.npve` write XML files for the NPVE patterning software:

    * :class:`~fibomat.default_backends.npve.NPVESingleSite` writes a shape file (``*.nps``) with the patterns of one site, relative to the center of the site. The file contains neither a stage position nor a field of view. A layout with several non-empty sites raises a ``ValueError``; use :meth:`~fibomat.layout.Layout.export_multi` to get one file per site.
    * :class:`~fibomat.default_backends.npve.NPVEStepAndRepeat` writes a step and repeat list with all sites and the stage moves between them. It replaces the deprecated ``StepAndRepeatBackend``.

Both backends support the same shapes and raster styles and need an ``NPVEMill``. ::

    from fibomat.default_backends.npve import NPVEMill, NPVESingleSite, NPVEStepAndRepeat

    mill = NPVEMill(dwell_time=1 * unit('µs'), repeats=2)

    # ...

    layout.export(NPVESingleSite).save('patterns.nps')
    layout.export(NPVEStepAndRepeat).save('patterns.xml')

The header of a shape file can be set with the arguments ``system_name``, ``beam`` and ``aperture`` of :class:`~fibomat.default_backends.npve.NPVESingleSite`.

Both backends validate the patterns on export: the dose of the mill must fit to the raster style, i.e. a spot dose (e.g. ``unit('ions')``) for spots, a line dose (e.g. ``unit('nC / µm')``) for curves and an area dose (e.g. ``unit('ions / nm**2')``) for areas. Otherwise, a :class:`~fibomat.units.DimensionError` is raised.

Doses of spots and lines
************************

NPVE holds every dose as area dose (nC/µm², the value ``TargetDose`` in the file). It converts the doses of spots and lines with the spot size of its active beam. Therefore, both backends need the spot size of the beam which is used for patterning if the layout contains spots or lines with a dose (``spot_size``, as shown by NPVE). Mills with repeats and areas do not need it. ::

    layout.export(NPVESingleSite, spot_size=0.5 * unit('nm'))

With the spot size :math:`s` which is passed to the backend, a dose of the script is written as

========  =========================  ===========================================  ===============================================
Shape     Dose in the script         Written to the file (nC/µm²)                 Example (:math:`s` = 0.5 nm)
========  =========================  ===========================================  ===============================================
line      :math:`\lambda` (nC/µm)    :math:`\lambda / s`                          ``5e-4 * unit('nC / µm')`` → 1.0
spot      :math:`Q` (nC, ions, ...)  :math:`Q / (\pi s^2 / 4)`                    ``1963.5 * unit('pA * µs')`` → 10.0
area      :math:`\sigma` (nC/µm²)    :math:`\sigma`                               ``5 * unit('nC / µm**2')`` → 5.0
========  =========================  ===========================================  ===============================================

When the pattern runs, NPVE converts the value back with the spot size :math:`d` of its active beam. The dose which NPVE applies is therefore

* for lines: :math:`\lambda_\text{NPVE} = \lambda \cdot d / s`
* for spots: :math:`Q_\text{NPVE} = Q \cdot (d / s)^2`

It is only the dose of the script if :math:`s = d`, i.e. if the file was exported with the spot size of the beam which is active in NPVE. If you change the aperture (or anything else which changes the spot size), export the file again.

For a spot with a dose, NPVE uses one pass and calculates the dwell time from the dose and the beam current; the dwell time of the mill is ignored.

Deflection lists
****************

A pattern with the shape :class:`~fibomat.shapes.RasterizedPoints` and the raster style :class:`~fibomat.raster_styles.zero_d.PreRasterized` is written as deflection list by both backends. The dwell time of a point is the dwell time of the mill multiplied by the weight of the point, so the weights must be positive integers. The mill must define the number of repeats; a dose is not supported. ::

    import numpy as np

    from fibomat.raster_styles import zero_d
    from fibomat.shapes import RasterizedPoints

    # x, y, multiple of the dwell time
    points = np.array([[0.0, 0.0, 1], [0.1, 0.0, 2], [0.2, 0.0, 4]])

    site.create_pattern(RasterizedPoints(points, False) * unit('µm'), mill, zero_d.PreRasterized())

Lists with up to ``max_embedded_points`` points (default: 5000) are embedded in the file. Longer lists are written to a binary file next to the file, called ``<file name>_<shape id>.dat``; keep it in the same folder. The binary file contains two unsigned integers (``1`` and the number of points), followed by ``x``, ``y`` (32 bit floats in µm) and the repeat value (32 bit unsigned integer) of each point. All values are little endian. Set ``max_embedded_points=0`` to write all lists to binary files. In both cases the positions are stored with single precision.

Export sites individually
+++++++++++++++++++++++++

The method :meth:`~fibomat.layout.Layout.export_multi` exports every site individually and returns a list containing one instance of the exporting backend for every site. ::

    from fibomat.layout import Layout
    from fibomat import default_backends

    layout = Layout()

    # ...

    exported_sites = layout.export_multi(default_backends.SpotListBackend)

    for i, exported in enumerate(exported_sites):
        exported.save(f'file_{i}.txt')  # save a txt file for each exported site.

The method :meth:`~fibomat.layout.Layout.export_with_description` exports only the sites whose description matches one of the given regular expressions (checked with :func:`re.match`). ::

    from fibomat.layout import Layout
    from fibomat import default_backends

    layout = Layout()

    # ...

    descr_pattern = {
        'foo',    # this matches sites where the description starts with 'foo', e.g. 'foo_1', 'foo', 'foobar'
        r'^bar$'  # this matches sites where the description is exactly 'bar'
    }
    exported = layout.export_with_description(default_backends.SpotListBackend, descr_pattern=descr_pattern)

    # do something with the exported sites

A ``ValueError`` is raised if no site matches. Use `<https://regex101.com/>`__, for example, to test regular expressions.

Visualization
-------------

Generating interactive plots
++++++++++++++++++++++++++++

fib-o-mat ships with a default plotting backend, the :class:`~fibomat.default_backends.BokehBackend`. This backend is based on the `bokeh <https://bokeh.org/>`__ library.
The backend generates an interactive, self-contained html file which is viewable in any modern browser. The file depends neither on the fib-o-mat
python package nor on an internet connection (all JavaScript is embedded). Hence, it can be distributed and used easily.

Plots can be generated via the :meth:`~fibomat.layout.Layout.plot` method of the :class:`~fibomat.layout.Layout` class.
The plotting can be configured with the following parameters:

    * ``filename``: if given, the plot is saved with the given name. Defaults to ``None``.
    * ``show``: if ``True``, the plot is opened in a browser window. Defaults to ``True``.
    * ``descr_pattern``: if given, only sites whose description matches one of the regular expressions are plotted.

All further keyword arguments are passed to the :class:`~fibomat.default_backends.BokehBackend`:

    * ``unit``: the length unit of the axes, by default µm
    * ``title``: title of the plot, by default the description of the layout
    * ``hide_sites``: if ``True``, :class:`~fibomat.layout.Site`\ s are not shown in the plot, by default ``False``
    * ``rasterize_pitch``: the pitch used to rasterize all plotting data to polylines, by default 0.01 µm
    * ``plot_rasterized``: if True, the pattern will be rasterized using the pattern's rasterstyle and the resulting points are plotted.
    * ``legend``: if ``True``, a legend with all sites as entries is shown, by default ``True``
    * ``cycle_colors``: if ``True``, each site and all its shapes get a different color, by default ``True``

See the :class:`~fibomat.default_backends.BokehBackend` documentation for the remaining options.
In the :doc:`getting started guide <../getting_started>`, all features of the generated plot are explained.

To save an html file, use ::

    layout = Layout()

    # ...

    layout.plot(filename='my_plot.html')
    # or
    layout.plot(filename='my_plot.html', show=False, hide_sites=True, legend=False)

.. warning:: Only the raw shapes are plotted, not shapes which a patterning backend would generate from them.

Annotation layer
****************

To add annotations to a plot, the :class:`~fibomat.layout.Layout` class has an :meth:`~fibomat.layout.Layout.add_annotation` method.
Annotations are only used for plotting; no other backend can access them.
Any shape introduced in :doc:`geometric-shapes` can be used as an annotation.
The shape can be drawn filled and its color can be customized.
To define the scaling of the annotations, they must be equipped with a length unit (similar to shapes in a :class:`~fibomat.layout.Pattern`).
The position of an annotation is always given in the global coordinate system. ::

    from fibomat.layout import Layout
    from fibomat.shapes import Line, Circle
    from fibomat.units import unit

    layout = Layout()

    layout.add_annotation(Line((0, 1), (1, 1)) * unit('µm'))
    layout.add_annotation(Circle(r=1, center=(1, 1)) * unit('µm'), filled=True, color='blue')

    layout.plot()

Colors must be defined in a way that bokeh can understand them (cf. the `bokeh documentation <https://docs.bokeh.org/en/latest/docs/user_guide/styling.html#specifying-colors>`__).

Vector graphics (SVG)
+++++++++++++++++++++

The :class:`~fibomat.default_backends.SVGBackend` draws a layout as svg file, e.g. for publications or to edit the design in `Inkscape <https://inkscape.org>`__. Every site is an Inkscape layer. Shapes with a two-dimensional raster style are drawn filled, all other shapes as lines and spots as dots. Arcs are exported as exact svg arcs. ::

    from fibomat.default_backends import SVGBackend

    layout.export(SVGBackend).save('layout.svg')

The most important options are

    * ``pixel_size``: length of one pixel (svg user unit), by default 0.01 µm,
    * ``show_fov``: if ``True`` (default), the fields of view of the sites are drawn as dashed rectangles,
    * ``color_by``: ``'dwell_time'``, ``'repeats'`` or ``'dose'`` (see :class:`~fibomat.default_backends.ColorBy`) encodes this mill setting by color,
    * ``colormap``: the `matplotlib color map <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`__ (name or object) used with ``color_by``, by default ``'viridis'``,
    * ``color_bar``: if ``True`` (default), a color bar with the value range is drawn next to the layout,
    * ``stroke_width`` and ``spot_radius``: line width and radius of the spots as lengths (by default relative to the size of the drawing),
    * ``color``, ``missing_color`` and ``fov_color``: the color of the shapes without color map, of shapes whose mill does not have the setting ``color_by``, and of the fields of view.

The values for ``color_by`` are read from the mill of each pattern (``mill['dwell_time']``, ``mill['repeats']`` or ``mill['dose']``). A :class:`~fibomat.mill.Mill` has a dwell time and a number of repeats; a dose is available for mills which store one, e.g. a :class:`~fibomat.mill.DoseMill` or the ``NPVEMill``. Values with different dimensions, e.g. spot doses and area doses, get separate color scales and color bars. Color maps need matplotlib, which is installed with the ``exporting`` extra.

The following plot colors the shapes by their dwell time (source: `examples/svg_export.py <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/svg_export.py>`__) ::

    layout.export(SVGBackend, color_by='dwell_time', colormap='viridis').save('svg_export.svg')

.. figure:: /_static/svg_export_dwell_time.svg
    :align: center
    :width: 700px

    Layout exported with the SVG backend; the dwell times are encoded by the color map ``viridis``.

Ion beam simulation
+++++++++++++++++++

Files exported with the default SpotList backend can be visualized with the ``beam_simulation`` tool (it needs the ``gui`` extra, ``pip install "fibomat[gui]"``). This tool animates the beam path and should show quantitatively the same what a real microscope would do.

The tool can be run from the command line with ::

    $ beam_simulation path/to/the/exported/file.txt

This works only if the python binary directory is in the PATH variable, e.g. if a virtual environment is used.

For better visualization, the beam is shown with a tail. The animation speed can be changed manually to adapt for different dwell point densities. The dwell time of the points is ignored.

.. figure:: /_static/ion_beam_sim.gif
    :align: center
    :width: 600px

    Screenshot of the beam simulation tool.


