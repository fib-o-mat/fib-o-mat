Getting started
===============

Installation
------------

fib-o-mat requires Python 3.8 or newer. Create a virtual environment and activate it with

.. code-block:: bash

    $ python -m venv .venv
    # for *nix systems
    $ source .venv/bin/activate
    # for MS Windows
    $ .venv\Scripts\activate.bat

See `the Python documentation <https://docs.python.org/3/tutorial/venv.html>`__ for further information on virtual environments.

Install the fib-o-mat package with

.. code-block:: bash

    $ pip install --upgrade "fibomat[exporting]"

The extra ``exporting`` installs the packages which are needed to plot layouts and to export them. If this does not work for you, fib-o-mat must be built from source for your system. This is explained :doc:`here <user_guide/installation>`.


"Hello world" example
---------------------

This guide shows how to create a simple patterning layout, which consists of a line and a rectangle.

The complete Python file can be found in the `example folder of the repository <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/getting_started.py>`__.

The starting point of each patterning design is the :class:`~fibomat.layout.Layout` class. A layout is the pattern design for a sample. This class is the glue between everything else in this library. ::

    from fibomat import default_backends, raster_styles, shapes
    from fibomat.layout import Layout, Pattern
    from fibomat.linalg import Vector
    from fibomat.mill import Mill
    from fibomat.units import unit

    layout = Layout(description="an optional description for yourself")

Next, a patterning site must be defined. The patterning site defines the stage position of the microscope and the field of view to be used during the patterning process.

(Physical) units, where needed, are specified with the :func:`~fibomat.units.unit` function from the :mod:`fibomat.units` submodule. For the creation of a site, units are needed to specify the length unit of the stage position and of the field of view. ::

    site = layout.create_site(
        dim_position=Vector(123.0, 456.0) * unit("µm"),
        dim_fov=Vector(5.0, 5.0) * unit("µm"),
        description="another description",
    )

The site is located at (123, 456) µm and has a field of view of (5, 5) µm in the horizontal and vertical direction, respectively.
The interpretation of the ``dim_position`` value depends on your microscope. For details, consult the documentation of the exporting backend you use (this concept is introduced a little later).

.. tip:: If no field of view is given, it is calculated automatically from the patterns of the site. It is the smallest square around the center of the site which contains all patterns, increased by a configurable factor, see :ref:`user_guide/general_structure:sites`.

The library has some predefined shapes which can be used to build patterning geometries.
A list of all shapes is given in :ref:`user_guide/geometric-shapes:geometric shapes`.

The patterning settings are specified with classes from the :mod:`fibomat.mill` and :mod:`fibomat.raster_styles` submodules. The former defines the dwell time per spot and the number of repeats, the latter the rasterization method.
Shapes and patterning settings are collected in the :class:`~fibomat.layout.Pattern` class. It merges geometries, mill settings and some optional additional parameters.

Putting everything together for a line shape ::

    # the mill object defines the dwell time per spot and the number of repeats
    single_repeat_mill = Mill(dwell_time=5 * unit("ms"), repeats=1)

    # a line shape
    line = shapes.Line(start=(-2, 2), end=(2, 0.5))

    # and finally the rasterization style. In this case, the line is rasterized consecutively from its start to its end.
    line_style = raster_styles.one_d.Curve(
        pitch=1 * unit("nm"), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
    )

    # everything is collected in a pattern
    line_pattern = Pattern(
        dim_shape=line * unit("µm"), mill=single_repeat_mill, raster_style=line_style
    )

    # and added to the site.
    site += line_pattern

Note that the ``line`` is equipped with a length unit during the pattern creation. Otherwise, its scaling would not be defined.

.. caution:: The shape position is always interpreted relative to the center of the corresponding site, not to the global coordinate system.

Secondly, a square is added to the layout. In contrast to the line pattern, the definition of the rasterization style is more involved.
Here, the square is rasterized line by line, with the lines parallel to the x-axis. The lines are ordered consecutively.
Additionally, the rasterization style of the individual lines must be given. A style similar to the one of the line pattern is used for this. ::

    square = shapes.Rect(width=2, height=2, center=(0, -1))

    # rasterize the square line by line, see the user guide for details
    square_style = raster_styles.two_d.LineByLine(
        line_pitch=10 * unit("nm"),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
        alpha=0,  # angle of the lines with respect to the x-axis
        invert=False,  # if True, the lines are created from the bottom to the top (instead of the top to the bottom)
        line_style=raster_styles.one_d.Curve(
            pitch=10 * unit("nm"), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
        ),
    )

    # the pattern can also be created in-place
    site.create_pattern(
        dim_shape=square * unit("µm"), mill=single_repeat_mill, raster_style=square_style
    )

Finally, the finished layout can be plotted and exported. Exporting is carried out by so-called backends.
A backend takes all sites with their patterns and creates a file (or something else) a microscope can understand.
Currently, only the following exporting backends are part of the library: a plotting backend for visualization, a rasterization backend, which rasterizes all geometries and creates a text file with all spots and their dwell times, and some backends for specific microscope software.
Backends are passed as classes to :meth:`~fibomat.layout.Layout.export`. ::

    layout.plot()

    # export as text file
    layout.export(default_backends.SpotListBackend).save("getting_started.txt")

The output format of the :class:`~fibomat.default_backends.SpotListBackend` can be customized, see :ref:`here <user_guide/exporting_visualization:exporting microscope readable output>`.

The resulting plot is shown below. The big box shows the field of view of the patterning site. The shapes in it are the patterning geometries.

Select one of the zoom tools in the right panel to zoom in and out, either by scrolling or with a box selection.
Hovering over a shape opens a tooltip which displays some information about the geometry, mill and rasterization style.

Additionally, the measure tool (the icon with the ruler) can be used to measure distances and angles: select the tool and drag with the mouse over the plot. A click removes the measurement.

.. fibomat-plot:: ../examples/getting_started.py

Further, the script exports a file containing the rasterized pattern according to the settings given above. This file (``getting_started.txt``) can be visualized with the :ref:`ion beam simulation tool <user_guide/exporting_visualization:ion beam simulation>`.
