General structure
=================

The base and root of all other objects in the fib-o-mat package is the :class:`~fibomat.layout.Layout` class. A layout is the pattern design for a sample.
All other objects are added directly or indirectly to an instance of this class.

:class:`~fibomat.layout.Site` objects can be added to a layout. Sites have a position and a field of view as properties and :class:`~fibomat.layout.Pattern`\ s can be added to a site.

A pattern is a collection of a patterning geometry, beam settings and rasterization settings.

A layout can have an unlimited number of sites. The same holds for sites and patterns.

The following example can be used as a template for any project. ::

    from fibomat.layout import Layout
    from fibomat.linalg import Vector
    from fibomat.units import unit

    layout = Layout()

    # add a site with the position (5 µm, 6 µm) and a field of view of (2 µm, 2 µm)
    first_site = layout.create_site(
        dim_position=Vector(5, 6) * unit('µm'),
        dim_fov=Vector(2, 2) * unit('µm')
    )

    # see the following chapters to see how shapes, mills and raster styles can be defined.
    first_site.create_pattern(
        dim_shape=...,
        mill=...,
        raster_style=...
    )

    # plot the pattern design
    layout.plot()

    # export the pattern with a backend. See the corresponding chapter for more details.
    exported = layout.export(SomeBackend)
    exported.save('my_file.txt')

Layouts
-------

A :class:`~fibomat.layout.Layout` collects sites. In addition, it can hold annotations, which are shapes that are only plotted (e.g. the physical dimensions of a sample), see :meth:`~fibomat.layout.Layout.add_annotation`.

The class has the methods :meth:`~fibomat.layout.Layout.plot`, which plots the pattern design, and :meth:`~fibomat.layout.Layout.export`, :meth:`~fibomat.layout.Layout.export_multi` and :meth:`~fibomat.layout.Layout.export_with_description`, which pass the sites to a backend. See :doc:`exporting_visualization` for details.

Sites
-----

A :class:`~fibomat.layout.Site` contains the stage position and the field of view (fov) to be used during patterning as well as all patterns of the site.

The position of a site must be passed during its creation. The fov is optional. If it is not given, it is calculated automatically: it is the smallest square around the center of the site which contains all patterns, multiplied with a factor which defaults to 1.1 (i.e. the fov is 10 % larger than the minimum). The factor can be set for all sites of a layout with the argument ``fov_scale`` of :class:`~fibomat.layout.Layout`, or for a single site with the argument ``fov_scale`` of :class:`~fibomat.layout.Site` ::

    layout = Layout(fov_scale=1.5)  # the fov of all sites created with this layout is 50 % larger than the minimum

    site = layout.create_site(dim_position=Vector(0, 0) * unit('µm'))  # the fov is calculated from the patterns
    other_site = layout.create_site(
        dim_position=Vector(100, 0) * unit('µm'), dim_fov=Vector(10, 10) * unit('µm')  # a fixed fov
    )

.. warning:: Currently, no checks are performed if all patterns fit inside a fov which is given explicitly!

The placement of the patterns is relative to the center of the site, not to the global coordinate system!
For example, suppose a site with the center (-1, -1) is created and a pattern containing a single spot with the position (0, 0) is added. The absolute position of the spot is (-1, -1) with regard to the global coordinate system.

Patterns
--------

A :class:`~fibomat.layout.Pattern` collects geometric :class:`~fibomat.shapes.Shape`\ s, :class:`~fibomat.mill.Mill` settings and :class:`~fibomat.raster_styles.RasterStyle`\ s.

The passed shape must be equipped with a length unit. This is done by multiplying a shape with a unit. ::

    from fibomat import raster_styles, shapes
    from fibomat.layout import Pattern
    from fibomat.mill import Mill

    # ...

    pattern = Pattern(
        dim_shape=shapes.Rect(width=1, height=1) * unit('µm'),
        mill=Mill(dwell_time=1 * unit('µs'), repeats=3),
        raster_style=raster_styles.one_d.Curve(
            pitch=1 * unit('nm'), scan_sequence=raster_styles.ScanSequence.CONSECUTIVE
        )
    )

    first_site += pattern
    # or use in-place creation of a pattern with first_site.create_pattern(dim_shape=..., mill=..., raster_style=...)

|:test_tube:| Create patterns only for plotting
++++++++++++++++++++++++++++++++++++++++++++++++

For these tasks, the mill can be set to ``None`` and the raster style to a special class defined in :mod:`fibomat.default_backends`. ::

    from fibomat import default_backends

    # ...

    pattern = Pattern(
        dim_shape=shapes.Rect(width=1, height=1) * unit('µm'),
        mill=None,
        raster_style=default_backends.StubRasterStyle(dimension=1)  # 1: only the boundary is plotted. 2: the shape is filled.
    )

    first_site += pattern

.. note:: To add annotations to a plot (e.g. physical dimensions of a sample) which are not part of a pattern but useful to display, use the :meth:`~fibomat.layout.Layout.add_annotation` method of the :class:`~fibomat.layout.Layout` class.

Transformation of patterns and sites
------------------------------------

Patterns and sites support the same transformations as introduced in :ref:`user_guide/geometric-shapes:rigid transformations and isotropic scaling`.

In contrast to shapes, sites and patterns have units. Therefore, :class:`~fibomat.linalg.DimVector` must be used instead of :class:`~fibomat.linalg.Vector` for any transformation method. To illustrate this ::

    from fibomat.linalg import DimVector

    pattern = Pattern(...)
    rotated_pattern = pattern.rotated(np.pi / 3, origin=DimVector(1 * unit('µm'), -2 * unit('µm')))

Further, sites can only be rotated by multiples of 90° and mirrored on the axes and their bisectors. The field of view of a site is exchanged accordingly.

.. warning:: If sites are transformed, all contained patterns are transformed as well. Patterns added after the transformation are not transformed.
