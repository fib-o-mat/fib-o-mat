Custom backends
===============

A backend maps the patterns of a :class:`~fibomat.layout.Layout` to an output format. Every backend derives from :class:`~fibomat.backend.BackendBase`.

How it works
------------

:class:`~fibomat.backend.BackendBase` has one method per shape type (``line``, ``polygon``, ``circle``, ...). Each method gets a :class:`~fibomat.layout.Pattern` and, by default, raises a :class:`~fibomat.backend.ShapeNotSupportedError`. A custom backend only overrides the methods for the shapes it supports. Patterns with other shapes are rejected on export. If a shape is not supported, the base classes of its type are tried, e.g. a backend which implements ``polyline`` also handles polygons, unless it implements ``polygon`` as well.

On :meth:`Layout.export <fibomat.layout.Layout.export>`, the layout creates one instance of the backend class (the keyword arguments of the call are passed to ``__init__``). Then, :meth:`~fibomat.backend.BackendBase.process_site` is called for every site, which in turn calls :meth:`~fibomat.backend.BackendBase.process_pattern` for every pattern. Groups, lattices and texts are split into their individual shapes automatically. The exported backend object is returned, so it can offer a ``save`` method (or any other methods).

A simple example
----------------

The following backend collects all lines of a layout and writes them to a text file. It overrides

    * ``process_site`` to remember the position of the current site, and
    * ``line``, which is called for every pattern with a :class:`~fibomat.shapes.Line` as shape.

The shape of a pattern is a :class:`~fibomat.shapes.DimShape`: ``ptn.dim_shape.shape`` is the unitless shape and ``ptn.dim_shape.unit`` its length unit. ::

    import typing as t

    from fibomat import shapes
    from fibomat.backend import BackendBase
    from fibomat.layout import Layout, Pattern, Site
    from fibomat.linalg import Vector
    from fibomat.mill import Mill
    from fibomat.raster_styles import ScanSequence, one_d
    from fibomat.units import scale_factor, unit
    from fibomat.utils import PathLike


    class SimpleLineBackend(BackendBase):
        """Collects all lines (also the lines of groups and lattices) as 'x0 y0 x1 y1' in µm."""

        def __init__(self, description: t.Optional[str] = None):
            super().__init__(description)
            self._lines: t.List[str] = []
            self._site_center = Vector(0, 0)

        def process_site(self, new_site: Site) -> None:
            # remember the position of the site (in µm) and process all of its patterns
            self._site_center = new_site.center.vector_as(unit('µm'))
            super().process_site(new_site)

        def line(self, ptn: Pattern[shapes.Line]) -> None:
            # scale from the unit of the shape to µm
            factor = scale_factor(unit('µm'), ptn.dim_shape.unit)
            line = ptn.dim_shape.shape
            start = line.start * factor + self._site_center
            end = line.end * factor + self._site_center
            self._lines.append(f'{start.x:.3f} {start.y:.3f} {end.x:.3f} {end.y:.3f}')

        def save(self, filename: PathLike) -> None:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write('\n'.join(self._lines) + '\n')

and can be used like every other backend ::

    layout = Layout()
    site = layout.create_site(Vector(10, 0) * unit('µm'))
    site.create_pattern(
        shapes.Line((0, 0), (1, 1)) * unit('µm'),
        Mill(1 * unit('ms'), 1),
        one_d.Curve(10 * unit('nm'), ScanSequence.CONSECUTIVE)
    )
    layout.export(SimpleLineBackend).save('lines.txt')

The file ``lines.txt`` then contains

.. code-block:: none

    10.000 0.000 11.000 1.000

Patterns with other shapes (e.g. circles) raise a :class:`~fibomat.backend.ShapeNotSupportedError`:
``SimpleLineBackend does not support the shape type Circle.``

Custom shapes
-------------

Backends can also support custom shape types. Declare the method with the :func:`~fibomat.backend.shape_type` decorator ::

    from fibomat.backend import BackendBase, shape_type

    class MyBackend(BackendBase):
        @shape_type(MyShape)
        def my_shape(self, ptn):
            ...

Rasterizing backends
--------------------

Backends which need dwell points do not have to rasterize shapes themselves: ``ptn.raster_style.rasterize(ptn.dim_shape, ptn.mill, length_unit, time_unit)`` returns a :class:`~fibomat.rasterizedpattern.RasterizedPattern`. The :class:`~fibomat.default_backends.SpotListBackend` is a good starting point; the FEI stream file backend (:class:`~fibomat.default_backends.fei.FEIStreamFile`) is derived from it and only provides a custom ``save_impl``.
