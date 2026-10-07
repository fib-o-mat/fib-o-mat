"""Container classes of a pattern design.

A :class:`Layout` (the pattern design for a sample) consists of :class:`Site` objects (a field of view at a position)
which hold :class:`Pattern` objects (a shape with a mill and a raster style). A layout is exported with the help of
backends.

Example::

    from fibomat.default_backends import SpotListBackend
    from fibomat.layout import Layout
    from fibomat.linalg import DimVector
    from fibomat.mill import Mill
    from fibomat.raster_styles import one_d, ScanSequence
    from fibomat.shapes import Line
    from fibomat.units import unit

    layout = Layout()
    site = layout.create_site(DimVector(0 * unit('µm'), 0 * unit('µm')))
    site.create_pattern(
        Line((0, 0), (1, 0)) * unit('µm'),
        Mill(1. * unit('ms'), 1),
        one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
    )
    exported = layout.export(SpotListBackend, length_unit=unit('nm'), time_unit=unit('µs'))
"""
# `Site` and `Pattern` must be imported before `Layout` (`Layout` imports both of them).
from fibomat.layout.pattern import Pattern
from fibomat.layout.site import Site
from fibomat.layout.layout import Layout

__all__ = ['Layout', 'Site', 'Pattern']
