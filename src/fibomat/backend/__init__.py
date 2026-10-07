"""Backends export a :class:`~fibomat.layout.layout.Layout` to a file (or a plot) a microscope (software) can
understand.

See the module :mod:`fibomat.default_backends` for example backends. To implement your own backend, derive from
:class:`BackendBase`, implement the methods of the shapes which should be supported and pass the class to
:meth:`fibomat.layout.layout.Layout.export`::

    from fibomat.backend import BackendBase

    class MyBackend(BackendBase):
        def line(self, ptn):
            ...

    exported = layout.export(MyBackend)
"""
from fibomat.backend.backendbase import BackendBase, ShapeNotSupportedError, shape_type

__all__ = ['BackendBase', 'ShapeNotSupportedError', 'shape_type']
