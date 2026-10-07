"""Provides the :class:`BackendBase` class and the :func:`shape_type` decorator.

A backend exports the patterns of a :class:`~fibomat.layout.layout.Layout`. It derives from :class:`BackendBase` and
implements the methods of the shapes it supports (e.g. ``line`` or ``polygon``); all other shapes raise a
:class:`ShapeNotSupportedError`. Backends are passed as classes to :meth:`fibomat.layout.layout.Layout.export`.

Example::

    class CountingBackend(BackendBase):
        def __init__(self, description=None):
            super().__init__(description)
            self.n_lines = 0

        def line(self, ptn):
            self.n_lines += 1

    backend = layout.export(CountingBackend)
    backend.n_lines
"""
from __future__ import annotations

import functools
import typing as t

from fibomat import composite_shapes, shapes
from fibomat.layout.pattern import Pattern
from fibomat.layout.site import Site
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.utils import PathLike


__all__ = ['BackendBase', 'ShapeNotSupportedError', 'shape_type']


class ShapeNotSupportedError(TypeError):
    """Raised by :class:`BackendBase` for shapes which the backend does not support."""


def shape_type(type_: t.Type[t.Any]) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
    """Decorator which declares that a method of a backend handles patterns with the shape type `type_`.

    :class:`BackendBase` declares the methods of all shapes of the library. The decorator is only needed to introduce
    a method for a custom shape type::

        class MyBackend(BackendBase):
            @shape_type(MyShape)
            def my_shape(self, ptn):
                ...

    A backend supports a shape type if the declared method is overridden (the stubs of :class:`BackendBase` raise
    :class:`ShapeNotSupportedError`) or newly declared with an implementation. If a shape type is not supported, the
    base classes of the shape are tried (e.g. a backend which supports `Polyline` also handles `Polygon` if it does
    not support polygons itself).

    Args:
        type_ (Type): type of the shape

    Returns:
        Callable: decorator
    """
    def decorator(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        func._shape_type = type_  # type: ignore[attr-defined]  # pylint: disable=protected-access
        return func

    return decorator


@functools.lru_cache(maxsize=None)
def _shape_methods(backend_class: t.Type[BackendBase]) -> t.Dict[t.Type[t.Any], str]:
    """Names of the shape methods of a backend class: all methods in the class hierarchy which are decorated with
    :func:`shape_type` (for each shape type the method of the most derived class)."""
    names: t.Dict[t.Type[t.Any], str] = {}
    for cls in reversed(backend_class.__mro__):
        for name, attr in vars(cls).items():
            declared_type = getattr(attr, '_shape_type', None)
            if declared_type is not None:
                names[declared_type] = name
    return names


class BackendBase:
    """Base class for any backend.

    For all shapes of the library, a method stub is implemented (see below). Derive from this class, implement the
    methods of the shapes which should be supported and ignore the rest; patterns with unsupported shapes raise a
    :class:`ShapeNotSupportedError`. Use :func:`shape_type` to support custom shapes.

    The shape methods get the :class:`~fibomat.layout.pattern.Pattern` of the shape. Arrangements (groups, lattices) and
    texts are split into their shapes automatically.
    """

    def __init__(self, description: t.Optional[str] = None):
        """
        Args:
            description (str, optional): description of the exported layout (set by
                :meth:`fibomat.layout.layout.Layout.export`)
        """
        self._description = description

    @property
    def description(self) -> t.Optional[str]:
        """Description of the exported layout.

        Access:
            get
        """
        return self._description

    @property
    def shape_methods(self) -> t.Dict[t.Type[t.Any], str]:
        """Names of the shape methods of the backend, by shape type.

        Access:
            get
        """
        return dict(_shape_methods(type(self)))

    @property
    def implemented_shape_methods(self) -> t.Dict[t.Type[t.Any], t.Callable[..., None]]:
        """The supported shape types and the (unbound) methods of the backend which handle them.

        Access:
            get
        """
        implemented = {}
        for shape_class, name in _shape_methods(type(self)).items():
            if self._is_implemented(name):
                implemented[shape_class] = getattr(type(self), name)
        return implemented

    def _is_implemented(self, name: str) -> bool:
        """True if the shape method `name` is overridden (or newly declared) by the backend."""
        return getattr(type(self), name) is not vars(BackendBase).get(name)

    def process_pattern(self, ptn: Pattern) -> None:
        """Add a pattern to the backend. The method of the shape of the pattern is called.

        If the pattern contains an arrangement (e.g. a group or a lattice) or another composite which provides
        ``arrangement_elements()`` (e.g. a text), the contained shapes are extracted and processed one by one.

        Args:
            ptn (Pattern): pattern to be added.

        Raises:
            ShapeNotSupportedError: Raised if the backend does not support the shape.
        """
        if callable(getattr(ptn.dim_shape, 'arrangement_elements', None)):
            for element in ptn.dim_shape.arrangement_elements():
                self.process_pattern(
                    Pattern(element, ptn.mill, ptn.raster_style, description=ptn.description, **ptn.kwargs)
                )
        else:
            self._dispatch(ptn)

    def _dispatch(self, ptn: Pattern) -> None:
        """Call the shape method for the shape of `ptn` (or of the nearest base class of the shape which is
        supported)."""
        names = _shape_methods(type(self))
        for shape_class in type(ptn.dim_shape.shape).__mro__:
            name = names.get(shape_class)
            if name is not None and self._is_implemented(name):
                getattr(self, name)(ptn)
                return

        self.process_unknown(ptn)

    def process_unknown(self, ptn: Pattern) -> None:
        """Process a pattern with a shape which is not supported by the backend.
        Raises an exception by default.

        Args:
            ptn (Pattern): pattern to be added.

        Raises:
            ShapeNotSupportedError: Always.
        """
        raise ShapeNotSupportedError(
            f'{type(self).__name__} does not support the shape type {type(ptn.dim_shape.shape).__name__}.'
        )

    def process_site(self, new_site: Site) -> None:
        """Adds a :class:`~fibomat.layout.site.Site` to the backend. Note that this method processes all patterns
        contained in the site.

        Overwrite it if a backend needs to do something for each site, and call the base method to process the
        patterns::

            def process_site(self, new_site):
                # ... do the backend specific stuff here (e.g. initialize a new patterning site)
                super().process_site(new_site)

        Args:
            new_site (Site): site to be added.
        """
        for ptn in new_site.patterns:
            self.process_pattern(ptn)

    def save(self, filename: PathLike) -> None:
        """Saves the exported layout to a file.

        Args:
            filename (PathLike): filename

        Raises:
            NotImplementedError: Raised by default (the backend cannot save).
        """
        raise NotImplementedError

    @shape_type(shapes.Spot)
    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        """Adds a pattern with `Spot` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Spot` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Spot.')

    @shape_type(shapes.Line)
    def line(self, ptn: Pattern[shapes.Line]) -> None:
        """Adds a pattern with `Line` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Line` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Line.')

    @shape_type(shapes.Rect)
    def rect(self, ptn: Pattern[shapes.Rect]) -> None:
        """Adds a pattern with `Rect` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Rect` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Rect.')

    @shape_type(shapes.Ellipse)
    def ellipse(self, ptn: Pattern[shapes.Ellipse]) -> None:
        """Adds a pattern with `Ellipse` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Ellipse` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Ellipse.')

    @shape_type(shapes.Circle)
    def circle(self, ptn: Pattern[shapes.Circle]) -> None:
        """Adds a pattern with `Circle` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Circle` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Circle.')

    @shape_type(composite_shapes.Ring)
    def ring(self, ptn: Pattern[composite_shapes.Ring]) -> None:
        """Adds a pattern with `Ring` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Ring` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Ring.')

    @shape_type(shapes.Arc)
    def arc(self, ptn: Pattern[shapes.Arc]) -> None:
        """Adds a pattern with `Arc` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Arc` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Arc.')

    @shape_type(shapes.ArcSpline)
    def arc_spline(self, ptn: Pattern[shapes.ArcSpline]) -> None:
        """Adds a pattern with `ArcSpline` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `ArcSpline` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support ArcSpline.')

    @shape_type(shapes.ParametricCurve)
    def parametric_curve(self, ptn: Pattern[shapes.ParametricCurve]) -> None:
        """Adds a pattern with `ParametricCurve` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `ParametricCurve` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support ParametricCurve.')

    @shape_type(shapes.RasterizedPoints)
    def rasterized_points(self, ptn: Pattern[shapes.RasterizedPoints]) -> None:
        """Adds a pattern with `RasterizedPoints` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `RasterizedPoints` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support RasterizedPoints.')

    @shape_type(RasterizedPattern)
    def rasterized_pattern(self, ptn: Pattern[RasterizedPattern]) -> None:
        """Adds a pattern with `RasterizedPattern` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `RasterizedPattern` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support RasterizedPattern.')

    @shape_type(shapes.Polyline)
    def polyline(self, ptn: Pattern[shapes.Polyline]) -> None:
        """Adds a pattern with `Polyline` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Polyline` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Polyline.')

    @shape_type(shapes.Polygon)
    def polygon(self, ptn: Pattern[shapes.Polygon]) -> None:
        """Adds a pattern with `Polygon` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Polygon` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support Polygon.')

    @shape_type(composite_shapes.HollowArcSpline)
    def hollow_arc_spline(self, ptn: Pattern[composite_shapes.HollowArcSpline]) -> None:
        """Adds a pattern with `HollowArcSpline` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `HollowArcSpline` as shape

        Raises:
            ShapeNotSupportedError: Raised by default (the shape is not supported).
        """
        raise ShapeNotSupportedError(f'{type(self).__name__} does not support HollowArcSpline.')
