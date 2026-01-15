"""
Provides the :class:`BackendBase` class.
"""
from typing import Dict, Callable, Type
import inspect

from fibomat.site import Site
from fibomat.pattern import Pattern
from fibomat import shapes
from fibomat.utils import PathLike
from fibomat.backend.backendbasemeta import BackendBaseMeta
from fibomat.rasterizedpattern import RasterizedPattern

from fibomat import layout


class ShapeNotSupportedError(TypeError):
    """Exception is used in :class:`BackendBase` for (all) unimplemented shape
    methods.
    """


def shape_type(type_: Type[shapes.Shape]) -> Callable:
    """Function is used as a decorator in :class:`BackendBase` to mark shape
    methods.
    See :class:`BackendBase` class and its implementation for details.

    .. note:: This decorator is only needed if a custom shape type is
              introduced and should be made available to the backend

    Args:
        type_ (Type[Shape]): Type of shape.

    Returns:
        Callable: Decorator
    """

    def decorator(func):
        # pylint: disable=protected-access
        func._shape_type = type_
        func._abstract_shape_method = True
        return func

    return decorator


class BackendBase(metaclass=BackendBaseMeta):
    """Base class for any backend. For all default shapes (see ...), a method
    stub is implemented.

    TODO: UPDATE THIS!

    Use :class:`BackendBase` as base class for a custom backend and implement
    all shape methods with should be supported and ignore the rest. If any
    other new shape is introduced and should be supported by the custom
    backend, use the following example as a base structure. ::

        class MyNewBackendBase(BackendBase):
            def __init__(self):  # is it needed?
                super().__init__()

            @availabe_shape_method('My_new_shape_name')
            def my_new_shape(self, parameters):
                raise ShapeNotSupportedError

        class MyNewBackend(MyNewBackendBase):
            ...

            def my_new_shape(self, parameters):
                ...


    Actually not needed:
    Make sure, that the base methods of each pattern function
    (e.g. :meth:`fibomat.backend.backendbase.BackendBase.rect`) is called, e.g. ::

        def rect(self, ptn: Pattern):
            super().rect(pattern)
            # ... do your stuff here

    """

    def __init__(self, **kwargs):
        """
        Args:
            `**kwargs`: optional parameters for the backend given by `**kwargs` in
                        :meth:`fibomat.project.Project.export`
        """
        if kwargs:
            print(
                'warning: unhandled kwargs in exporting backend ("{}")'.format(
                    ", ".join(kwargs.keys())
                )
            )

        super().__init__()

        self._abstract_shape_methods: Dict[shapes.Shape, Callable]
        self._implemented_shape_methods: Dict[shapes.Shape, Callable]

    def process_pattern(self, ptn: Pattern) -> None:
        """
        Add a pattern to the backend. The appropriate method is determined  by the pattern's shape class automatically.

        If pattern contains a Layout, the contained shapes will be extracted automatically.

        Args:
            ptn (Pattern): pattern to be added.
            on_unknown_shape:

        Returns:
            None
        """

        def dispatch(extracted_ptn):
            try:
                method = self.implemented_shape_methods[
                    type(extracted_ptn.dim_shape.shape)
                ]
                return method(self, extracted_ptn)
            except KeyError:
                # try bases classes
                for base in inspect.getmro(extracted_ptn.dim_shape.shape.__class__):
                    try:
                        method = self.implemented_shape_methods[base]
                        return method(self, extracted_ptn)
                    except KeyError:
                        pass
                self.process_unknown(extracted_ptn)

        if isinstance(ptn.dim_shape, layout.LayoutBase):
            # ptn.dim_shape[0]: layout.LayoutBase
            for extracted_shape in ptn.dim_shape.layout_elements():
                if isinstance(extracted_shape, layout.LayoutBase):
                    self.process_pattern(
                        Pattern(
                            extracted_shape,
                            ptn.mill,
                            ptn.raster_style,
                            **ptn.kwargs,
                            description=ptn.description,
                        )
                    )
                else:
                    dispatch(
                        Pattern(
                            extracted_shape,
                            ptn.mill,
                            ptn.raster_style,
                            **ptn.kwargs,
                            description=ptn.description,
                        )
                    )
        else:
            dispatch(ptn)

    def process_unknown(self, ptn: Pattern) -> None:
        """
        Process a shape, which do not have a registered shape handler.
        Raises an exception by default.

        Args:
            ptn (Pattern):  pattern to be added.

        Returns:
            None
        """
        raise ShapeNotSupportedError(f"Shape type = {ptn.dim_shape.shape.__class__}")

    # @staticmethod
    # def _check_pattern_shape_type(expected_shape_type_: typing.Type[Shape], patter: Pattern):
    #     if expected_shape_type_.__class__ != patter.shape.__class__:
    #         raise RuntimeError('Expected shape of class {} but got {}'.format(
    #             expected_shape_type_.__class__.__name__, patter.shape.__class__.__name__
    #         ))

    @property
    def shape_methods(self) -> Dict[shapes.Shape, Callable]:
        """Dict[Shape, str]: all (abstract) shape methods"""
        return self._abstract_shape_methods

    @property
    def implemented_shape_methods(self) -> Dict[shapes.Shape, Callable]:
        """Dict[Shape, str]: implemented shape methods of backend"""
        return self._implemented_shape_methods

    def process_site(self, new_site: Site) -> None:
        """
        Adds a :class:`fibomat.site.Site` to the backend. Note, that this method processes all patterns contained in
        the site.
        Use the following example as reference, if a backend overwrites this method. ::

            ...
            def process_site(self, site: Site):
                # ...
                # do the backend specific stuff here (e.g. initialize a new patterning site)

                # call base method to add all patterns automatically
                super().process_site(site)
            ...

        Args:
            new_site (Site): site to be added.

        Returns:
            None
        """
        for ptn in new_site.patterns:
            self.process_pattern(ptn)

    def save(self, filename: PathLike) -> None:
        """
        Saves the exported project to file.
        Args:
            filename (PathLike): filename

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.Spot)
    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        """
        Adds pattern with `Spot` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Spot` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.Line)
    def line(self, ptn: Pattern[shapes.Line]) -> None:
        """
        Adds pattern with `Line` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Line` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.Rect)
    def rect(self, ptn: Pattern[shapes.Rect]) -> None:
        """
        Adds pattern with `Rect` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Rect` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.Ellipse)
    def ellipse(self, ptn: Pattern[shapes.Ellipse]) -> None:
        """
        Adds pattern with `Ellipse` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Ellipse` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.Circle)
    def circle(self, ptn: Pattern[shapes.Circle]) -> None:
        """
        Adds pattern with `Circle` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Circle` as shape

        Returns:
            None
        """
        raise NotImplementedError


    @shape_type(shapes.Ring)
    def ring(self, ptn: Pattern[shapes.Ring]) -> None:
        """
        Adds pattern with `Ring` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Ring` as shape

        Returns:
            None
        """
        
        raise NotImplementedError


    @shape_type(shapes.Arc)
    def arc(self, ptn: Pattern[shapes.Arc]) -> None:
        """
        Adds pattern with `Arc` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Arc` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.ArcSpline)
    def arc_spline(self, ptn: Pattern[shapes.ArcSpline]) -> None:
        """
        Adds pattern with `ArcSpline` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `ArcSpline` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.ParametricCurve)
    def parametric_curve(self, ptn: Pattern[shapes.ParametricCurve]) -> None:
        """
        Adds pattern with `Curve` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Curve` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.RasterizedPoints)
    def rasterized_points(self, ptn: Pattern[shapes.RasterizedPoints]):
        """
        Adds pattern with `RasterizedPoints` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `RasterizedPoints` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(RasterizedPattern)
    def rasterized_pattern(self, ptn: Pattern[RasterizedPattern]):
        """
        Adds pattern with `RasterizedPattern` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `RasterizedPattern` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.Polyline)
    def polyline(self, ptn: Pattern[shapes.Polyline]) -> None:
        """
        Adds pattern with `Ellipse` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Ellipse` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.Polygon)
    def polygon(self, ptn: Pattern[shapes.Polygon]) -> None:
        """
        Adds pattern with `Ellipse` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Ellipse` as shape

        Returns:
            None
        """
        raise NotImplementedError

    @shape_type(shapes.HollowArcSpline)
    def hollow_arc_spline(self, ptn: Pattern[shapes.HollowArcSpline]) -> None:
        """
        Adds pattern with `HollowArcSpline` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `HollowArcSpline` as shape

        Returns:
            None
        """
        raise NotImplementedError
