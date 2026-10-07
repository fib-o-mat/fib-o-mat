"""Provides the :class:`Layout` class.

Example:
    >>> from fibomat.layout import Layout
    >>> from fibomat.linalg import DimVector
    >>> from fibomat.mill import Mill
    >>> from fibomat.raster_styles import ScanSequence, one_d
    >>> from fibomat.shapes import Line
    >>> from fibomat.units import unit
    >>> layout = Layout(description='test', fov_scale=1.2)
    >>> site = layout.create_site(DimVector(0 * unit('µm'), 0 * unit('µm')))
    >>> _ = site.create_pattern(
    ...     Line((0, 0), (1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1),
    ...     one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
    ... )
    >>> layout.number_of_sites
    1
    >>> site.fov_scale
    1.2
"""
from __future__ import annotations

import dataclasses
import re
import typing as t
import warnings

from fibomat.arrangements import ArrangementBase
from fibomat.describable import Describable
from fibomat.layout.pattern import Pattern
from fibomat.layout.site import DEFAULT_FOV_SCALE, Site, _check_fov_scale
from fibomat.linalg import DimBoundingBox, DimVectorLike
from fibomat.shapes import DimShape
from fibomat.utils import PathLike

if t.TYPE_CHECKING:  # pragma: no cover
    # The backends depend on `Site` and `Pattern` of this package, hence, they are imported lazily (in the methods).
    from fibomat.backend import BackendBase
    from fibomat.default_backends import BokehBackend


__all__ = ['Layout']


@dataclasses.dataclass(frozen=True)
class _Annotation:
    """A shape which is only plotted."""
    dim_shape: DimShape
    filled: bool
    color: t.Optional[str]
    description: t.Optional[str]


BackendT = t.TypeVar('BackendT')


def _registry() -> t.Any:
    """Return the backend registry (imported lazily because the backends depend on this package).

    The default backends are imported, too, because this registers them.
    """
    # pylint: disable=import-outside-toplevel,unused-import
    import fibomat.default_backends  # noqa: F401
    from fibomat.backend import registry
    return registry


class Layout(Describable):
    """A layout is the pattern design for a sample.

    This class is the glueing between all subcomponents of the library: a layout consists of
    :class:`~fibomat.layout.site.Site` objects (a field of view at a position), which hold
    :class:`~fibomat.layout.pattern.Pattern` objects (a shape with a mill and a raster style). A layout is exported
    with the help of registered backends or plotted.
    """

    def __init__(self, *, description: t.Optional[str] = None, fov_scale: float = DEFAULT_FOV_SCALE):
        """
        Args:
            description (str, optional): Optional description of the layout
            fov_scale (float): factor by which the minimal field of view of a site is increased if the site is created
                with :meth:`create_site` without an explicit field of view (at least 1). The default adds a margin of
                10 %.

        Raises:
            ValueError: Raised if fov_scale is smaller than 1 or not finite.
        """
        super().__init__(description)

        self._fov_scale = _check_fov_scale(fov_scale)

        self._sites: t.List[Site] = []
        self._annotations: t.List[_Annotation] = []

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(description={self._description!r}, n_sites={len(self._sites)})'

    @property
    def fov_scale(self) -> float:
        """Factor by which the minimal field of view of the sites created with :meth:`create_site` is increased.

        Access:
            get
        """
        return self._fov_scale

    def create_site(
        self,
        dim_position: DimVectorLike,
        dim_fov: t.Optional[DimVectorLike] = None,
        description: t.Optional[str] = None,
    ) -> Site:
        """Creates a site in-place (the site is automatically added to the layout). Patterns can be added to the
        returned object.

        If no `dim_fov` is given, the field of view is calculated from the added patterns and increased by the
        :attr:`Layout.fov_scale` of the layout.

        See :class:`~fibomat.layout.site.Site` for the description of the arguments.

        Args:
            dim_position (DimVectorLike): center of the site
            dim_fov (DimVectorLike, optional): field of view
            description (str, optional): description

        Returns:
            Site
        """
        new_site = Site(dim_position, dim_fov, description=description, fov_scale=self._fov_scale)
        self._sites.append(new_site)
        return new_site

    def add_site(self, site_like: t.Union[Site, ArrangementBase]) -> None:
        """Adds a site (or an arrangement of sites) to the layout.
        Alternatively, the '+=' operator can be used.

        Note that the field of view of a site which is added with this method is not influenced by
        :attr:`Layout.fov_scale`; use the `fov_scale` argument of :class:`~fibomat.layout.site.Site`.

        Args:
            site_like (Site, ArrangementBase): new site(s)

        Raises:
            TypeError: Raised if `site_like` is no site or no arrangement of sites.
        """
        if isinstance(site_like, ArrangementBase):
            new_sites = list(site_like.arrangement_elements())
        else:
            new_sites = [site_like]

        for new_site in new_sites:
            if not isinstance(new_site, Site):
                raise TypeError(f'Only sites can be added to a layout, got {type(new_site).__name__}.')

        self._sites.extend(new_sites)

    def __iadd__(self, site_like: t.Union[Site, ArrangementBase]) -> Layout:
        """See :meth:`~Layout.add_site`."""
        self.add_site(site_like)
        return self

    @property
    def sites(self) -> t.List[Site]:
        """The sites of the layout (a copy of the list).

        Access:
            get
        """
        return list(self._sites)

    @property
    def number_of_sites(self) -> int:
        """Number of sites.

        Access:
            get
        """
        return len(self._sites)

    @property
    def bounding_box(self) -> t.Optional[DimBoundingBox]:
        """Bounding box of the patterns of all sites (absolute coordinates), None if there are no patterns.

        Access:
            get
        """
        bbox: t.Optional[DimBoundingBox] = None

        for site in self._sites:
            if site.empty:
                continue
            site_bbox = site.bounding_box_abs
            bbox = site_bbox if bbox is None else bbox.extended(site_bbox)

        return bbox

    @staticmethod
    def _export(
        backend_class: t.Type[BackendT],
        sites: t.Union[Site, t.Sequence[Site]],
        descr_pattern: t.Optional[t.Set[str]] = None,
        **kwargs: t.Any,
    ) -> BackendT:
        """Create a backend and let it process the sites.

        Args:
            backend_class (Type[BackendBase]): backend
            sites (Site, Sequence[Site]): sites
            descr_pattern (Set[str], optional): if given, only sites with a description which matches one of the
                regular expressions are processed. Ignored for a single site.
            **kwargs: arguments of the backend

        Returns:
            BackendBase

        Raises:
            ValueError: Raised if `descr_pattern` is given and no site matches.
        """
        exporter: BackendBase = backend_class(**kwargs)  # type: ignore[call-arg]

        if isinstance(sites, Site):
            if descr_pattern:
                warnings.warn('Ignoring descr_pattern for the export of a single site.', stacklevel=3)
            exporter.process_site(sites)
            return exporter  # type: ignore[return-value]

        processed = 0
        for site in sites:
            if descr_pattern:
                description = site.description
                if not description or not any(re.match(regex, description) for regex in descr_pattern):
                    continue
            exporter.process_site(site)
            processed += 1

        if descr_pattern and not processed:
            raise ValueError(f'No site has a description which matches one of {sorted(descr_pattern)}.')

        return exporter  # type: ignore[return-value]

    def plot(
        self,
        show: bool = True,
        filename: t.Optional[PathLike] = None,
        descr_pattern: t.Optional[t.Set[str]] = None,
        **kwargs: t.Any,
    ) -> BokehBackend:
        """Plot (and save) the layout with the :class:`~fibomat.default_backends.bokeh_backend.BokehBackend`.

        Args:
            show (bool): if True, the plot is opened in a browser automatically
            filename (PathLike, optional): if filename is given, the plot is saved in this file. The file suffix should
                be `*.htm` or `*.html`.
            descr_pattern (Set[str], optional): if given, only sites with a description matching one of the regular
                expressions are plotted.
            **kwargs: parameters for the bokeh backend. These are directly passed to the __init__ method of the
                BokehBackend class. The title parameter defaults to :attr:`Layout.description`.

        Returns:
            BokehBackend

        Raises:
            ValueError: Raised if `descr_pattern` is given and no site matches.
        """
        from fibomat.default_backends import (  # pylint: disable=import-outside-toplevel
            BokehBackend, StubRasterStyle
        )

        kwargs.setdefault('title', self._description)

        plotter: BokehBackend = self._export(BokehBackend, self._sites, descr_pattern=descr_pattern, **kwargs)

        for annot in self._annotations:
            raster = StubRasterStyle(2) if annot.filled else StubRasterStyle(1)

            plotter.process_pattern(
                Pattern(
                    annot.dim_shape,
                    None,
                    raster,
                    _annotation=True,
                    _color=annot.color,
                    description=annot.description,
                )
            )

        plotter.plot()

        if filename:
            plotter.save(filename)
        if show:
            plotter.show()

        return plotter

    @staticmethod
    def _backend_class(exp_backend: t.Union[str, t.Type[BackendT]]) -> t.Type[BackendT]:
        """Return the backend class of a registered name or the class itself."""
        if isinstance(exp_backend, str):
            return _registry().get(exp_backend)  # type: ignore[no-any-return]
        return exp_backend

    def export(self, exp_backend: t.Union[str, t.Type[BackendT]], **kwargs: t.Any) -> BackendT:
        """Exports the layout. Note that the method returns the backend object so you will be able to save a file or
        show a plot. See the docs of the backends for details.

        .. note:: The export method does not save any files on its own. This must be done by the user manually. See
                  docs of the used backend for details.

        Args:
            exp_backend (str, Type[BackendBase]): name of the backend or class. The backend must be registered before
                if a name is used.
            **kwargs: optional arguments are passed to the backend's __init__ method

        Returns:
            BackendBase

        Raises:
            KeyError: Raised if no backend is registered with the name `exp_backend`.
        """
        kwargs.setdefault('description', self._description)
        return self._export(self._backend_class(exp_backend), self._sites, **kwargs)

    def export_multi(self, exp_backend: t.Union[str, t.Type[BackendT]], **kwargs: t.Any) -> t.List[BackendT]:
        """Similar to :meth:`Layout.export` but for each :class:`~fibomat.layout.site.Site` an individual backend
        instance is returned.

        This can be useful if multiple sites are used within fibomat but the pattern system only supports one site at a
        time.

        Args:
            exp_backend (str, Type[BackendBase]): name of the backend or class
            **kwargs: optional arguments are passed to the backend's __init__ method

        Returns:
            List[BackendBase]: one backend per site
        """
        backend_class = self._backend_class(exp_backend)
        kwargs.setdefault('description', self._description)

        return [self._export(backend_class, site, **kwargs) for site in self._sites]

    def export_with_description(
        self,
        exp_backend: t.Union[str, t.Type[BackendT]],
        descr_pattern: t.Set[str],
        **kwargs: t.Any,
    ) -> BackendT:
        """Exports only the sites with a matching description. Otherwise identical to :meth:`Layout.export`.

        Args:
            exp_backend (str, Type[BackendBase]): name of the backend or class
            descr_pattern (Set[str]): regular expressions; only sites with a description which matches (with
                :func:`re.match`) one of them are exported.
            **kwargs: optional arguments are passed to the backend's __init__ method

        Returns:
            BackendBase

        Raises:
            ValueError: Raised if no site has a matching description.
        """
        kwargs.setdefault('description', self._description)
        return self._export(
            self._backend_class(exp_backend), self._sites, descr_pattern=set(descr_pattern), **kwargs
        )

    def add_annotation(
        self,
        dim_shape: DimShape,
        filled: bool = False,
        color: t.Optional[str] = None,
        description: t.Optional[str] = None,
    ) -> None:
        """Add `dim_shape` to an annotation layer. This layer is only used to visualize extra shapes and is ignored by
        the exporting backends.

        Args:
            dim_shape (DimShape): shape
            filled (bool): If True, the shape is plotted filled (only possible if the shape is closed)
            color (str, optional): a color bokeh can understand
            description (str, optional): description

        Raises:
            TypeError: Raised if `dim_shape` is no DimShape.
        """
        if not isinstance(dim_shape, DimShape):
            raise TypeError(f'dim_shape must be a DimShape (shape * unit), got {type(dim_shape).__name__}.')

        self._annotations.append(
            _Annotation(dim_shape=dim_shape, filled=bool(filled), color=color, description=description)
        )
