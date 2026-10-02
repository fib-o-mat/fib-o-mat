"""Provides the :class:`Sample` class."""

import dataclasses
import re
from typing import List, Optional, Set, Type, TypeVar, Union

from fibomat.backend import BackendBase, registry
from fibomat.default_backends import BokehBackend, StubRasterStyle
from fibomat.describable import Describable
from fibomat.layout import LayoutBase
from fibomat.linalg import DimVectorLike
from fibomat.linalg.boundingboxes import boundingbox
from fibomat.linalg.boundingboxes.dim_boundingbox import DimBoundingBox
from fibomat.pattern import Pattern
from fibomat.shapes import DimShape
from fibomat.site import Site
from fibomat.utils import PathLike


@dataclasses.dataclass(frozen=True)
class _Annotation:
    dim_shape: DimShape
    filled: bool
    color: Optional[str]
    description: Optional[str]


BackendType = TypeVar("BackendType")


class Sample(Describable):
    """
    This class is the glueing between all subcomponents of the library.
    All shapes and their milling settings are added to this class and can be exported with the help of registered
    backends .

    """

    def __init__(self, *, description: Optional[str] = None):
        """
        Args:
            description (str, optional): Optional description of the project, default to None
        """
        super().__init__(description)

        self._sites: List[Site] = []
        self._annotations: List[_Annotation] = []

    def create_site(
        self,
        dim_position: DimVectorLike,
        dim_fov: Optional[DimVectorLike] = None,
        description: Optional[str] = None,
    ) -> Site:
        """
        Creates and Site in-place (hence, the Site is automatically added to the sample). Patterns can be added to the
        returned object.

        See :class:`fibomat.site.Site.__init__` for argument description.

        Returns:
            Site
        """
        new_site: Site = Site(dim_position, dim_fov, description=description)
        self._sites.append(new_site)
        return new_site

    def add_site(self, site_like: Site) -> None:
        """
        Adds a Site to the project.
        Alternatively, the '+=' operator can be used.

        Args:
            site_like (Site): new site

        Returns:
            None
        """
        if isinstance(site_like, LayoutBase):
            for site_ in site_like.layout_elements():
                self._sites.append(site_)
        else:
            self._sites.append(site_like)

    @property
    def number_of_sites(self) -> int:
        """int: number of added sites"""
        return len(self._sites)

    @property
    def bounding_box(self) -> Optional[DimBoundingBox]:
        if not self._sites:
            return None
        else:
            bbox = self._sites[0].bounding_box

            for site in self._sites[1:]:
                bbox = bbox.extended(site.bounding_box)

            return bbox

    def __iadd__(self, site_like):
        """See :meth:`~Sample.add_site`."""
        self.add_site(site_like)
        return self

    @staticmethod
    def _export(
        backend_class: Type[BackendType],
        sites: Union[Site, List[Site]],
        descr_pattern: Optional[Set[str]] = None,
        **kwargs,
    ) -> BackendType:
        def _matches(description_) -> bool:
            for pattern in descr_pattern:
                if re.match(pattern, description_):
                    return True
            return False

        exporter: Type[BackendBase] = backend_class(**kwargs)
        if isinstance(sites, Site):
            if descr_pattern:
                print(
                    "Warning: ignoring descriptions in _export for single site export."
                )
            exporter.process_site(sites)
        else:
            for site_ in sites:
                if descr_pattern:
                    description = site_.description
                    if description and _matches(description):
                        exporter.process_site(site_)
                        # descriptions.remove(description)
                else:
                    exporter.process_site(site_)

            # if descriptions:
            #    raise RuntimeError(f'Could not find sites with descriptions: {descriptions} in sample')

        return exporter

    def plot(
        self,
        show: bool = True,
        filename: Optional[PathLike] = None,
        descr_pattern: Optional[Set[str]] = None,
        **kwargs,
    ) -> BokehBackend:
        """
        Plots and save the project using the :class:`~fibomat.default_backends.bokeh_backend.BokehBackend`.

        Args:
            show (bool): if true, the plot is opened in a browser automatically
            filename (PathLike, optional): if filename is given, the plot is saved in this file. The file suffix should
                                           be `*.htm` or `*.html`, default to None
            `**kwargs`: parameters for the bokeh backend. These are directly passed to the __init__ method of the
                        BokehBackend class. The title parameter is automatically set to the :attr:`Sample.description`

        Returns:
            None
        """

        plotter: BokehBackend = self._export(
            BokehBackend,
            self._sites,
            descr_pattern=descr_pattern,
            title=self._description,
            **kwargs,
        )

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

    def export(
        self, exp_backend: Union[str, Type[BackendBase]], **kwargs
    ) -> BackendBase:
        """
        Exports the project. Note that the method returns the backend object so you will be able to save a file or show
        a plot. See backends example nd docs for details.

        .. note:: The export method does not save any files on its one. This must be done by the user manually. See docs
                  of the used backend for details.

        Args:
            exp_backend (str or Type[backend.BackendBase]):
                name of the backend or class. The backend must be registered before.
            **kwargs: optional arguments are passed to the backend's __init__ method

        Returns:
            BackendBase
        """

        if isinstance(exp_backend, str):
            exp_backend = registry.get(exp_backend)

        return self._export(
            exp_backend, self._sites, description=self._description, **kwargs
        )

    def export_multi(
        self, exp_backend: Union[str, Type[BackendBase]], **kwargs
    ) -> List[BackendBase]:
        """
        Similar to :meth:`Project.export` but for each :class:`fibomat.site.Site` an individual backend instance is
        returned.

        This can be usefull if multiple sites are used within fibomat but the pattern system only supports one site at a
        time.

        Returns:
            List[BackendBase]
        """
        backends: List[BackendBase] = []

        if isinstance(exp_backend, str):
            exp_backend = registry.get(exp_backend)

        for added_site in self._sites:
            backends.append(
                self._export(
                    exp_backend, added_site, description=self._description, **kwargs
                )
            )

        return backends

    def export_with_description(
        self,
        exp_backend: Union[str, Type[BackendBase]],
        descr_pattern: Set[str],
        **kwargs,
    ) -> BackendBase:
        """
        Exports the project. Note that the method returns the backend object so you will be able to save a file or show
        a plot. See backends example nd docs for details.

        Only sites with a description given in `descriptions` will be exported. An exception is raised, if no site is
        found with a given description.

        .. note:: The export method does not save any files on its one. This must be done by the user manually. See docs
                  of the used backend for details.

        Args:
            exp_backend (str or Type[backend.BackendBase]):
                name of the backend or class. The backend must be registered before.
            descr_pattern (Set[str]): a set of descriptions of the sites which should be exported.
            **kwargs: optional arguments are passed to the backend's __init__ method

        Returns:
            BackendBase
        """
        if isinstance(exp_backend, str):
            exp_backend = registry.get(exp_backend)
        return self._export(
            exp_backend,
            self._sites,
            description=self._description,
            descr_pattern=descr_pattern,
            **kwargs,
        )

    def add_annotation(
        self,
        dim_shape: DimShape,
        filled: bool = False,
        color: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Add `dim_shape` to a annotation layer. This layer is only used to visualize extra shapes and is ignored by the
        exporting backend.

        Args:
            dim_shape (DimShape): shape
            filled (bool): If True, shape is plotted filled (only possible if shape is closed)
            color (str, Optional): a color bokeh can understand, default to None
            description (str, optional): description, default to None

        Returns:
            None
        """
        self._annotations.append(
            _Annotation(
                dim_shape=dim_shape, filled=filled, color=color, description=description
            )
        )
