"""Models of the sites and the file of the NPVE step and repeat format (see ``sar_schemas.py`` for the format)."""
from __future__ import annotations

import typing as t

from fibomat.default_backends.npve.step_and_repeat.common_models import FIBShape
from fibomat.layout.site import Site
from fibomat.linalg import Vector
from fibomat.units import unit


__all__ = ['SaRSite', 'SaRSharedShapes', 'SaRFile']


_MICRON = unit('µm')


class _SaRAxis:
    """Position of a stage axis, written with four decimals."""

    def __init__(self, value: float):
        self.value = f"{value:.4f}"


class SaRSite:
    """A site of the file: position relative to the previous site, field of view and shapes."""

    def __init__(self, index: int, site: Site, last_site_center: Vector):
        """
        Args:
            index (int): index of the site
            site (Site): site
            last_site_center (Vector): center of the previous site in µm (of this site for the first site)

        Raises:
            ValueError: Raised if the field of view of the site cannot be determined.
        """
        self.index = index
        center = site.center.vector_as(_MICRON)
        self.dx = center.x - last_site_center.x
        self.dy = center.y - last_site_center.y

        self.center = center

        self.fov = site.square_fov[0].m_as("µm")

        self.shapes: t.Dict[str, t.List[FIBShape]] = {"shapes_list": []}

    def add_fib_shape(self, shape: FIBShape) -> None:
        """Add a shape to the site.

        Args:
            shape (FIBShape): shape
        """
        self.shapes["shapes_list"].append(shape)


class SaRSharedShapes:
    """The shapes which are shared by all sites."""

    def __init__(self, fib_shapes: t.List[FIBShape]):
        """
        Args:
            fib_shapes (List[FIBShape]): shapes
        """
        self.fib_shapes = fib_shapes


class SaRFile:
    """A step and repeat file."""

    def __init__(
        self,
        sites: t.List[SaRSite],
        shared_shapes: t.Optional[SaRSharedShapes] = None,
        pre_image: bool = False,
        post_image: bool = False,
    ):
        """
        Args:
            sites (List[SaRSite]): sites
            shared_shapes (SaRSharedShapes, optional): shapes which are shared by all sites
            pre_image (bool): take an image before the patterning
            post_image (bool): take an image after the patterning

        Raises:
            ValueError: Raised if there are no sites.
        """
        self.options = {
            "pre_image": bool(pre_image),
            "post_image": bool(post_image),
            "share_shapes": shared_shapes is not None,
        }

        self.stage_position = {
            "x": _SaRAxis(0.0),
            "y": _SaRAxis(0.0),
            "z": _SaRAxis(0.0),
            "m": _SaRAxis(0.0),
            "t": _SaRAxis(0.0),
            "r": _SaRAxis(0.0),
        }

        self.shared_shapes = shared_shapes

        if not sites:
            raise ValueError("At least one non-empty site is required.")

        self.sites = {"sites_list": sites}
