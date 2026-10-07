from typing import List

from fibomat.layout.site import Site
from fibomat.linalg import Vector
from fibomat.units import U_
from fibomat.default_backends.npve.step_and_repeat.common_models import FIBShape


class _SaRAxis:
    def __init__(self, value: float):
        self.value = f"{value:.4f}"


class SaRSite:
    def __init__(self, index: int, site: Site, last_site_center: Vector):
        self.index = index
        center = site.center.vector_as(U_("µm"))
        self.dx = center.x - last_site_center.x
        self.dy = center.y - last_site_center.y

        self.center = center

        # print(self.dx, self.dy)

        self.fov = site.square_fov[0].m_as("µm")
        #
        # print(self.fov)

        self.shapes = {"shapes_list": []}

    def add_fib_shape(self, shape: FIBShape):
        self.shapes["shapes_list"].append(shape)


class SaRSharedShapes:
    def __init__(self, fib_shapes: List[FIBShape]):
        self.fib_shapes = fib_shapes


class SaRFile:
    def __init__(
        self,
        sites: List[SaRSite],
        shared_shapes: SaRSharedShapes = None,
        pre_image: bool = False,
        post_image: bool = False,
    ):
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

        # mapped_site_list = []
        #
        # for i, site in enumerate(sites):
        #     if not site.empty:
        #         mapped_site_list.append(_SaRSite(0, sites[0], sites[0].center))
        #         first_site_index = i
        #
        # if not mapped_site_list:
        #     raise RuntimeError('No non-empty site in sample. There is nothing to export.')
        #
        # last_site_index = first_site_index
        # for i, site in enumerate(sites[first_site_index+1:], start=first_site_index+1):
        #     if not site.empty:
        #         mapped_site_list.append(_SaRSite(i, site, sites[last_site_index].center))
        #         last_site_index = i

        self.sites = {"sites_list": sites}
