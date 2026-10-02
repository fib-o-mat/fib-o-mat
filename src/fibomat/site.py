"""Provides the :class:`Site` class."""

from __future__ import annotations

from turtle import width
from typing import List, Optional, Union

import numpy as np

from fibomat import layout
from fibomat.linalg import DimBoundingBox, DimTransformable, DimVector, DimVectorLike
from fibomat.linalg.vectors.vector import Vector
from fibomat.mill import MillBase
from fibomat.pattern import Pattern
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.shapes import DimShape


class Site(DimTransformable):
    """
    The `Site` class is used to collect shapes with its patterning settings.

    .. note:: All shape positions added to each site are interpreted relative to the site's position!

    .. note:: If fov is not passed, it will be determined by the bounding box of the added patterns. But if a fixed FOV
              is given, it is **not** checked if the added shapes fit inside the fov.

    """

    def __init__(
        self,
        dim_center: DimVectorLike,
        dim_fov: Optional[DimVectorLike] = None,
        *,
        description: Optional[str] = None,
    ):
        """
        Args:
            dim_center (DimVectorLike): Center coordinate of the site.
            dim_fov (DimVectorLike, optional): The fov (field of view) to be used. If not given, the fov will be
                                               calculated automatically.
            description (str, optional): description
        """
        super().__init__(description=description)

        self._center = DimVector(dim_center)

        self._theta_vec = Vector(1, 0)

        # TODO: check if fov is valid?
        self._fov = DimVector(dim_fov) if dim_fov is not None else None

        self._fov_scale = 1.1  # only used if no default fov is given

        self._patterns: List[Pattern] = []

    @property
    def fov(self) -> DimVector:
        """Field-of-view of the site.

        Access:
            get

        Returns:
            DimVector
        """
        if self._fov:
            return self._fov
        else:
            bbox = self.bounding_box
            center = bbox.center

            width = max(
                abs(center.x - bbox.lower_left.x), abs(center.x - bbox.upper_right.x)
            )
            height = max(
                abs(center.y - bbox.lower_left.y), abs(center.y - bbox.upper_right.y)
            )

            size = max(width, height)  # make it square

            return self._fov_scale * 2 * DimVector(size, size)

    @property
    def square_fov(self) -> DimVector:
        """Squared field-of-view of the site.

        Access:
            get

        Returns:
            DimVector
        """
        fov = self.fov
        size = fov.x if fov.x > fov.y else fov.y
        return DimVector(size, size)

    @property
    def fov_bounding_box(self) -> DimBoundingBox:
        """Bounding box given by fov and center rather from the contained patterns.

        Returns:
            DimBoundingBox
        """
        fov = self.fov

        fov_x_2 = fov.x / 2
        fov_y_2 = fov.y / 2
        return DimBoundingBox(
            self._center - (fov_x_2, fov_x_2), self._center + (fov_x_2, fov_x_2)
        )

    @property
    def empty(self) -> bool:
        """If True, site does not contain any shapes

        Access:
            get

        Returns:
            bool
        """
        return not self._patterns

    @property
    def bounding_box(self) -> DimBoundingBox:
        """Bounding box of the added patterns.

        Access:
            get

        Returns:
            DimBoundingBox
        """
        # bbox = DimBoundingBox(self._center, self._center)

        if not self._patterns:
            raise RuntimeError("Cannot calculate bounding box of empty site.")

        bbox = self._patterns[0].bounding_box

        for pattern in self._patterns[1:]:
            bbox = bbox.extended(pattern.bounding_box)

        return bbox

    @property
    def bounding_box_abs(self) -> DimBoundingBox:
        """Bounding box of the added patterns.

        Access:
            get

        Returns:
            DimBoundingBox
        """
        # bbox = DimBoundingBox(self._center, self._center)

        if not self._patterns:
            raise RuntimeError("Cannot calculate bounding box of empty site.")

        bbox = self._patterns[0].bounding_box

        for pattern in self._patterns[1:]:
            bbox = bbox.extended(pattern.bounding_box)

        bbox._lower_left += self.center
        bbox._upper_right += self.center

        return bbox

    @property
    def patterns(self):
        """Contained patterns in site

        Access:
            get

        Returns:
            List[Pattern]
        """
        return self._patterns

    @property
    def patterns_absolute(self):
        """Return a list of all patterns contained in the side which are shifted by the side's center.
        Hence, these patterns' positions are **not** relative to the site's center anymore but absolute to the
        coordinate origin.

        Access:
            get

        Returns:
            List[Pattern]
        """
        return [pattern.translated(self._center) for pattern in self._patterns]

    def create_pattern(
        self,
        dim_shape: DimShape,
        mill: MillBase,
        raster_style: RasterStyle,
        description: Optional[str] = None,
        **kwargs,
    ) -> Pattern:
        """Creates a pattern in-place (returned pattern is automatically added to the site).
        The parameters are identical to the __init__method of the :class:`fibomat.pattern.Pattern` class.

        Args:
            dim_shape:
            mill:
            raster_style:
            description:
            **kwargs:

        Returns:
            Pattern
        """
        pattern = Pattern(
            dim_shape, mill, raster_style, description=description, **kwargs
        )
        self.add_pattern(pattern)
        return pattern

    def add_pattern(self, ptn: Union[Pattern, layout.LayoutBase]) -> None:
        """Adds a :class:`fibomat.pattern.Pattern` or Layoutbase[Pattern] to the site.

        Args:
            ptn (Pattern):  new pattern

        Returns:
            None
        """
        if isinstance(ptn, layout.LayoutBase):
            for extracted_pattern in ptn.layout_elements():
                self._patterns.append(extracted_pattern)
        else:
            self._patterns.append(ptn)

    def __iadd__(self, ptn: Union[Pattern, layout.LayoutBase]) -> Site:
        """Adds a :class:`fibomat.pattern.Pattern` to the site.
        Identical to :meth:`add_pattern`

        Args:
            ptn: new pattern

        Returns:
            None
        """
        self.add_pattern(ptn)
        return self

    @property
    def center(self) -> DimVector:
        """Center of the site.

        Access:
            get

        Returns:
            DimVector
        """
        return self._center

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        trans_vec = DimVector(trans_vec)
        self._center += DimVector(trans_vec)

    def _impl_rotate(self, theta: float, _allow_any_rot=False) -> None:
        if not _allow_any_rot:
            if not np.isclose(np.mod(theta, np.pi / 2), 0.0):
                raise ValueError("Sites can only be rotated by multiples of pi/2")

            if not np.isclose(np.mod(theta, np.pi), 0.0):
                self._fov = DimVector(self._fov.y, self._fov.x)

        self._center = self._center.rotated(theta)

        self._theta_vec = self._theta_vec.rotated(theta)

        for ptn in self._patterns:
            ptn._impl_rotate(theta)

    def _impl_scale(self, fac: float) -> None:
        self._fov *= float(fac)
        self._center *= float(fac)

        for ptn in self._patterns:
            ptn._impl_scale(fac)

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        mirror_axis = DimVector(mirror_axis)

        if not np.isclose(
            np.mod(mirror_axis.vector.angle_about_x_axis, np.pi / 4), 0.0
        ):
            raise ValueError(
                "Sites can only be mirrored on the axes or their diagonals."
            )

        self._center = self._center.mirrored(mirror_axis)

        self._theta_vec = self._theta_vec.mirrored(mirror_axis.vector)

        if not np.isclose(np.mod(mirror_axis.vector.angle_about_x_axis, np.pi), 0.0):
            self._fov = DimVector(self._fov.y, self._fov.x)

        for ptn in self._patterns:
            ptn._impl_mirror(mirror_axis)
