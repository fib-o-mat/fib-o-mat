"""Provides the :class:`BokehImage` class.

Example::

    image = BokehImage('micrograph.png', pixel_size=0.01)  # one pixel is 0.01 units wide
    layout.add_annotation(image * unit('µm'))  # shown behind the patterns of the plot
"""
from __future__ import annotations

import typing as t

import numpy as np
import PIL.Image

from fibomat.linalg import BoundingBox, Vector, VectorLike
from fibomat.shapes.shape import Shape
from fibomat.utils import PathLike


__all__ = ['BokehImage']


class BokehImage(Shape):
    """An image which is plotted by the :class:`~fibomat.default_backends.bokeh_backend.BokehBackend` as annotation.

    The image is a rectangle in the plot: pixels have the size `pixel_size`, measured in the unit which is given when
    the image is multiplied with a unit. The image is only plotted, it cannot be exported by other backends or be
    transformed (except for translations).
    """

    def __init__(
        self,
        file: t.Union[PathLike, PIL.Image.Image],
        pixel_size: float,
        center: t.Optional[VectorLike] = None,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            file (PathLike, PIL.Image.Image): image file or image
            pixel_size (float): size of one (square) pixel
            center (VectorLike, optional): center of the image, default (0, 0)
            description (str, optional): description

        Raises:
            ValueError: Raised if pixel_size is not positive and finite.
        """
        super().__init__(description)

        pixel_size = float(pixel_size)
        if not np.isfinite(pixel_size) or pixel_size <= 0.:
            raise ValueError(f'pixel_size must be positive and finite, got {pixel_size}.')

        image = file if isinstance(file, PIL.Image.Image) else PIL.Image.open(file)
        image = image.convert('RGBA')

        n_columns, n_rows = image.size
        # bokeh expects the first row at the bottom
        self._image_data = np.ascontiguousarray(image, dtype=np.uint8)[::-1, :, :]

        self._pixel_size = pixel_size
        self._width = n_columns * pixel_size
        self._height = n_rows * pixel_size
        self._center = Vector(center) if center is not None else Vector(0, 0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(width={self._width}, height={self._height}, center={self._center!r})'

    @property
    def is_closed(self) -> bool:
        return True

    @property
    def width(self) -> float:
        """Width of the image.

        Access:
            get
        """
        return self._width

    @property
    def height(self) -> float:
        """Height of the image.

        Access:
            get
        """
        return self._height

    @property
    def data(self) -> np.ndarray:
        """RGBA pixel data (uint8, shape (rows, columns, 4)), the first row is the bottom row.

        Access:
            get
        """
        return self._image_data

    @property
    def bounding_box(self) -> BoundingBox:
        half = Vector(self._width / 2., self._height / 2.)
        return BoundingBox(self._center - half, self._center + half)

    @property
    def center(self) -> Vector:
        return self._center

    def _impl_translate(self, trans_vec: Vector) -> None:
        self._center = self._center + trans_vec

    def _impl_rotate(self, theta: float) -> None:
        raise NotImplementedError('Images cannot be rotated.')

    def _impl_scale(self, fac: float) -> None:
        raise NotImplementedError('Images cannot be scaled.')

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        raise NotImplementedError('Images cannot be mirrored.')
