"""Provides the :class:`RasterizedPoints` class.

Example::

    import numpy as np
    from fibomat.shapes import RasterizedPoints

    # x, y and a dwell time multiplicand of each point
    points = RasterizedPoints(np.array([[0., 0., 1.], [1., 0., 1.], [2., 0., 0.5]]), is_closed=False)

    points.positions, points.weights, points.n_points
    points.translated((1, 1)).bounding_box
    points.repeats_applied(3)  # every point three times
    RasterizedPoints.merged([points, points.translated((0, 1))])
"""
from __future__ import annotations

import typing as t

import numpy as np

from fibomat.linalg import BoundingBox, Vector
from fibomat.shapes.shape import Shape


__all__ = ['RasterizedPoints']


class RasterizedPoints(Shape):
    """Class represents pre-rasterized points (a position and a dwell time multiplicand for every point)."""

    def __init__(self, dwell_points: np.ndarray, is_closed: bool, description: t.Optional[str] = None):
        """
        Args:
            dwell_points (np.ndarray):
                dim must be 2 with shape = (-1, 3). First to components are coordinates and  third a dwell time
                multiplicand. The array is copied.
            is_closed (bool): if True, shape is considered as closed.
            description (str, optional): description

        Raises:
            ValueError: Raised if dimension or shape of dwell points is wrong or the values are not finite
        """
        super().__init__(description)

        try:
            array = np.array(dwell_points, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('dwell_points must be an array of floats.') from error

        if array.ndim != 2:
            raise ValueError('dwell_points must have dimension 2.')
        if array.shape[1] != 3:
            raise ValueError('dwell_points must have shape (N, 3)')
        if not np.all(np.isfinite(array)):
            raise ValueError('dwell_points must be finite.')

        self._dwell_points = array
        self._is_closed = bool(is_closed)

    @classmethod
    def merged(
        cls, other_raster_points: t.Sequence[RasterizedPoints], description: t.Optional[str] = None
    ) -> RasterizedPoints:
        """Create merged :class:`RasterizedPoints` class from list of :class:`RasterizedPoints`.

        Args:
            other_raster_points: points to be merged.
            description (str, optional): description

        Returns:
            RasterizedPoints (not closed)
        """
        if not all(isinstance(raster_points, RasterizedPoints) for raster_points in other_raster_points):
            raise TypeError('Only RasterizedPoints can be merged.')

        if not other_raster_points:
            return cls(np.empty((0, 3)), False, description)

        dwell_points = np.concatenate([raster_points._dwell_points for raster_points in other_raster_points])
        return cls(dwell_points, False, description)

    @staticmethod
    def _read_only(view: np.ndarray) -> np.ndarray:
        view.flags.writeable = False
        return view

    @property
    def positions(self) -> np.ndarray:
        """Coordinates of dwell points (read-only array of shape (N, 2)).

        Access:
            get

        Returns:
            np.ndarray
        """
        return self._read_only(self._dwell_points[:, :2])

    @property
    def weights(self) -> np.ndarray:
        """Dwell time multiplicator of dwell points (read-only array of shape (N,)).

        Access:
            get

        Returns:
            np.ndarray
        """
        return self._read_only(self._dwell_points[:, 2])

    @property
    def dwell_points(self) -> np.ndarray:
        """Dwell points (read-only array of shape (N, 3): x, y, dwell time multiplicator).

        Access:
            get

        Returns:
            np.ndarray
        """
        return self._read_only(self._dwell_points[:])

    @property
    def n_points(self) -> int:
        """Number of dwell points.

        Access:
            get

        Returns:
            int
        """
        return len(self._dwell_points)

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n_points={self.n_points}, is_closed={self._is_closed})'

    @property
    def bounding_box(self) -> BoundingBox:
        """Bounding box of the points.

        Access:
            get

        Raises:
            ValueError: Raised if there are no points.
        """
        if not self.n_points:
            raise ValueError('RasterizedPoints without points have no bounding box.')

        positions = self._dwell_points[:, :2]
        return BoundingBox(np.min(positions, axis=0), np.max(positions, axis=0))

    @property
    def center(self) -> Vector:
        """Center of the bounding box of the points.

        Access:
            get

        Raises:
            ValueError: Raised if there are no points.
        """
        return self.bounding_box.center

    def _impl_translate(self, trans_vec: Vector) -> None:
        trans_vec = Vector(trans_vec)
        self._dwell_points[:, :2] += np.array([trans_vec.x, trans_vec.y])

    def _impl_rotate(self, theta: float) -> None:
        theta = float(theta)
        cos, sin = np.cos(theta), np.sin(theta)
        rotation = np.array([[cos, -sin], [sin, cos]])

        self._dwell_points[:, :2] = self._dwell_points[:, :2] @ rotation.T

    def _impl_scale(self, fac: float) -> None:
        self._dwell_points[:, :2] *= float(fac)

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        angle = Vector(mirror_axis).phi
        cos, sin = np.cos(2 * angle), np.sin(2 * angle)
        reflection = np.array([[cos, sin], [sin, -cos]])

        self._dwell_points[:, :2] = self._dwell_points[:, :2] @ reflection.T

    def repeats_applied(self, repeats: int) -> RasterizedPoints:
        """Return :class:`RasterizedPoints` with the dwell points repeated `repeats` times (one after the other).

        Args:
            repeats (int): number of repeats (at least 1).

        Returns:
            RasterizedPoints (a new object)

        Raises:
            ValueError: Raised if `repeats` is no positive integer.
        """
        if int(repeats) != repeats or repeats < 1:
            raise ValueError('repeats must be a positive integer.')

        return RasterizedPoints(
            np.tile(self._dwell_points, (int(repeats), 1)), self._is_closed, self.description
        )
