"""Provides the :class:`SpotListBackend`, which rasterizes all patterns and saves a list of dwell points.

Example::

    exported = layout.export(SpotListBackend, length_unit=unit('nm'), time_unit=unit('µs'))
    exported.save('points.txt')
"""
from __future__ import annotations

import collections
import configparser
import datetime
import typing as t
import warnings

import numpy as np

from fibomat import composite_shapes, shapes
from fibomat.backend import BackendBase
from fibomat.default_backends._unit_helpers import to_length_unit, to_time_unit
from fibomat.layout.pattern import Pattern
from fibomat.layout.site import Site
from fibomat.linalg import BoundingBox, Vector
from fibomat.units import DimFloat, LengthUnit, TimeUnit, has_time_dim, scale_to
from fibomat.utils import PathLike


__all__ = ['SpotListBackend']


def _integer_dwell_times(dwell_points: np.ndarray) -> np.ndarray:
    """Round the dwell times to integers (they are written as integers); warn if this changes them noticeably.

    Args:
        dwell_points (np.ndarray): dwell points, shape (n, 3)

    Returns:
        np.ndarray: copy of the dwell points with integer dwell times
    """
    dwell_points = np.array(dwell_points, dtype=float)
    rounded = np.rint(dwell_points[:, 2])
    if np.any(np.abs(rounded - dwell_points[:, 2]) > 1e-3):
        warnings.warn('The dwell times are not integers; they are rounded in the saved file.', stacklevel=3)
    dwell_points[:, 2] = rounded
    return dwell_points


def _default_save_impl(filename: PathLike, dwell_points: np.ndarray, parameters: t.Dict[str, t.Any]) -> None:
    """Save the dwell points as text file with a header (see :class:`SpotListBackend` for the parameters)."""
    header = configparser.ConfigParser()

    fov = parameters['fov']
    base_dwell_time = parameters['base_dwell_time']

    header_dict: t.Dict[str, str] = collections.OrderedDict()
    header_dict['length_unit'] = f'{parameters["length_unit"]:~P}'
    header_dict['time_unit'] = f'{parameters["time_unit"]:~P}'
    header_dict['fov'] = f'{fov.width}, {fov.height}'
    header_dict['center'] = f'{tuple(fov.center)}'
    header_dict['base_dwell_time'] = 'None' if base_dwell_time is None else f'{base_dwell_time.quantity:~P}'
    header_dict['total_dwell_time'] = f'{parameters["total_dwell_time"]}'
    header_dict['number_of_points'] = f'{parameters["number_of_points"]}'
    header_dict['description'] = f'{parameters["description"]}'
    header_dict['time_stamp'] = f'{parameters["time_stamp"]}'

    header['Info'] = header_dict

    with open(filename, 'w', encoding='utf-8') as file:
        header.write(file)
        file.write('[Points]\n')
        np.savetxt(file, _integer_dwell_times(dwell_points), '%.5f %.5f %d')


class SpotListBackend(BackendBase):
    """Rasterizes all patterns and saves the dwell points: a list of the form ::

        x1, y1, t1
        x2, y2, t2
        ...

    The positions are absolute (the positions of the sites are added). The format of the saved file is defined by a
    custom function `save_impl`.

    The `save_impl` function must have the form ::

        def func(filename: PathLike, dwell_points: np.ndarray, parameters: Dict[str, Any]):
            # open the file `filename` and save dwell_points there
            ...

    The dictionary `parameters` holds further metadata with the keys ``length_unit`` and ``time_unit`` (pint units),
    ``fov`` (unitless :class:`~fibomat.linalg.BoundingBox` in the length unit: the bounding box of the points; the field
    of view of the sites if all points are on a line), ``base_dwell_time`` (DimFloat or None), ``total_dwell_time``,
    ``number_of_points``, ``description`` and ``time_stamp``. The default implementation is `_default_save_impl` in
    the source file of the backend.
    """

    def __init__(
        self,
        save_impl: t.Optional[t.Callable[[PathLike, np.ndarray, t.Dict[str, t.Any]], None]] = None,
        base_dwell_time: t.Optional[DimFloat[t.Any]] = None,
        length_unit: t.Optional[LengthUnit] = None,
        time_unit: t.Optional[TimeUnit] = None,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            save_impl (Callable, optional): function which saves the dwell points (see above)
            base_dwell_time (DimFloat, optional): if given, the dwell times are expressed as integer multiples of
                `base_dwell_time`.
            length_unit (LengthUnit, optional): length unit of the points, default ``unit('µm')``
            time_unit (TimeUnit, optional): time unit of the dwell times, default ``unit('µs')``
            description (str, optional): description

        Raises:
            TypeError: Raised if `base_dwell_time` is no dimensioned value or `save_impl` is not callable.
            ValueError: Raised if a unit has the wrong dimension or `base_dwell_time` is not a positive time.
        """
        super().__init__(description)

        if save_impl is not None and not callable(save_impl):
            raise TypeError('save_impl must be callable.')
        self._save_impl = save_impl if save_impl is not None else _default_save_impl

        self._length_unit = to_length_unit(length_unit if length_unit is not None else 'µm')
        self._time_unit = to_time_unit(time_unit if time_unit is not None else 'µs')

        if base_dwell_time is not None:
            if not isinstance(base_dwell_time, DimFloat):
                raise TypeError('base_dwell_time must be a dimensioned value like 0.1 * unit("µs").')
            if not has_time_dim(base_dwell_time) or not base_dwell_time.magnitude > 0.:
                raise ValueError('base_dwell_time must be a positive time.')
        self._base_dwell_time = base_dwell_time

        self._site_count = 0
        self._site_pos = np.zeros(2)
        self._dwell_points: t.List[np.ndarray] = []
        self._site_bounding_boxes: t.List[BoundingBox] = []  # fields of view of the sites in `length_unit`

    def process_site(self, new_site: Site) -> None:
        self._site_pos = np.asarray(new_site.center.vector_as(self._length_unit), dtype=float)

        try:
            fov_bounding_box = new_site.fov_bounding_box
        except ValueError:
            pass  # empty site without fov
        else:
            self._site_bounding_boxes.append(BoundingBox(
                fov_bounding_box.lower_left.vector_as(self._length_unit),
                fov_bounding_box.upper_right.vector_as(self._length_unit),
            ))

        self._site_count += 1

        super().process_site(new_site)

    def _fov(self, dwell_points: np.ndarray) -> BoundingBox:
        """Bounding box of the points, or of the fields of view of the sites if the points have no extent."""
        bbox = BoundingBox.from_points(dwell_points[:, :2])
        if (bbox.height == 0 or bbox.width == 0) and self._site_bounding_boxes:
            bbox = self._site_bounding_boxes[0]
            for other in self._site_bounding_boxes[1:]:
                bbox = bbox.extended(other)
        return bbox

    def dwell_points(self) -> np.ndarray:
        """All dwell points (with absolute positions in the length unit, dwell times in the time unit; expressed in
        multiples of the base dwell time if one is set).

        Returns:
            np.ndarray: array with shape (n, 3)

        Raises:
            ValueError: Raised if there are no dwell points.
        """
        if not self._dwell_points:
            raise ValueError('There are no dwell points.')

        dwell_points = np.concatenate(self._dwell_points)

        if self._base_dwell_time is not None:
            multiples = dwell_points[:, 2] / scale_to(self._time_unit, self._base_dwell_time)
            if np.any(np.abs(multiples - np.rint(multiples)) > 1e-3):
                warnings.warn('Some dwell times are no integer multiples of the base dwell time.', stacklevel=3)
            dwell_points[:, 2] = np.rint(multiples)

        return dwell_points

    def save(self, filename: PathLike) -> None:
        """Save the dwell points.

        Args:
            filename (PathLike): filename

        Raises:
            ValueError: Raised if there are no dwell points.
        """
        if self._site_count > 1:
            warnings.warn('The dwell points of multiple sites are merged.', stacklevel=2)

        dwell_points = self.dwell_points()

        self._save_impl(
            filename, dwell_points,
            {
                'length_unit': self._length_unit, 'time_unit': self._time_unit,
                'fov': self._fov(dwell_points), 'base_dwell_time': self._base_dwell_time,
                'total_dwell_time': np.sum(dwell_points[:, 2]),
                'number_of_points': len(dwell_points), 'description': self._description,
                'time_stamp': str(datetime.datetime.now()),
            }
        )

    def _rasterize_and_add(self, ptn: Pattern) -> None:
        dwell_points = np.array(ptn.raster_style.rasterize(
            ptn.dim_shape, ptn.mill, self._length_unit, self._time_unit
        ).dwell_points)

        dwell_points[:, :2] += self._site_pos

        self._dwell_points.append(dwell_points)

    def arc_spline(self, ptn: Pattern[shapes.ArcSpline]) -> None:
        self._rasterize_and_add(ptn)

    def parametric_curve(self, ptn: Pattern[shapes.ParametricCurve]) -> None:
        self._rasterize_and_add(ptn)

    def rasterized_points(self, ptn: Pattern[shapes.RasterizedPoints]) -> None:
        self._rasterize_and_add(ptn)

    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        self._rasterize_and_add(ptn)

    def line(self, ptn: Pattern[shapes.Line]) -> None:
        self._rasterize_and_add(ptn)

    def polyline(self, ptn: Pattern[shapes.Polyline]) -> None:
        self._rasterize_and_add(ptn)

    def polygon(self, ptn: Pattern[shapes.Polygon]) -> None:
        self._rasterize_and_add(ptn)

    def arc(self, ptn: Pattern[shapes.Arc]) -> None:
        self._rasterize_and_add(ptn)

    def circle(self, ptn: Pattern[shapes.Circle]) -> None:
        self._rasterize_and_add(ptn)

    def ellipse(self, ptn: Pattern[shapes.Ellipse]) -> None:
        self._rasterize_and_add(ptn)

    def rect(self, ptn: Pattern[shapes.Rect]) -> None:
        self._rasterize_and_add(ptn)

    def ring(self, ptn: Pattern[composite_shapes.Ring]) -> None:
        self._rasterize_and_add(ptn)

    def hollow_arc_spline(self, ptn: Pattern[composite_shapes.HollowArcSpline]) -> None:
        self._rasterize_and_add(ptn)
