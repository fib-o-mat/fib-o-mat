"""Provides the :class:`Site` class.

Example:
    >>> from fibomat.layout import Site
    >>> from fibomat.linalg import DimVector
    >>> from fibomat.mill import Mill
    >>> from fibomat.raster_styles import ScanSequence, one_d
    >>> from fibomat.shapes import Line
    >>> from fibomat.units import unit
    >>> site = Site(DimVector(10 * unit('µm'), 0 * unit('µm')), fov_scale=1.5)
    >>> _ = site.create_pattern(
    ...     Line((1, 0), (3, 0)) * unit('µm'), Mill(1. * unit('ms'), 1),
    ...     one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
    ... )
    >>> site.fov.x.m_as('µm')  # square, centered at the site, 3 µm * 2 * 1.5 (patterns are relative to the center)
    9.0
"""
from __future__ import annotations

import math
import typing as t

import numpy as np

from fibomat import arrangements
from fibomat.layout.pattern import Pattern
from fibomat.linalg import DimBoundingBox, DimTransformable, DimVector, DimVectorLike, Vector
from fibomat.mill import MillBase
from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.shapes import DimShape


__all__ = ['Site']


DEFAULT_FOV_SCALE = 1.1
"""Default factor by which the minimal field of view is increased if no fov is given."""


def _check_fov_scale(fov_scale: float) -> float:
    """Check that the fov scale is finite and not smaller than 1.

    Args:
        fov_scale (float): scale

    Returns:
        float: scale

    Raises:
        ValueError: Raised if the scale is smaller than 1 or not finite.
    """
    fov_scale = float(fov_scale)
    if not math.isfinite(fov_scale) or fov_scale < 1.:
        raise ValueError(f'fov_scale must be finite and not smaller than 1, got {fov_scale}.')
    return fov_scale


class Site(DimTransformable):
    """A site is a field of view at a position. It collects patterns.

    .. note:: All pattern positions added to a site are interpreted relative to the position (center) of the site!

    .. note:: If the fov is not passed, it is determined automatically: it is the smallest square centered at the site
              which contains all patterns, multiplied with `fov_scale`. If a fov is passed, it is **not** checked if
              the added patterns fit inside of it.
    """

    def __init__(
        self,
        dim_center: DimVectorLike,
        dim_fov: t.Optional[DimVectorLike] = None,
        *,
        description: t.Optional[str] = None,
        fov_scale: float = DEFAULT_FOV_SCALE,
    ):
        """
        Args:
            dim_center (DimVectorLike): Center coordinate of the site.
            dim_fov (DimVectorLike, optional): The fov (field of view, width and height) to be used. If not given, the
                fov will be calculated automatically.
            description (str, optional): description
            fov_scale (float): factor by which the minimal fov is increased if `dim_fov` is not given (at least 1).
                The default adds a margin of 10 %.

        Raises:
            ValueError: Raised if the fov is not positive or fov_scale is smaller than 1 or not finite.
        """
        super().__init__(description=description)

        self._center = DimVector(dim_center)

        self._theta_vec = Vector(1, 0)

        self._fov: t.Optional[DimVector] = None
        if dim_fov is not None:
            fov = DimVector(dim_fov)
            if not (fov.x.magnitude > 0. and fov.y.magnitude > 0.):
                raise ValueError(f'The fov must be positive, got {fov!r}.')
            self._fov = fov

        self._fov_scale = _check_fov_scale(fov_scale)

        self._patterns: t.List[Pattern] = []

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(center={self._center!r}, fov={self._fov!r}, '
            f'n_patterns={len(self._patterns)})'
        )

    @property
    def fov_scale(self) -> float:
        """Factor by which the minimal fov is increased if no explicit fov is given.

        Access:
            get
        """
        return self._fov_scale

    @property
    def has_explicit_fov(self) -> bool:
        """True if a fov was passed to the site (False if it is calculated from the patterns).

        Access:
            get
        """
        return self._fov is not None

    @property
    def theta(self) -> float:
        """Rotation angle of the site (a multiple of pi/2) in rad.

        Access:
            get
        """
        return float(self._theta_vec.angle_about_x_axis)

    @property
    def fov(self) -> DimVector:
        """Field of view (width, height) of the site, centered at :attr:`Site.center`.

        If no fov was passed, the fov is the smallest square around the center of the site which contains all patterns,
        scaled with :attr:`Site.fov_scale`.

        Access:
            get

        Raises:
            ValueError: Raised if no fov was passed and it cannot be calculated, because the site is empty or the
                patterns do not have an extent.
        """
        if self._fov is not None:
            return self._fov

        if not self._patterns:
            raise ValueError('The fov of an empty site cannot be calculated, pass dim_fov to the site.')

        # the patterns are relative to the center of the site, which is the center of the fov, too
        bbox = self.bounding_box
        unit = bbox.lower_left.unit
        lower_left = bbox.lower_left.vector_as(unit)
        upper_right = bbox.upper_right.vector_as(unit)

        half_size = max(
            abs(lower_left.x), abs(lower_left.y), abs(upper_right.x), abs(upper_right.y)
        )
        if half_size == 0.:
            raise ValueError(
                'The fov cannot be calculated because the patterns of the site have no extent; pass dim_fov.'
            )

        size = self._fov_scale * 2. * half_size
        return DimVector.from_vector(Vector(size, size), unit)

    @property
    def square_fov(self) -> DimVector:
        """Square field of view (the larger of the two sides of :attr:`Site.fov`).

        Access:
            get
        """
        fov = self.fov
        size = fov.x if fov.x > fov.y else fov.y
        return DimVector(size, size)

    @property
    def fov_bounding_box(self) -> DimBoundingBox:
        """Bounding box given by fov and center (rather than by the contained patterns).

        Access:
            get
        """
        fov = self.fov
        half = DimVector(fov.x / 2, fov.y / 2)
        return DimBoundingBox(self._center - half, self._center + half)

    @property
    def empty(self) -> bool:
        """True if the site does not contain any pattern.

        Access:
            get
        """
        return not self._patterns

    @property
    def bounding_box(self) -> DimBoundingBox:
        """Bounding box of the contained patterns, relative to the center of the site.

        Access:
            get

        Raises:
            RuntimeError: Raised if the site is empty.
        """
        if not self._patterns:
            raise RuntimeError('Cannot calculate bounding box of empty site.')

        bbox = self._patterns[0].bounding_box

        for pattern in self._patterns[1:]:
            bbox = bbox.extended(pattern.bounding_box)

        return bbox

    @property
    def bounding_box_abs(self) -> DimBoundingBox:
        """Bounding box of the contained patterns in absolute coordinates (shifted by the center of the site).

        Access:
            get

        Raises:
            RuntimeError: Raised if the site is empty.
        """
        bbox = self.bounding_box
        return DimBoundingBox(bbox.lower_left + self._center, bbox.upper_right + self._center)

    @property
    def patterns(self) -> t.List[Pattern]:
        """The contained patterns (a copy of the list, positions are relative to the center of the site).

        Access:
            get
        """
        return list(self._patterns)

    @property
    def patterns_absolute(self) -> t.List[Pattern]:
        """The contained patterns, shifted by the center of the site.
        Hence, the positions of these patterns are **not** relative to the center of the site anymore but absolute to
        the coordinate origin. The patterns are copies.

        Access:
            get
        """
        return [pattern.translated(self._center) for pattern in self._patterns]

    def create_pattern(
        self,
        dim_shape: DimShape,
        mill: MillBase,
        raster_style: RasterStyle,
        description: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> Pattern:
        """Creates a pattern in-place (the returned pattern is automatically added to the site).
        The parameters are identical to the __init__ method of the :class:`~fibomat.layout.pattern.Pattern` class.

        Args:
            dim_shape (DimShape): shape with a length unit
            mill (MillBase): mill
            raster_style (RasterStyle): raster style
            description (str, optional): description
            **kwargs: additional arguments for the backends

        Returns:
            Pattern
        """
        pattern = Pattern(dim_shape, mill, raster_style, description=description, **kwargs)
        self.add_pattern(pattern)
        return pattern

    def add_pattern(self, ptn: t.Union[Pattern, arrangements.ArrangementBase]) -> None:
        """Adds a :class:`~fibomat.layout.pattern.Pattern` or an arrangement of patterns to the site.

        Args:
            ptn (Pattern, ArrangementBase): new pattern(s)

        Raises:
            TypeError: Raised if `ptn` is no pattern or no arrangement of patterns.
        """
        if isinstance(ptn, arrangements.ArrangementBase):
            extracted_patterns = list(ptn.arrangement_elements())
        else:
            extracted_patterns = [ptn]

        for extracted_pattern in extracted_patterns:
            if not isinstance(extracted_pattern, Pattern):
                raise TypeError(f'Only patterns can be added to a site, got {type(extracted_pattern).__name__}.')

        self._patterns.extend(extracted_patterns)

    def __iadd__(self, ptn: t.Union[Pattern, arrangements.ArrangementBase]) -> Site:
        """Adds a :class:`~fibomat.layout.pattern.Pattern` to the site. Identical to :meth:`add_pattern`.

        Args:
            ptn (Pattern, ArrangementBase): new pattern(s)

        Returns:
            Site
        """
        self.add_pattern(ptn)
        return self

    @property
    def center(self) -> DimVector:
        """Center of the site.

        Access:
            get
        """
        return self._center

    def _swap_fov(self) -> None:
        """Exchange width and height of an explicit fov."""
        if self._fov is not None:
            self._fov = DimVector(self._fov.y, self._fov.x)

    def _impl_translate(self, trans_vec: DimVectorLike) -> None:
        self._center = self._center + DimVector(trans_vec)

    def _impl_rotate(self, theta: float) -> None:
        quarter_turns = round(theta / (math.pi / 2.))
        if not np.isclose(theta, quarter_turns * math.pi / 2., rtol=0., atol=1e-9):
            raise ValueError('Sites can only be rotated by multiples of pi/2.')

        if quarter_turns % 2:
            self._swap_fov()

        self._center = self._center.rotated(theta)
        self._theta_vec = self._theta_vec.rotated(theta)

        for ptn in self._patterns:
            ptn._impl_rotate(theta)  # pylint: disable=protected-access

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)
        if self._fov is not None:
            self._fov = self._fov * abs(fac)
        self._center = self._center * fac

        for ptn in self._patterns:
            ptn._impl_scale(fac)  # pylint: disable=protected-access

    def _impl_mirror(self, mirror_axis: DimVectorLike) -> None:
        mirror_axis = DimVector(mirror_axis)

        eighth_turns = round(mirror_axis.vector.angle_about_x_axis / (math.pi / 4.))
        if not np.isclose(
            mirror_axis.vector.angle_about_x_axis, eighth_turns * math.pi / 4., rtol=0., atol=1e-9
        ):
            raise ValueError('Sites can only be mirrored on the axes or their diagonals.')

        # a mirror on a diagonal exchanges x and y
        if eighth_turns % 2:
            self._swap_fov()

        self._center = self._center.mirrored(mirror_axis)
        self._theta_vec = self._theta_vec.mirrored(mirror_axis.vector)

        for ptn in self._patterns:
            ptn._impl_mirror(mirror_axis)  # pylint: disable=protected-access
