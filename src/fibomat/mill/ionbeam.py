"""Provide the abstract class :class:`IonBeam` and an implementation :class:`GaussBeam`.

Example:
    >>> from fibomat.units import unit
    >>> beam = GaussBeam(fwhm=10. * unit('nm'), current=1. * unit('pA'))
    >>> round(beam.std.m_as('nm'), 3)
    4.247
    >>> flux = beam.flux_at((0, 0), (0, 0), unit('µm'))  # flux of a single spot in its maximum
    >>> flux.units
    <Unit('picoampere / micrometer ** 2')>
"""
from __future__ import annotations

import abc
import math
import operator
import typing as t

import numpy as np
import pint  # type: ignore

from fibomat.linalg import VectorLike
from fibomat.units import DimFloat, LengthDimension, UnitType, scale_factor, ureg, has_length_dim, unit


__all__ = ['IonBeam', 'GaussBeam']


_CURRENT_DIMENSIONALITY = ureg.get_dimensionality('[current]')
_FLUX_UNIT = ureg.Unit('pA / µm**2')
"""Unit of the flux returned by :class:`GaussBeam`."""


def _check_beam_cutoff(beam_cutoff: int) -> int:
    """Check that the beam cutoff is an integer which is not smaller than 1.

    Args:
        beam_cutoff (int): cutoff

    Returns:
        int: cutoff

    Raises:
        TypeError: Raised if beam_cutoff is not an integer.
        ValueError: Raised if beam_cutoff < 1.
    """
    if isinstance(beam_cutoff, bool):
        raise TypeError('beam_cutoff must be an int.')
    try:
        beam_cutoff = operator.index(beam_cutoff)
    except TypeError:
        raise TypeError(f'beam_cutoff must be an int, got {type(beam_cutoff).__name__}.') from None

    if beam_cutoff < 1:
        raise ValueError('beam_cutoff must not be smaller than 1.')

    return beam_cutoff


def _positions(points: t.Any, name: str) -> t.Tuple[np.ndarray, bool]:
    """Convert a point or a sequence of points to an array of shape (n, 2).

    Args:
        points (VectorLike, Sequence[VectorLike]): point or points
        name (str): name used in error messages

    Returns:
        Tuple[np.ndarray, bool]: array of points and True if `points` was a sequence of points.

    Raises:
        ValueError: Raised if `points` is neither a point nor a sequence of points or is not finite.
    """
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a point or a sequence of points.') from error

    if array.ndim == 1 and array.shape == (2,):
        result, is_sequence = array[np.newaxis, :], False
    elif array.ndim == 2 and array.shape[1] == 2:
        result, is_sequence = array, True
    else:
        raise ValueError(f'{name} must be a point or a sequence of points.')

    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite.')

    return result, is_sequence


class IonBeam(abc.ABC):
    """Base class to model an ion beam."""

    def __init__(self, beam_cutoff: int):
        """
        Args:
            beam_cutoff (int): beam_cutoff * std is used as cut-off length for flux calculations.

        Raises:
            TypeError: Raised if beam_cutoff is not an integer.
            ValueError: Raised if beam_cutoff < 1.
        """
        self._beam_cutoff = _check_beam_cutoff(beam_cutoff)

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @property
    def beam_cutoff(self) -> int:
        """Cut-off of the beam in units of the standard deviation.

        Access:
            get
        """
        return self._beam_cutoff

    @property
    @abc.abstractmethod
    def std(self) -> DimFloat[LengthDimension]:
        """Standard deviation of the ion beam distribution.

        Access:
            get
        """

    @abc.abstractmethod
    def flux_at(
        self,
        position: VectorLike,
        beam_maxima: t.Union[t.Sequence[VectorLike], VectorLike],
        position_unit: UnitType
    ) -> t.Union[t.Tuple[np.ndarray, UnitType], DimFloat[t.Any]]:
        """Calculate the ion flux at `position` if the ion beam maximum is at `beam_maxima`.

        Args:
            position (VectorLike): position at which the flux is calculated.
            beam_maxima (VectorLike, Sequence[VectorLike]): position of the beam maximum/maxima
            position_unit (UnitType): length unit of the positions and the beam maxima

        Returns:
            Union[DimFloat, Tuple[np.ndarray, UnitType]]:
                A flux is returned if `beam_maxima` is a single point. If it is a sequence of points, the fluxes
                (one for each maximum) and their unit are returned.

        Raises:
            ValueError: Raised if the positions are invalid.
        """

    def nominal_flux_per_spot(self) -> DimFloat[t.Any]:
        """Calculate the ion flux of a single, isolated spot.

        Returns:
            DimFloat: flux
        """
        flux = self.flux_at((0, 0), (0, 0), unit('µm'))
        assert isinstance(flux, DimFloat)
        return flux

    def nominal_flux_per_spot_on_line(self, pitch: DimFloat[LengthDimension]) -> DimFloat[t.Any]:
        """Calculate the ion flux of a spot on a line of spots with pitch `pitch`.

        All spots inside of the cut-off length are considered.

        Args:
            pitch (DimFloat[LengthDimension]): pitch on the line

        Returns:
            DimFloat: flux

        Raises:
            ValueError: Raised if pitch is not a positive length.
        """
        pitch_um = _positive_length(pitch, 'pitch')

        n_considered_spots = int(self.std.m_as('µm') * self._beam_cutoff / pitch_um)

        points = np.c_[
            pitch_um * np.arange(-n_considered_spots, n_considered_spots + 1),
            np.zeros(2 * n_considered_spots + 1)
        ]

        fluxes, flux_unit = self._fluxes_at_spots((0, 0), points, unit('µm'))

        return DimFloat(float(np.sum(fluxes)) * flux_unit)

    def nominal_flux_per_spot_in_rect(
        self, pitch_x: DimFloat[LengthDimension], pitch_y: DimFloat[LengthDimension]
    ) -> DimFloat[t.Any]:
        """Calculate the ion flux of a spot on a rectangular grid of spots with the pitches `pitch_x` and `pitch_y`.

        All spots inside of the cut-off length are considered.

        Args:
            pitch_x (DimFloat[LengthDimension]): pitch in x direction
            pitch_y (DimFloat[LengthDimension]): pitch in y direction

        Returns:
            DimFloat: flux

        Raises:
            ValueError: Raised if a pitch is not a positive length.
        """
        pitch_x_um = _positive_length(pitch_x, 'pitch_x')
        pitch_y_um = _positive_length(pitch_y, 'pitch_y')

        cutoff_um = self.std.m_as('µm') * self._beam_cutoff

        n_considered_spots_x = int(cutoff_um / pitch_x_um) + 1
        n_considered_spots_y = int(cutoff_um / pitch_y_um) + 1

        x_grid = np.arange(-n_considered_spots_x, n_considered_spots_x + 1) * pitch_x_um
        y_grid = np.arange(-n_considered_spots_y, n_considered_spots_y + 1) * pitch_y_um

        points = np.dstack(np.meshgrid(x_grid, y_grid)).reshape(-1, 2)
        points = points[points[:, 0] ** 2 + points[:, 1] ** 2 <= cutoff_um ** 2]

        fluxes, flux_unit = self._fluxes_at_spots((0, 0), points, unit('µm'))

        return DimFloat(float(np.sum(fluxes)) * flux_unit)

    def _fluxes_at_spots(
        self, position: VectorLike, beam_maxima: np.ndarray, position_unit: UnitType
    ) -> t.Tuple[np.ndarray, UnitType]:
        """Fluxes at `position` for several beam maxima (always an array, even for a single maximum)."""
        fluxes = self.flux_at(position, beam_maxima, position_unit)
        assert isinstance(fluxes, tuple)
        return fluxes


def _positive_length(length: DimFloat[LengthDimension], name: str) -> float:
    """Return a length in µm.

    Args:
        length (DimFloat[LengthDimension]): length
        name (str): name used in error messages

    Returns:
        float: magnitude in µm

    Raises:
        TypeError: Raised if length is not a dimensioned value.
        ValueError: Raised if length is not a positive and finite length.
    """
    if not isinstance(length, DimFloat):
        raise TypeError(f'{name} must be a dimensioned value like 2. * unit("nm"), got {type(length).__name__}.')
    if not has_length_dim(length):
        raise ValueError(f'{name} must have dimension [length].')

    value = length.m_as('µm')
    if not math.isfinite(value) or value <= 0.:
        raise ValueError(f'{name} must be positive and finite, got {length!r}.')
    return value


class GaussBeam(IonBeam):
    """Gauss function model of an ion beam."""

    def __init__(self, fwhm: DimFloat[LengthDimension], current: DimFloat[t.Any], beam_cutoff: int = 10):
        """
        Args:
            fwhm (DimFloat[LengthDimension]): full width at half maximum of the beam profile.
            current (DimFloat): total beam current, e.g. ``1. * unit('pA')``.
            beam_cutoff (int, optional): beam cut-off, defaults to 10.

        Raises:
            TypeError: Raised if fwhm or current is not a dimensioned value or beam_cutoff is not an integer.
            ValueError: Raised if fwhm is not a positive length, current is not a positive current or
                beam_cutoff < 1.
        """
        super().__init__(beam_cutoff)

        fwhm_um = _positive_length(fwhm, 'fwhm')

        if not isinstance(current, DimFloat):
            raise TypeError(f'current must be a dimensioned value like 1. * unit("pA"), got {type(current).__name__}.')
        if current.quantity.dimensionality != _CURRENT_DIMENSIONALITY:
            raise ValueError('current must have dimension [current].')
        if not math.isfinite(current.magnitude) or current.magnitude <= 0.:
            raise ValueError(f'current must be positive and finite, got {current!r}.')

        self._fwhm = fwhm
        self._current = current

        self._std_um = fwhm_um / (2. * math.sqrt(2. * math.log(2.)))

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}'
            f'(fwhm={self._fwhm!r}, current={self._current!r}, beam_cutoff={self._beam_cutoff})'
        )

    @property
    def fwhm(self) -> DimFloat[LengthDimension]:
        """Full width at half maximum of the beam profile.

        Access:
            get
        """
        return self._fwhm

    @property
    def current(self) -> DimFloat[t.Any]:
        """Total beam current.

        Access:
            get
        """
        return self._current

    @property
    def std(self) -> DimFloat[LengthDimension]:
        return DimFloat(self._std_um * ureg.Unit('µm'))

    def flux_at(  # type: ignore[override]
        self,
        position: VectorLike,
        beam_maxima: t.Union[t.Sequence[VectorLike], VectorLike],
        position_unit: UnitType
    ) -> t.Union[t.Tuple[np.ndarray, pint.Unit], DimFloat[t.Any]]:
        """Calculate the ion flux at `position` if the ion beam maximum is at `beam_maxima`.

        The flux of a beam with the current `I` and the standard deviation `sigma` is
        ``I / (2 pi sigma^2) * exp(-r^2 / (2 sigma^2))`` where `r` is the distance to the beam maximum.

        Args:
            position (VectorLike): position at which the flux is calculated.
            beam_maxima (VectorLike, Sequence[VectorLike]): position of the beam maximum/maxima
            position_unit (UnitType): length unit of the positions and the beam maxima

        Returns:
            Union[DimFloat, Tuple[np.ndarray, UnitType]]:
                A flux (in pA / µm^2) is returned if `beam_maxima` is a single point. If it is a sequence of points,
                the fluxes (one for each maximum) and their unit (pA / µm^2) are returned.

        Raises:
            ValueError: Raised if the position or the beam maxima are no points or not finite, or position_unit is
                no length unit.
        """
        if not has_length_dim(position_unit):
            raise ValueError('position_unit must have dimension [length].')

        position_array, position_is_sequence = _positions(position, 'position')
        if position_is_sequence:
            raise ValueError('position must be a single point.')
        maxima, maxima_is_sequence = _positions(beam_maxima, 'beam_maxima')

        to_um = scale_factor(unit('µm'), position_unit)

        distance_squared = np.sum((position_array[0] * to_um - maxima * to_um) ** 2, axis=1)

        amplitude = self._current.m_as('pA') / (2. * math.pi * self._std_um ** 2)
        fluxes = amplitude * np.exp(-distance_squared / (2. * self._std_um ** 2))

        if maxima_is_sequence:
            return fluxes, _FLUX_UNIT

        return DimFloat(float(fluxes[0]) * _FLUX_UNIT)
