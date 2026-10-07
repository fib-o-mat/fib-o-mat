"""Tests of `fibomat.mill.ionbeam`."""
import math

import numpy as np
import pytest

from fibomat.linalg import Vector
from fibomat.mill import GaussBeam, IonBeam
from fibomat.units import DimFloat, unit, ureg


def beam(fwhm=10., current=1., cutoff=10):
    return GaussBeam(fwhm * unit('nm'), current * unit('pA'), cutoff)


SIGMA_FACTOR = 2 * math.sqrt(2 * math.log(2))


class TestGaussBeam:
    def test_is_ion_beam(self):
        assert isinstance(beam(), IonBeam)

    def test_std(self):
        std = beam(fwhm=10.).std
        assert isinstance(std, DimFloat)
        assert std.m_as('nm') == pytest.approx(10. / SIGMA_FACTOR)

    def test_properties_and_repr(self):
        gauss = beam(cutoff=7)
        assert gauss.beam_cutoff == 7
        assert gauss.fwhm == 10. * unit('nm')
        assert gauss.current == 1. * unit('pA')
        assert 'beam_cutoff=7' in repr(gauss)

    def test_fwhm_in_other_unit(self):
        assert GaussBeam(0.01 * unit('µm'), 1. * unit('pA')).std.m_as('nm') == pytest.approx(10. / SIGMA_FACTOR)

    def test_flux_at_maximum(self):
        sigma = 0.01 / SIGMA_FACTOR  # µm
        flux = beam().flux_at((0, 0), (0, 0), unit('µm'))
        assert isinstance(flux, DimFloat)
        assert flux.m_as('pA / µm**2') == pytest.approx(1. / (2 * math.pi * sigma ** 2))

    def test_flux_at_fwhm_distance_is_half(self):
        maximum = beam().flux_at((0, 0), (0, 0), unit('µm'))
        half = beam().flux_at((5., 0), (0, 0), unit('nm'))
        # at a distance of fwhm / 2 the flux is half of the maximum
        assert half.m_as('pA / µm**2') / maximum.m_as('pA / µm**2') == pytest.approx(0.5)

    def test_position_unit(self):
        in_um = beam().flux_at((0.005, 0), (0, 0), unit('µm'))
        in_nm = beam().flux_at((5., 0), (0, 0), unit('nm'))
        assert in_um.m_as('pA / µm**2') == pytest.approx(in_nm.m_as('pA / µm**2'))

    def test_several_maxima(self):
        fluxes, flux_unit = beam().flux_at((0, 0), [(0, 0), (5., 0), (0, 5.), (100., 0)], unit('nm'))
        assert fluxes.shape == (4,)
        assert flux_unit == ureg.Unit('pA / µm**2')
        assert fluxes[1] == pytest.approx(0.5 * fluxes[0])
        assert fluxes[2] == pytest.approx(fluxes[1])
        assert fluxes[3] < 1e-6 * fluxes[0]

    def test_vectors_and_arrays_are_accepted(self):
        reference = beam().flux_at((1., 2.), (3., 4.), unit('nm')).m_as('pA / µm**2')
        assert beam().flux_at(Vector(1., 2.), Vector(3., 4.), unit('nm')).m_as('pA / µm**2') == pytest.approx(reference)
        fluxes, _ = beam().flux_at(np.array([1., 2.]), np.array([[3., 4.]]), unit('nm'))
        assert fluxes[0] == pytest.approx(reference)

    def test_current_scales_flux(self):
        assert beam(current=2.).flux_at((0, 0), (0, 0), unit('nm')).m_as('pA / µm**2') == pytest.approx(
            2 * beam(current=1.).flux_at((0, 0), (0, 0), unit('nm')).m_as('pA / µm**2')
        )

    def test_total_current_is_the_integral(self):
        gauss = beam(fwhm=100.)
        grid = np.linspace(-0.5, 0.5, 401)  # µm
        step = grid[1] - grid[0]
        points = np.dstack(np.meshgrid(grid, grid)).reshape(-1, 2)
        fluxes, flux_unit = gauss.flux_at((0, 0), points, unit('µm'))
        assert np.sum(fluxes) * step ** 2 == pytest.approx(1., rel=1e-3)

    def test_nominal_flux_per_spot(self):
        gauss = beam()
        assert gauss.nominal_flux_per_spot().m_as('pA / µm**2') == pytest.approx(
            gauss.flux_at((0, 0), (0, 0), unit('µm')).m_as('pA / µm**2')
        )

    def test_nominal_flux_on_line_far_apart_equals_single_spot(self):
        gauss = beam(fwhm=10.)
        assert gauss.nominal_flux_per_spot_on_line(1. * unit('µm')).m_as('pA / µm**2') == pytest.approx(
            gauss.nominal_flux_per_spot().m_as('pA / µm**2')
        )

    def test_nominal_flux_on_line_is_the_sum_of_the_neighbours(self):
        gauss = beam(fwhm=10.)
        pitch = 2. * unit('nm')
        flux = gauss.nominal_flux_per_spot_on_line(pitch).m_as('pA / µm**2')
        positions = pitch.m_as('nm') * np.arange(-100, 101)
        fluxes, _ = gauss.flux_at((0, 0), [(x, 0) for x in positions], unit('nm'))
        assert flux == pytest.approx(np.sum(fluxes), rel=1e-6)
        # sum of a dense line of spots: current / (pitch * sqrt(2 pi) * sigma) / ... is much larger than a single spot
        assert flux > gauss.nominal_flux_per_spot().m_as('pA / µm**2')

    def test_nominal_flux_in_rect(self):
        gauss = beam(fwhm=10.)
        single = gauss.nominal_flux_per_spot().m_as('pA / µm**2')
        assert gauss.nominal_flux_per_spot_in_rect(1. * unit('µm'), 1. * unit('µm')).m_as('pA / µm**2') == \
            pytest.approx(single)
        dense = gauss.nominal_flux_per_spot_in_rect(2. * unit('nm'), 2. * unit('nm')).m_as('pA / µm**2')
        # dense grid: sum ~ current / pitch^2 (= 1 pA / 4 nm^2)
        assert dense == pytest.approx(1. / 4e-6, rel=1e-3)

    def test_invalid_fwhm_and_current(self):
        with pytest.raises(TypeError):
            GaussBeam(10., 1. * unit('pA'))
        with pytest.raises(ValueError, match='length'):
            GaussBeam(10. * unit('s'), 1. * unit('pA'))
        with pytest.raises(ValueError):
            GaussBeam(0. * unit('nm'), 1. * unit('pA'))
        with pytest.raises(TypeError):
            GaussBeam(10. * unit('nm'), 1.)
        with pytest.raises(ValueError, match='current'):
            GaussBeam(10. * unit('nm'), 1. * unit('s'))
        with pytest.raises(ValueError):
            GaussBeam(10. * unit('nm'), -1. * unit('pA'))

    def test_invalid_cutoff(self):
        with pytest.raises(ValueError):
            beam(cutoff=0)
        with pytest.raises(TypeError):
            beam(cutoff=2.5)
        with pytest.raises(TypeError):
            beam(cutoff=True)

    def test_invalid_positions(self):
        gauss = beam()
        with pytest.raises(ValueError):
            gauss.flux_at((0, 0, 0), (0, 0), unit('nm'))
        with pytest.raises(ValueError):
            gauss.flux_at([(0, 0), (1, 1)], (0, 0), unit('nm'))
        with pytest.raises(ValueError):
            gauss.flux_at((0, 0), (math.nan, 0), unit('nm'))
        with pytest.raises(ValueError):
            gauss.flux_at((0, 0), 'foo', unit('nm'))
        with pytest.raises(ValueError, match='position_unit'):
            gauss.flux_at((0, 0), (0, 0), unit('s'))

    def test_invalid_pitch(self):
        gauss = beam()
        with pytest.raises(TypeError):
            gauss.nominal_flux_per_spot_on_line(1.)
        with pytest.raises(ValueError):
            gauss.nominal_flux_per_spot_on_line(0. * unit('nm'))
        with pytest.raises(ValueError):
            gauss.nominal_flux_per_spot_in_rect(1. * unit('nm'), -1. * unit('nm'))

    def test_ion_beam_is_abstract(self):
        with pytest.raises(TypeError):
            IonBeam(10)
