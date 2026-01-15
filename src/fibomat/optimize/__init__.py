from typing import Tuple, Optional, Callable, Union

import numpy as np
# import open3d as o3d

import scipy.spatial as spatial
import scipy.interpolate as interpolate

from fibomat.shapes import ArcSpline
from fibomat.mill.ionbeam import GaussBeam, IonBeam
from fibomat.units import Q_, UnitType, U_, scale_factor, QuantityType, LengthQuantity, scale_to, LengthUnit
# from fibomat.sample import Sample
from fibomat.curve_tools import rasterize, smooth, deflate
from fibomat._dimensioned_object import DimObjLike, DimObj

try:
    import numba
except ModuleNotFoundError as error:
    raise RuntimeError(
        "You need to install 'numba' to be able to use the optimize package."
    ) from error


_FLUX_UNIT = U_('ions / nm**2 / µs')


def _create_adjacency_matrix(dwell_points: np.ndarray, length_unit: LengthUnit, beam: IonBeam):
    beam_cutoff = (beam.std * beam._beam_cutoff).to(length_unit).m

    spatial_tree = spatial.cKDTree(dwell_points[:, :2])

    # adjacent_point_indices = []
    #
    # for i in range(len(dwell_points)):
    #     adjacent_point_indices.append(spatial_tree.query_ball_point(dwell_points[i, :2], beam_cutoff))

    adjacent_point_indices = np.zeros(shape=(len(dwell_points), len(dwell_points)), dtype=float)

    for i in range(len(dwell_points)):
        indices = spatial_tree.query_ball_point(dwell_points[i, :2], beam_cutoff)
        adjacent_point_indices[i, indices] = 1.


    # f = plt.matshow(adjacent_point_indices, cmap='Greys')
    # plt.colorbar(f)
    # plt.show()

    return adjacent_point_indices


def _create_flux_matrix(
    dwell_points: np.ndarray,
    connectivity_matrix: np.ndarray,
    beam: IonBeam,
):
    global _FLUX_UNIT

    flux_matrix = np.empty((len(dwell_points), len(dwell_points)), dtype=float)

    for i in range(len(dwell_points)):
        connectivity = connectivity_matrix[i]
        flux, flux_unit = beam.flux_at(
            dwell_points[i, :2],
            dwell_points[:, :2],
            U_('µm')
        )

        scale = scale_factor(_FLUX_UNIT, flux_unit)

        flux_matrix[i] = connectivity * dwell_points[:, 2] * scale * flux
        # connectivity = connectivity_matrix[i]
        # dose[i] = np.sum(
        #         dwell_points[connectivity, 2] * beam(
        #             dwell_points[i, 0], dwell_points[i, 1],
        #             dwell_points[connectivity[i], 0], dwell_points[connectivity[i], 1]
        #         )
        # )
    return flux_matrix


# https://stackoverflow.com/a/43259123
# def split_above_threshold(signal, threshold, above_or_below):
#     if above_or_below:
#         mask = np.concatenate(([False], signal > threshold, [False]))
#     else:
#         mask = np.concatenate(([False], signal < threshold, [False]))
#     idx = np.flatnonzero(mask[1:] != mask[:-1])
#     return [(idx[i], idx[i+1]) for i in range(0, len(idx), 2)]


def optimize_rasterized(dwell_points, length_unit: LengthUnit, beam: IonBeam, nominal_flux: QuantityType, hint: Optional[Callable] = None):
    _MAX_ITER = 1000

    adj_matrix = _create_adjacency_matrix(dwell_points, length_unit, beam)
    inv_mean_connectivity = (1/np.sum(adj_matrix, axis=1))
    flux_matrix = _create_flux_matrix(dwell_points, adj_matrix, beam)

    initial_flux = flux_matrix @ np.ones(len(dwell_points))

    if hint:
        alpha = np.asarray(hint(np.linspace(0, 1, len(dwell_points))))
    else:
        alpha = np.full_like(initial_flux, 1.)

    max_initial_flux = np.max(initial_flux)

    alpha_max = 1.2

    # if not hint:
    # for threshold_factor in np.arange(.2, 1, 0.01):
    #     for _ in range(_MAX_ITER):
    #         # intervals = split_above_threshold(flux_matrix @ alpha, threshold_factor*max_initial_flux, False)  # .99 * nominal_flux.m
    #         # print(len(intervals))
    #
    #         idx = np.argwhere((flux_matrix @ alpha) < threshold_factor*max_initial_flux)
    #
    #         if len(idx) == 0:
    #             break
    #
    #         if np.allclose(alpha[idx], alpha_max):
    #             break
    #
    #         alpha[idx] += inv_mean_connectivity[idx]
    #
    #         alpha[alpha > alpha_max] = alpha_max

    nominal_flux = scale_to(_FLUX_UNIT, nominal_flux)

    for threshold_factor in reversed(np.arange(1, 2, 0.05)): # [1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.01, 1.]:
        # print(threshold_factor * nominal_flux.m)
        for _ in range(_MAX_ITER):

            idx = np.argwhere((flux_matrix @ alpha) > threshold_factor * nominal_flux)

            if len(idx) == 0:
                break

            if np.allclose(alpha[idx], 0.):
                break

            alpha[idx] -= inv_mean_connectivity[idx] #

            alpha[alpha < 0] = 0.

    dwell_points[:, 2] = alpha
    if len(dwell_points) > 1:
        hint = interpolate.interp1d(np.linspace(0, 1, len(dwell_points)), alpha)
    else:
        hint = None

    return dwell_points, hint, flux_matrix


@numba.jit(nopython=True, fastmath=True)
def _impl(flux_0, alpha, conn, threshold):
    last_std = np.std(flux_0 @ alpha)

    ref_step = 0
    step = 0.1

    while last_std > threshold:
        new_alpha = alpha.copy()
        flux_t = flux_0 @ alpha
        for i in range(len(alpha)):
            new_alpha[i] = alpha[i]  # - step * (14 - flux_t[i])
            new_alpha[i] -= step * np.sum(flux_t[i] - flux_t[conn[i]])

        new_alpha[new_alpha < 0.] = 0.
        new_alpha[new_alpha > 3.] = 3.
        std = np.std(flux_0 @ new_alpha)
        # print(std)

        if last_std < std:
            # raise RuntimeError('WARNING: last_std < std:')
            print('WARNING: last_std < std:')

            ref_step += 1
            step /= 10

            if ref_step > 4:
                print('WARNING: last_std < std. BREAKING NOW.')
                break

            continue

        last_std = std
        alpha = new_alpha

    return alpha, last_std


# https://www.labri.fr/perso/ejeannot/publications/europar06.pdf
# http://www.dartmouth.edu/~gvc/Cybenko_JPDP.pdf
def optimze_rasterized_second_try(dwell_points, length_unit: LengthUnit, beam: IonBeam, nominal_flux: QuantityType, hint: Optional[Callable] = None):

    _MAX_ITER = 1000

    threshold = 0.025

    adj_matrix = _create_adjacency_matrix(dwell_points, length_unit, beam)
    flux_matrix = _create_flux_matrix(dwell_points, adj_matrix, beam)

    # # algorithm will converge to mean value. hence, we shift the mean of initial flux to the nominal flux
    # # flux_matrix += nominal_flux.to(_FLUX_UNIT).m - np.mean(flux_matrix)
    # print(np.sum(flux_matrix, axis=1) - nominal_flux.to(_FLUX_UNIT).m)
    # print(np.sum(flux_matrix + (nominal_flux.to(_FLUX_UNIT).m - np.mean(flux_matrix)) / np.sum(adj_matrix) * adj_matrix, axis=1))
    #
    # print(np.mean(np.sum(flux_matrix, axis=1)))
    #

    flux_0 = flux_matrix
    if hint:
        alpha = hint(np.linspace(0, 1, len(flux_matrix)))
    else:
        alpha = np.ones(len(flux_matrix))

    # mean_flux = np.mean(flux_0 @ alpha)
    # alpha *= nominal_flux.to(_FLUX_UNIT).m / mean_flux

    conn = flux_matrix > 0
    # conn = adj_matrix

    # last_std = np.std(flux_0 @ alpha)
    #
    # ref_step = 0
    # step = 0.01
    #
    # while last_std > threshold:
    #     new_alpha = alpha.copy()
    #     flux_t = flux_0 @ alpha
    #     for i in range(len(alpha)):
    #         new_alpha[i] = alpha[i]
    #         new_alpha[i] -= step * np.sum(flux_t[i] - flux_t[conn[i]])
    #
    #     new_alpha[new_alpha < 0.] = 0.
    #     new_alpha[new_alpha > 1.5] = 1.5
    #     std = np.std(flux_0 @ new_alpha)
    #     print(std)
    #
    #     if last_std < std:
    #         # raise RuntimeError('WARNING: last_std < std:')
    #         print('WARNING: last_std < std:')
    #
    #         ref_step += 1
    #         step /= 10
    #
    #         if ref_step > 3:
    #             print('WARNING: last_std < std. BREAKING NOW.')
    #             break
    #
    #         continue
    #
    #     last_std = std
    #     alpha = new_alpha

    alpha, last_std = _impl(flux_0, alpha, conn, threshold)

    if last_std > threshold:
        # raise RuntimeError('last_std > threshold. Optimization did not converge.')
        print('WARNING: last_std > threshold. Optimization did not converge.')

    # TODO: NOT GOOD!
    alpha[alpha > 3.] = 1.5

    mean_flux = np.mean(flux_matrix @ alpha)
    print('mean_flux:', mean_flux)

    print('fac', mean_flux / nominal_flux.to(_FLUX_UNIT).m )

    # scale alpha so that flux mean is nominal_flux
    dwell_points[:, 2] = alpha * nominal_flux.to(_FLUX_UNIT).m / mean_flux
    if len(dwell_points) > 1:
        hint = interpolate.interp1d(np.linspace(0, 1, len(dwell_points)), alpha)
    else:
        hint = None

    return dwell_points, hint, flux_matrix


def optimze_rasterized_hybrid(dwell_points, length_unit: LengthUnit, beam: IonBeam, nominal_flux: QuantityType, hint: Optional[Callable] = None):
    _MAX_ITER = 1000

    adj_matrix = _create_adjacency_matrix(dwell_points, length_unit, beam)
    inv_mean_connectivity = (1/np.sum(adj_matrix, axis=1))

    flux_matrix = _create_flux_matrix(dwell_points, adj_matrix, beam)

    initial_flux = flux_matrix @ np.ones(len(dwell_points))

    if hint:
        alpha = np.asarray(hint(np.linspace(0, 1, len(dwell_points))))
    else:
        # alpha = np.full_like(initial_flux, 1.)
        alpha = np.array(dwell_points[:, 2])

    nominal_flux = scale_to(_FLUX_UNIT, nominal_flux)

    # first pass
    for threshold_factor in reversed(np.arange(1, 2, 0.05)): # [1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.01, 1.]:
        # print(threshold_factor)
        for _ in range(_MAX_ITER):

            idx = np.argwhere((flux_matrix @ alpha) > threshold_factor * nominal_flux)

            if len(idx) == 0:
                break

            if np.allclose(alpha[idx], 0.):
                break

            alpha[idx] -= inv_mean_connectivity[idx] #

            alpha[alpha < 0] = 0.

    # second pass
    threshold = 0.025

    alpha, last_std = _impl(flux_matrix, alpha, flux_matrix > 0, threshold)

    if last_std > threshold:
        # raise RuntimeError('last_std > threshold. Optimization did not converge.')
        print('WARNING: last_std > threshold. Optimization did not converge.')

    # TODO: NOT GOOD! WARN!
    alpha[alpha > 1.5] = 1.5

    # scale alpha so that flux mean is nominal_flux
    dwell_points[:, 2] = alpha * nominal_flux / np.mean(flux_matrix @ alpha)
    if len(dwell_points) > 1:
        hint = interpolate.interp1d(np.linspace(0, 1, len(dwell_points)), alpha)
    else:
        hint = None

    return dwell_points, hint, flux_matrix




def optimize_curve(
    dim_spline: DimObjLike[ArcSpline, LengthUnit],
    pitch: LengthQuantity,
    beam: IonBeam,
    nominal_flux: QuantityType,
    hint: Optional[Callable] = None,
    info: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Callable, np.ndarray]]:
    """Optimize the the dwell points an a curve so every point has an ion flux of `nominal_flux`.

    Args:
        dim_spline (DimObjLike[ArcSpline, LengthUnit]): curve with unit to be rasterized and optimized
        pitch (LengthQuantity): rasterization pitch
        beam (IonBeam): beam
        nominal_flux (QuantityType): nominal ion flux per spot
        hint (Callable, optional):
            a function f: [0, 1] -> R mapping rasterized points indizes (which are itself mappedto [0, 1] to dwell time
            multiplicands.
        info (bool): if True, dwell_points, hint and flux_matrix are returned.

    Returns:

    """

    dim_spline = DimObj.create(dim_spline)

    dwell_points = np.array(rasterize(dim_spline.obj, scale_to(dim_spline.unit, pitch)).dwell_points)

    dwell_points, hint, flux_matrix = optimze_rasterized_hybrid(dwell_points, dim_spline.unit, beam, nominal_flux, hint)
    # dwell_points, hint, flux_matrix = optimze_rasterized_second_try(dwell_points, dim_spline.unit, beam, nominal_flux, hint)

    if info:
        return dwell_points, hint, flux_matrix

    return dwell_points

# from typing import List, Tuple
#
# import numpy as np
# import cvxpy as cp
# from scipy import spatial
#
# from fibomat import units, rasterizing
#
# from fibomat.mill.gaussbeam import GaussBeam


# def _create_adjacency_matrix(dwell_points: np.ndarray, beam: GaussBeam):
#     beam_cutoff = (beam.sigma * beam.BEAM_CUTOFF).to('nm').m
#
#     spatial_tree = spatial.cKDTree(dwell_points[:, :2])
#
#     # adjacent_point_indices = []
#     #
#     # for i in range(len(dwell_points)):
#     #     adjacent_point_indices.append(spatial_tree.query_ball_point(dwell_points[i, :2], beam_cutoff))
#
#     adjacent_point_indices = np.zeros(shape=(len(dwell_points), len(dwell_points)), dtype=float)
#
#     for i in range(len(dwell_points)):
#         indices = spatial_tree.query_ball_point(dwell_points[i, :2], beam_cutoff)
#         adjacent_point_indices[i, indices] = 1.
#
#     import matplotlib.pyplot as plt
#     import matplotlib
#     matplotlib.use('Qt5Cairo')
#
#     print(adjacent_point_indices)
#     f = plt.matshow(adjacent_point_indices * 10000, cmap='Greys')
#     plt.colorbar(f)
#     plt.show()
#
#     return adjacent_point_indices
#
#
# def _create_dose_matrix(
#         dwell_points: np.ndarray,
#         connectivity_matrix: np.ndarray,
#         beam: GaussBeam,
#         flux_unit: units.UnitType
# ):
#     dose = np.empty((len(dwell_points), len(dwell_points)), dtype=float)
#
#     for i in range(len(dwell_points)):
#         connectivity = connectivity_matrix[i]
#         dose[i] = connectivity * dwell_points[:, 2] * beam.flux_at(
#             dwell_points[i, 0], dwell_points[i, 1],
#             dwell_points[:, 0], dwell_points[:, 1],
#             flux_unit
#         )
#         # connectivity = connectivity_matrix[i]
#         # dose[i] = np.sum(
#         #         dwell_points[connectivity, 2] * beam(
#         #             dwell_points[i, 0], dwell_points[i, 1],
#         #             dwell_points[connectivity[i], 0], dwell_points[connectivity[i], 1]
#         #         )
#         # )
#     return dose
#
#
# def optimize_curve_dwell_times(
#     dim_curves: List[Tuple[rasterizing.RasterizedPoints, units.LengthUnit]],
#     dwell_times_per_curve: List[units.TimeQuantity],
#     # curve_units: Tuple[units.UnitType, units.UnitType],
#     nominal_fluxes_per_curve: List[units.QuantityType],
#     beam: GaussBeam,
#     mode: str = 'min_dose_var',
#     plot_solution: bool = False
# ) -> List[rasterizing.RasterizedPoints]:
#     """
#     This function optimizes the dwell times of the given rasterized curves.
#
#     ...
#
#     .. note:: curves are modified!
#
#     Args:
#         dim_curves:
#         dwell_times_per_curve:
#         nominal_fluxes_per_curve:
#         beam:
#         mode:
#         plot_solution:
#
#     Returns:
#
#     """
#     if len(dim_curves) < 1:
#         raise RuntimeError
#     if len(dim_curves) != len(nominal_fluxes_per_curve):
#         raise RuntimeError
#     if len(dim_curves) != len(dwell_times_per_curve):
#         raise RuntimeError
#
#     flux_unit = units.U_('ions / µm**2 / µs')
#
#     # prepare curves
#     n_dwell_points = sum([len(curve.dwell_points) for curve, _ in dim_curves])
#     dwell_points = np.empty(shape=(n_dwell_points, 3), dtype=float)
#     nominal_dose_vector = np.empty(shape=(n_dwell_points,), dtype=float)
#
#     offset = 0
#     for (curve, curve_unit), dwell_time, nominal_flux in zip(dim_curves, dwell_times_per_curve, nominal_fluxes_per_curve):
#         # if curve_unit != units.U_('µm'):
#
#         point_range = slice(offset, offset+len(curve.dwell_points))
#
#         dwell_points[point_range, :2] = units.scale_factor(units.U_('µm'), curve_unit) * curve.dwell_points[:, :2]
#         dwell_points[point_range, 2] = units.scale_to(units.U_('µs'), dwell_time)
#
#         nominal_dose_vector[point_range] = nominal_flux.to(flux_unit).m
#
#         nominal_dose_vector *= dwell_points[:, 2]
#
#         offset += len(curve.dwell_points)
#
#     # precalculate adjacency matrix ("which other spots influence spot i")  and dose matrix ("which dose contributes
#     # spot j to spot j")
#     # dose_unit = units.U_('ions / µm**2 / µs')
#     adjacency_matrix = _create_adjacency_matrix(dwell_points, beam)
#     dose_matrix = _create_dose_matrix(dwell_points, adjacency_matrix, beam, flux_unit)
#
#     # return dose_matrix @ np.ones(len(dwell_points), dtype=float)
#
#     # # normalize doses
#     # dose_scaling = 1 / np.mean(nominal_dose_vector)
#     # dose_matrix *= dose_scaling
#     # nominal_dose_vector *= dose_scaling
#
#     # run optimization
#     n_points = len(dwell_points)
#     alpha = cp.Variable(n_points)
#     alpha.value = np.ones(len(dwell_points), dtype=float)
#
#     if mode == 'min_dose_var':
#         cost = cp.sum_squares(dose_matrix @ alpha - nominal_dose_vector) # + cp.sum_squares(alpha - 1.)
#         constrains = [alpha >= 0.]
#     elif mode == 'min_alpha_var':
#         # cost = cp.sum_squares((alpha - cp.sum(alpha) / n_points) / n_points)
#         # cost = cp.norm2((alpha - cp.sum(alpha) / n_points)) / n_points
#         ones = np.ones(len(dwell_points), dtype=float)
#         cost = cp.norm2(alpha - 1.)
#         constrains = [cp.abs(dose_matrix @ alpha - nominal_dose_vector) <= 0.01 * nominal_dose_vector, alpha >= 0.]
#     else:
#         raise RuntimeError(f'Unknown mode "{mode}"')
#
#     prob = cp.Problem(cp.Minimize(cost), constrains)
#
#     print(prob.is_dcp())
#
#     res = prob.solve(warm_start=True)
#     if prob.status != cp.OPTIMAL:
#         print(res)
#         print(prob.solver_stats.solver_name)
#         raise RuntimeError("Solver did not converge!")
#
#     # save optimized dwell times and restore original scaling
#     alpha_sol = alpha.value
#
#     # import scipy.optimize
#     #
#     # def f(alpha):
#     #     return dose_matrix @ alpha - nominal_dose_vector
#     #
#     #
#     # def f_jac(alpha):
#     #     return dose_matrix
#     #
#     # alpha_0 = np.ones(len(dwell_points), dtype=float)
#     #
#     # sol = scipy.optimize.least_squares(f, alpha_0, bounds=(0, np.inf), jac=f_jac)
#     #
#     # print(sol)
#     #
#     # alpha_sol = sol.x
#
#
#     i_dwell_point = 0
#     for i, (curve, _) in enumerate(dim_curves):
#         i_range = slice(i_dwell_point, i_dwell_point+len(curve.dwell_points))
#         # print(curve.dwell_points[:, 2], alpha_sol[i_range], '\n')
#         curve.dwell_points[:, 2] = alpha_sol[i_range]
#         i_dwell_point += len(curve.dwell_points)
#
#     if plot_solution:
#         import matplotlib.pyplot as plt
#         import matplotlib
#         matplotlib.use('Qt5Cairo')
#
#         # https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
#         ones = np.ones(n_points, dtype=float)
#
#         plt.matshow(adjacency_matrix * 10000, cmap='Greys')
#         print(np.where(adjacency_matrix > 0.))
#         plt.savefig('foo.png',dpi=1000)
#         plt.show()
#
#         doses_before_opt = dose_matrix @ ones
#         dose_diffs_before = dose_matrix @ ones - nominal_dose_vector
#         doses_after_opt = dose_matrix @ alpha.value
#
#         fig, axs = plt.subplots(3, 2, constrained_layout=True)
#
#         data_sources = [nominal_dose_vector, doses_before_opt, doses_after_opt] # dose_diffs_before,
#         vmin = min([np.min(source) for source in data_sources])
#         vmax = max([np.max(source) for source in data_sources])
#
#         # plot_nominal_doses = axs[0, 0].scatter(
#         #     dwell_points[:, 0], dwell_points[:, 1], c=nominal_dose_vector, vmin=vmin, vmax=vmax
#         # )
#         # fig.colorbar(plot_nominal_doses, ax=axs[0, 0])
#         # axs[0, 0].set_aspect('equal')
#
#         axs[0, 0].imshow(dose_matrix)
#
#         plot_doses_bef_opt = axs[0, 1].scatter(dwell_points[:, 0], dwell_points[:, 1], c=doses_before_opt, vmin=vmin, vmax=vmax)
#         axs[0, 1].set_xlabel('x / µm')
#         axs[0, 1].set_ylabel('y / µm')
#         cbar = fig.colorbar(plot_doses_bef_opt, ax=axs[0, 1])
#         cbar.ax.set_ylabel('dose multiplier')
#         axs[0, 1].set_aspect('equal')
#
#         plot_doses_after_opt = axs[1, 1].scatter(dwell_points[:, 0], dwell_points[:, 1], c=doses_after_opt, vmin=vmin, vmax=vmax)
#         cbar = fig.colorbar(plot_doses_after_opt, ax=axs[1, 1])
#         axs[1, 1].set_xlabel('x / µm')
#         axs[1, 1].set_ylabel('y / µm')
#         cbar.ax.set_ylabel('dose multiplier')
#         axs[1, 1].set_aspect('equal')
#
#         # adjacency_matrix.dtype = float
#         # adjacency_matrix[adjacency_matrix == 0] = None
#
#         axs[1, 0].hist(doses_after_opt, bins=100)
#
#         axs[2, 0].hist(dwell_points[:, 2] * alpha_sol, bins=100)
#
#         # data_sources = [nominal_dose_vector, doses_before_opt, dose_diffs_before, doses_after_opt]
#         # vmin = min([np.min(source) for source in data_sources])
#         # vmax = max([np.max(source) for source in data_sources])
#         #
#         #
#         # axs[1, 0].scatter(dwell_points[:, 0], dwell_points[:, 1], c=doses_after_opt, vmin=vmin, vmax=vmax)
#         # axs[1, 1].plot(np.arange(len(dwell_points)), nominal_dose_vector - doses_after_opt)
#         # fig.colorbar(plot_nominal_doses, ax=axs.flat)
#
#         plt.show()
#
#     return dim_curves
