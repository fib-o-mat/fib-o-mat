"""Approximation of smooth planar curves by arc splines.

The algorithm follows D.S. Meek and D.J. Walton, *Approximating smooth planar curves by arc splines*, Journal of
Computational and Applied Mathematics 59 (1995) 221-231:

1. The curve is split into *spirals*, i.e. pieces without curvature extrema and inflection points (and without points
   with vanishing velocity). Within a spiral the absolute curvature is strictly monotone.
2. Each spiral is bisected (in its parameter) until, for the segment from `A` to `B`, the *enclosing condition* holds:
   the curvature at `A` is smaller (larger) than the curvature of the bounding arc through `A` and `B` with the tangent
   of `A` and the curvature at `B` is larger (smaller) than the one of the bounding arc with the tangent of `B` (for a
   spiral with increasing (decreasing) curvature). Then the curve lies in the crescent between the two bounding arcs
   and so does every biarc which interpolates `A`, `B` and the tangents (Theorems 2 and 5).
3. If the bounding arcs are closer than `epsilon` (the maximal distance of the arcs is ``d / 2 * |tan(alpha / 2) -
   tan(beta / 2)|``, Eq. (8)), a biarc is returned for the segment. Hence, the distance between the curve and the
   returned arc spline is at most `epsilon` (for curves which are spirals between the detected special points).

All steps are vectorized over all segments of a bisection level, so no python loop runs over single segments.
"""
from __future__ import annotations

import typing as t

import numpy as np
import scipy.optimize as optimize


__all__ = ['approximate_vertices']


_ANGLE_CAP = 1.2
"""Maximal angle (rad) between the chord and the tangents. Smaller than pi / 2, so that arcs sweep at most a half
circle and the biarc is a graph over the chord."""

_N_BISECTIONS = 60
"""Number of bisection steps to refine special points."""


class CurveLike(t.Protocol):  # pragma: no cover
    """Interface of the curve which is approximated (all methods are vectorized over `t`, points have shape (n, 2))."""

    @property
    def domain(self) -> t.Tuple[float, float]: ...
    def f(self, t: np.ndarray) -> np.ndarray: ...
    def df(self, t: np.ndarray) -> np.ndarray: ...
    def curvature(self, t: np.ndarray) -> np.ndarray: ...
    def d_curvature(self, t: np.ndarray) -> np.ndarray: ...


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _norm(a: np.ndarray) -> np.ndarray:
    return np.hypot(a[..., 0], a[..., 1])


class _Evaluator:
    """Evaluates points, unit tangents and curvatures. At parameters with vanishing velocity the tangent and the
    curvature are taken from a nearby parameter (inside of the domain), as limits from the corresponding side."""

    def __init__(self, curve: CurveLike, ref_speed: float):
        self.curve = curve
        self.domain = curve.domain
        self.span = self.domain[1] - self.domain[0]
        self.tiny_speed = 1e-9 * ref_speed

    def points(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(self.curve.f(params), dtype=float)

    def tangents_and_curvatures(self, params: np.ndarray, side: int) -> t.Tuple[np.ndarray, np.ndarray]:
        """Unit tangents and curvatures. `side` is +1 if the limit from the right and -1 if the limit from the left
        is taken at parameters with vanishing velocity."""
        velocity = np.asarray(self.curve.df(params), dtype=float)
        speed = _norm(velocity)
        curvature = np.asarray(self.curve.curvature(params), dtype=float)

        singular = ~(speed > self.tiny_speed)
        if np.any(singular):
            shifted_params = params.copy()
            for shift in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
                todo = singular & ~(_norm(velocity) > self.tiny_speed)
                if not np.any(todo):
                    break
                shifted_params[todo] = np.clip(
                    params[todo] + side * shift * self.span, self.domain[0], self.domain[1]
                )
                velocity[todo] = self.curve.df(shifted_params[todo])
                curvature[todo] = self.curve.curvature(shifted_params[todo])
            speed = _norm(velocity)
            if np.any(~(speed > 0.)):
                raise ValueError('The velocity of the curve vanishes on an interval.')

        return velocity / speed[..., np.newaxis], curvature


def _bisect_sign_change(
    func: t.Callable[[np.ndarray], np.ndarray], low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Roots of `func` in the intervals [low, high] where `func` changes its sign (vectorized bisection)."""
    low = low.copy()
    high = high.copy()
    sign_low = np.sign(func(low))
    for _ in range(_N_BISECTIONS):
        middle = (low + high) / 2
        same_as_low = np.sign(func(middle)) == sign_low
        low = np.where(same_as_low, middle, low)
        high = np.where(same_as_low, high, middle)
    return (low + high) / 2


def _sign_change_roots(
    values: np.ndarray, grid: np.ndarray, threshold: float, func: t.Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Roots of `func` between grid points where the (thresholded) sign of `values` changes."""
    signs = np.where(np.abs(values) > threshold, np.sign(values), 0.)
    nonzero = np.nonzero(signs)[0]
    if len(nonzero) < 2:
        return np.empty(0)

    changed = signs[nonzero[:-1]] != signs[nonzero[1:]]
    if not np.any(changed):
        return np.empty(0)

    return _bisect_sign_change(func, grid[nonzero[:-1][changed]], grid[nonzero[1:][changed]])


def _vanishing_velocity_params(
    curve: CurveLike, grid: np.ndarray, speed: np.ndarray, tiny_speed: float
) -> np.ndarray:
    """Parameters at which the velocity vanishes (cusps or points with zero speed inside of the domain)."""
    found = []
    inner = np.nonzero((speed[1:-1] <= speed[:-2]) & (speed[1:-1] <= speed[2:]) & (speed[1:-1] < 1e-3 * np.max(speed)))[0] + 1
    for i in inner:
        res = optimize.minimize_scalar(
            lambda param: float(_norm(np.asarray(curve.df(np.asarray(param, dtype=float))))),
            bounds=(grid[i - 1], grid[i + 1]), method='bounded', options={'xatol': 1e-14 * max(1., abs(grid[i]))},
        )
        if res.fun <= tiny_speed * 1e3:
            found.append(res.x)
    return np.array(found, dtype=float)


def _special_points(
    curve: CurveLike, n_samples: int
) -> t.Tuple[np.ndarray, float]:
    """Breakpoints of the curve (domain borders, inflections, curvature extrema, points with vanishing velocity).

    Returns:
        the sorted breakpoints and a reference speed (maximal speed on the grid)
    """
    start, end = curve.domain
    grid = np.linspace(start, end, n_samples)

    speed = _norm(np.asarray(curve.df(grid), dtype=float))
    ref_speed = float(np.max(speed))
    if not ref_speed > 0.:
        raise ValueError('The velocity of the curve vanishes everywhere.')
    tiny_speed = 1e-9 * ref_speed

    # inside of the domain, the velocity must not vanish for the curvature to be defined; the curvature at the borders
    # is evaluated with the one-sided limits
    evaluator = _Evaluator(curve, ref_speed)
    _, k_first = evaluator.tangents_and_curvatures(grid[:1], 1)
    _, k_last = evaluator.tangents_and_curvatures(grid[-1:], -1)

    with np.errstate(all='ignore'):
        curvature = np.asarray(curve.curvature(grid), dtype=float)
        curvature[0], curvature[-1] = k_first[0], k_last[0]
        d_curvature = np.asarray(curve.d_curvature(grid), dtype=float)

    # points with a vanishing velocity are singular (curvature is infinite or undefined); they are handled separately
    singular_params = _vanishing_velocity_params(curve, grid, speed, tiny_speed)
    good = np.isfinite(curvature) & np.isfinite(d_curvature)

    k_scale = float(np.max(np.abs(curvature[good]))) if np.any(good) else 0.
    dk_scale = float(np.max(np.abs(d_curvature[good]))) if np.any(good) else 0.

    def safe(func: t.Callable[[np.ndarray], np.ndarray]) -> t.Callable[[np.ndarray], np.ndarray]:
        def wrapped(params: np.ndarray) -> np.ndarray:
            with np.errstate(all='ignore'):
                return np.nan_to_num(np.asarray(func(params), dtype=float), nan=0., posinf=0., neginf=0.)
        return wrapped

    # inflections: sign changes of the curvature; curvature extrema: sign changes of its derivative.
    inflections = _sign_change_roots(
        np.where(good, curvature, 0.), grid, 1e-9 * k_scale, safe(curve.curvature)
    )
    extrema = _sign_change_roots(
        np.where(good, d_curvature, 0.), grid, 1e-9 * dk_scale, safe(curve.d_curvature)
    )

    points = np.concatenate(([start, end], inflections, extrema, singular_params))
    points = np.sort(points[(points >= start) & (points <= end)])

    # merge points which are (numerically) identical
    keep = np.concatenate(([True], np.diff(points) > 1e-12 * (end - start)))
    return points[keep], ref_speed


def _classify_pieces(
    curve: CurveLike, evaluator: _Evaluator, breakpoints: np.ndarray
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Sign `s` of the curvature and the monotonicity `m` (+1: |k| increasing, -1: decreasing, 0: constant) of the
    spirals between the breakpoints."""
    middle = (breakpoints[:-1] + breakpoints[1:]) / 2
    with np.errstate(all='ignore'):
        k_mid = np.nan_to_num(np.asarray(curve.curvature(middle), dtype=float))
        dk_mid = np.nan_to_num(np.asarray(curve.d_curvature(middle), dtype=float))

    k_scale = max(float(np.max(np.abs(k_mid))), 1e-300)
    sign = np.where(np.abs(k_mid) > 1e-9 * k_scale, np.sign(k_mid), 0.)
    # the sign of the curvature may be zero in one piece only if the curve is a line there
    growth = np.sign(dk_mid) * np.where(sign == 0., 1., sign)
    growth = np.where(np.abs(dk_mid) > 1e-9 * k_scale / max(evaluator.span, 1e-300), growth, 0.)
    return sign, growth


def _signed_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.arctan2(_cross(first, second), _dot(first, second))


def _biarc_joints_and_bulges(
    p_a: np.ndarray, p_b: np.ndarray, t_a: np.ndarray, t_b: np.ndarray
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joints and bulges of the biarcs interpolating the points and unit tangents (vectorized; see
    :class:`fibomat.shapes.biarc.Biarc` for the formulas)."""
    chord = p_b - p_a
    chord_sq = _dot(chord, chord)
    t_sum = t_a + t_b
    v_dot_t = _dot(chord, t_sum)
    den = 2. * (1. - _dot(t_a, t_b))
    parallel = den <= 1e-14

    safe_den = np.where(parallel, 1., den)
    root = np.sqrt(np.maximum(v_dot_t ** 2 + den * chord_sq, 0.))
    with np.errstate(all='ignore'):
        dist = np.where(v_dot_t >= 0., chord_sq / (v_dot_t + root), (root - v_dot_t) / safe_den)
    dist = np.where(parallel, 0., dist)
    joint = np.where(parallel[:, None], (p_a + p_b) / 2, (p_a + p_b + dist[:, None] * (t_a - t_b)) / 2)

    # sweep of an arc = 2 * angle between tangent and chord, bulge = tan(sweep / 4) = tan(angle / 2)
    bulge_1 = np.tan(_signed_angle(t_a, joint - p_a) / 2)
    bulge_2 = np.tan(_signed_angle(p_b - joint, t_b) / 2)
    return joint, bulge_1, bulge_2


def approximate_vertices(
    curve: CurveLike,
    epsilon: float,
    n_samples: int = 2000,
    max_depth: int = 48,
    max_segments: int = 5_000_000,
) -> t.Tuple[np.ndarray, bool]:
    """Approximate a curve by an arc spline with a maximal distance of `epsilon` to the curve.

    Args:
        curve: curve to be approximated (see :class:`CurveLike`)
        epsilon (float): maximal distance between the curve and the arc spline
        n_samples (int): number of samples to detect curvature extrema and inflections
        max_depth (int): maximal number of bisections of a segment
        max_segments (int): maximal number of segments (protection against exhausting the memory)

    Returns:
        Tuple[np.ndarray, bool]: vertices (n, 3) of the arc spline (x, y, bulge) and a flag if the arc spline is closed

    Raises:
        ValueError: Raised if `epsilon` is not positive or the curve is degenerated.
        RuntimeError: Raised if more than `max_segments` segments are necessary.
    """
    if not np.isfinite(epsilon) or epsilon <= 0.:
        raise ValueError('epsilon must be positive and finite.')
    if n_samples < 8:
        raise ValueError('n_samples must be at least 8.')

    start, end = curve.domain
    breakpoints, ref_speed = _special_points(curve, n_samples)
    evaluator = _Evaluator(curve, ref_speed)

    sign, growth = _classify_pieces(curve, evaluator, breakpoints)

    # intervals which are still to be processed
    t_0 = breakpoints[:-1].copy()
    t_1 = breakpoints[1:].copy()
    spiral_sign = sign.copy()
    spiral_growth = growth.copy()

    results: t.List[t.Tuple[np.ndarray, ...]] = []
    n_results = 0

    for depth in range(max_depth + 1):
        if len(t_0) == 0:
            break

        p_a = evaluator.points(t_0)
        p_b = evaluator.points(t_1)
        tangent_a, k_a = evaluator.tangents_and_curvatures(t_0, 1)
        tangent_b, k_b = evaluator.tangents_and_curvatures(t_1, -1)

        t_mid = (t_0 + t_1) / 2
        p_mid = evaluator.points(t_mid)
        with np.errstate(all='ignore'):
            k_mid = np.nan_to_num(np.asarray(curve.curvature(t_mid), dtype=float))

        chord = p_b - p_a
        dist = _norm(chord)

        with np.errstate(all='ignore'):
            scale = np.maximum(np.maximum(_norm(p_a), _norm(p_b)), epsilon)
            non_degenerate = dist > 1e-12 * scale

            alpha = _signed_angle(tangent_a, chord)
            beta = _signed_angle(chord, tangent_b)
            angles_ok = non_degenerate & (np.abs(alpha) <= _ANGLE_CAP) & (np.abs(beta) <= _ANGLE_CAP)

            # curvatures of the bounding circular arcs (arc through A and B with the tangent of A or B)
            k_e = 2. * np.sin(alpha) / dist
            k_f = 2. * np.sin(beta) / dist

            s = np.where(spiral_sign == 0., 1., spiral_sign)
            tol = 1e-9 * (np.abs(k_a) + np.abs(k_b) + np.abs(k_e) + np.abs(k_f)) + 1e-9 / dist
            increasing = spiral_growth >= 0.
            enclosed = np.where(
                increasing,
                (s * k_a <= s * k_e + tol) & (s * k_b >= s * k_f - tol),
                (s * k_a >= s * k_e - tol) & (s * k_b <= s * k_f + tol),
            ) & (s * k_e >= -tol) & (s * k_f >= -tol)

            # maximal distance of the bounding arcs, Eq. (8) of the paper
            delta = dist / 2. * np.abs(np.tan(alpha / 2.) - np.tan(beta / 2.))

            joint, bulge_1, bulge_2 = _biarc_joints_and_bulges(p_a, p_b, tangent_a, tangent_b)
            bulges_ok = (np.abs(bulge_1) <= 1. + 1e-12) & (np.abs(bulge_2) <= 1. + 1e-12) & np.isfinite(
                bulge_1 + bulge_2
            )

            accept = angles_ok & bulges_ok & enclosed & (delta <= epsilon)

            # segments which are nearly straight need no enclosing curve: the curve cannot leave the neighborhood of
            # the chord by more than ~ epsilon (case 1 of the algorithm in the paper)
            k_max = np.maximum(np.maximum(np.abs(k_a), np.abs(k_b)), np.abs(k_mid))
            flat = angles_ok & bulges_ok & (
                (dist ** 2 * k_max / 8. <= 0.25 * epsilon)
                & (np.abs(_cross(chord, p_mid - p_a)) / dist <= 0.25 * epsilon)
            )
            accept |= flat

            if depth == max_depth:
                accept = np.ones_like(accept)

        if np.any(accept):
            forced_line = ~(angles_ok & bulges_ok)
            results.append((
                t_0[accept], p_a[accept], p_b[accept], joint[accept],
                np.where(forced_line, 0., bulge_1)[accept], np.where(forced_line, 0., bulge_2)[accept],
                forced_line[accept],
            ))
            n_results += int(np.count_nonzero(accept))

        split = ~accept
        if n_results + 2 * int(np.count_nonzero(split)) > max_segments:
            raise RuntimeError(
                f'More than {max_segments} segments are necessary to approximate the curve with epsilon = {epsilon}. '
                'Increase epsilon.'
            )

        t_0, t_1, t_mid = t_0[split], t_1[split], t_mid[split]
        spiral_sign, spiral_growth = spiral_sign[split], spiral_growth[split]
        t_0, t_1 = np.concatenate((t_0, t_mid)), np.concatenate((t_mid, t_1))
        spiral_sign = np.concatenate((spiral_sign, spiral_sign))
        spiral_growth = np.concatenate((spiral_growth, spiral_growth))

    return _assemble_vertices(results, curve, evaluator)


def _assemble_vertices(
    results: t.List[t.Tuple[np.ndarray, ...]], curve: CurveLike, evaluator: _Evaluator
) -> t.Tuple[np.ndarray, bool]:
    """Collect the vertices of all biarcs in the order of the curve parameter."""
    start_params = np.concatenate([res[0] for res in results])
    order = np.argsort(start_params, kind='stable')

    p_a = np.concatenate([res[1] for res in results])[order]
    p_b = np.concatenate([res[2] for res in results])[order]
    joint = np.concatenate([res[3] for res in results])[order]
    bulge_1 = np.concatenate([res[4] for res in results])[order]
    bulge_2 = np.concatenate([res[5] for res in results])[order]
    forced_line = np.concatenate([res[6] for res in results])[order]

    chord_length = _norm(p_b - p_a)
    tiny = 1e-12 * np.maximum(chord_length, 1e-300)
    first_empty = _norm(joint - p_a) <= tiny
    second_empty = _norm(p_b - joint) <= tiny
    is_line = (np.abs(bulge_1) < 1e-13) & (np.abs(bulge_2) < 1e-13)

    # vertex at the start of the biarc: it carries the bulge of its first non-empty arc
    start_bulge = np.where(first_empty, bulge_2, bulge_1)
    # vertex at the joint is only needed if both arcs exist and the biarc is no line
    joint_needed = ~(first_empty | second_empty | is_line | forced_line)

    rows = np.empty((2 * len(p_a), 3))
    rows[0::2, :2] = p_a
    rows[0::2, 2] = np.where(forced_line | is_line, 0., start_bulge)
    rows[1::2, :2] = joint
    rows[1::2, 2] = bulge_2
    keep = np.empty(2 * len(p_a), dtype=bool)
    keep[0::2] = True
    keep[1::2] = joint_needed

    vertices = rows[keep]
    last = np.array([[p_b[-1, 0], p_b[-1, 1], 0.]])
    vertices = np.concatenate((vertices, last))

    start_point, end_point = vertices[0, :2], vertices[-1, :2]
    closed = bool(np.linalg.norm(start_point - end_point) <= 1e-9 * max(np.max(np.abs(vertices[:, :2])), 1e-300))
    if closed and len(vertices) > 2:
        vertices = vertices[:-1]
    else:
        closed = False

    return vertices, closed
