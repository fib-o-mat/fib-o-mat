"""Provides the :class:`ParametricCurve` class.

A parametric curve is a smooth curve ``f: [a, b] -> R^2``. It can be created from python functions, from sympy
expressions or from a splipy curve. Curves are transformable (translation, rotation, uniform scaling and mirroring are
stored as a similarity which is applied to the curve) and can be converted to an
:class:`~fibomat.shapes.arc_spline.ArcSpline` with a given maximal error.

Example::

    import numpy as np
    import sympy
    from fibomat.shapes import ParametricCurve

    # from sympy
    t = sympy.Symbol('t')
    curve = ParametricCurve.from_sympy_curve(sympy.Curve([3 * sympy.cos(t), sympy.sin(t)], (t, 0, 2 * sympy.pi)))

    # from functions; all functions must be vectorized (``func(t)`` has shape ``t.shape + (2,)``)
    curve = ParametricCurve(
        func=lambda t: np.stack([np.cos(t), np.sin(t)], axis=-1),
        d_func=lambda t: np.stack([-np.sin(t), np.cos(t)], axis=-1),
        d2_func=lambda t: np.stack([-np.cos(t), -np.sin(t)], axis=-1),
        domain=(0, np.pi),
    )

    curve.length, curve.bounding_box, curve.curvature(1.)
    curve.rotated(np.pi / 3).to_arc_spline(epsilon=1e-4)  # maximal distance 1e-4 to the curve
    points = curve.rasterize_at(pitch=0.1)  # points with equal distance along the curve

References:
    - D.S. Meek and D.J. Walton, Approximating smooth planar curves by arc splines, Journal of Computational and
      Applied Mathematics 59 (1995) 221-231. https://www.sciencedirect.com/science/article/pii/037704279400029Z
"""
# pylint: disable=invalid-name,too-many-arguments
from __future__ import annotations

import typing as t
import warnings

import numpy as np
import scipy.integrate as integrate

from fibomat.linalg import BoundingBox, Vector
from fibomat.shapes.arc_spline import ArcSpline
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.shape import Shape


if t.TYPE_CHECKING:  # pragma: no cover
    import sympy
    import splipy


__all__ = ['ParametricCurve', 'ARC_SPLINE_RELATIVE_TOLERANCE']


ARC_SPLINE_RELATIVE_TOLERANCE = 1e-5
"""Default maximal distance of the arc spline approximation of a curve in units of the diagonal of its bounding box."""

_N_EXTREMA_SAMPLES = 2048
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(5)


def _vectorized_points(func: t.Callable[[np.ndarray], np.ndarray], name: str) -> t.Callable[[np.ndarray], np.ndarray]:
    """Wrap `func` so that it returns arrays of shape ``t.shape + (2,)``."""

    def call(param: t.Any) -> np.ndarray:
        param = np.asarray(param, dtype=float)
        values = np.asarray(func(param), dtype=float)
        if values.shape != param.shape + (2,):
            raise ValueError(
                f'{name} must return an array of shape {param.shape + (2,)} for an input of shape {param.shape} '
                f'but returned an array of shape {values.shape}.'
            )
        return values

    return call


def _vectorized_scalars(func: t.Callable[[np.ndarray], np.ndarray], name: str) -> t.Callable[[np.ndarray], np.ndarray]:
    def call(param: t.Any) -> np.ndarray:
        param = np.asarray(param, dtype=float)
        values = np.asarray(func(param), dtype=float)
        if values.shape != param.shape:
            raise ValueError(
                f'{name} must return an array of shape {param.shape} for an input of shape {param.shape} '
                f'but returned an array of shape {values.shape}.'
            )
        return values

    return call


def _cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _bisect_sign_change(func: t.Callable[[np.ndarray], np.ndarray], low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Roots of a function in intervals in which it changes its sign (vectorized bisection)."""
    low = low.copy()
    high = high.copy()
    sign_low = np.sign(func(low))
    for _ in range(60):
        middle = (low + high) / 2
        same_as_low = np.sign(func(middle)) == sign_low
        low = np.where(same_as_low, middle, low)
        high = np.where(same_as_low, high, middle)
    return (low + high) / 2


class ParametricCurve(Shape, ArcSplineCompatible):
    r"""Parametric curve  f: [a, b] -> R^2 with f in C^3 (at least C^2 is needed for the curvature).

    E.g. f(u) = (cos(u), sin(u)).
    """

    def __init__(
        self,
        func: t.Callable[[np.ndarray], np.ndarray],
        d_func: t.Callable[[np.ndarray], np.ndarray],
        d2_func: t.Callable[[np.ndarray], np.ndarray],
        domain: t.Tuple[float, float],
        bounding_box: t.Optional[BoundingBox] = None,
        curvature: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
        length: t.Optional[t.Callable[[float, float], float]] = None,
        description: t.Optional[str] = None,
        *,
        d3_func: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
        d_curvature: t.Optional[t.Callable[[np.ndarray], np.ndarray]] = None,
    ):
        r"""
        All `Callable`\ s must be vectorized. Hence

            - func(1.) should return np.array([x, y])
            - func(np.array([1., 2.])) should return np.array([[x1, y1], [x2, y2]])

        If `length` is not provided, it will be calculated numerically. If `curvature` is not provided, it is
        calculated from the derivatives. The derivative of the curvature is needed to find curvature extrema and is
        calculated from `d3_func` or `d_curvature` if one of them is given and by finite differences otherwise.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): function value of parametric curve
            d_func (Callable[[np.ndarray], np.ndarray]): function value of first derivative
            d2_func (Callable[[np.ndarray], np.ndarray]): function value of second derivative
            domain (Tuple[float, float]): parametric domain of curve
            bounding_box (BoundingBox, optional): deprecated, ignored (the bounding box is calculated)
            curvature (Optional[Callable[[np.ndarray], np.ndarray]], optional): curvature of parametric curve.
            length (Optional[Callable[[float, float], float]], optional):
                arc length of parametric curve in interval u_1, u_2
            description (str, optional): description
            d3_func (Callable[[np.ndarray], np.ndarray], optional): function value of third derivative
            d_curvature (Callable[[np.ndarray], np.ndarray], optional): derivative of the curvature

        Raises:
            ValueError: Raised if the domain is not a finite interval or a function is no callable.
        """
        super().__init__(description)

        for name, function in (('func', func), ('d_func', d_func), ('d2_func', d2_func)):
            if not callable(function):
                raise ValueError(f'{name} must be callable.')
        for name, function in (
            ('curvature', curvature), ('length', length), ('d3_func', d3_func), ('d_curvature', d_curvature)
        ):
            if function is not None and not callable(function):
                raise ValueError(f'{name} must be callable.')

        try:
            start, end = (float(value) for value in domain)
        except (TypeError, ValueError) as error:
            raise ValueError('domain must be a tuple of two numbers.') from error
        if not (np.isfinite(start) and np.isfinite(end)) or not start < end:
            raise ValueError('domain must be a finite interval (start < end).')

        if bounding_box is not None:
            warnings.warn(
                'The bounding_box argument of ParametricCurve is deprecated and ignored.',
                category=DeprecationWarning,
                stacklevel=2,
            )

        self._func = _vectorized_points(func, 'func')
        self._d_func = _vectorized_points(d_func, 'd_func')
        self._d2_func = _vectorized_points(d2_func, 'd2_func')
        self._d3_func = _vectorized_points(d3_func, 'd3_func') if d3_func is not None else None
        self._curvature = _vectorized_scalars(curvature, 'curvature') if curvature is not None else None
        self._d_curvature = _vectorized_scalars(d_curvature, 'd_curvature') if d_curvature is not None else None
        self._length = length
        self._domain = (start, end)

        # similarity (translation, rotation, uniform scaling, mirroring) applied to the curve defined by the functions
        self._matrix = np.eye(2)
        self._offset = np.zeros(2)
        self._bounding_box_cache: t.Optional[BoundingBox] = None

    # construction

    @classmethod
    def from_sympy_curve(
        cls,
        curve: 'sympy.geometry.Curve',
        try_length_integration: bool = False,
        description: t.Optional[str] = None,
        integration_timeout: t.Optional[float] = None,
    ) -> ParametricCurve:
        """Create a :class:`ParametricCurve` from a sympy curve. The derivatives (up to the third one) are calculated
        automatically.

        Args:
            curve (sympy.geometry.Curve): parametric curve.
            try_length_integration (bool): deprecated and ignored, the length is integrated numerically.
            description (str, optional): description
            integration_timeout (float, optional): deprecated and ignored

        Returns:
            ParametricCurve
        """
        import sympy  # pylint: disable=import-outside-toplevel

        if try_length_integration or integration_timeout is not None:
            warnings.warn(
                'try_length_integration and integration_timeout are deprecated and ignored, '
                'the arc length is integrated numerically.',
                category=DeprecationWarning,
                stacklevel=2,
            )

        parameter = curve.parameter
        derivatives = [list(curve.functions)]
        for _ in range(3):
            derivatives.append([sympy.diff(expression, parameter) for expression in derivatives[-1]])

        def make_func(expressions: t.Sequence['sympy.Expr']) -> t.Callable[[np.ndarray], np.ndarray]:
            # (no simplify: it is slow and not needed; common subexpressions are eliminated)
            lambdified = sympy.lambdify(parameter, list(expressions), modules='numpy', cse=True)

            def func(param: np.ndarray) -> np.ndarray:
                param = np.asarray(param, dtype=float)
                columns = [np.broadcast_to(np.asarray(value, dtype=float), param.shape) for value in lambdified(param)]
                return np.stack(columns, axis=-1)

            return func

        return cls(
            func=make_func(derivatives[0]),
            d_func=make_func(derivatives[1]),
            d2_func=make_func(derivatives[2]),
            domain=(float(curve.limits[1]), float(curve.limits[2])),
            d3_func=make_func(derivatives[3]),
            description=description,
        )

    @classmethod
    def from_splipy_curve(
        cls, splipy_curve: 'splipy.Curve', description: t.Optional[str] = None
    ) -> ParametricCurve:
        """Create a :class:`ParametricCurve` from a splipy curve (only the first two dimensions are used).

        Args:
            splipy_curve (splipy.Curve): curve
            description (str, optional): description

        Returns:
            ParametricCurve
        """
        splipy_curve = splipy_curve.clone()

        def to_points(values: t.Any, param: np.ndarray) -> np.ndarray:
            # (splipy returns arrays of shape (1, dim) for scalar parameters)
            values = np.asarray(values, dtype=float).reshape(param.shape + (-1,))
            return values[..., :2]

        def derivative_func(order: int) -> t.Callable[[np.ndarray], np.ndarray]:
            def func(param: np.ndarray) -> np.ndarray:
                return to_points(splipy_curve.derivative(param, d=order), param)
            return func

        def func(param: np.ndarray) -> np.ndarray:
            return to_points(splipy_curve.evaluate(param), param)

        def length(t_0: float, t_1: float) -> float:
            return float(splipy_curve.length(t_0, t_1))

        return cls(
            func=func,
            d_func=derivative_func(1),
            d2_func=derivative_func(2),
            domain=(float(splipy_curve.start(direction=0)), float(splipy_curve.end(direction=0))),
            length=length,
            d3_func=derivative_func(3),
            description=description,
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(domain={self._domain!r})'

    # transformation state

    @property
    def _scale(self) -> float:
        """Scale factor of the similarity."""
        return float(np.sqrt(abs(np.linalg.det(self._matrix))))

    @property
    def _orientation(self) -> float:
        """+1 if the similarity keeps and -1 if it reverses the orientation."""
        return float(np.sign(np.linalg.det(self._matrix)))

    # evaluation of the curve and its derivatives

    @property
    def domain(self) -> t.Tuple[float, float]:
        """Parametric domain of curve.

        Access:
            get

        Returns:
            Tuple[float, float]
        """
        return self._domain

    def f(self, t: t.Union[float, np.ndarray]) -> np.ndarray:  # pylint: disable=redefined-outer-name
        """Function values of param. curve.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray: array of shape ``t.shape + (2,)``
        """
        return self._func(t) @ self._matrix.T + self._offset

    def df(self, t: t.Union[float, np.ndarray]) -> np.ndarray:  # pylint: disable=redefined-outer-name
        """Function values of first derivative of param. curve.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray
        """
        return self._d_func(t) @ self._matrix.T

    def d2f(self, t: t.Union[float, np.ndarray]) -> np.ndarray:  # pylint: disable=redefined-outer-name
        """Function values of second derivative of param. curve.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray
        """
        return self._d2_func(t) @ self._matrix.T

    def d3f(self, t: t.Union[float, np.ndarray]) -> np.ndarray:  # pylint: disable=redefined-outer-name
        """Function values of third derivative of param. curve.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray

        Raises:
            NotImplementedError: Raised if the third derivative is not available.
        """
        if self._d3_func is None:
            raise NotImplementedError('The third derivative is not available.')
        return self._d3_func(t) @ self._matrix.T

    def curvature(self, t: t.Union[float, np.ndarray]) -> np.ndarray:  # pylint: disable=redefined-outer-name
        """Signed curvature of param. curve (positive if the curve turns left).

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray
        """
        if self._curvature is not None:
            return self._orientation / self._scale * self._curvature(t)

        velocity = self.df(t)
        with np.errstate(divide='ignore', invalid='ignore'):
            return _cross(velocity, self.d2f(t)) / np.linalg.norm(velocity, axis=-1) ** 3

    def d_curvature(self, t: t.Union[float, np.ndarray]) -> np.ndarray:  # pylint: disable=redefined-outer-name
        """Derivative of the curvature with respect to the curve parameter.

        It is calculated analytically if the third derivative (or the derivative of the curvature) is available and by
        finite differences otherwise.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray
        """
        param = np.asarray(t, dtype=float)

        if self._d_curvature is not None:
            return self._orientation / self._scale * self._d_curvature(param)

        if self._d3_func is not None:
            velocity, acceleration, jerk = self.df(param), self.d2f(param), self.d3f(param)
            speed_sq = np.sum(velocity * velocity, axis=-1)
            with np.errstate(divide='ignore', invalid='ignore'):
                return (
                    _cross(velocity, jerk) / speed_sq ** 1.5
                    - 3. * _cross(velocity, acceleration) * np.sum(velocity * acceleration, axis=-1) / speed_sq ** 2.5
                )

        # central finite differences (one-sided at the borders of the domain)
        step = 1e-6 * (self._domain[1] - self._domain[0])
        upper = np.minimum(param + step, self._domain[1])
        lower = np.maximum(param - step, self._domain[0])
        return (self.curvature(upper) - self.curvature(lower)) / (upper - lower)

    # length and rasterization

    def arc_length(self, t_0: float, t_1: float) -> float:
        """Arc length of the curve between the parameters `t_0` and `t_1`.

        Args:
            t_0 (float): start parameter
            t_1 (float): end parameter

        Returns:
            float
        """
        if self._length is not None:
            return float(self._scale * self._length(t_0, t_1))

        return float(
            integrate.quad(
                lambda param: float(np.linalg.norm(self.df(param))), t_0, t_1, limit=500, epsabs=0., epsrel=1e-11
            )[0]
        )

    @property
    def length(self) -> float:
        """Arc length of curve.

        Access:
            get

        Returns:
            float
        """
        return self.arc_length(*self._domain)

    def rasterize(
        self,
        pitch: float,
        domain: t.Optional[t.Tuple[float, float]] = None,
        safety: t.Optional[float] = None,
        add_endpoint: bool = False,
    ) -> np.ndarray:
        """Rasterize the param. curve equally.

        The parameters of the points with the arc length ``0, pitch, 2 * pitch, ...`` (measured from the start of the
        domain) are returned. The arc length is integrated on a fine grid (Gauss-Legendre) and inverted with Newton
        steps; all of this is vectorized.

        Args:
            pitch (float): distance of rasterized points on the curve.
            domain (Tuple[float, float], optional): parametric domain to be used. Default to self.domain.
            safety (float, optional): deprecated and ignored.
            add_endpoint (bool): if True, the parameter of the end of the domain is added to the rasterized points
                                 (if it is not already the last point).

        Returns:
            np.ndarray: function parameters (NOT function values)

        Raises:
            ValueError: Raised if the pitch is not positive or the domain is invalid.
        """
        if safety is not None:
            warnings.warn('The safety argument is deprecated and ignored.', category=DeprecationWarning, stacklevel=2)

        pitch = float(pitch)
        if not np.isfinite(pitch) or pitch <= 0.:
            raise ValueError('pitch must be positive and finite.')

        t_min, t_max = self._domain if domain is None else (float(domain[0]), float(domain[1]))
        if not (self._domain[0] <= t_min < t_max <= self._domain[1]):
            raise ValueError('domain must be a non-empty subinterval of the domain of the curve.')

        # fine grid with the cumulative arc length
        rough_length = self.arc_length(t_min, t_max)
        n_cells = int(min(max(2000, 20 * rough_length / pitch), 2_000_000))
        grid = np.linspace(t_min, t_max, n_cells + 1)
        cell_length = self._integrate_speed(grid[:-1], grid[1:])
        arc_length = np.concatenate(([0.], np.cumsum(cell_length)))
        total = arc_length[-1]

        n_points = int(np.floor(total / pitch * (1. + 1e-12))) + 1
        targets = np.arange(n_points) * pitch

        cell = np.clip(np.searchsorted(arc_length, targets, side='right') - 1, 0, n_cells - 1)
        # linear interpolation inside of the cells followed by Newton steps
        fraction = (targets - arc_length[cell]) / np.where(cell_length[cell] > 0., cell_length[cell], 1.)
        params = grid[cell] + np.clip(fraction, 0., 1.) * (grid[cell + 1] - grid[cell])
        for _ in range(3):
            speed = np.linalg.norm(self.df(params), axis=-1)
            error = arc_length[cell] + self._integrate_speed(grid[cell], params) - targets
            params = np.clip(params - error / np.where(speed > 0., speed, 1.), grid[cell], grid[cell + 1])

        params[0] = t_min

        if add_endpoint and t_max - params[-1] > 1e-9 * (t_max - t_min) / n_points:
            params = np.append(params, t_max)

        return params

    def _integrate_speed(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Integrals of the speed over the intervals [lower, upper] (5 point Gauss-Legendre rule)."""
        half = (upper - lower) / 2
        nodes = (lower + upper)[:, np.newaxis] / 2 + half[:, np.newaxis] * _GAUSS_NODES[np.newaxis, :]
        speed = np.linalg.norm(self.df(nodes.ravel()), axis=-1).reshape(nodes.shape)
        return half * (speed * _GAUSS_WEIGHTS).sum(axis=1)

    def rasterize_at(self, pitch: float, domain: t.Optional[t.Tuple[float, float]] = None) -> np.ndarray:
        """Rasterize the param. curve equally.

        Args:
            pitch (float): distance of rasterized points on the curve.
            domain (Tuple[float, float], optional): parametric domain to be used. Default to self.domain.

        Returns:
             np.ndarray: function values
        """
        return self.f(self.rasterize(pitch, domain))

    # arc spline approximation

    def to_arc_spline(
        self, rasterize_pitch: t.Optional[float] = None, epsilon: t.Optional[float] = None
    ) -> ArcSpline:
        """Approximate the curve by an arc spline (a tangent continuous curve of arcs and lines).

        The maximal distance between the curve and the arc spline is `epsilon` (cf. Meek and Walton, 1995).

        Args:
            rasterize_pitch (float, optional): deprecated and ignored.
            epsilon (float, optional): maximal distance between the curve and the arc spline. Default to
                :data:`ARC_SPLINE_RELATIVE_TOLERANCE` times the diagonal of the bounding box of the curve.

        Returns:
            ArcSpline

        Raises:
            ValueError: Raised if epsilon is not positive.
        """
        from fibomat.curve_tools.biarc_approximation import approximate_parametric_curve  # pylint: disable=C0415

        if rasterize_pitch is not None:
            warnings.warn(
                'rasterize_pitch is deprecated and ignored. The approximation is controlled by epsilon only.',
                category=DeprecationWarning,
                stacklevel=2,
            )

        if epsilon is None:
            bbox = self.bounding_box
            epsilon = ARC_SPLINE_RELATIVE_TOLERANCE * float((bbox.upper_right - bbox.lower_left).mag)
            if not epsilon > 0.:
                raise ValueError('Cannot determine a default epsilon for a curve without extent.')

        return approximate_parametric_curve(self, float(epsilon), description=self.description)

    # shape interface

    @property
    def is_closed(self) -> bool:
        start, end = self.f(np.array(self._domain))
        return bool(np.linalg.norm(start - end) <= 1e-9 * max(1., float(np.linalg.norm(start))))

    @property
    def bounding_box(self) -> BoundingBox:
        """Bounding box of the curve. It is calculated from the extrema of the coordinates (roots of the derivative).

        Access:
            get

        Returns:
            BoundingBox
        """
        if self._bounding_box_cache is None:
            start, end = self._domain
            grid = np.linspace(start, end, _N_EXTREMA_SAMPLES)
            velocity = self.df(grid)

            params = [grid]
            for axis in (0, 1):
                signs = np.sign(velocity[:, axis])
                changed = np.nonzero(signs[:-1] * signs[1:] < 0)[0]
                if len(changed):
                    params.append(
                        _bisect_sign_change(
                            lambda param, axis=axis: self.df(param)[..., axis], grid[changed], grid[changed + 1]
                        )
                    )

            points = self.f(np.concatenate(params))
            self._bounding_box_cache = BoundingBox(np.min(points, axis=0), np.max(points, axis=0))

        return self._bounding_box_cache

    @property
    def center(self) -> Vector:
        """Center of the bounding box of the curve.

        Access:
            get

        Returns:
            Vector
        """
        return self.bounding_box.center

    def _impl_translate(self, trans_vec: Vector) -> None:
        trans_vec = Vector(trans_vec)
        self._offset = self._offset + np.array([trans_vec.x, trans_vec.y])
        self._bounding_box_cache = None

    def _impl_rotate(self, theta: float) -> None:
        cos, sin = np.cos(float(theta)), np.sin(float(theta))
        rotation = np.array([[cos, -sin], [sin, cos]])
        self._matrix = rotation @ self._matrix
        self._offset = rotation @ self._offset
        self._bounding_box_cache = None

    def _impl_scale(self, fac: float) -> None:
        fac = float(fac)
        self._matrix = fac * self._matrix
        self._offset = fac * self._offset
        self._bounding_box_cache = None

    def _impl_mirror(self, mirror_axis: Vector) -> None:
        angle = Vector(mirror_axis).phi
        cos, sin = np.cos(2 * angle), np.sin(2 * angle)
        reflection = np.array([[cos, sin], [sin, -cos]])
        self._matrix = reflection @ self._matrix
        self._offset = reflection @ self._offset
        self._bounding_box_cache = None
