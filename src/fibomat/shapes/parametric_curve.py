"""Provides the :class:`ParametricCurve` class."""
# pylint: disable=invalid-name,too-many-arguments

from __future__ import annotations
from typing import Callable, Optional, Tuple, Union
import functools

import scipy.integrate as integrate
import scipy.optimize as optimize

import numpy as np
import multiprocessing

import sympy
import sympy.geometry as sympy_geom

import splipy

from fibomat.shapes.shape import Shape
from fibomat.shapes.arc_spline import ArcSplineCompatible, ArcSpline
from fibomat.linalg import VectorLike, Vector, BoundingBox
from fibomat.curve_tools.biarc_approximation import approximate_parametric_curve


class ParametricCurve(Shape, ArcSplineCompatible):
    """Parametric curve  f: [a, b] -> R^2 with f \\in C^\\inf

    E.g. f(u) = (cos(u), sin(u)).

    TODO: handle bounding box!
    TODO: Put references here (thesis + 2 papers)
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        d_func: Callable[[np.ndarray], np.ndarray],
        d2_func: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float],
        bounding_box: BoundingBox,
        curvature: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        length: Optional[Callable[[float, float], float]] = None,
        description: Optional[str] = None
    ):
        r"""
        All `Callable`\ s must be vectorized. Hence

            - func(1) should return np.array([x, y])
            - func([1, 2]) should return np.array([[x1, y1], [x2, y2]])

        If `length` is not provided, it will be calculated numerically.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): function value of parametric curve
            d_func (Callable[[np.ndarray], np.ndarray]): function value of first derivative
            d2_func (Callable[[np.ndarray], np.ndarray]): function value of second derivative
            domain (Tuple[float, float]): parametric domain of curve
            bounding_box (BoundingBox): bounding box of curve
            curvature (Optional[Callable[[np.ndarray], np.ndarray]], optional): curvature of parametric curve.
            length (Optional[Callable[[float, float], float]], optional):
                arc length of parametric curve in interval u_1, u_2
            description (str, optional): description
        """
        super().__init__(description)

        self._func = func
        self._d_func = d_func
        self._d2_func = d2_func

        if curvature:
            self._curvature = curvature
        else:
            def curvature_impl(t: np.ndarray):
                df = self._d_func(t)
                d2f = self._d2_func(t)
                # print(t, type(t), t.ndim)
                if t.ndim == 0:
                    return (df[0] * d2f[1] - df[1] * d2f[0]) / (df[0]**2 + df[1]**2)**1.5

                return (df[:, 0] * d2f[:, 1] - df[:, 1] * d2f[:, 0]) / (df[:, 0]**2 + df[:, 1]**2)**1.5

            self._curvature = curvature_impl

        if length:
            self._length = length
        else:
            def length_impl(t_0, t_1):
                return integrate.quad(lambda t: np.linalg.norm(self._d_func(t)), t_0, t_1)[0]

            self._length = length_impl

        self._domain = domain

    @classmethod
    def from_sympy_curve(
        cls,
        curve: sympy_geom.Curve,
        try_length_integration: bool = False,
        description: Optional[str] = None,
        integration_timeout=5
    ):
        """Create a :class:`ParametricCurve` from a sympy curve. All derivatives are calculated automatically.

        Args:
            curve (sympy.geometry.Curve): parametric curve.
            try_length_integration (bool):
                if True, it is attempted to solve arc length parametrization integral analytically.
            description (str, optional): description

        Returns:
            ParametricCurve
        """
        # pylint: disable=function-redefined

        # https://github.com/sympy/sympy/issues/5642
        # https://stackoverflow.com/a/59757810
        def lambdify_np(arg, exp, modules=None):
            exp_lambdified = sympy.lambdify(arg, exp, modules)
            if exp.is_constant:
                return lambda t: np.full_like(t, exp_lambdified(t))

            return exp_lambdified

        def lambdify_np_2d(arg, expressions, modules=None):
            l1 = lambdify_np(arg, expressions[0], modules)
            l2 = lambdify_np(arg, expressions[1], modules)
            # return lambda t: np.vstack((l1(t), l2(t))).T
            return lambda t: np.squeeze(np.vstack((l1(t), l2(t))).T)
            # return lambda t: np.c_[l1(t), l2(t)]

        # TODO: check if diff failed !
        dfunc = [sympy.diff(curve.functions[0], curve.parameter), sympy.diff(curve.functions[1], curve.parameter)]
        d2func = [sympy.diff(dfunc[0], curve.parameter), sympy.diff(dfunc[1], curve.parameter)]

        curvature = sympy.simplify(
            (dfunc[0]*d2func[1] - dfunc[1]*d2func[0]) / (dfunc[0]**2 + dfunc[1]**2)**sympy.Rational(3, 2)
        )


        lambda_length: Optional[Callable]

        if try_length_integration:
            def integrate_with_timeout(dfunc, curveparameter, queue):
                result = sympy.simplify(
                    sympy.integrate(sympy.sqrt(dfunc[0] * dfunc[0] + dfunc[1] * dfunc[1]), curveparameter)
                )
                queue.put(result)
            print("Trying length integration...")
            # length integration is done in a separate process not for optimization but only for timeout
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=integrate_with_timeout, args=(dfunc, curve.parameter, queue))
            process.start()
            process.join(timeout=integration_timeout)

            if process.is_alive():  #integration failed
                process.terminate()
                process.join()
                print(f"Length integration timed out after {integration_timeout} seconds.")
                length = sympy.Integral(sympy.sqrt(dfunc[0] * dfunc[0] + dfunc[1] * dfunc[1]), curve.parameter)  # unevaluated integral
            else:
                if not queue.empty():
                    length = queue.get()
                else:
                    length = sympy.Integral(sympy.sqrt(dfunc[0] * dfunc[0] + dfunc[1] * dfunc[1]), curve.parameter)

            if isinstance(length, sympy.Integral):  # integral was not evaluated
                lambda_length = None
            else:
                antiderivate = lambdify_np(curve.parameter, length, 'numpy')

                def lambda_length(t_0: float, t_1: float):
                    return antiderivate(t_1) - antiderivate(t_0)
        else:
            lambda_length = None

        return cls(
            func=lambdify_np_2d(curve.parameter, curve.functions, 'numpy'),
            d_func=lambdify_np_2d(curve.parameter, dfunc, 'numpy'),
            d2_func=lambdify_np_2d(curve.parameter, d2func, 'numpy'),
            domain=(float(curve.limits[1]), float(curve.limits[2])),
            bounding_box=BoundingBox((0, 0), (0, 0)),
            curvature=lambdify_np(curve.parameter, curvature, 'numpy'),
            length=lambda_length,
            description=description
        )

    @classmethod
    def from_splipy_curve(cls, splipy_curve: splipy.Curve, description: Optional[str] = None):
        splipy_curve = splipy_curve.clone()

        func = lambda t: splipy_curve.evaluate(t)
        d_func = lambda t: splipy_curve.derivative(t, d=1)
        d2_func = lambda t: splipy_curve.derivative(t, d=2)
        curvature = lambda t: splipy_curve.curvature(t)
        length = lambda t_0, t_1: splipy_curve.length(t_0, t_1)
        domain = splipy_curve.start(direction=0), splipy_curve.end(direction=0)

        print(domain, type(domain[0]))

        return cls(
            func=func,
            d_func=d_func,
            d2_func=d2_func,
            domain=domain,
            bounding_box=BoundingBox((0, 0), (0, 0)),
            curvature=curvature,
            length=length,
            description=description
        )

    def __repr__(self) -> str:
        return '{}(...)'.format(self.__class__.__name__)

    def to_arc_spline(self, rasterize_pitch: Optional[float] = None, epsilon: Optional[float] = None) -> ArcSpline:
        if (rasterize_pitch is not None and epsilon is None) or (rasterize_pitch is None and epsilon is not None):
            raise ValueError('Either rasterize_pitch and epsilon must be given or none of them.')
        # if (rasterize_pitch is not None and epsilon is not None) or (rasterize_pitch is None and epsilon is None):
        #     raise ValueError('Either rasterize_pitch and epsilon must be given or none of them.')

        if rasterize_pitch is None:
            rasterize_pitch = self.length / 100
            epsilon = rasterize_pitch / 10
        

        return ArcSpline.from_segments(approximate_parametric_curve(self, rasterize_pitch, epsilon))

    @property
    def domain(self) -> Tuple[float, float]:
        """Parametric domain of curve.

        Access:
            get

        Returns:
            Tuple[float, float]
        """
        return self._domain

    @property
    def is_closed(self) -> bool:
        return np.allclose(self._func(self._domain[0]), self._func(self._domain[1]))

    @property
    def length(self) -> float:
        """Arc length of curve.

        Access:
            get

        Returns:
            float
        """
        return self._length(*self._domain)

    def f(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Function values of param. curve.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray
        """
        return self._func(np.asarray(t))

    def df(self, t):
        """Function values of first derivative of param. curve.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray
        """
        return self._d_func(np.asarray(t))

    def d2f(self, t):
        """Function values of second derivative of param. curve.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray
        """
        return self._d2_func(np.asarray(t))

    def curvature(self, t):
        """Curvature of param. curve.

        Args:
            t (float, np.ndarray): time points for evaluation.

        Returns:
            np.ndarray
        """
        return self._curvature(np.asarray(t))

    def rasterize(
        self,
        pitch: float,
        domain: Optional[Tuple[float, float]] = None,
        safety: float = 1.25,
        add_endpoint: bool = False
    ) -> np.ndarray:
        """Rasterize the param. curve equally.

        Args:
            pitch (float): distance of rasteruized points on the curve.
            domain (Tuple[float, float], optional): parametric domain to be used. Default to self.domain.
            safety (float):
                The upper bound of the function parameter of a point is estimated with the tangent vector `t` of the
                previous point with `t_next_up = t_prev + pitch * safety / ||t||`. If the function has large gradients,
                thesafety factor must be increased. Otherwise, `t_next_up` is not an upper bound anymore. Default to 1.5
            add_endpoint (bool): if True, the point at f(domain[1]) is added to the rasterized points, if the distance
                                 to the point before is smaller than the pitch.


        Returns:
            np.ndarray: function parameters (NOT function values)

        Raises:
            RuntimeError: Raised if `safety` is to small.
        """
        safety_step = .05
        safety = 1.

        if domain:
            n_points = int(self._length(*domain) / pitch) + 1
        else:
            n_points = int(self._length(*self._domain) / pitch) + 1

        if domain:
            t_min, t_max = domain
        else:
            t_min, t_max = self._domain

        t = t_min

        # create a little more space than needed to account for inaccuracies at the rasterization step
        buffer_size = int(n_points * 1.2)
        t_res = np.empty(buffer_size, dtype=float)
        t_res[0] = t_min

        def length_diff(t_0: float, t_1: float):
            return self._length(t_0, t_1) - pitch

        i = 1
        # for i in range(1, n_points):
        while t < t_max:
            # f = lambda t_: self._length(t, t_) - pitch
            root_found = False

            df = np.linalg.norm(self.df(t))
            if np.isclose(df, 0):
                df = (t_max - t_min) / n_points

            while not root_found:
                try:
                    t_bound = t + safety * pitch / df
                    if t_bound > t_max:
                        t_bound = t_max

                    # print(functools.partial(length_diff, t)(t), functools.partial(length_diff, t)(t_bound))

                    t = optimize.toms748(
                        functools.partial(length_diff, t),
                        t, t_bound,
                        xtol=0.01*pitch, rtol=0.001*pitch
                    )  # t_max
                except ValueError:
                    if t_bound == t_max:
                        # if self._length(*domain) is a multiple of the pitch: f(t) < 0 and f(t_max) \approx 0
                        # (hence toms748 would fail because no sign change occurs)
                        t = t_max
                        root_found = True
                    else:
                        # print('increase')
                        safety += safety_step
                else:
                    root_found = True
            t_res[i] = t
            i += 1

            if i > buffer_size:
                raise RuntimeError('i >= buffer_size. This is probably a bug. Please report it.')

        if add_endpoint and not np.isclose(t_res[i-2], t_max):
            t_res[i-1] = t_max
            i += 1

        t_res = np.resize(t_res, i-1)

        return t_res

    def rasterize_at(self, pitch: float, domain: Optional[Tuple[float, float]] = None):
        """Rasterize the param. curve equally.

        Args:
            pitch (float): distance of rasteruized points on the curve.
            domain (Tuple[float, float], optional): parametric domain to be used. Default to self.domain.

        Returns:
             np.ndarray: function values
        """
        return self.f(self.rasterize(pitch, domain))

    @property
    def bounding_box(self) -> BoundingBox:
        raise NotImplementedError(
            f'Cannot calculate a bounding box for {self.__class__.__name__}. '
            'Convert it to an ArcSpline first.'
        )

    @property
    def center(self) -> Vector:
        raise NotImplementedError(
            f'Cannot calculate center for {self.__class__.__name__}. '
            'Convert it to an ArcSpline first.'
        )

    def _impl_translate(self, trans_vec: VectorLike) -> None:
        raise NotImplementedError(f'Cannot translate a {self.__class__.__name__}. Convert it to an ArcSpline first.')

    def _impl_rotate(self, theta: float) -> None:
        raise NotImplementedError(f'Cannot rotate a {self.__class__.__name__}. Convert it to an ArcSpline first.')

    def _impl_scale(self, fac: float) -> None:
        raise NotImplementedError(f'Cannot scale a {self.__class__.__name__}. Convert it to an ArcSpline first.')

    def _impl_mirror(self, mirror_axis: VectorLike) -> None:
        raise RuntimeError(f'Cannot mirror a {self.__class__.__name__}. Convert it to an ArcSpline first.')
