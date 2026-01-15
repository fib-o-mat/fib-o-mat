from typing import Tuple

from fibomat.raster_styles.rasterstyle import RasterStyle
from fibomat.raster_styles.one_d import Curve
from fibomat.shapes import rasterizedpoints
from fibomat import shapes
from fibomat.units import LengthUnit, TimeUnit, LengthQuantity, TimeQuantity, has_length_dim, scale_to, scale_factor
from fibomat.shapes import Shape, RasterizedPoints, DimShape
from fibomat.mill import DDDMill, SILMill
from fibomat.rasterizedpattern import RasterizedPattern
from fibomat.curve_tools import fill_with_spiral, rasterize, fill_with_lines
import numpy as np
from typing import Optional
from fibomat.raster_styles.scansequence import ScanSequence


class Spiral(RasterStyle):
    """
    pitch: pitch between points on the spiral
    spiral_pitch: distance between arms of the spiral
    direction: "outwards" for beginning in the center, "inwards" for beginning outside, "out-in" for going out and back in

    """
    def __init__(self, pitch: LengthQuantity, spiral_pitch: LengthQuantity, scan_sequence: ScanSequence, direction: Optional[str]):
        self._pitch = pitch
        self._spiral_pitch = spiral_pitch
        self._scan_sequence = scan_sequence
        if direction:
            if direction not in ["outwards", "inwards", "out-in"]:
                raise ValueError("Direction must be 'outwards', 'inwards' or 'in-out', if not specified it is 'outwards'.")
        self._direction = direction

    @property
    def dimension(self) -> int:
        return 2

    @property
    def pitch(self):
        return self._pitch

    @property
    def scan_sequence(self):
        return self._scan_sequence

    @property
    def direction(self):
        return None

    @property
    def line_style(self):
        return None



    def rasterize(
        self,
        dim_shape: DimShape,
        mill: DDDMill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit,
    ) -> RasterizedPattern:
        """
        dim_shape: Shape to be rasterized
        mill: Mill to be used
        out_length_unit: Unit handed over to the rasterized pattern
        out_time_unit: Timeunit handed over to the rasterized pattern

        uses an arclength-approximation for the spiral described in 
        https://math.stackexchange.com/questions/2335055/placing-points-equidistantly-along-an-archimedean-spiral-from-parametric-equatio

        Note/TODO: Not completely tested if the approximation is well enough. Not tested if it is okay to parametrize by arclength, as this
        results in more points per area in the shape's center than outside.
        TODO: This method is not tested yet for complex shapes, for example shapes with holes.
        TODO: This method is a pain in the ass. No good compatability with fibomat at all. Implement backstitch-option etc. as for other raster styles.
        Creating an actual spiral filling the area and then sampling would be more similar to linebyline-rasterstyle. Breaking this rasterization down to 
        one-d-rasterization instead of reimplementing everything would be much cleaner. 
        Biggest problem is that creating the spiral and rasterizing it as a curve takes way longer than the currently implemented version.
        """

        spiral_pitch=scale_to(out_length_unit, self._spiral_pitch)
        tang_pitch=scale_to(out_length_unit, self._pitch)
        dim_shape.set_unit(out_length_unit)  # scaling the shape too prevents memory overload from to big theta_max

        spline_shape = dim_shape.shape.to_arc_spline() 

        center = spline_shape.center 
        bbox = spline_shape.bounding_box
        radius = np.linalg.norm(bbox.upper_right - bbox.center)  # radius of circumscribed circle

        theta_max = (radius / spiral_pitch)*2 * np.pi  # maximal angle needed

        a = spiral_pitch / (2 * np.pi)  # radius-theta-proportion factor
        s_max = 0.5 * a * theta_max ** 2  # approximate arclength of spiral
        s_vals = np.arange(0, s_max, tang_pitch)
        t_n = np.sqrt(2 * s_vals / a)  # approximate arclength-steps along [0, s_max]

        r_vals = a * t_n
        x_vals = r_vals * np.cos(t_n)
        y_vals = r_vals * np.sin(t_n)

        # Stack as N×2 array and translate to the correct center
        spiral_points = np.column_stack((x_vals, y_vals))+center

        if isinstance(mill, SILMill):  # unit-compatibility between mill and points. TODO: do this for every Mill and every Rasterstyle. 
            mill.set_unit(out_length_unit)

        dwell_times = [scale_to(out_time_unit, mill.dwell_time(p)) for p in spiral_points] 
        dwell_times = np.array(dwell_times)

        dwell_points = np.column_stack((spiral_points, dwell_times))

        # remove points contained in the circumscribed circle of the boundingbox but not in the shape
        points_inside = [p for p in dwell_points if spline_shape.contains(p[:2])]
        points_inside = np.array(points_inside)

        if self._direction == "inwards":
            points_inside = points_inside[::-1]
        elif self._direction == "out-in":
            points_inside = np.concatenate([points_inside, points_inside[::-1]])

        #Wrap in RasterizedPattern
        rast_pat = RasterizedPattern(points_inside, out_length_unit, out_time_unit)
        return rast_pat





def rasterize_sympy_curve(curve: Curve, pitch: float, total_arc_length=None):
    """
    TODO: this is a relict from when trying to calculate arclength parametrization directly. Probably just delete this method?
    Sample points on a sympy Curve at regular arc-length intervals (tangential pitch).
    
    Parameters:
    - curve: sympy.geometry.Curve object
    - pitch: desired arc-length distance between points
    - total_arc_length: optionally override automatic arc length estimation
    
    Returns:
    - NumPy array with shape (N, 2) of (x, y) points
    """
    import numpy as np
    from sympy import lambdify, sqrt, diff
    from sympy import cos, sin, pi
    from sympy.abc import t

    from sympy.geometry import Curve
    from scipy.integrate import quad
    from scipy.optimize import root_scalar

    # Unpack parameter and limits
    param, t_start, t_end = curve.limits
    x_expr, y_expr = curve.functions

    # First derivatives
    dx = diff(x_expr, param)
    dy = diff(y_expr, param)

    # Arc length integrand: sqrt(dx² + dy²)
    integrand_expr = sqrt(dx**2 + dy**2)
    integrand_func = lambdify(param, integrand_expr, 'numpy')

    # Arc length function: s(t) = ∫ sqrt(dx² + dy²) dt from t_start to t
    def arc_length(t_val):
        result, _ = quad(integrand_func, float(t_start), t_val)
        return result

    # Invert arc length: find t such that arc_length(t) ≈ s_target
    def find_t_for_s(s_target):
        sol = root_scalar(
            lambda t: arc_length(t) - s_target,
            bracket=[float(t_start), float(t_end)],
            method='brentq'
        )
        return sol.root

    # Determine total arc length if not provided
    if total_arc_length is None:
        total_arc_length = arc_length(float(t_end))

    # Compute how many points based on pitch
    num_points = int(np.floor(total_arc_length / pitch)) + 1
    s_vals = np.linspace(0, total_arc_length, num_points)

    # Find corresponding t-values
    t_vals = np.array([find_t_for_s(s) for s in s_vals])

    # Evaluate curve expressions
    x_func = lambdify(param, x_expr, 'numpy')
    y_func = lambdify(param, y_expr, 'numpy')
    x_vals = x_func(t_vals)
    y_vals = y_func(t_vals)

    return np.column_stack((x_vals, y_vals))



from scipy.special import lambertw
from sympy import sqrt, sinh
from scipy.integrate import quad

def rasterize_spiral_lambertw(curve, pitch):
    """
    TODO this is another outdated idea to compute the arclength, probably to be deleted as well
    Rasterizes a sympy-defined Archimedean spiral using arc-length spacing via Lambert W.

    Parameters:
    - curve: sympy.Curve object with r(t) = b*t form
    - pitch: arc-length pitch (assumes radial pitch == pitch)

    Returns:
    - np.ndarray of shape (N, 2) containing (x, y) coordinates
    """

    # Extract parametric info
    param, t_start, t_end = curve.limits
    x_expr, y_expr = curve.functions

    # Spiral must be of the form x(t) = b*t*cos(t) + x0, y(t) = b*t*sin(t) + y0
    # Try to extract center
    center_x = x_expr.subs(param, 0)
    center_y = y_expr.subs(param, 0)

    # Assume r(t) = b*t → b = radial pitch
    b = pitch  # because your spiral is constructed this way

    # Compute arc length at final theta
    def arc_length(phi):
        return (b / 2) * (phi * np.sqrt(1 + phi**2) + np.arcsinh(phi))

    phi_max = float(t_end)
    s_max = arc_length(phi_max)

    # Create arc-length steps
    s_vals = np.arange(0, s_max, pitch)

    # Compute φ from arc length using Lambert W approx
    lambert_arg = 0.5 * np.exp(2 * s_vals / b - 1)
    phi_vals = np.sqrt(lambertw(lambert_arg).real) / np.sqrt(2)

    # Compute r = b * φ
    r_vals = b * phi_vals
    x_vals = r_vals * np.cos(phi_vals) + float(center_x)
    y_vals = r_vals * np.sin(phi_vals) + float(center_y)

    return np.column_stack((x_vals, y_vals))
