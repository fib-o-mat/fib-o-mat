"""Provide rasterization routines."""
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np

import sympy
from sympy.abc import t
from sympy import sin, cos

from fibomat.shapes.arc import Arc
from fibomat.shapes.line import Line
from fibomat.shapes.polygon import Polygon
from fibomat.shapes.polyline import Polyline
from fibomat.shapes.rasterizedpoints import RasterizedPoints
from fibomat.shapes.shape import Shape
from fibomat.shapes.arc_spline import ArcSplineCompatible, ArcSpline
from fibomat.shapes.hollow_arc_spline import HollowArcSpline
from fibomat.linalg import Vector, translate, rotate, VectorLike, BoundingBox
from fibomat.curve_tools.intersections import curve_intersections
from fibomat.shapes._line_non_continuous import LineNonContinuous
from fibomat import _libfibomat


def _rasterize_arc_spline_non_continuous_curve(
    curve: Union[ArcSpline, LineNonContinuous], pitch: float
) -> RasterizedPoints:
    # pylint: disable=invalid-name,too-many-locals

    pitch = float(pitch)
    # n_points = int(curve.length / pitch) + 1

    points = []

    i_points = 0
    offset = 0.
    for segment in curve.segments:
        if isinstance(segment, Arc):
            arc: Arc = segment

            offset_angle = offset / arc.radius
            pitch_angle = pitch / arc.radius

            if offset_angle <= arc.theta:
                points_on_arc = int((arc.theta - offset_angle) / pitch_angle) + 1

                if np.isclose((arc.theta - offset_angle) / pitch_angle - points_on_arc - 1, 0.):
                    points_on_arc += 1

                if not arc.sweep_dir:
                    pitch_angle *= -1.
                    arc_start_angle = arc.start_angle - offset_angle
                else:
                    arc_start_angle = arc.start_angle + offset_angle

                center = np.array(arc.center)
                theta = pitch_angle*np.arange(points_on_arc)

                points.append(
                    center + arc.radius * np.column_stack(
                        (np.cos(theta + arc_start_angle), np.sin(theta + arc_start_angle))
                    )
                )

                i_points += points_on_arc
                offset = pitch - (arc.length - offset - (len(theta)-1) * pitch)
            else:
                offset -= arc.length

        elif isinstance(segment, Line):
            line: Line = segment
            if offset <= line.length:
                direction = (line.end - line.start).normalized()

                start = line.start + direction * offset

                points_on_line = int((line.length-offset) / pitch) + 1

                direction = np.array(direction)
                start = np.array(start)

                t = pitch * np.arange(points_on_line)
                # points[i_points:i_points+points_on_line, :2] = start + np.repeat(t[None, :], 2, axis=0).T * direction
                points.append(start + np.repeat(t[None, :], 2, axis=0).T * direction)
                # points[i_points:i_points+points_on_line, 2] = 1.

                i_points += points_on_line
                offset = pitch - (line.length - t[-1] - offset)
            else:
                offset -= line.length
        else:
            raise RuntimeError(f'Cannot rasterize segment type {segment.__class__.__name__}')

    # assert i_points == n_points
    dwell_points = np.ones(shape=(i_points, 3))
    dwell_points[:, :2] = np.concatenate(points)

    return RasterizedPoints(dwell_points, curve.is_closed)


def rasterize(curve: Shape, pitch: float) -> RasterizedPoints:
    """Rasterize the outline of a Shape with a given pitch uniformly.

    For this, the shape must be convertible to an ArcSpline or must have a `rasterize` method expecting the pitch as
    input.

    Args:
        curve (Shape): curve
        pitch (float): pitch (spacing of points)

    Returns:
        RasterizedPoints

    Raises:
        ValueError: Raised if curve is no ArcSpline, ArcSplineCompatible or not have a `rasterize` method.
    """
    if isinstance(curve, ArcSpline) or isinstance(curve, LineNonContinuous):
        return _rasterize_arc_spline_non_continuous_curve(curve, pitch)

    if raster_method := getattr(curve, 'rasterize', None):
        return raster_method(curve, pitch)

    if isinstance(curve, ArcSplineCompatible):
        return _rasterize_arc_spline_non_continuous_curve(curve.to_arc_spline(), pitch)

    raise ValueError(f'Cannot rasterize the passed object of type {curve.__class__}.')


def rasterize_with_const_error(curve: ArcSpline, error: float) -> Union[Polyline, Polygon]:
    """Convert a arc spline to polygon or polyline.

    Args:
        curve: arc spline
        error: maximum distance between curve an returned polyine/polygon

    Returns:
        Polygon if arc spline is closed and polyline otherwise
    """

    approx = ArcSpline(_libfibomat.convert_arcs_to_lines(curve.arc_spline_impl, error))

    if curve.is_closed:
        return Polygon(approx.vertices[:, :2])
    else:
        return Polyline(approx.vertices[:, :2])


def _make_line(
    curve: ArcSpline, intersection_interval: Tuple[Dict[str, Any], Dict[str, Any]], holes: List[ArcSpline]
) -> Optional[Line]:
    start = Vector(intersection_interval[0]['pos'])
    end = Vector(intersection_interval[1]['pos'])

    assert np.isclose(start.y, end.y)
    assert end.x > start.x

    midpoint = Vector((start.x + end.x) / 2, start.y)

    if curve.contains(midpoint):
        for hole in holes:
            if hole.contains(midpoint):
                return
        return Line(start, end)


def fill_with_lines(
    shape: Union[ArcSpline, HollowArcSpline],
    pitch: float,
    alpha: float,
    invert: bool,
    seed: Optional[VectorLike] = None
) -> List[List[Line]]:
    """Fill a closed shape with lines which are rotated by `alpha`-

    Args:
        shape (Union[ArcSpline, HollowArcSpline]): closed curve to be filled
        pitch (float): distance between lines
        alpha (float): rotation angle of lines with respect to x-axis
        invert (bool):


    Returns:
        List[ArcSpline]

    Raises:
        ValueError: Raised if angle < 0 or angle > pi.
        NotImplementedError: Raised if some kind of singularities occur. This can be fixed (probably) if the shape is
                             rotated.
    """
    # pylint: disable=invalid-name,too-many-locals

    # TODO: cache spatial tree for intersection calculations

    if isinstance(shape, ArcSpline):
        curve = shape
        if not curve.is_closed:
            raise ValueError('ArcSpline is not closed.')
        holes = []
    elif isinstance(shape, HollowArcSpline):
        curve = shape.boundary
        holes = shape.holes
    else:
        raise TypeError(f'Shape must be ArcSpline or HollowArcSpline (got {type(shape)}).')

    if not (-np.pi/2 <= alpha <= np.pi/2):
        raise ValueError('alpha < -pi/2 or alpha > pi/2')

    seed = (Vector(seed) - curve.center).rotated(-alpha) if seed is not None else Vector()
    # seed = Vector(seed) if seed is not None else Vector()

    center = curve.center

    curve = curve.transformed(
        translate(-center) | rotate(-alpha)
    )

    holes = list(map(lambda hole: hole.transformed(translate(-center) | rotate(-alpha)), holes))

    extend = 2 * pitch  # 2 * pitch
    bbox_curve = curve.bounding_box
    bbox = BoundingBox(bbox_curve.lower_left - (extend, extend), bbox_curve.upper_right + (extend, extend))

    bbox_left = bbox.lower_left.x
    bbox_right = bbox.upper_right.x
    bbox_top = pitch * (np.fix(bbox.upper_right.y / pitch)) + seed.y - pitch * np.fix(seed.y / pitch) # pitch * np.fix((bbox.upper_right.y - seed.y) / pitch)
    bbox_bottom = bbox_top - bbox.height # pitch * np.fix((bbox.center.y - bbox.height / 2) / pitch)

    height = np.abs(bbox_top - bbox_bottom)

    y = np.arange(-height / 2, height / 2, pitch)

    ny = len(y)

    if invert:
        intersection_line_base = ArcSpline([(bbox_left, bbox_bottom, 0), (bbox_right, bbox_bottom, 0)], False)
        pitch = -pitch
    else:
        intersection_line_base = ArcSpline([(bbox_left, bbox_top, 0), (bbox_right, bbox_top, 0)], False)

    fill_lines: List[List[Line]] = []

    for i_y in range(ny):
        intersection_line = intersection_line_base.translated((0, -i_y*pitch))

        intersections = curve_intersections(curve, intersection_line)['intersections']

        # ignore coincidences for now
        # # TODO: handle case, if cl_inter contains coincidence
        # assert len(cl_intersections['coincidences']) == 0

        if len(intersections) > 1:
            for hole in holes:
                intersections.extend(curve_intersections(hole, intersection_line)['intersections'])

            intersections_sorted = sorted(
                intersections,
                key=lambda intersection: intersection['pos'][0]
            )

            fill_line = []

            for intersection_interval in zip(intersections_sorted, intersections_sorted[1:]):
                if part_fill_line := _make_line(curve, intersection_interval, holes):
                    fill_line.append(
                        part_fill_line.transformed(
                            translate(center) | rotate(alpha, origin=center)
                        )
                    )

            if fill_line:
                fill_lines.append(fill_line)

    return fill_lines


def fill_with_spiral(shape: Union[ArcSpline, HollowArcSpline], pitch: float) -> List[Arc]:
    from fibomat.shapes import ParametricCurve  #TODO fix this (throws circular import if imported globally)
    from fibomat import curve_tools
    """Fill the circumcircle of the bounding box with an Archimedean spiral.

    Args:
        shape (Union[ArcSpline, HollowArcSpline]): closed curve to be filled
        pitch (float): distance between spiral arms


    Returns:
        ParametricCurve: Spiral filling the circumscribed circle of the bounding box

        
    """
    if isinstance(shape, ArcSpline):
        curve = shape
        if not curve.is_closed:
            raise ValueError('ArcSpline is not closed.')
    elif isinstance(shape, HollowArcSpline):
        curve = shape.boundary  # TODO include holes
    else:
        raise TypeError(f'Shape must be ArcSpline or HollowArcSpline (got {type(shape)}).')
    

    center = curve.center 
    bbox = curve.bounding_box
    radius = np.linalg.norm(bbox.upper_right - bbox.center)  # radius circumscribed circle
    
    theta_max = radius/pitch + 2*np.pi

    curve = sympy.Curve(
    [t*pitch*cos(t)+center[0], t*pitch*sin(t)+center[1]],
    (t, 0, theta_max)
    )


    parametric_curve = ParametricCurve.from_sympy_curve(curve, try_length_integration=False)
    return parametric_curve

