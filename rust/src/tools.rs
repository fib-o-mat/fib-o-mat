//! Curve algorithms: intersections, boolean operations and offsetting.

use cavalier_contours::core::Control;
use cavalier_contours::polyline::{BooleanOp, PlineIntersect, PlineOrientation, PlineSource, PlineSourceMut, Polyline};
use cavalier_contours::shape_algorithms::{Shape, ShapeOffsetOptions};
use pyo3::prelude::*;

use crate::arc_spline::ArcSpline;
use crate::error::runtime_error;

type Point = (f64, f64);
type Intersection = (usize, usize, Point);
type Coincidence = (usize, usize, Point, Point);

/// Intersections of a curve with itself: `(segment index 1, segment index 2, point)`.
///
/// Overlapping segments are not reported.
pub fn self_intersections_impl(pline: &Polyline<f64>) -> Vec<Intersection> {
    let mut intersections = Vec::new();
    if pline.vertex_count() < 2 {
        return intersections;
    }
    pline.visit_self_intersects(&mut |intersect: PlineIntersect<f64>| {
        if let PlineIntersect::Basic(basic) = intersect {
            intersections.push((basic.start_index1, basic.start_index2, (basic.point.x, basic.point.y)));
        }
        Control::<()>::Continue
    });
    intersections
}

/// Intersections of two curves: crossing points and coincident (overlapping) parts.
pub fn curve_intersections_impl(pline1: &Polyline<f64>, pline2: &Polyline<f64>) -> (Vec<Intersection>, Vec<Coincidence>) {
    if pline1.vertex_count() < 2 || pline2.vertex_count() < 2 {
        return (Vec::new(), Vec::new());
    }
    let found = pline1.find_intersects(pline2);
    let intersections = found
        .basic_intersects
        .iter()
        .map(|i| (i.start_index1, i.start_index2, (i.point.x, i.point.y)))
        .collect();
    let coincidences = found
        .overlapping_intersects
        .iter()
        .map(|i| (i.start_index1, i.start_index2, (i.point1.x, i.point1.y), (i.point2.x, i.point2.y)))
        .collect();
    (intersections, coincidences)
}

/// Boolean operation of two closed curves. Returns `(remaining, subtracted)` curves.
pub fn combine_curves_impl(
    pline1: &Polyline<f64>,
    pline2: &Polyline<f64>,
    mode: &str,
) -> Result<(Vec<Polyline<f64>>, Vec<Polyline<f64>>), String> {
    if !pline1.is_closed() || !pline2.is_closed() {
        return Err("Only closed curves can be combined.".into());
    }
    let operation = match mode {
        "union" => BooleanOp::Or,
        "xor" => BooleanOp::Xor,
        "exclude" => BooleanOp::Not,
        "intersect" => BooleanOp::And,
        _ => return Err("Unknown combining mode.".into()),
    };
    let result = pline1.boolean(pline2, operation);
    Ok((
        result.pos_plines.into_iter().map(|p| p.pline).collect(),
        result.neg_plines.into_iter().map(|p| p.pline).collect(),
    ))
}

/// Offset of islands (clockwise curves) and an optional outer curve (counter-clockwise).
///
/// The input curves are oriented as needed. Returns `(islands, outer curves)`.
pub fn offset_with_islands_impl(
    islands: Vec<Polyline<f64>>,
    outer: Option<Polyline<f64>>,
    delta: f64,
) -> (Vec<Polyline<f64>>, Vec<Polyline<f64>>) {
    let orient = |mut pline: Polyline<f64>, orientation: PlineOrientation| {
        if pline.orientation() != orientation {
            pline.invert_direction_mut();
        }
        pline
    };
    let plines = islands
        .into_iter()
        .map(|p| orient(p, PlineOrientation::Clockwise))
        .chain(outer.into_iter().map(|p| orient(p, PlineOrientation::CounterClockwise)));
    let shape = Shape::from_plines(plines).parallel_offset(delta, &ShapeOffsetOptions::default());
    (
        shape.cw_plines.into_iter().map(|p| p.polyline).collect(),
        shape.ccw_plines.into_iter().map(|p| p.polyline).collect(),
    )
}

fn wrap(plines: Vec<Polyline<f64>>) -> Vec<ArcSpline> {
    plines.into_iter().map(ArcSpline::from_pline).collect()
}

/// Self intersections of a curve: list of (segment index 1, segment index 2, (x, y)).
#[pyfunction]
pub fn self_intersections(py: Python<'_>, curve: PyRef<'_, ArcSpline>) -> Vec<Intersection> {
    let pline = curve.pline.clone();
    py.detach(move || self_intersections_impl(&pline))
}

/// Intersections of two curves: (intersections, coincident segments).
#[pyfunction]
pub fn curve_intersections(
    py: Python<'_>,
    curve_1: PyRef<'_, ArcSpline>,
    curve_2: PyRef<'_, ArcSpline>,
) -> (Vec<Intersection>, Vec<Coincidence>) {
    let (p1, p2) = (curve_1.pline.clone(), curve_2.pline.clone());
    py.detach(move || curve_intersections_impl(&p1, &p2))
}

/// Combine two closed curves. `mode` is one of 'union', 'xor', 'exclude' or 'intersect'.
/// Returns (remaining curves, subtracted curves).
#[pyfunction]
pub fn combine_curves(
    py: Python<'_>,
    curve_1: PyRef<'_, ArcSpline>,
    curve_2: PyRef<'_, ArcSpline>,
    mode: &str,
) -> PyResult<(Vec<ArcSpline>, Vec<ArcSpline>)> {
    let (p1, p2, mode) = (curve_1.pline.clone(), curve_2.pline.clone(), mode.to_owned());
    let (remaining, subtracted) = py
        .detach(move || combine_curves_impl(&p1, &p2, &mode))
        .map_err(runtime_error)?;
    Ok((wrap(remaining), wrap(subtracted)))
}

/// Parallel offset of a curve by `delta`.
#[pyfunction]
pub fn offset_curve(py: Python<'_>, curves: PyRef<'_, ArcSpline>, delta: f64) -> Vec<ArcSpline> {
    let pline = curves.pline.clone();
    wrap(py.detach(move || pline.parallel_offset(delta)))
}

/// Parallel offset of islands and an optional outer curve. Returns (islands, outer curves).
#[pyfunction]
pub fn offset_with_islands(
    py: Python<'_>,
    islands: Vec<PyRef<'_, ArcSpline>>,
    outer_curve: Option<PyRef<'_, ArcSpline>>,
    delta: f64,
) -> PyResult<(Vec<ArcSpline>, Vec<ArcSpline>)> {
    let islands: Vec<_> = islands.iter().map(|c| c.pline.clone()).collect();
    let outer = outer_curve.map(|c| c.pline.clone());
    if islands.iter().chain(outer.iter()).any(|p| !p.is_closed()) {
        return Err(runtime_error("Only closed curves can be offset with islands."));
    }
    let (res_islands, res_outer) = py.detach(move || offset_with_islands_impl(islands, outer, delta));
    Ok((wrap(res_islands), wrap(res_outer)))
}

/// Approximate all arcs of a curve by lines with a maximum distance of `error` to the arc.
#[pyfunction]
pub fn convert_arcs_to_lines(curve: PyRef<'_, ArcSpline>, error: f64) -> PyResult<ArcSpline> {
    curve
        .pline
        .arcs_to_approx_lines(error)
        .map(ArcSpline::from_pline)
        .ok_or_else(|| runtime_error("Could not convert arcs to lines."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cavalier_contours::polyline::PlineSource;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Polyline<f64> {
        let mut p = Polyline::new_closed();
        for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
            p.add(x, y, 0.);
        }
        p
    }

    fn line(x0: f64, y0: f64, x1: f64, y1: f64) -> Polyline<f64> {
        let mut p = Polyline::new();
        p.add(x0, y0, 0.);
        p.add(x1, y1, 0.);
        p
    }

    #[test]
    fn crossing_lines_intersect_once() {
        let (basic, coincident) = curve_intersections_impl(&line(-1., 0., 1., 0.), &line(0., -1., 0., 1.));
        assert_eq!(basic.len(), 1);
        assert!(coincident.is_empty());
        assert!(basic[0].2 .0.abs() < 1e-9 && basic[0].2 .1.abs() < 1e-9);
    }

    #[test]
    fn overlapping_lines_are_coincident() {
        let (basic, coincident) = curve_intersections_impl(&line(0., 0., 2., 0.), &line(1., 0., 3., 0.));
        assert!(basic.is_empty());
        assert_eq!(coincident.len(), 1);
    }

    #[test]
    fn short_curves_have_no_intersections() {
        let mut single = Polyline::new();
        single.add(0., 0., 0.);
        assert_eq!(curve_intersections_impl(&single, &line(0., 0., 1., 0.)), (vec![], vec![]));
        assert!(self_intersections_impl(&single).is_empty());
    }

    #[test]
    fn bow_tie_has_one_self_intersection() {
        let mut p = Polyline::new_closed();
        for (x, y) in [(0., 0.), (2., 2.), (2., 0.), (0., 2.)] {
            p.add(x, y, 0.);
        }
        let intersections = self_intersections_impl(&p);
        assert_eq!(intersections.len(), 1);
        assert!((intersections[0].2 .0 - 1.).abs() < 1e-9 && (intersections[0].2 .1 - 1.).abs() < 1e-9);
    }

    #[test]
    fn square_has_no_self_intersection() {
        assert!(self_intersections_impl(&rect(0., 0., 1., 1.)).is_empty());
    }

    #[test]
    fn combine_modes() {
        let (a, b) = (rect(0., 0., 2., 2.), rect(1., 1., 3., 3.));
        let area = |plines: &[Polyline<f64>]| plines.iter().map(|p| p.area().abs()).sum::<f64>();

        let (union, sub) = combine_curves_impl(&a, &b, "union").unwrap();
        assert!((area(&union) - 7.).abs() < 1e-9);
        assert!(sub.is_empty());

        let (inter, _) = combine_curves_impl(&a, &b, "intersect").unwrap();
        assert!((area(&inter) - 1.).abs() < 1e-9);

        let (excl, _) = combine_curves_impl(&a, &b, "exclude").unwrap();
        assert!((area(&excl) - 3.).abs() < 1e-9);

        let (xor, _) = combine_curves_impl(&a, &b, "xor").unwrap();
        assert!((area(&xor) - 6.).abs() < 1e-9);
    }

    #[test]
    fn combine_errors() {
        let (a, b) = (rect(0., 0., 1., 1.), rect(0., 0., 1., 1.));
        assert!(combine_curves_impl(&a, &b, "bogus").is_err());
        assert!(combine_curves_impl(&a, &line(0., 0., 1., 1.), "union").is_err());
    }

    #[test]
    fn offset_shrinks_square() {
        // positive offsets move to the left of the direction of the curve, i.e. inwards for counter-clockwise curves
        let result = rect(0., 0., 4., 4.).parallel_offset(1.);
        assert_eq!(result.len(), 1);
        assert!((result[0].area() - 4.).abs() < 1e-9);
        assert!(rect(0., 0., 4., 4.).parallel_offset(3.).is_empty());
    }

    #[test]
    fn offset_island_in_outer() {
        // 10x10 outer with a 2x2 hole; shrinking by 1 -> 8x8 outer and a hole with rounded corners (2*2 + 4*2*1 + pi)
        let outer = rect(0., 0., 10., 10.);
        let island = rect(4., 4., 6., 6.);
        let (islands, outers) = offset_with_islands_impl(vec![island], Some(outer), 1.);
        assert_eq!(outers.len(), 1);
        assert_eq!(islands.len(), 1);
        assert!((outers[0].area() - 64.).abs() < 1e-6);
        assert!((islands[0].area().abs() - (12. + std::f64::consts::PI)).abs() < 1e-6);
    }

    #[test]
    fn offset_orients_input() {
        // a counter-clockwise island is handled as hole
        let outer = rect(0., 0., 10., 10.);
        let island = rect(4., 4., 6., 6.);
        assert!(island.area() > 0.);
        let (islands, _) = offset_with_islands_impl(vec![island], Some(outer), 1.);
        assert!(islands[0].area() < 0.);
    }

    #[test]
    fn arcs_to_lines_have_no_bulge() {
        let mut p = Polyline::new_closed();
        p.add(1., 0., 1.);
        p.add(-1., 0., 1.);
        let lines = p.arcs_to_approx_lines(0.01).unwrap();
        assert!(lines.vertex_count() > 8);
        assert!(lines.iter_vertexes().all(|v| v.bulge == 0.));
    }
}
