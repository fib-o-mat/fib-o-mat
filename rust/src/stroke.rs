//! Outlines of strokes (polylines with a width), e.g. for the glyphs of text.
//!
//! The strokes of all paths are united, so that overlapping strokes (e.g. crossing strokes of a glyph) give one
//! polygon. Closed paths result in polygons with holes.

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::mesh::float::stroke::offset::StrokeOffset;
use i_overlay::mesh::float::style::{LineCap, LineJoin, StrokeStyle};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::f64::consts::PI;

use crate::error::runtime_error;

type Point = [f64; 2];
/// Contours of a polygon: the outer contour (counterclockwise) followed by the holes (clockwise).
type Polygon = Vec<Vec<Point>>;

/// Angle step of arcs (joins and caps): the maximal segment length over the radius.
const ARC_STEP: f64 = 0.05 * PI;
/// Miter joins with a smaller interior angle are cut (bevel).
const MITER_MIN_ANGLE: f64 = PI / 12.;

/// Parse the name of a line join.
fn parse_join(name: &str) -> Result<LineJoin<f64>, String> {
    match name {
        "bevel" => Ok(LineJoin::Bevel),
        "miter" => Ok(LineJoin::Miter(MITER_MIN_ANGLE)),
        "round" => Ok(LineJoin::Round(ARC_STEP)),
        _ => Err(format!("Unknown join '{name}', expected 'bevel', 'miter' or 'round'.")),
    }
}

/// Parse the name of a line cap.
fn parse_cap(name: &str) -> Result<LineCap<Point>, String> {
    match name {
        "butt" => Ok(LineCap::Butt),
        "square" => Ok(LineCap::Square),
        "round" => Ok(LineCap::Round(ARC_STEP)),
        _ => Err(format!("Unknown cap '{name}', expected 'butt', 'square' or 'round'.")),
    }
}

/// Union of the strokes of all paths.
///
/// Every path is a list of points together with a flag whether the path is closed. Returns polygons (outer contour
/// counterclockwise, holes clockwise).
pub fn stroke_paths_impl(
    paths: &[(Vec<Point>, bool)],
    width: f64,
    join: &str,
    cap: &str,
) -> Result<Vec<Polygon>, String> {
    if !width.is_finite() || width <= 0. {
        return Err("width must be positive and finite.".into());
    }
    let join = parse_join(join)?;
    let cap = parse_cap(cap)?;

    let mut contours: Vec<Vec<Point>> = Vec::new();
    for (points, is_closed) in paths {
        if points.len() < 2 {
            return Err("Every path needs at least two points.".into());
        }
        if points.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
            return Err("The points of the paths must be finite.".into());
        }

        let style = StrokeStyle::new(width)
            .line_join(join.clone())
            .start_cap(cap.clone())
            .end_cap(cap.clone());
        for polygon in points.stroke_as::<i64>(style, *is_closed) {
            contours.extend(polygon);
        }
    }

    if paths.len() <= 1 {
        // (a single stroke needs no union, it is simple already)
        return Ok(group_contours(contours));
    }

    // united with the positive fill rule: the outer contours are counterclockwise (winding number +1) and the holes
    // are clockwise (-1), so overlapping strokes are merged and holes are kept (unless covered by other strokes).
    let empty: Vec<Vec<Point>> = Vec::new();
    Ok(contours.overlay_as::<i64>(&empty, OverlayRule::Subject, FillRule::Positive))
}

/// Group contours of a single stroke (the first contour is the outer one, the others are holes).
fn group_contours(contours: Vec<Vec<Point>>) -> Vec<Polygon> {
    if contours.is_empty() {
        Vec::new()
    } else {
        vec![contours]
    }
}

fn to_array<'py>(py: Python<'py>, points: &[Point]) -> Bound<'py, PyArray2<f64>> {
    let data: Vec<f64> = points.iter().flat_map(|p| [p[0], p[1]]).collect();
    Array2::from_shape_vec((points.len(), 2), data)
        .expect("shape matches the data")
        .into_pyarray(py)
}

/// Outline of the united strokes of several paths.
///
/// Args:
///     paths: list of arrays of shape (n, 2), each path has at least two points
///     closed: list with a flag for each path whether the path is closed (the last point is connected to the first
///         one; the first point must not be repeated at the end)
///     width: width of the strokes
///     join: 'bevel', 'miter' or 'round'
///     cap: 'butt', 'square' or 'round' (only for open paths)
///
/// Returns:
///     list of polygons, each polygon is a list of contours (arrays of shape (n, 2)): the outer contour
///     (counterclockwise) followed by the holes (clockwise)
#[pyfunction]
#[pyo3(signature = (paths, closed, width, join="bevel", cap="butt"))]
pub fn stroke_paths<'py>(
    py: Python<'py>,
    paths: Vec<PyReadonlyArray2<'py, f64>>,
    closed: Vec<bool>,
    width: f64,
    join: &str,
    cap: &str,
) -> PyResult<Vec<Vec<Bound<'py, PyArray2<f64>>>>> {
    if paths.len() != closed.len() {
        return Err(runtime_error("paths and closed must have the same length."));
    }

    let mut native_paths = Vec::with_capacity(paths.len());
    for (path, is_closed) in paths.iter().zip(closed) {
        let view = path.as_array();
        if view.ncols() != 2 {
            return Err(runtime_error("Every path must be an array of shape (n, 2)."));
        }
        native_paths.push((view.rows().into_iter().map(|row| [row[0], row[1]]).collect::<Vec<_>>(), is_closed));
    }
    let (join, cap) = (join.to_owned(), cap.to_owned());

    let polygons = py
        .detach(move || stroke_paths_impl(&native_paths, width, &join, &cap))
        .map_err(runtime_error)?;

    Ok(polygons
        .iter()
        .map(|polygon| polygon.iter().map(|contour| to_array(py, contour)).collect())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(contour: &[Point]) -> f64 {
        let n = contour.len();
        (0..n)
            .map(|i| contour[i][0] * contour[(i + 1) % n][1] - contour[(i + 1) % n][0] * contour[i][1])
            .sum::<f64>()
            / 2.
    }

    fn polygon_area(polygon: &Polygon) -> f64 {
        polygon.iter().map(|contour| area(contour)).sum()
    }

    #[test]
    fn straight_line() {
        let polygons = stroke_paths_impl(&[(vec![[0., 0.], [10., 0.]], false)], 2., "bevel", "butt").unwrap();
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].len(), 1);
        assert!((polygon_area(&polygons[0]) - 20.).abs() < 1e-6);
    }

    #[test]
    fn caps() {
        let path = [(vec![[0., 0.], [10., 0.]], false)];
        let square = stroke_paths_impl(&path, 2., "bevel", "square").unwrap();
        assert!((polygon_area(&square[0]) - 24.).abs() < 1e-6);
        let round = stroke_paths_impl(&path, 2., "bevel", "round").unwrap();
        assert!((polygon_area(&round[0]) - (20. + PI)).abs() < 0.05);
    }

    #[test]
    fn closed_path_has_a_hole() {
        let square = vec![[0., 0.], [10., 0.], [10., 10.], [0., 10.]];
        let polygons = stroke_paths_impl(&[(square, true)], 2., "miter", "butt").unwrap();
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].len(), 2);
        assert!(area(&polygons[0][0]) > 0. && area(&polygons[0][1]) < 0.);
        // outer 12 x 12 minus inner 8 x 8
        assert!((polygon_area(&polygons[0]) - (144. - 64.)).abs() < 1e-6);
    }

    #[test]
    fn crossing_strokes_are_united() {
        let paths = [(vec![[-5., 0.], [5., 0.]], false), (vec![[0., -5.], [0., 5.]], false)];
        let polygons = stroke_paths_impl(&paths, 2., "bevel", "butt").unwrap();
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].len(), 1);
        // two 10 x 2 strokes minus the crossing area
        assert!((polygon_area(&polygons[0]) - 36.).abs() < 1e-6);
    }

    #[test]
    fn sharp_curve_with_a_small_radius_has_no_artefacts() {
        // a hook like the J: the stroke is wider than the radius of the curve
        let hook = vec![[1., 1.], [1., 0.25], [0.8, 0.05], [0.5, 0.], [0.2, 0.05], [0.05, 0.25]];
        let polygons = stroke_paths_impl(&[(hook, false)], 0.5, "bevel", "butt").unwrap();
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].len(), 1);
        assert!(area(&polygons[0][0]) > 0.);
    }

    #[test]
    fn invalid_arguments() {
        let path = [(vec![[0., 0.], [1., 0.]], false)];
        assert!(stroke_paths_impl(&path, 0., "bevel", "butt").is_err());
        assert!(stroke_paths_impl(&path, f64::NAN, "bevel", "butt").is_err());
        assert!(stroke_paths_impl(&path, 1., "foo", "butt").is_err());
        assert!(stroke_paths_impl(&path, 1., "bevel", "foo").is_err());
        assert!(stroke_paths_impl(&[(vec![[0., 0.]], false)], 1., "bevel", "butt").is_err());
        assert!(stroke_paths_impl(&[(vec![[0., 0.], [f64::NAN, 1.]], false)], 1., "bevel", "butt").is_err());
    }
}
