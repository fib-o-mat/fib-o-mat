//! The `ArcSpline` python class: a polyline of line and arc segments.
//!
//! The curve is stored as a list of vertices `(x, y, bulge)`. The bulge of a vertex describes the segment from this
//! vertex to the next one: `bulge = tan(angle / 4)` where `angle` is the arc's sweep angle (0 for lines, positive for
//! counter-clockwise arcs). If the curve is closed, the last vertex is connected to the first one.

use cavalier_contours::core::math::Vector2;
use cavalier_contours::polyline::{
    PlineOrientation, PlineSource, PlineSourceMut, PlineVertex, Polyline,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::error::runtime_error;

/// Fuzzy comparison epsilon for positions, equal to the default of the cavalier_contours algorithms.
const POS_EQUAL_EPS: f64 = 1e-5;

type Vertex = (f64, f64, f64);

/// A curve consisting of line and arc segments.
#[pyclass(name = "ArcSpline", module = "fibomat._libfibomat", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct ArcSpline {
    pub(crate) pline: Polyline<f64>,
}

impl ArcSpline {
    /// Wrap an existing polyline.
    pub(crate) fn from_pline(pline: Polyline<f64>) -> Self {
        Self { pline }
    }

    /// Create a curve from `(x, y, bulge)` vertices.
    pub fn from_vertices<I: IntoIterator<Item = Vertex>>(vertices: I, is_closed: bool) -> Self {
        let mut pline = Polyline::new();
        for (x, y, bulge) in vertices {
            pline.add(x, y, bulge);
        }
        pline.set_is_closed(is_closed);
        Self { pline }
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.pline.vertex_count()
    }

    /// True if the curve has no vertices.
    pub fn is_empty(&self) -> bool {
        self.pline.is_empty()
    }

    /// Reflect the curve on the line through the origin spanned by `axis`.
    pub fn mirror(&mut self, axis: (f64, f64)) -> Result<(), String> {
        let norm2 = axis.0 * axis.0 + axis.1 * axis.1;
        if norm2.sqrt() < POS_EQUAL_EPS {
            return Err("mirror axis may not be the null vector".into());
        }
        let a = (axis.0 * axis.0 - axis.1 * axis.1) / norm2;
        let b = 2. * axis.0 * axis.1 / norm2;
        for i in 0..self.pline.vertex_count() {
            let v = self.pline.at(i);
            // a reflection reverts the orientation, hence the bulge changes its sign
            self.pline
                .set_vertex(i, PlineVertex::new(a * v.x + b * v.y, b * v.x - a * v.y, -v.bulge));
        }
        Ok(())
    }

    /// Rotate the curve around the origin. A positive angle rotates counter-clockwise.
    pub fn rotate(&mut self, angle: f64) {
        let (sin, cos) = angle.sin_cos();
        for i in 0..self.pline.vertex_count() {
            let v = self.pline.at(i);
            self.pline
                .set_vertex(i, PlineVertex::new(cos * v.x - sin * v.y, sin * v.x + cos * v.y, v.bulge));
        }
    }

    /// Bounding box `((x_min, y_min), (x_max, y_max))` of the curve.
    pub fn bounding_box(&self) -> Result<((f64, f64), (f64, f64)), String> {
        if self.is_empty() {
            return Err("An empty curve has no bounding box.".into());
        }
        // `extents` covers the arcs, the vertices cover single vertex curves
        let first = self.pline.at(0);
        let mut min = (first.x, first.y);
        let mut max = min;
        if let Some(extents) = self.pline.extents() {
            min = (extents.min_x, extents.min_y);
            max = (extents.max_x, extents.max_y);
        }
        for v in self.pline.iter_vertexes() {
            min = (min.0.min(v.x), min.1.min(v.y));
            max = (max.0.max(v.x), max.1.max(v.y));
        }
        Ok((min, max))
    }

    /// Center of the bounding box.
    pub fn center(&self) -> Result<(f64, f64), String> {
        let ((x_min, y_min), (x_max, y_max)) = self.bounding_box().map_err(|_| "An empty curve has no center.".to_string())?;
        Ok(((x_min + x_max) / 2., (y_min + y_max) / 2.))
    }

    /// True if the (closed) curve is oriented counter-clockwise.
    pub fn orientation(&self) -> Result<bool, String> {
        if !self.pline.is_closed() {
            return Err("Cannot determine orientation if curve is not closed.".into());
        }
        if self.len() < 2 {
            return Err("Cannot determine orientation if curve has less than 2 points.".into());
        }
        Ok(self.pline.orientation() == PlineOrientation::CounterClockwise)
    }

    /// Check if the point lies in the area enclosed by the (closed) curve.
    pub fn contains(&self, x: f64, y: f64) -> Result<bool, String> {
        if !self.pline.is_closed() {
            return Err("Curve is not closed, hence it cannot be checked if it contains something.".into());
        }
        Ok(self.pline.winding_number(Vector2::new(x, y)) != 0)
    }

    /// Closest point on the curve: `(segment index, point, distance)`.
    pub fn closest_point(&self, x: f64, y: f64) -> Result<(usize, (f64, f64), f64), String> {
        let res = self
            .pline
            .closest_point(Vector2::new(x, y), POS_EQUAL_EPS)
            .ok_or_else(|| "An empty curve has no closest point.".to_string())?;
        Ok((res.seg_start_index, (res.seg_point.x, res.seg_point.y), res.distance))
    }

    /// Vertices as `(x, y, bulge)` tuples.
    pub fn vertices(&self) -> Vec<Vertex> {
        self.pline.iter_vertexes().map(|v| (v.x, v.y, v.bulge)).collect()
    }
}

/// Convert a python object (numpy array of shape (n, 3) or a sequence of `(x, y, bulge)`) to vertices.
fn extract_vertices(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vertex>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray2<f64>>() {
        let view = array.as_array();
        if view.shape()[1] != 3 {
            return Err(runtime_error("curve_points must have 3 elements second axis."));
        }
        return Ok(view.rows().into_iter().map(|row| (row[0], row[1], row[2])).collect());
    }
    if obj.hasattr("shape")? {
        // some other array-like which is not a 2d float array
        return Err(runtime_error("curve_points must be a 2d array."));
    }
    let rows: Vec<Vec<f64>> = obj
        .extract()
        .map_err(|_| runtime_error("curve_points must be a 2d array."))?;
    rows.into_iter()
        .map(|row| match row.as_slice() {
            [x, y, bulge] => Ok((*x, *y, *bulge)),
            _ => Err(runtime_error("curve_points must have 3 elements second axis.")),
        })
        .collect()
}

#[pymethods]
impl ArcSpline {
    /// ArcSpline(other) or ArcSpline(vertices, is_closed)
    #[new]
    #[pyo3(signature = (vertices, is_closed=None))]
    fn py_new(vertices: &Bound<'_, PyAny>, is_closed: Option<bool>) -> PyResult<Self> {
        if let Ok(other) = vertices.extract::<PyRef<'_, ArcSpline>>() {
            return Ok(other.clone());
        }
        let is_closed =
            is_closed.ok_or_else(|| runtime_error("is_closed must be given if the curve is constructed from vertices."))?;
        Ok(Self::from_vertices(extract_vertices(vertices)?, is_closed))
    }

    /// Create a copy of the curve.
    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    /// Number of vertices.
    #[getter]
    fn size(&self) -> usize {
        self.len()
    }

    /// True if the curve is closed, i.e. the last vertex is connected to the first one.
    #[getter]
    fn is_closed(&self) -> bool {
        self.pline.is_closed()
    }

    /// Bounding box ((x_min, y_min), (x_max, y_max)).
    #[getter(bounding_box)]
    fn py_bounding_box(&self) -> PyResult<((f64, f64), (f64, f64))> {
        self.bounding_box().map_err(runtime_error)
    }

    /// Center of the bounding box.
    #[getter(center)]
    fn py_center(&self) -> PyResult<(f64, f64)> {
        self.center().map_err(runtime_error)
    }

    fn impl_translate(&mut self, trans_vec: Vec<f64>) -> PyResult<()> {
        let [x, y] = to_pair(&trans_vec)?;
        self.pline.translate_mut(x, y);
        Ok(())
    }

    fn impl_rotate(&mut self, angle: f64) {
        self.rotate(angle);
    }

    fn impl_scale(&mut self, fac: f64) {
        self.pline.scale_mut(fac);
    }

    fn impl_mirror(&mut self, mirror_axis: Vec<f64>) -> PyResult<()> {
        let [x, y] = to_pair(&mirror_axis)?;
        self.mirror((x, y)).map_err(runtime_error)
    }

    /// Start point (x, y, bulge).
    #[getter]
    fn start(&self) -> PyResult<Vertex> {
        self.pline
            .get(0)
            .map(|v| (v.x, v.y, v.bulge))
            .ok_or_else(|| runtime_error("An empty curve has no start point."))
    }

    /// End point (x, y, bulge).
    #[getter]
    fn end(&self) -> PyResult<Vertex> {
        self.pline
            .last()
            .map(|v| (v.x, v.y, v.bulge))
            .ok_or_else(|| runtime_error("An empty curve has no end point."))
    }

    /// List of vertices (x, y, bulge).
    #[getter(vertices)]
    fn py_vertices(&self) -> Vec<Vertex> {
        self.vertices()
    }

    /// Vertices as numpy array of shape (n, 3) with the columns x, y, bulge (much faster than `vertices`).
    #[getter]
    fn vertices_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut data = Vec::with_capacity(3 * self.len());
        for v in self.pline.iter_vertexes() {
            data.extend_from_slice(&[v.x, v.y, v.bulge]);
        }
        Array2::from_shape_vec((self.len(), 3), data)
            .expect("shape matches the data")
            .into_pyarray(py)
    }

    /// True if the (closed) curve is oriented in mathematically positive direction.
    #[getter(orientation)]
    fn py_orientation(&self) -> PyResult<bool> {
        self.orientation().map_err(runtime_error)
    }

    /// Length of the curve.
    #[getter]
    fn length(&self) -> f64 {
        self.pline.path_length()
    }

    /// Signed area enclosed by the curve (positive for counter-clockwise curves).
    #[getter]
    fn area(&self) -> f64 {
        self.pline.area()
    }

    /// Reverse the direction of the curve in-place.
    fn reverse(&mut self) {
        self.pline.invert_direction_mut();
    }

    /// Visit all segments: `func(segment_index, start_vertex, end_vertex) -> bool`. Iteration stops if `func`
    /// returns False.
    fn visit(&self, func: &Bound<'_, PyAny>) -> PyResult<()> {
        for (index, (i, j)) in self.pline.iter_segment_indexes().enumerate() {
            let (v1, v2) = (self.pline.at(i), self.pline.at(j));
            let keep_going: bool = func
                .call1((index, (v1.x, v1.y, v1.bulge), (v2.x, v2.y, v2.bulge)))?
                .extract()?;
            if !keep_going {
                break;
            }
        }
        Ok(())
    }

    /// Check if the point lies in the area enclosed by the (closed) curve.
    #[pyo3(name = "contains")]
    fn py_contains(&self, p_x: f64, p_y: f64) -> PyResult<bool> {
        self.contains(p_x, p_y).map_err(runtime_error)
    }

    /// Closest point on the curve: (segment index, point, distance).
    #[pyo3(name = "closest_point")]
    fn py_closest_point(&self, p_x: f64, p_y: f64) -> PyResult<(usize, (f64, f64), f64)> {
        self.closest_point(p_x, p_y).map_err(runtime_error)
    }

    /// Support for `copy.copy`: the curve is copied natively (a `copy.deepcopy` through `__reduce__` is orders of
    /// magnitude slower).
    fn __copy__(&self) -> Self {
        Clone::clone(self)
    }

    /// Support for `copy.deepcopy`, same as `__copy__` (all data is owned by the curve).
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        Clone::clone(self)
    }

    /// Support for pickle.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyType>, (Vec<Vertex>, bool))> {
        Ok((py.get_type::<ArcSpline>(), (self.vertices(), self.pline.is_closed())))
    }
}

/// Convert an iterable of exactly two numbers.
fn to_pair(values: &[f64]) -> PyResult<[f64; 2]> {
    match values {
        [x, y] => Ok([*x, *y]),
        _ => Err(runtime_error("Cannot construct Vector2 from iterable with size != 2")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit circle as a closed curve of two half circles.
    fn circle() -> ArcSpline {
        ArcSpline::from_vertices([(1., 0., 1.), (-1., 0., 1.)], true)
    }

    fn square() -> ArcSpline {
        ArcSpline::from_vertices([(0., 0., 0.), (2., 0., 0.), (2., 2., 0.), (0., 2., 0.)], true)
    }

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn basic_properties() {
        let c = square();
        assert_eq!(c.len(), 4);
        assert!(c.pline.is_closed());
        assert_close(c.pline.path_length(), 8.);
        assert_close(c.pline.area(), 4.);
        assert_eq!(c.orientation(), Ok(true));
        assert_eq!(c.center(), Ok((1., 1.)));
        assert_eq!(c.bounding_box(), Ok(((0., 0.), (2., 2.))));
        // the center is the center of the bounding box, not the mean of the vertices
        let triangle = ArcSpline::from_vertices([(0., 0., 0.), (3., 0., 0.), (3., 3., 0.)], false);
        assert_eq!(triangle.center(), Ok((1.5, 1.5)));
    }

    #[test]
    fn circle_properties() {
        let c = circle();
        assert_close(c.pline.path_length(), 2. * std::f64::consts::PI);
        assert_close(c.pline.area(), std::f64::consts::PI);
        let ((x0, y0), (x1, y1)) = c.bounding_box().unwrap();
        assert_close(x0, -1.);
        assert_close(y0, -1.);
        assert_close(x1, 1.);
        assert_close(y1, 1.);
    }

    #[test]
    fn empty_curve_errors() {
        let c = ArcSpline::from_vertices([], false);
        assert!(c.is_empty());
        assert!(c.bounding_box().is_err());
        assert!(c.center().is_err());
        assert!(c.closest_point(0., 0.).is_err());
    }

    #[test]
    fn open_curve_errors() {
        let c = ArcSpline::from_vertices([(0., 0., 0.), (1., 0., 0.)], false);
        assert!(c.orientation().is_err());
        assert!(c.contains(0., 0.).is_err());
    }

    #[test]
    fn contains_and_closest_point() {
        let c = square();
        assert_eq!(c.contains(1., 1.), Ok(true));
        assert_eq!(c.contains(3., 1.), Ok(false));
        let (index, point, dist) = c.closest_point(1., -1.).unwrap();
        assert_eq!(index, 0);
        assert_close(point.0, 1.);
        assert_close(point.1, 0.);
        assert_close(dist, 1.);
    }

    #[test]
    fn transformations() {
        let mut c = ArcSpline::from_vertices([(1., 0., 0.), (2., 0., 0.)], false);
        c.pline.translate_mut(1., 1.);
        assert_eq!(c.vertices(), vec![(2., 1., 0.), (3., 1., 0.)]);
        c.pline.scale_mut(2.);
        assert_eq!(c.vertices(), vec![(4., 2., 0.), (6., 2., 0.)]);

        let mut c = ArcSpline::from_vertices([(1., 0., 0.5), (2., 0., 0.)], false);
        c.rotate(std::f64::consts::FRAC_PI_2);
        let v = c.vertices();
        assert_close(v[0].0, 0.);
        assert_close(v[0].1, 1.);
        assert_close(v[0].2, 0.5);

        // mirror on the x axis flips y and the bulge
        let mut c = ArcSpline::from_vertices([(1., 2., 0.5), (3., 4., 0.)], false);
        c.mirror((1., 0.)).unwrap();
        let v = c.vertices();
        assert_close(v[0].0, 1.);
        assert_close(v[0].1, -2.);
        assert_close(v[0].2, -0.5);
        assert!(c.mirror((0., 0.)).is_err());
    }

    #[test]
    fn reverse_flips_orientation() {
        let mut c = square();
        c.pline.invert_direction_mut();
        assert_eq!(c.orientation(), Ok(false));
        assert_close(c.pline.area(), -4.);
    }
}
