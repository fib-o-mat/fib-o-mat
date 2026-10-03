//! Native geometry kernel of fib-o-mat, exposed to python as `fibomat._libfibomat`.

mod arc_spline;
mod error;
mod tools;

pub use arc_spline::ArcSpline;

use pyo3::prelude::*;

/// Rust extension for fib-o-mat.
#[pymodule]
fn _libfibomat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArcSpline>()?;
    m.add_function(wrap_pyfunction!(tools::self_intersections, m)?)?;
    m.add_function(wrap_pyfunction!(tools::curve_intersections, m)?)?;
    m.add_function(wrap_pyfunction!(tools::combine_curves, m)?)?;
    m.add_function(wrap_pyfunction!(tools::offset_curve, m)?)?;
    m.add_function(wrap_pyfunction!(tools::offset_with_islands, m)?)?;
    m.add_function(wrap_pyfunction!(tools::convert_arcs_to_lines, m)?)?;
    Ok(())
}
