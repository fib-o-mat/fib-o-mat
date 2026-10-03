//! Error helpers shared by the python bindings.

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Create the python exception which is raised for all geometry errors (`RuntimeError`, as in the former C++ module).
pub(crate) fn runtime_error(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(message.into())
}
