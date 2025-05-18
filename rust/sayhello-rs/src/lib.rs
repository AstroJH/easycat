use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod sayhello_rs {
    use pyo3::prelude::*;
    use std::env;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn hello() -> PyResult<String> {
        Ok(String::from("Hello! [from Rust]"))
    }
}