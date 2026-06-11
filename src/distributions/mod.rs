pub mod bernoulli;
pub mod binomial;
pub mod lognormal;
pub mod normal;
pub mod uniform;

use polars::prelude::arity::unary_elementwise;
use polars::prelude::*;

/// Shared driver for the constant-parameter value-keyed fast paths.
///
/// The constant-parameter counterpart of each distribution's `value_keyed` helper: when every
/// distribution parameter is a Python scalar, the caller validates them and builds the
/// distribution **once**, and only the evaluation-point column crosses FFI. This maps `f` over
/// that single column, where `f` is the same per-method body the per-row path uses (a named
/// function in each distribution file), so the two paths cannot drift and output is bit-identical.
///
/// Null contract: `value` is the only nullable input this driver sees. The distribution
/// parameters are not passed here at all; the caller validates them once (via `build_dist`,
/// which raises on an invalid or non-finite parameterisation before this runs) and bakes them
/// into `f` through a pre-built `dist`. So a null `value` propagates to null, and `f` returning
/// `None` nulls the row on the method's own terms (e.g. `ppf` outside `[0, 1]`), matching the
/// per-row path element for element.
pub(crate) fn value_keyed_scalar<F>(value: &Series, f: F) -> PolarsResult<Series>
where
    F: Fn(f64) -> Option<f64>,
{
    let value = value.cast(&DataType::Float64)?;
    let value_ca = value.f64()?;
    let name = value_ca.name().clone();

    let ca: Float64Chunked = unary_elementwise(value_ca, |opt| opt.and_then(&f));
    Ok(ca.with_name(name).into_series())
}
