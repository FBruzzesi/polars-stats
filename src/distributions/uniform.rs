#![allow(clippy::unused_unit)]
use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution;
use statrs::distribution::Uniform;

use crate::rng::SampleKwargs;

fn build_dist(min: f64, max: f64) -> PolarsResult<Uniform> {
    Uniform::new(min, max).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("max must be strictly greater than min, got min={min}, max={max}: {e}").into(),
        )
    })
}

/// Element-wise continuous Uniform sampler over `[min, max)`.
///
/// `inputs[0]` carries the lower bound, `inputs[1]` the upper bound, and `inputs[2]` a per-row
/// index used to derive a per-row sub-seed, so the function is genuinely elementwise: chunking
/// and threading cannot change the output. With `seed=None`, a fresh root seed is drawn once per
/// call.
///
/// Per-row validation:
///   * `null` (in any input) propagates;
///   * `max <= min` or non-finite bounds raise `InvalidOperation` (surfaces as a `ComputeError`),
///     consistent with how every distribution reports an invalid parameterisation.
///
/// Returns a `Float64` series.
#[polars_expr(output_type=Float64)]
fn uniform_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let min = inputs[0].cast(&DataType::Float64)?;
    let min_ca = min.f64()?;
    let max = inputs[1].cast(&DataType::Float64)?;
    let max_ca = max.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs();

    let ca: Float64Chunked = try_ternary_elementwise(
        min_ca,
        max_ca,
        index_ca,
        |min_opt, max_opt, i_opt| -> PolarsResult<Option<f64>> {
            match (min_opt, max_opt, i_opt) {
                (Some(lo), Some(hi), Some(i)) => {
                    let dist = build_dist(lo, hi)?;
                    let mut rng = rngs.rng(i);
                    Ok(Some(dist.sample(&mut rng)))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Element-wise support width `max - min`, validating the parameterisation.
///
/// `inputs[0]` is the lower bound, `inputs[1]` the upper bound. `null` in either propagates;
/// `max <= min` or non-finite bounds raise `InvalidOperation` (surfaces as a `ComputeError`).
///
/// Every closed-form Python method derives from this width, so routing it through Rust is what lets
/// them report an invalid parameterisation consistently with `uniform_sample`, instead of silently
/// producing a negative or infinite result.
#[polars_expr(output_type=Float64)]
fn uniform_range(inputs: &[Series]) -> PolarsResult<Series> {
    let min = inputs[0].cast(&DataType::Float64)?;
    let min_ca = min.f64()?;
    let max = inputs[1].cast(&DataType::Float64)?;
    let max_ca = max.f64()?;
    let name = inputs[0].name().clone();

    let ca: Float64Chunked = try_binary_elementwise(
        min_ca,
        max_ca,
        |min_opt, max_opt| -> PolarsResult<Option<f64>> {
            match (min_opt, max_opt) {
                (Some(lo), Some(hi)) => {
                    build_dist(lo, hi)?;
                    Ok(Some(hi - lo))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}
