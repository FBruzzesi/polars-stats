#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution;
use serde::Deserialize;
use statrs::distribution::Bernoulli;

use crate::rng::{sample_by_index, SampleKwargs};

fn build_dist(proba: f64) -> PolarsResult<Bernoulli> {
    Bernoulli::new(proba).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {proba}: {e}").into())
    })
}

/// Element-wise validation of the success probability: returns `p` unchanged, raising
/// `InvalidOperation` if `p` is outside `[0, 1]`. `null` propagates.
///
/// The closed-form Python methods derive from this so they report an invalid `p` consistently with
/// `bernoulli_sample`, instead of silently computing a negative probability.
#[polars_expr(output_type=Float64)]
fn bernoulli_proba(inputs: &[Series]) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let proba_ca = proba.f64()?;
    let name = inputs[0].name().clone();

    let ca: Float64Chunked =
        proba_ca.try_apply_nonnull_values_generic(|p| -> PolarsResult<f64> {
            build_dist(p)?;
            Ok(p)
        })?;

    Ok(ca.with_name(name).into_series())
}

/// Element-wise Bernoulli sampler.
///
/// `inputs[0]` carries the success probability (one per row).
/// `inputs[1]` carries a per-row index used to derive a per-row sub-seed, so the
/// function is genuinely elementwise: chunking and threading cannot change the
/// output. With `seed=None`, a fresh root seed is drawn once per call.
///
/// Per-row validation:
///   * `null` (in either input) propagates;
///   * `NaN` or out-of-range `p` raises `InvalidOperation`.
///
/// Returns a `Boolean` series.
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let proba_ca = proba.f64()?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs();

    let ca: BooleanChunked = try_binary_elementwise(
        proba_ca,
        index_ca,
        |p_opt, i_opt| -> PolarsResult<Option<bool>> {
            match (p_opt, i_opt) {
                (Some(p), Some(i)) => {
                    let dist = build_dist(p)?;
                    let mut rng = rngs.rng(i);
                    Ok(Some(<Bernoulli as Distribution<bool>>::sample(
                        &dist, &mut rng,
                    )))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Static parameters for the constant-parameter sampler fast path.
///
/// When `p` is a Python scalar, the Python layer routes it here as a kwarg instead of expanding it
/// into a full-length column (`pl.repeat`) that crosses FFI and is re-validated on every row. The
/// distribution is validated and built once, and only the per-row index travels as an input.
#[derive(Deserialize)]
struct BernoulliScalarKwargs {
    seed: Option<u64>,
    p: f64,
}

/// Constant-probability Bernoulli sampler.
///
/// Semantically identical to [`bernoulli_sample`] for the common case of a scalar `p`, but built for
/// it: `inputs[0]` is the per-row index (never null, sole FFI input), and `p` arrives in `kwargs`.
/// The distribution is validated and constructed once up front, then reused for every row. Seeding
/// and the draw are unchanged, so output matches `bernoulli_sample` for the same `(seed, index, p)`.
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample_scalar(
    inputs: &[Series],
    kwargs: BernoulliScalarKwargs,
) -> PolarsResult<Series> {
    let dist = build_dist(kwargs.p)?;
    let name = inputs[0].name().clone();

    sample_by_index::<BooleanType, _>(&inputs[0], kwargs.seed, |index_ca, rngs| {
        BooleanChunked::from_iter_values(
            name,
            index_ca.into_no_null_iter().map(|i| {
                let mut rng = rngs.rng(i);
                <Bernoulli as Distribution<bool>>::sample(&dist, &mut rng)
            }),
        )
    })
}
