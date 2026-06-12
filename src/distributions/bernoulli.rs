#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution;
use serde::Deserialize;
use statrs::distribution::Bernoulli;

use crate::rng::{
    sample_scalar_plugin, samples_bool_output, samples_by_index, samples_per_row, SampleKwargs,
    SamplesKwargs,
};

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

sample_scalar_plugin! {
    /// Static parameters for the constant-parameter sampler fast path: when `p` is a Python
    /// scalar, it travels here as a kwarg instead of as a full-length `pl.repeat` column
    /// re-validated on every row.
    struct BernoulliScalarKwargs { p: f64 }

    /// Constant-probability Bernoulli sampler: [`bernoulli_sample`] for the scalar-`p` case. The
    /// distribution is validated and built once, and only the per-row index travels as an input;
    /// seeding and the draw are unchanged, so output matches `bernoulli_sample` for the same
    /// `(seed, index, p)`.
    fn bernoulli_sample_scalar(output_type = Boolean, physical = BooleanType);
    build = |kw| build_dist(kw.p)?;
    draw = |dist, rng| <Bernoulli as Distribution<bool>>::sample(&dist, rng);
}

/// Static parameters for the constant-probability multi-draw fast path.
///
/// Like [`BernoulliScalarKwargs`] plus the draw count `size` (the shared [`SamplesKwargs`] shape
/// with the scalar `p` alongside).
#[derive(Deserialize)]
struct BernoulliSamplesScalarKwargs {
    seed: Option<u64>,
    size: usize,
    p: f64,
}

/// Constant-probability multi-draw Bernoulli sampler: `size` draws per row in one call.
///
/// Row `i`'s draws are consecutive values from the one stream seeded `(seed, i)`, the same
/// stream [`bernoulli_sample_scalar`] takes its single draw from, so `samples(size=1)` matches
/// `sample` bit for bit and growing `size` extends each row's array. Returns
/// `Array(Boolean, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples_scalar(
    inputs: &[Series],
    kwargs: BernoulliSamplesScalarKwargs,
) -> PolarsResult<Series> {
    let dist = build_dist(kwargs.p)?;
    let size = kwargs.size;
    let name = inputs[0].name().clone();

    samples_by_index::<BooleanType, _>(&inputs[0], kwargs.seed, size, |index_ca, rngs| {
        BooleanChunked::from_iter_values(
            name,
            index_ca.into_no_null_iter().flat_map(|i| {
                let mut rng = rngs.rng(i);
                std::iter::repeat_with(move || {
                    <Bernoulli as Distribution<bool>>::sample(&dist, &mut rng)
                })
                .take(size)
            }),
        )
    })
}

/// Element-wise multi-draw Bernoulli sampler: `size` draws per row in one call.
///
/// The column-parameter counterpart of [`bernoulli_samples_scalar`], replacing `samples`' former
/// construction of `k` [`bernoulli_sample`] calls glued by `concat_arr`: the distribution is
/// built once per row instead of once per draw. Row `i`'s draws come from the one stream seeded
/// `(seed, i)`, so output is bit-identical to the scalar path for the same `p` (the seeding is
/// positional, parameters never enter it, so equal-`p` rows still draw independently).
/// Null/error contract follows [`bernoulli_sample`] per row; a null row yields an array of null
/// elements (the Python layer masks it to a null array). Returns `Array(Boolean, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let proba_ca = proba.f64()?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rows = proba_ca
        .iter()
        .zip(index_ca.iter())
        .map(|(p_opt, i_opt)| match (p_opt, i_opt) {
            (Some(p), Some(i)) => Ok(Some((i, build_dist(p)?))),
            _ => Ok(None),
        });

    samples_per_row::<BooleanType, _, _, _, _>(
        name,
        kwargs.seed,
        kwargs.size,
        index_ca.len(),
        rows,
        <Bernoulli as Distribution<bool>>::sample,
    )
}
