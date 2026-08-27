use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution;
use statrs::distribution::Bernoulli;

use crate::distributions::{align_inputs, validate_params_unary};
use crate::rng::{
    binary_param_rows, sample_by_index, sample_per_row_binary, samples_bool_output,
    samples_by_index, samples_per_row, SampleKwargs, SampleScalarKwargs, SamplesKwargs,
    SamplesScalarKwargs,
};

fn build_dist(proba: f64) -> PolarsResult<Bernoulli> {
    Bernoulli::new(proba).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {proba}: {e}").into())
    })
}

/// Bernoulli's constant success probability, deserialised once per call.
#[derive(serde::Deserialize)]
struct BernoulliParamsKwargs {
    p: f64,
}

impl BernoulliParamsKwargs {
    fn build(&self) -> PolarsResult<Bernoulli> {
        build_dist(self.p)
    }
}

/// Element-wise validation of the success probability: returns `p` unchanged, raising
/// `InvalidOperation` if `p` is outside `[0, 1]`. `null` propagates.
///
/// The closed-form Python methods derive from this so they report an invalid `p` consistently
/// with `bernoulli_sample`, instead of silently computing a negative probability.
#[polars_expr(output_type=Float64)]
fn bernoulli_proba(inputs: &[Series]) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;

    validate_params_unary(proba.f64()?, |proba| {
        build_dist(proba)?;
        Ok(proba)
    })
}

/// One Bernoulli draw from a `&mut` per-row RNG already seeded from `(root_seed, index)`.
///
/// Every Bernoulli sampler draws through here, per-row and fast path alike, so their bit-equality is
/// structural rather than only sampled by `sample_test.py`.
#[inline]
fn draw(dist: &Bernoulli, rng: &mut impl rand::Rng) -> bool {
    <Bernoulli as Distribution<bool>>::sample(dist, rng)
}

/// Element-wise Bernoulli sampler over `(p, row_index)`, returning `Boolean`.
///
/// Per row, `null` propagates and an invalid `p` raises via [`build_dist`]. Seeding and
/// chunk-invariance follow [`sample_per_row_binary`].
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    sample_per_row_binary(
        name,
        proba.f64()?,
        index.u64()?,
        kwargs.seed,
        build_dist,
        draw,
    )
}

/// Constant-parameter fast path for [`bernoulli_sample`].
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<BernoulliParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    sample_by_index(name, &inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

/// Constant-parameter multi-draw fast path: the `samples` twin of [`bernoulli_sample_scalar`].
///
/// `size` consecutive draws from each row's stream, so `samples(size=1)` matches `sample` bit for
/// bit and the distribution is built once per call. Returns `Array(Boolean, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<BernoulliParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    samples_by_index(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&dist, rng)
    })
}

/// Element-wise multi-draw Bernoulli sampler: `size` draws per row in one call, the distribution
/// built once per row. Returns `Array(Boolean, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`bernoulli_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = binary_param_rows(proba.f64()?, index.u64()?, build_dist);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, draw)
}
