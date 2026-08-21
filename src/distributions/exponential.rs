#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Exp;

use crate::distributions::{align_inputs, validate_params_unary};
use crate::rng::{
    binary_param_rows, sample_by_index, sample_per_row_binary, samples_by_index,
    samples_f64_output, samples_per_row, SampleKwargs, SampleScalarKwargs, SamplesKwargs,
    SamplesScalarKwargs,
};

/// Construct a `statrs::Exp`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::Exp::new` rejects a `NaN` rate or `rate <= 0` (a positive-infinite rate is accepted, as a
/// degenerate point mass at 0). That surfaces as `InvalidOperation`, so an invalid rate fails the whole
/// evaluation rather than silently nulling the row.
fn build_dist(rate: f64) -> PolarsResult<Exp> {
    Exp::new(rate).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("rate must be strictly positive, got rate={rate}: {e}").into(),
        )
    })
}

/// Exponential's constant rate, deserialised once per call.
#[derive(serde::Deserialize)]
struct ExponentialParamsKwargs {
    rate: f64,
}

impl ExponentialParamsKwargs {
    fn build(&self) -> PolarsResult<Exp> {
        build_dist(self.rate)
    }
}

/// Element-wise validation of the rate (λ): returns `rate` unchanged, raising `InvalidOperation`
/// if `rate` is `NaN` or `rate <= 0`. `null` propagates.
///
/// Exponential is an elementary closed-form distribution, so its pdf / cdf / ppf and moments are
/// pure Polars expressions; routing the rate through this validator is what lets them report an
/// invalid parameterisation consistently with `exponential_sample`, instead of silently computing
/// with a non-positive rate.
#[polars_expr(output_type=Float64)]
fn exponential_rate(inputs: &[Series]) -> PolarsResult<Series> {
    let rate = inputs[0].cast(&DataType::Float64)?;

    validate_params_unary(rate.f64()?, |rate| {
        build_dist(rate)?;
        Ok(rate)
    })
}

/// One Exponential draw from a `&mut` per-row RNG already seeded from `(root_seed, index)`.
///
/// Every Exponential sampler draws through here, per-row and fast path alike, so their bit-equality is
/// structural rather than only sampled by `sample_test.py`.
#[inline]
fn draw(dist: &Exp, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

/// Element-wise Exponential sampler over `(rate, row_index)`, returning `Float64`.
///
/// Per row, `null` propagates and an invalid rate raises via [`build_dist`]. Seeding and
/// chunk-invariance follow [`sample_per_row_binary`].
///
/// The draw keeps `statrs` (`O(1)` ziggurat: `sample_exp_1(rng) / rate`); routing it through
/// `rand_distr` would buy nothing, since that is already the algorithm class `statrs` uses (unlike
/// the binomial draw, see docs/explanation/design.md).
#[polars_expr(output_type=Float64)]
fn exponential_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let rate = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    sample_per_row_binary(
        name,
        rate.f64()?,
        index.u64()?,
        kwargs.seed,
        build_dist,
        draw,
    )
}

/// Constant-rate fast path for [`exponential_sample`].
#[polars_expr(output_type=Float64)]
fn exponential_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<ExponentialParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    sample_by_index(name, &inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

/// Constant-parameter multi-draw fast path: the `samples` twin of [`exponential_sample_scalar`].
///
/// `size` consecutive draws from each row's stream, so `samples(size=1)` matches `sample` bit for
/// bit and the distribution is built once per call. Returns `Array(Float64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn exponential_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<ExponentialParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    samples_by_index(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&dist, rng)
    })
}

/// Element-wise multi-draw Exponential sampler: `size` draws per row in one call, the distribution
/// built once per row. Returns `Array(Float64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`exponential_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn exponential_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let rate = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = binary_param_rows(rate.f64()?, index.u64()?, build_dist);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, draw)
}
