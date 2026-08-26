#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Geometric;

use crate::distributions::{align_inputs, validate_params_unary};
use crate::rng::{
    binary_param_rows, sample_by_index, sample_per_row_binary, samples_by_index, samples_per_row,
    samples_u64_output, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

/// `statrs::Geometric::new` rejects `NaN` and any `p` outside `(0, 1]`, so unlike `Bernoulli` the
/// degenerate `p = 0` point mass is not representable.
fn build_dist(proba: f64) -> PolarsResult<Geometric> {
    Geometric::new(proba).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in (0, 1], got {proba}: {e}").into())
    })
}

/// Geometric's constant success probability, deserialised once per call.
#[derive(serde::Deserialize)]
struct GeometricParamsKwargs {
    p: f64,
}

impl GeometricParamsKwargs {
    fn build(&self) -> PolarsResult<Geometric> {
        build_dist(self.p)
    }
}

/// Element-wise validation of the success probability: returns `p` unchanged, raising
/// `InvalidOperation` if `p` is outside `(0, 1]`. `null` propagates.
///
/// The closed-form Python methods derive from this so they report an invalid `p` consistently
/// with `geometric_sample`, instead of silently computing with a non-positive probability.
#[polars_expr(output_type=Float64)]
fn geometric_p(inputs: &[Series]) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;

    validate_params_unary(proba.f64()?, |proba| {
        build_dist(proba)?;
        Ok(proba)
    })
}

/// One Geometric draw from a `&mut` per-row RNG already seeded from `(root_seed, index)`.
///
/// Every Geometric sampler draws through here, per-row and fast path alike, so their bit-equality is
/// structural rather than only sampled by `sample_test.py`.
#[inline]
fn draw(dist: &Geometric, rng: &mut impl rand::Rng) -> u64 {
    RandDistribution::<u64>::sample(dist, rng)
}

/// Element-wise Geometric sampler over `(p, row_index)`, returning `UInt64`.
///
/// Per row, `null` propagates and an invalid `p` raises via [`build_dist`]. Seeding and
/// chunk-invariance follow [`sample_per_row_binary`].
#[polars_expr(output_type=UInt64)]
fn geometric_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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

/// Constant-parameter fast path for [`geometric_sample`].
#[polars_expr(output_type=UInt64)]
fn geometric_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<GeometricParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    sample_by_index(name, &inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

/// Constant-parameter multi-draw fast path: the `samples` twin of [`geometric_sample_scalar`].
///
/// `size` consecutive draws from each row's stream, so `samples(size=1)` matches `sample` bit for
/// bit and the distribution is built once per call. Returns `Array(UInt64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn geometric_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<GeometricParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    samples_by_index(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&dist, rng)
    })
}

/// Element-wise multi-draw Geometric sampler: `size` draws per row in one call, the distribution
/// built once per row. Returns `Array(UInt64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`geometric_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn geometric_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = binary_param_rows(proba.f64()?, index.u64()?, build_dist);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, draw)
}
