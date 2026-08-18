#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution;
use statrs::distribution::Bernoulli;

use crate::distributions::validate_params_unary;
use crate::rng::{
    binary_param_rows, sample_scalar_plugin, samples_bool_output, samples_per_row, SampleKwargs,
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
/// The closed-form Python methods derive from this so they report an invalid `p` consistently
/// with `bernoulli_sample`, instead of silently computing a negative probability.
#[polars_expr(output_type=Float64)]
fn bernoulli_proba(inputs: &[Series]) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let name = inputs[0].name().clone();

    validate_params_unary(proba.f64()?, name, |proba| {
        build_dist(proba)?;
        Ok(proba)
    })
}

/// Element-wise Bernoulli sampler over `(p, row_index)`, returning `Boolean`.
///
/// Per row, `null` propagates and an invalid `p` raises via [`build_dist`]. Seeding and
/// chunk-invariance follow [`SampleKwargs::row_rngs`].
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let proba_ca = proba.f64()?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs()?;

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
    struct BernoulliScalarKwargs { p: f64 }

    /// Constant-parameter fast path for [`bernoulli_sample`].
    fn bernoulli_sample_scalar(output_type = Boolean, physical = BooleanType);

    samples = bernoulli_samples_scalar as BernoulliSamplesScalarKwargs -> samples_bool_output;

    build = |kw| build_dist(kw.p)?;
    draw = |dist, rng| <Bernoulli as Distribution<bool>>::sample(&dist, rng);
}

/// Element-wise multi-draw Bernoulli sampler: `size` draws per row in one call, the distribution
/// built once per row. Returns `Array(Boolean, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`bernoulli_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = binary_param_rows(proba.f64()?, index.u64()?, build_dist);

    samples_per_row::<BooleanType, _, _, _, _>(
        name,
        kwargs.seed,
        kwargs.size,
        rows,
        <Bernoulli as Distribution<bool>>::sample,
    )
}
