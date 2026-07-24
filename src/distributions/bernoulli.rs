#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution;
use statrs::distribution::Bernoulli;

use crate::distributions::param_validator;
use crate::rng::{
    binary_param_rows, sample_scalar_plugin, samples_bool_output, samples_per_row, SampleKwargs,
    SamplesKwargs,
};

fn build_dist(proba: f64) -> PolarsResult<Bernoulli> {
    Bernoulli::new(proba).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {proba}: {e}").into())
    })
}

param_validator! {
    /// Element-wise validation of the success probability: returns `p` unchanged, raising
    /// `InvalidOperation` if `p` is outside `[0, 1]`. `null` propagates.
    ///
    /// The closed-form Python methods derive from this so they report an invalid `p` consistently
    /// with `bernoulli_sample`, instead of silently computing a negative probability.
    fn bernoulli_proba;
    params = (proba: DataType::Float64 => f64);
    build = build_dist;
    returns = proba;
    output_name = inputs[0];
}

/// Element-wise Bernoulli sampler.
///
/// `inputs[0]` carries the success probability (one per row), `inputs[1]` the per-row index each
/// row's sub-seed derives from, so chunking and threading cannot change the output; `seed=None`
/// draws a fresh root seed once per call. Per row, `null` propagates and an invalid `p` raises
/// via [`build_dist`]. Returns `Boolean`.
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
    /// Static `p` for the constant-parameter sampler fast path: validated once, passed as a kwarg
    /// instead of a full-length column.
    struct BernoulliScalarKwargs { p: f64 }

    /// Constant-probability Bernoulli sampler: [`bernoulli_sample`] with the distribution built
    /// once; seeding and draw unchanged, so output is bit-identical for the same inputs.
    fn bernoulli_sample_scalar(output_type = Boolean, physical = BooleanType);

    samples = bernoulli_samples_scalar as BernoulliSamplesScalarKwargs -> samples_bool_output;

    build = |kw| build_dist(kw.p)?;
    draw = |dist, rng| <Bernoulli as Distribution<bool>>::sample(&dist, rng);
}

/// Element-wise multi-draw Bernoulli sampler: `size` draws per row in one call, the distribution
/// built once per row.
///
/// Seeding is positional (see [`samples_per_row`]), so output is bit-identical to
/// [`bernoulli_samples_scalar`] for the same `p`. Null/error contract follows
/// [`bernoulli_sample`] per row; a null row yields a null array element.
/// Returns `Array(Boolean, size)`.
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
