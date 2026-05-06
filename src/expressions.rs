#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::Deserialize;
use statrs::distribution::Bernoulli;

#[derive(Deserialize)]
struct BernoulliSampleKwargs {
    seed: Option<u64>,
}

fn build_dist(proba: f64) -> PolarsResult<Bernoulli> {
    Bernoulli::new(proba).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {proba}: {e}").into())
    })
}

/// Element-wise Bernoulli sampler.
///
/// `inputs[0]` carries the success probability, one per row (the Python side
/// expands a scalar `p` to a length-N expression so this function never has
/// to special-case broadcast). Per-row validation: `null` propagates;
/// `NaN` or out-of-range raises `InvalidOperation`.
///
/// Returns a `UInt8` series of 0/1, with nulls propagated from `p`.
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample(inputs: &[Series], kwargs: BernoulliSampleKwargs) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let proba_ca = proba.f64()?;
    let name = inputs[0].name().clone();

    let mut rng = match kwargs.seed {
        Some(s) => ChaCha20Rng::seed_from_u64(s),
        None => ChaCha20Rng::from_entropy(),
    };

    let ca: BooleanChunked = proba_ca.try_apply_nonnull_values_generic(|p| {
        let dist = build_dist(p)?;
        Ok::<bool, PolarsError>(<Bernoulli as Distribution<bool>>::sample(&dist, &mut rng))
    })?;

    Ok(ca.with_name(name).into_series())
}
