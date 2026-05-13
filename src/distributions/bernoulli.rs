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
    size: u64,
    seed: Option<u64>,
}

fn build_dist(proba: f64) -> PolarsResult<Bernoulli> {
    Bernoulli::new(proba).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {proba}: {e}").into())
    })
}

/// Element-wise Bernoulli sampler.
///
/// `inputs[0]` carries the success probability (one per row).
///
/// Per-row validation:
///   * `null` propagates;
///   * `NaN` or out-of-range raises `InvalidOperation`.
///
/// Returns a `Boolean` series, with nulls propagated from `p`.
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

fn list_bool_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::Boolean)),
    ))
}

#[polars_expr(output_type_func=list_bool_output)]
fn bernoulli_samples(inputs: &[Series], kwargs: BernoulliSampleKwargs) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let proba_ca = proba.f64()?;
    let name = inputs[0].name().clone();

    let mut rng = match kwargs.seed {
        Some(s) => ChaCha20Rng::seed_from_u64(s),
        None => ChaCha20Rng::from_entropy(),
    };

    let size = kwargs.size as usize;
    let len = proba_ca.len();

    let mut builder = ListBooleanChunkedBuilder::new(name, len, len * size);

    for opt_p in proba_ca.iter() {
        match opt_p {
            None => builder.append_null(),
            Some(p) => {
                let dist = build_dist(p)?;
                let row: Vec<Option<bool>> =
                    dist.sample_iter(&mut rng).take(size).map(Some).collect();
                builder.append_iter(row.into_iter());
            }
        }
    }

    Ok(builder.finish().into_series())
}
