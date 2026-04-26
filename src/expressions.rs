#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Deserialize;
use statrs::distribution::Bernoulli;

#[derive(Deserialize)]
struct BernoulliSampleKwargs {
    seed: Option<u64>,
}

/// Element-wise Bernoulli sampler.
///
/// ``inputs[0]`` carries success probabilities, one per output row. Returns
/// a ``UInt8`` series of 0/1, with null on rows where ``p`` itself is null.
/// Rows where ``p`` is NaN or outside ``[0, 1]`` raise
/// ``InvalidOperationError`` and abort the whole call.
#[polars_expr(output_type=UInt8)]
fn bernoulli_sample(inputs: &[Series], kwargs: BernoulliSampleKwargs) -> PolarsResult<Series> {
    let name = inputs[0].name().clone();
    let p_series = inputs[0].cast(&DataType::Float64)?;
    let p_ca = p_series.f64()?;
    let n = p_ca.len();

    let mut rng: Box<dyn RngCore + Send> = match kwargs.seed {
        Some(s) => Box::new(ChaCha20Rng::seed_from_u64(s)),
        None => Box::new(ChaCha20Rng::from_entropy()),
    };

    let mut values: Vec<Option<u8>> = Vec::with_capacity(n);
    for i in 0..n {
        match p_ca.get(i) {
            None => values.push(None),
            Some(p) => {
                if p.is_nan() || !(0.0..=1.0).contains(&p) {
                    return Err(PolarsError::InvalidOperation(
                        format!("p must be in [0, 1], got {p}").into(),
                    ));
                }
                let dist = Bernoulli::new(p).map_err(|e| {
                    PolarsError::InvalidOperation(
                        format!("invalid Bernoulli parameter p={p}: {e}").into(),
                    )
                })?;
                let s = <Bernoulli as Distribution<bool>>::sample(&dist, &mut rng) as u8;
                values.push(Some(s));
            },
        }
    }
    let ca: UInt8Chunked = values.into_iter().collect();
    Ok(ca.with_name(name).into_series())
}
