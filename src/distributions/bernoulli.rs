#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution;
use rand::rngs::OsRng;
use rand::{RngCore, SeedableRng};
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

/// Per-row RNG seeded by (root_seed, row_index). Identical pairs always yield identical streams,
/// so output is independent of how Polars chunks or threads the input.
fn row_rng(root_seed: u64, index: u64) -> ChaCha20Rng {
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&root_seed.to_le_bytes());
    seed_bytes[8..16].copy_from_slice(&index.to_le_bytes());
    ChaCha20Rng::from_seed(seed_bytes)
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
fn bernoulli_sample(inputs: &[Series], kwargs: BernoulliSampleKwargs) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    let proba_ca = proba.f64()?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let root_seed = kwargs.seed.unwrap_or_else(|| OsRng.next_u64());

    let ca: BooleanChunked = try_binary_elementwise(
        proba_ca,
        index_ca,
        |p_opt, i_opt| -> PolarsResult<Option<bool>> {
            match (p_opt, i_opt) {
                (Some(p), Some(i)) => {
                    let dist = build_dist(p)?;
                    let mut rng = row_rng(root_seed, i);
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
