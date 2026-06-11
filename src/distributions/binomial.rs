#![allow(clippy::unused_unit)]
use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution as RandDistribution;
use serde::Deserialize;
use statrs::distribution::{Binomial, Discrete, DiscreteCDF};
use statrs::statistics::Distribution as StatrsDistribution;

use crate::rng::{sample_by_index, SampleKwargs};

/// Construct a `statrs::Binomial`, mapping both invalid-parameter cases to a `ComputeError`.
///
/// `statrs::Binomial::new(p, n)` takes the arguments in the opposite order to this crate's `(n, p)`
/// and only rejects a `NaN` `p` or `p` outside `[0, 1]`; its `n` is a `u64`, so a negative trial
/// count cannot reach it. We validate `n >= 0` here and surface both failures as `InvalidOperation`
/// so an invalid parameterisation fails the whole evaluation rather than silently nulling the row,
/// consistent with every other distribution. Validation lives here so each method that builds a
/// distribution reports it identically.
fn build_dist(n: i64, p: f64) -> PolarsResult<Binomial> {
    let trials = u64::try_from(n).map_err(|_| {
        PolarsError::InvalidOperation(format!("n must be a non-negative integer, got {n}").into())
    })?;
    Binomial::new(p, trials).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {p}: {e}").into())
    })
}

/// `value` as a binomial support point: `Some(k)` when `value` is a non-negative integer, else
/// `None` (off the support, where the mass is zero). `value > n` still maps to `Some`; `statrs`
/// returns zero mass there, so the check need only reject negatives and non-integers. The `as`
/// cast saturates to `u64::MAX` for values beyond the integer range, which `statrs` then treats as
/// `> n`.
#[inline]
fn support_point(value: f64) -> Option<u64> {
    if value < 0.0 || value.fract() != 0.0 {
        None
    } else {
        Some(value as u64)
    }
}

/// Apply a value-keyed function `f(dist, value)` element-wise over `(value, n, p)`.
///
/// `inputs[0]` is the evaluation point, `inputs[1]` is `n`, `inputs[2]` is `p`. `null` in any input
/// propagates to `null`; an invalid parameterisation raises via [`build_dist`]. `f` returns an
/// `Option` so a method can null a row on its own terms (e.g. `ppf` outside `[0, 1]`). Shared by
/// `pmf`, `ln_pmf`, `cdf`, `sf`, `ppf`.
fn value_keyed<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
where
    F: Fn(&Binomial, f64) -> Option<f64>,
{
    let value = inputs[0].cast(&DataType::Float64)?;
    let value_ca = value.f64()?;
    let n = inputs[1].cast(&DataType::Int64)?;
    let n_ca = n.i64()?;
    let p = inputs[2].cast(&DataType::Float64)?;
    let p_ca = p.f64()?;
    let name = inputs[0].name().clone();

    let ca: Float64Chunked = try_ternary_elementwise(
        value_ca,
        n_ca,
        p_ca,
        |value_opt, n_opt, p_opt| -> PolarsResult<Option<f64>> {
            match (value_opt, n_opt, p_opt) {
                (Some(v), Some(n), Some(p)) => {
                    let dist = build_dist(n, p)?;
                    Ok(f(&dist, v))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Apply a parameter-keyed moment `f(dist)` element-wise over `(n, p)`.
///
/// `inputs[0]` is `n`, `inputs[1]` is `p`. `null` in either propagates; an invalid parameterisation
/// raises via [`build_dist`]. Only `entropy` routes through here (the one moment without an
/// elementary closed form); `mean` and `variance` are closed forms computed in Polars and gated on
/// [`binomial_params`].
fn params_keyed<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
where
    F: Fn(&Binomial) -> f64,
{
    let n = inputs[0].cast(&DataType::Int64)?;
    let n_ca = n.i64()?;
    let p = inputs[1].cast(&DataType::Float64)?;
    let p_ca = p.f64()?;
    let name = inputs[1].name().clone();

    let ca: Float64Chunked =
        try_binary_elementwise(n_ca, p_ca, |n_opt, p_opt| -> PolarsResult<Option<f64>> {
            match (n_opt, p_opt) {
                (Some(n), Some(p)) => {
                    let dist = build_dist(n, p)?;
                    Ok(Some(f(&dist)))
                },
                _ => Ok(None),
            }
        })?;

    Ok(ca.with_name(name).into_series())
}

/// Element-wise Binomial sampler.
///
/// `inputs[0]` carries `n`, `inputs[1]` `p`, and `inputs[2]` a per-row index used to derive a
/// per-row sub-seed, so the function is genuinely element-wise: chunking and threading cannot change
/// the output. With `seed=None`, a fresh root seed is drawn once per call.
///
/// Per-row validation:
///   * `null` (in any input) propagates;
///   * a negative `n`, or a `NaN` / out-of-range `p`, raises `InvalidOperation`.
///
/// Returns a `UInt64` series.
#[polars_expr(output_type=UInt64)]
fn binomial_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let n = inputs[0].cast(&DataType::Int64)?;
    let n_ca = n.i64()?;
    let p = inputs[1].cast(&DataType::Float64)?;
    let p_ca = p.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs();

    let ca: UInt64Chunked = try_ternary_elementwise(
        n_ca,
        p_ca,
        index_ca,
        |n_opt, p_opt, i_opt| -> PolarsResult<Option<u64>> {
            match (n_opt, p_opt, i_opt) {
                (Some(n), Some(p), Some(i)) => {
                    let dist = build_dist(n, p)?;
                    let mut rng = rngs.rng(i);
                    Ok(Some(<Binomial as RandDistribution<u64>>::sample(
                        &dist, &mut rng,
                    )))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Static parameters for the constant-parameter sampler fast path.
///
/// When both `n` and `p` are Python scalars, the Python layer routes them here as kwargs instead of
/// expanding each into a full-length column (`pl.repeat`) that crosses FFI and is re-validated on
/// every row. The distribution is validated and built once, and only the per-row index travels as an
/// input.
#[derive(Deserialize)]
struct BinomialScalarKwargs {
    seed: Option<u64>,
    n: i64,
    p: f64,
}

/// Constant-parameter Binomial sampler.
///
/// Semantically identical to [`binomial_sample`] for the common case of scalar parameters, but built
/// for it: `inputs[0]` is the per-row index (never null, sole FFI input), and the parameters arrive
/// in `kwargs`. The distribution is validated and constructed once up front, then reused for every
/// row. Seeding and the draw are unchanged, so output matches `binomial_sample` for the same
/// `(seed, index, n, p)`.
#[polars_expr(output_type=UInt64)]
fn binomial_sample_scalar(inputs: &[Series], kwargs: BinomialScalarKwargs) -> PolarsResult<Series> {
    let dist = build_dist(kwargs.n, kwargs.p)?;
    let name = inputs[0].name().clone();

    sample_by_index::<UInt64Type, _>(&inputs[0], kwargs.seed, |index_ca, rngs| {
        UInt64Chunked::from_iter_values(
            name,
            index_ca.into_no_null_iter().map(|i| {
                let mut rng = rngs.rng(i);
                <Binomial as RandDistribution<u64>>::sample(&dist, &mut rng)
            }),
        )
    })
}

/// Element-wise pmf via `statrs` `Discrete::pmf`; zero off the integer support (`value < 0`,
/// non-integer, or `value > n`). See [`value_keyed`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn binomial_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, v| {
        Some(support_point(v).map_or(0.0, |k| dist.pmf(k)))
    })
}

/// Element-wise log-pmf via native `Discrete::ln_pmf` (more accurate than `pmf().ln()`);
/// `-inf` off the integer support.
#[polars_expr(output_type=Float64)]
fn binomial_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, v| {
        Some(support_point(v).map_or(f64::NEG_INFINITY, |k| dist.ln_pmf(k)))
    })
}

/// Element-wise cdf `P(X <= floor(value))` via `statrs` `DiscreteCDF::cdf`; `0` for `value < 0`,
/// `1` for `value >= n`.
#[polars_expr(output_type=Float64)]
fn binomial_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, v| {
        if v < 0.0 {
            Some(0.0)
        } else {
            Some(dist.cdf(v.floor() as u64))
        }
    })
}

/// Element-wise survival function `P(X > floor(value))` via native `DiscreteCDF::sf` (accurate in
/// the upper tail); `1` for `value < 0`, `0` for `value >= n`.
#[polars_expr(output_type=Float64)]
fn binomial_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, v| {
        if v < 0.0 {
            Some(1.0)
        } else {
            Some(dist.sf(v.floor() as u64))
        }
    })
}

/// Slack absorbing the rounding error in `statrs`' regularized-incomplete-beta cdf when comparing
/// against a quantile that sits exactly on a cdf step. The cdf is accurate to a few ULP (~1e-15),
/// so 1e-12 reliably treats `cdf(k) == q` as satisfied (matching scipy's integer-valued ppf) while
/// staying far below the gap between distinct adjacent cdf values at any quantile a caller would
/// pass.
const PPF_CDF_TOL: f64 = 1e-12;

/// Smallest support point `k in {0, ..., n}` with `cdf(k) >= q`, by binary search over the bounded
/// discrete CDF.
///
/// This is the inverse-cdf statrs documents for `Binomial` (a search over the CDF), reimplemented
/// here because `statrs`' generic `DiscreteCDF::inverse_cdf` panics for small `n` (it `unwrap`s a
/// bisection that returns `None`). The comparison carries [`PPF_CDF_TOL`] so a quantile exactly on a
/// cdf step is not pushed to the next support point by the cdf's last-ULP error. `q == 0` returns the
/// support minimum `0`; `q == 1` returns `n`.
fn inverse_cdf(dist: &Binomial, q: f64) -> u64 {
    let mut lo = 0u64;
    let mut hi = dist.n();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if dist.cdf(mid) + PPF_CDF_TOL >= q {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Element-wise ppf (inverse cdf), returning the integer support point as `f64`.
///
/// A quantile outside `[0, 1]` yields `null`. At the endpoints this returns the support bounds
/// (`ppf(0) = 0`, `ppf(1) = n`); scipy's below-support sentinel `ppf(0) = -1` is not reproduced, so
/// parity comparisons restrict to interior quantiles.
#[polars_expr(output_type=Float64)]
fn binomial_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, q| {
        if !(0.0..=1.0).contains(&q) {
            None
        } else {
            Some(inverse_cdf(dist, q) as f64)
        }
    })
}

/// Validate the `(n, p)` parameterisation and return the validated `p`.
///
/// `inputs[0]` is `n`, `inputs[1]` is `p`. Mirrors `normal_std_dev` / `bernoulli_proba`: the
/// closed-form moments (`mean = n * p`, `variance = n * p * (1 - p)`) are computed from Polars
/// expressions and gated on this single FFI round-trip, so they report an invalid parameterisation
/// identically to the value-keyed methods that build the distribution directly, rather than silently
/// computing a moment from a negative `n` or an out-of-range `p`. `null` in either input propagates;
/// a negative `n` or a `NaN` / out-of-range `p` raises `InvalidOperation` via [`build_dist`].
#[polars_expr(output_type=Float64)]
fn binomial_params(inputs: &[Series]) -> PolarsResult<Series> {
    let n = inputs[0].cast(&DataType::Int64)?;
    let n_ca = n.i64()?;
    let p = inputs[1].cast(&DataType::Float64)?;
    let p_ca = p.f64()?;
    let name = inputs[1].name().clone();

    let ca: Float64Chunked =
        try_binary_elementwise(n_ca, p_ca, |n_opt, p_opt| -> PolarsResult<Option<f64>> {
            match (n_opt, p_opt) {
                (Some(n), Some(p)) => {
                    build_dist(n, p)?;
                    Ok(Some(p))
                },
                _ => Ok(None),
            }
        })?;

    Ok(ca.with_name(name).into_series())
}

/// Element-wise Shannon entropy (in nats) via `statrs` `Distribution::entropy`, the exact support
/// sum `-sum_k pmf(k) ln pmf(k)`; `0` at the degenerate endpoints `p in {0, 1}`.
///
/// Kept in Rust, unlike `mean` / `variance`: the binomial entropy has no elementary closed form (it
/// is a sum over the whole `{0, ..., n}` support, which scipy also evaluates exactly), so there is no
/// numerically equivalent Polars expression to move it to.
#[polars_expr(output_type=Float64)]
fn binomial_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    params_keyed(inputs, |dist| dist.entropy().unwrap())
}
