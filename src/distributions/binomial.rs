#![allow(clippy::unused_unit)]
use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use rand_distr::Binomial as BinomialSampler;
use statrs::distribution::{Binomial, Discrete, DiscreteCDF};
use statrs::statistics::Distribution as StatrsDistribution;

use crate::distributions::{param_validator, value_keyed_per_row, value_keyed_scalar_plugins};
use crate::rng::{
    sample_scalar_plugin, samples_per_row, samples_u64_output, ternary_param_rows, SampleKwargs,
    SamplesKwargs,
};

/// Construct a `statrs::Binomial`, mapping both invalid-parameter cases to a `ComputeError`.
///
/// `statrs::Binomial::new(p, n)` takes the arguments in the opposite order to this crate's `(n, p)`
/// and only rejects a `NaN` `p` or `p` outside `[0, 1]`; its `n` is a `u64`, so a negative trial
/// count cannot reach it. `n >= 0` is validated here, and both failures surface as `InvalidOperation`,
/// so an invalid parameterisation fails the whole evaluation rather than silently nulling the row.
fn build_dist(n: i64, p: f64) -> PolarsResult<Binomial> {
    let trials = u64::try_from(n).map_err(|_| {
        PolarsError::InvalidOperation(format!("n must be a non-negative integer, got {n}").into())
    })?;
    Binomial::new(p, trials).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {p}: {e}").into())
    })
}

/// Construct a `rand_distr::Binomial` for sampling, mirroring [`build_dist`]'s validation contract.
///
/// `statrs`' `Distribution<u64>` draw for `Binomial` is `(0..n).fold(...)`: one uniform per trial, so
/// `O(n)` draws per sampled row. `rand_distr`'s sampler is `O(1)`-amortised (inversion for small
/// `n*p`, BTPE otherwise), so the sampler builds *this* distribution rather than the `statrs` one.
/// The accepted/rejected parameter set is identical to [`build_dist`] (`n >= 0`, `p` finite in
/// `[0, 1]`; `rand_distr` rejects `NaN` via `!(p >= 0.0)`), and the error messages share the same
/// prefixes, so validation behaviour is unchanged from the value-keyed methods. The value-keyed
/// methods keep building the `statrs` distribution; only the sampler uses this.
fn build_sampler(n: i64, p: f64) -> PolarsResult<BinomialSampler> {
    let trials = u64::try_from(n).map_err(|_| {
        PolarsError::InvalidOperation(format!("n must be a non-negative integer, got {n}").into())
    })?;
    BinomialSampler::new(trials, p).map_err(|e| {
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

value_keyed_per_row! {
    /// Apply a value-keyed `f(dist, value)` element-wise over `(value, n, p)`; shared by `pmf`,
    /// `ln_pmf`, `cdf`, `sf`, `ppf`. `null` propagates; an invalid parameterisation raises via
    /// [`build_dist`]; `f` may return `None` to null a row on its own terms.
    fn value_keyed(&Binomial);
    params = (DataType::Int64 => i64, DataType::Float64 => f64);
    build = build_dist;
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

/// Element-wise Binomial sampler over `(n, p, row_index)`, returning `UInt64`.
///
/// Per row, `null` propagates and an invalid parameterisation raises via [`build_sampler`].
/// Seeding and chunk-invariance follow [`SampleKwargs::row_rngs`].
#[polars_expr(output_type=UInt64)]
fn binomial_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let n = inputs[0].cast(&DataType::Int64)?;
    let n_ca = n.i64()?;
    let p = inputs[1].cast(&DataType::Float64)?;
    let p_ca = p.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs()?;

    let ca: UInt64Chunked = try_ternary_elementwise(
        n_ca,
        p_ca,
        index_ca,
        |n_opt, p_opt, i_opt| -> PolarsResult<Option<u64>> {
            match (n_opt, p_opt, i_opt) {
                (Some(n), Some(p), Some(i)) => {
                    let dist = build_sampler(n, p)?;
                    let mut rng = rngs.rng(i);
                    Ok(Some(dist.sample(&mut rng)))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

sample_scalar_plugin! {
    struct BinomialScalarKwargs { n: i64, p: f64 }

    /// Constant-parameter fast path for [`binomial_sample`], on the same [`build_sampler`].
    fn binomial_sample_scalar(output_type = UInt64, physical = UInt64Type);

    samples = binomial_samples_scalar as BinomialSamplesScalarKwargs -> samples_u64_output;

    build = |kw| build_sampler(kw.n, kw.p)?;
    draw = |dist, rng| dist.sample(rng);
}

/// Element-wise multi-draw Binomial sampler: `size` draws per row in one call, the `rand_distr`
/// sampler (whose BINV/BTPE setup is the expensive part) built once per row.
/// Returns `Array(UInt64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`binomial_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn binomial_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let n = inputs[0].cast(&DataType::Int64)?;
    let p = inputs[1].cast(&DataType::Float64)?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = ternary_param_rows(n.i64()?, p.f64()?, index.u64()?, build_sampler);

    samples_per_row::<UInt64Type, _, _, _, _>(name, kwargs.seed, kwargs.size, rows, |dist, rng| {
        dist.sample(rng)
    })
}

// Per-method bodies, shared by the per-row plugins and their `*_scalar` twins.
//
// The bodies never see a `NaN` value: the shared drivers short-circuit it to `NaN` first (see
// `value_keyed_scalar` in `mod.rs`). That guard is load-bearing here: `NaN < 0.0` is false and
// `NaN.floor() as u64` saturates to `0`, so an unguarded body would confidently return
// `P(X <= 0)` (and `pmf` a confident `0.0`) instead of propagating `NaN` (scipy semantics).

fn pmf_value(dist: &Binomial, v: f64) -> Option<f64> {
    Some(support_point(v).map_or(0.0, |k| dist.pmf(k)))
}

fn ln_pmf_value(dist: &Binomial, v: f64) -> Option<f64> {
    Some(support_point(v).map_or(f64::NEG_INFINITY, |k| dist.ln_pmf(k)))
}

fn cdf_value(dist: &Binomial, v: f64) -> Option<f64> {
    if v < 0.0 {
        Some(0.0)
    } else {
        Some(dist.cdf(v.floor() as u64))
    }
}

fn sf_value(dist: &Binomial, v: f64) -> Option<f64> {
    if v < 0.0 {
        Some(1.0)
    } else {
        Some(dist.sf(v.floor() as u64))
    }
}

/// Element-wise pmf via `statrs` `Discrete::pmf`; zero off the integer support (`value < 0`,
/// non-integer, or `value > n`), `NaN` for a `NaN` value. See [`value_keyed`] for the null/error
/// contract.
#[polars_expr(output_type=Float64)]
fn binomial_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pmf_value)
}

/// Element-wise log-pmf via native `Discrete::ln_pmf` (more accurate than `pmf().ln()`);
/// `-inf` off the integer support, `NaN` for a `NaN` value.
#[polars_expr(output_type=Float64)]
fn binomial_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pmf_value)
}

/// Element-wise cdf `P(X <= floor(value))` via `statrs` `DiscreteCDF::cdf`; `0` for `value < 0`,
/// `1` for `value >= n`, `NaN` for a `NaN` value.
#[polars_expr(output_type=Float64)]
fn binomial_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

/// Element-wise survival function `P(X > floor(value))` via native `DiscreteCDF::sf` (accurate in
/// the upper tail); `1` for `value < 0`, `0` for `value >= n`, `NaN` for a `NaN` value.
#[polars_expr(output_type=Float64)]
fn binomial_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
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
/// here so the cdf-step comparison carries the **relative** slack below, which `statrs`' generic
/// `DiscreteCDF::inverse_cdf` does not provide.
///
/// The step slack is **relative** ([`PPF_CDF_TOL`] scaled by `q`), not absolute: an absolute slack
/// exceeds every quantile below it, so `cdf(0) + tol >= q` held for any `q <= 1e-12` and the search
/// collapsed to `0`. The caller maps the closed endpoints, so `q` here is interior.
fn inverse_cdf(dist: &Binomial, q: f64) -> u64 {
    let threshold = q * (1.0 - PPF_CDF_TOL);
    let mut lo = 0u64;
    let mut hi = dist.n();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if dist.cdf(mid) >= threshold {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// A quantile outside `[0, 1]` yields `null`. At the endpoints this returns the support bounds
/// (`ppf(0) = 0`, `ppf(1) = n`); scipy's below-support sentinel `ppf(0) = -1` is not reproduced, so
/// parity comparisons restrict to interior quantiles.
///
/// `q == 1` is mapped here rather than left to the search: `cdf(k)` reaches `1.0` a long way below
/// `n` once the upper tail underflows, so the search would stop at the first such `k`.
fn ppf_value(dist: &Binomial, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else if q == 0.0 {
        Some(0.0)
    } else if q == 1.0 {
        Some(dist.n() as f64)
    } else {
        Some(inverse_cdf(dist, q) as f64)
    }
}

/// Element-wise ppf (inverse cdf), returning the integer support point as `f64`.
/// See [`ppf_value`] for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn binomial_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

value_keyed_scalar_plugins! {
    struct BinomialParamsKwargs { n: i64, p: f64 }

    build = |kw| build_dist(kw.n, kw.p)?;

    methods {
        fn binomial_pmf_scalar => pmf_value;
        fn binomial_ln_pmf_scalar => ln_pmf_value;
        fn binomial_cdf_scalar => cdf_value;
        fn binomial_sf_scalar => sf_value;
        fn binomial_ppf_scalar => ppf_value;
    }
}

param_validator! {
    /// Validate the `(n, p)` parameterisation and return the validated `p`.
    ///
    /// `inputs[0]` is `n`, `inputs[1]` is `p`. The Python closed-form moments (`mean = n * p`,
    /// `variance = n * p * (1 - p)`) are gated on this single FFI round-trip, so they raise on an
    /// invalid parameterisation exactly like the value-keyed methods. `null` in either input
    /// propagates; invalid raises via [`build_dist`].
    fn binomial_params;
    params = (n: DataType::Int64 => i64, p: DataType::Float64 => f64);
    build = build_dist;
    returns = p;
    output_name = inputs[1];
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
