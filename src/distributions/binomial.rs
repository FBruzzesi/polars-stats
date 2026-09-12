use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use rand_distr::Binomial as BinomialSampler;
use statrs::distribution::{Binomial, Discrete, DiscreteCDF};
use statrs::statistics::Distribution as StatrsDistribution;

use crate::distributions::{
    coerce_f64, param_keyed, validated_pair, value_keyed_scalar, value_keyed_ternary, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_per_row_ternary,
    samples_u64_output, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

/// `n` has no rule of its own beyond the `u64` [`coerce_n`] makes it.
const P: ParamDomain = ParamDomain::probability("p");

fn check_params(_n: &UInt64Chunked, p: &Float64Chunked) -> PolarsResult<()> {
    P.check_column(p)
}

/// `statrs::Binomial::new(p, n)` takes the arguments in the opposite order to this crate's `(n, p)`.
fn build_dist(n: u64, p: f64) -> PolarsResult<Binomial> {
    Binomial::new(p, n).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// The samplers draw from `rand_distr::Binomial`, `O(1)`-amortised (inversion for small `n p`, BTPE
/// otherwise), where `statrs`' draw is one uniform per trial. Both accept the same parameter set.
fn build_sampler(n: u64, p: f64) -> PolarsResult<BinomialSampler> {
    BinomialSampler::new(n, p).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {p}: {e}").into())
    })
}

/// An `n` column as the `UInt64` both distribution types take. A float dtype is refused rather than
/// cast, since a cast would truncate `2.7` to `2` while the Python closed-form moments keep `2.7`;
/// the strict cast reports a negative `n`, where the default cast would null the row. Widening to
/// `UInt64` rather than `Int64` keeps the whole `u64` range reachable.
///
/// A `Null`-dtype column widens to all-null so null-in-null-out holds; the dtype gate stays
/// column-level, so a float column is refused even when every value in it is null.
fn coerce_n(n: &Series) -> PolarsResult<UInt64Chunked> {
    let dtype = n.dtype();
    if dtype == &DataType::Null {
        return Ok(n.cast(&DataType::UInt64)?.u64()?.clone());
    }
    if !dtype.is_integer() {
        let msg = format!(
            "n must be an integer column, got {dtype}: cast it explicitly, e.g. \
             `pl.col(\"n\").cast(pl.Int64)`. n counts trials, so casting it here would silently \
             truncate a fractional value"
        );
        return Err(PolarsError::InvalidOperation(msg.into()));
    }
    let widened = n.strict_cast(&DataType::UInt64).map_err(|e| {
        PolarsError::InvalidOperation(format!("n must be a non-negative integer: {e}").into())
    })?;
    Ok(widened.u64()?.clone())
}

/// Constant parameters, deserialised once per call. The value-keyed `_scalar` twins build the
/// `statrs` distribution through [`Self::build`], the samplers the `rand_distr` sampler through
/// [`Self::build_sampler`]; neither can rebuild per row. Swapping `(n, p)` is a compile error.
#[derive(serde::Deserialize)]
struct BinomialParams {
    n: u64,
    p: f64,
}

impl BinomialParams {
    fn build(&self) -> PolarsResult<Binomial> {
        P.check(self.p)?;
        build_dist(self.n, self.p)
    }

    fn build_sampler(&self) -> PolarsResult<BinomialSampler> {
        P.check(self.p)?;
        build_sampler(self.n, self.p)
    }

    fn value_keyed<Body>(&self, value: &Series, body: Body) -> PolarsResult<Series>
    where
        Body: Fn(&Binomial, f64) -> Option<f64>,
    {
        value_keyed_scalar(value, &self.build()?, body)
    }
}

fn value_keyed<Body>(inputs: &[Series], body: Body) -> PolarsResult<Series>
where
    Body: Fn(&Binomial, f64) -> Option<f64>,
{
    value_keyed_ternary(inputs, coerce_n, coerce_f64, check_params, build_dist, body)
}

/// `Some(k)` for a non-negative integer point, `None` off the support. `value > n` still maps to
/// `Some`; `statrs` returns zero mass there. The cast saturates to `u64::MAX` beyond the integer
/// range, which `statrs` treats as `> n`.
#[inline]
fn support_point(value: f64) -> Option<u64> {
    if value < 0.0 || value.fract() != 0.0 {
        None
    } else {
        Some(value as u64)
    }
}

#[inline]
fn draw(dist: &BinomialSampler, rng: &mut impl rand::Rng) -> u64 {
    dist.sample(rng)
}

#[polars_expr(output_type=UInt64)]
fn binomial_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_ternary(
        inputs,
        kwargs,
        coerce_n,
        coerce_f64,
        check_params,
        build_sampler,
        draw,
    )
}

#[polars_expr(output_type=UInt64)]
fn binomial_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<BinomialParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build_sampler()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn binomial_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<BinomialParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build_sampler()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn binomial_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_ternary(
        inputs,
        kwargs,
        coerce_n,
        coerce_f64,
        check_params,
        build_sampler,
        draw,
    )
}

// The bodies never see a `NaN` point: `NaN < 0.0` is false and `NaN.floor() as u64` is `0`, so an
// unguarded `cdf` would answer a confident `P(X <= 0)`. The drivers' short-circuit keeps it out.

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

/// Relative slack on the cdf-step comparison in [`inverse_cdf`]. The regularized-incomplete-beta
/// cdf is accurate to a few ulp (`~1e-15`), so `1e-12` treats `cdf(k) == q` as satisfied (scipy's
/// integer-valued `ppf`) while staying far below the gap between adjacent cdf values. Relative, not
/// absolute: an absolute slack exceeds every quantile below it and would collapse the search to `0`.
const PPF_CDF_TOL: f64 = 1e-12;

/// Smallest `k in {0, ..., n}` with `cdf(k) >= q (1 - tol)`, by binary search; the caller maps the
/// closed endpoints, so `q` here is interior.
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

/// Null outside `[0, 1]`; the endpoints map to the support bounds (`ppf(0) = 0`, `ppf(1) = n`),
/// not scipy's `-1` sentinel. `q == 1` is mapped here because `cdf(k)` reaches `1.0` far below `n`
/// once the upper tail underflows, and the search would stop there.
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

fn isf_value(dist: &Binomial, q: f64) -> Option<f64> {
    ppf_value(dist, 1.0 - q)
}

#[polars_expr(output_type=Float64)]
fn binomial_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pmf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pmf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_pmf_scalar(inputs: &[Series], kwargs: BinomialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pmf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_ln_pmf_scalar(inputs: &[Series], kwargs: BinomialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pmf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_cdf_scalar(inputs: &[Series], kwargs: BinomialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_sf_scalar(inputs: &[Series], kwargs: BinomialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_ppf_scalar(inputs: &[Series], kwargs: BinomialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_isf_scalar(inputs: &[Series], kwargs: BinomialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

#[polars_expr(output_type=Float64)]
fn binomial_params(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_n, check_params)
}

/// Shannon entropy (nats) as the exact support sum `-sum_k pmf(k) ln pmf(k)`, which has no Polars
/// expression to move to. `n == u64::MAX` is refused here and nowhere else: it is a valid trial count
/// for every other method, but `statrs` iterates `0..n + 1`, which wraps to an empty range and returns
/// a confident `0.0`.
#[polars_expr(output_type=Float64)]
fn binomial_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    param_keyed(inputs, coerce_n, coerce_f64, check_params, |n, p| {
        // TODO(FBruzzesi): Remove once fixed upstream in statrs
        polars_ensure!(
            n != u64::MAX,
            ComputeError: "n = {} overflows the entropy support sum: statrs iterates 0..=n via n + 1, which wraps at u64::MAX", n
        );
        Ok(build_dist(n, p)?
            .entropy()
            .expect("Binomial::entropy is Some for every valid (n, p)"))
    })
}
