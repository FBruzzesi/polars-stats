use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution;
use statrs::distribution::Bernoulli;

use crate::distributions::{
    align_inputs, on_unit_interval, value_keyed_derived_per_row, value_keyed_derived_scalar,
    ParamDomain,
};
use crate::rng::{
    binary_param_rows, sample_by_index, sample_per_row_binary, samples_bool_output,
    samples_by_index, samples_per_row, SampleKwargs, SampleScalarKwargs, SamplesKwargs,
    SamplesScalarKwargs,
};

const P: ParamDomain = ParamDomain::probability("p");

/// Cannot fail behind [`P`]'s pass; the `statrs` error is kept as the backstop.
fn build_dist(proba: f64) -> PolarsResult<Bernoulli> {
    Bernoulli::new(proba).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Bernoulli's constant success probability, deserialised once per call.
#[derive(serde::Deserialize)]
struct BernoulliParamsKwargs {
    p: f64,
}

impl BernoulliParamsKwargs {
    fn build(&self) -> PolarsResult<Bernoulli> {
        P.check(self.p)?;
        build_dist(self.p)
    }

    /// Binds the constant parameter and its domain into [`value_keyed_derived_scalar`], which
    /// validates and derives once per call rather than per row.
    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(f64) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        value_keyed_derived_scalar(value, self.p, &P, derive, select)
    }
}

/// `p` unchanged after [`P`]'s column pass, `null` included. The moments derive from this so an
/// invalid `p` raises as it does from `bernoulli_sample`, instead of computing a negative variance.
#[polars_expr(output_type=Float64)]
fn bernoulli_proba(inputs: &[Series]) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;
    P.check_column(proba.f64()?)?;
    Ok(proba)
}

// Per-method bodies, shared by the per-row plugins and their `*_scalar` twins. Each method pairs a
// `derive` that turns `p` into its branch answers with an `at` that picks one by where the value
// sits. `derive` runs once per call on the constant path and once per row only when `p` is a column,
// which is what keeps the constant path from recomputing a logarithm 10M times.

/// `pmf` / `log_pmf`: the answer at each of the two support points, and off the support.
struct Mass {
    at_zero: f64,
    at_one: f64,
    off_support: f64,
}

impl Mass {
    /// `1 - p` at 0, `p` at 1, `0` off the support.
    fn pmf(p: f64) -> Self {
        Mass {
            at_zero: 1.0 - p,
            at_one: p,
            off_support: 0.0,
        }
    }

    /// `ln_1p(-p)` at 0, `ln(p)` at 1, `-inf` off the support.
    ///
    /// `ln_1p`, not `ln(1 - p)`: the latter collapses to `0.0` below `p ~ 1.1e-16`. Both logarithms
    /// are derived where a row reads one, which costs the column path a spare `ln` and saves the
    /// constant path a per-row one.
    fn ln_pmf(p: f64) -> Self {
        Mass {
            at_zero: (-p).ln_1p(),
            at_one: p.ln(),
            off_support: f64::NEG_INFINITY,
        }
    }

    /// Exact `f64` equality, never a cast to an integer type, so a non-integral value is off the
    /// support rather than an error: `pmf(0.5)` is `0.0`.
    fn at(&self, value: f64) -> Option<f64> {
        Some(if value == 0.0 {
            self.at_zero
        } else if value == 1.0 {
            self.at_one
        } else {
            self.off_support
        })
    }
}

/// `cdf` / `log_cdf` / `sf` / `log_sf`: the answer on each side of the two steps, at 0 and at 1.
struct Steps {
    below_zero: f64,
    below_one: f64,
    at_least_one: f64,
}

impl Steps {
    /// `0` below 0, `1 - p` on `[0, 1)`, `1` at or above 1.
    fn cdf(p: f64) -> Self {
        Steps {
            below_zero: 0.0,
            below_one: 1.0 - p,
            at_least_one: 1.0,
        }
    }

    /// `-inf` below 0, `ln_1p(-p)` on `[0, 1)`, `0` at or above 1; same `ln_1p` reason as
    /// [`Mass::ln_pmf`].
    fn ln_cdf(p: f64) -> Self {
        Steps {
            below_zero: f64::NEG_INFINITY,
            below_one: (-p).ln_1p(),
            at_least_one: 0.0,
        }
    }

    /// `1` below 0, `p` on `[0, 1)`, `0` at or above 1.
    ///
    /// Read straight off `p`, never as `1 - cdf`: that recomputes `p` as `1 - (1 - p)` and so
    /// quantises it to the `1.1e-16` spacing of `1.0`, reaching `0.0` below that.
    fn sf(p: f64) -> Self {
        Steps {
            below_zero: 1.0,
            below_one: p,
            at_least_one: 0.0,
        }
    }

    /// The plain log of [`Steps::sf`], slot for slot: `ln(1) = 0` and `ln(0) = -inf` are exact, and
    /// the middle slot is `p` itself, so there is no complement for an `ln_1p` to rescue.
    fn ln_sf(p: f64) -> Self {
        Steps {
            below_zero: 0.0,
            below_one: p.ln(),
            at_least_one: f64::NEG_INFINITY,
        }
    }

    fn at(&self, value: f64) -> Option<f64> {
        Some(if value < 0.0 {
            self.below_zero
        } else if value < 1.0 {
            self.below_one
        } else {
            self.at_least_one
        })
    }
}

/// `ppf`: the cdf step the quantile is compared against, and the answer at `q == 1`.
struct PpfCutoffs {
    /// `1 - p`, the only step in the cdf.
    step: f64,
    /// `1.0` unless the mass at 1 is zero, in which case `0.0`.
    at_quantile_one: f64,
}

impl PpfCutoffs {
    fn derive(p: f64) -> Self {
        PpfCutoffs {
            step: 1.0 - p,
            at_quantile_one: f64::from(p > 0.0),
        }
    }

    /// Smallest `x` with `cdf(x) >= q`: `0.0` if `q <= 1 - p` else `1.0`; `None` (null) outside
    /// `[0, 1]`. The support point is a `Float64`, not a `Boolean`, so `NaN` stays representable.
    ///
    /// `q == 1` is a separate branch because `1 - p` rounds to exactly `1.0` below `p ~ 1.1e-16`,
    /// which made `q > 1 - p` false and answered `0.0` where `1.0` is the only correct answer. Every
    /// representable `q < 1` is safely below the true `1 - p` for such a `p`, so the branch is
    /// needed only at the endpoint; `p == 0` keeps `0.0`.
    fn at(&self, quantile: f64) -> Option<f64> {
        (0.0..=1.0).contains(&quantile).then(|| {
            if quantile == 1.0 {
                self.at_quantile_one
            } else {
                f64::from(quantile > self.step)
            }
        })
    }
}

/// Smallest `x` with `sf(x) <= q`: `1.0` if `p > q` else `0.0`; null outside `[0, 1]`.
///
/// Compared against `q` itself, never as `ppf(1 - q)`: `sf(0)` *is* `p`, so this comparison forms no
/// complement, while `ppf(1 - q)` forms two and loses the answer whenever either saturates
/// (`Bernoulli(1e-17).isf(1e-20)` gave `0.0` against a true `1.0`). So `isf` reads `p` directly and
/// hoists nothing.
fn derive_isf(p: f64) -> impl Fn(f64) -> f64 {
    move |quantile: f64| f64::from(p > quantile)
}

/// Element-wise pmf: `1 - p` at 0, `p` at 1, `0` off the support.
/// See [`value_keyed_derived_per_row`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn bernoulli_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, &P, Mass::pmf, Mass::at)
}

/// Element-wise log-pmf; see [`Mass::ln_pmf`] for the `ln_1p` reason.
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, &P, Mass::ln_pmf, Mass::at)
}

/// Element-wise cdf `P(X <= value)`; see [`Steps::cdf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, &P, Steps::cdf, Steps::at)
}

/// Element-wise log-cdf; see [`Steps::ln_cdf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, &P, Steps::ln_cdf, Steps::at)
}

/// Element-wise survival function `P(X > value)`; see [`Steps::sf`] for why it is not `1 - cdf`.
#[polars_expr(output_type=Float64)]
fn bernoulli_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, &P, Steps::sf, Steps::at)
}

/// Element-wise log-sf; see [`Steps::ln_sf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, &P, Steps::ln_sf, Steps::at)
}

/// Element-wise ppf (inverse cdf), returning the support point as `f64`; see [`PpfCutoffs::at`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, &P, PpfCutoffs::derive, PpfCutoffs::at)
}

/// Element-wise inverse survival function; see [`derive_isf`] for why it never forms a complement.
#[polars_expr(output_type=Float64)]
fn bernoulli_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, &P, derive_isf, on_unit_interval)
}

/// Constant-parameter fast path for [`bernoulli_pmf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_pmf_scalar(inputs: &[Series], kwargs: BernoulliParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Mass::pmf, Mass::at)
}

/// Constant-parameter fast path for [`bernoulli_ln_pmf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_pmf_scalar(
    inputs: &[Series],
    kwargs: BernoulliParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Mass::ln_pmf, Mass::at)
}

/// Constant-parameter fast path for [`bernoulli_cdf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_cdf_scalar(inputs: &[Series], kwargs: BernoulliParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Steps::cdf, Steps::at)
}

/// Constant-parameter fast path for [`bernoulli_ln_cdf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_cdf_scalar(
    inputs: &[Series],
    kwargs: BernoulliParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Steps::ln_cdf, Steps::at)
}

/// Constant-parameter fast path for [`bernoulli_sf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_sf_scalar(inputs: &[Series], kwargs: BernoulliParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Steps::sf, Steps::at)
}

/// Constant-parameter fast path for [`bernoulli_ln_sf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_sf_scalar(
    inputs: &[Series],
    kwargs: BernoulliParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Steps::ln_sf, Steps::at)
}

/// Constant-parameter fast path for [`bernoulli_ppf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ppf_scalar(inputs: &[Series], kwargs: BernoulliParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], PpfCutoffs::derive, PpfCutoffs::at)
}

/// Constant-parameter fast path for [`bernoulli_isf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_isf_scalar(inputs: &[Series], kwargs: BernoulliParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

/// One Bernoulli draw from a `&mut` per-row RNG already seeded from `(root_seed, index)`.
///
/// Every Bernoulli sampler draws through here, per-row and fast path alike, so their bit-equality is
/// structural rather than only sampled by `sample_test.py`.
#[inline]
fn draw(dist: &Bernoulli, rng: &mut impl rand::Rng) -> bool {
    <Bernoulli as Distribution<bool>>::sample(dist, rng)
}

/// Element-wise Bernoulli sampler over `(p, row_index)`, returning `Boolean`.
///
/// Per row, `null` propagates; an invalid `p` raises from [`P`]'s column pass first. Seeding and
/// chunk-invariance follow [`sample_per_row_binary`].
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();
    P.check_column(proba.f64()?)?;

    sample_per_row_binary(
        name,
        proba.f64()?,
        index.u64()?,
        kwargs.seed,
        build_dist,
        draw,
    )
}

/// Constant-parameter fast path for [`bernoulli_sample`].
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<BernoulliParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    sample_by_index(name, &inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

/// Constant-parameter multi-draw fast path: the `samples` twin of [`bernoulli_sample_scalar`].
///
/// `size` consecutive draws from each row's stream, so `samples(size=1)` matches `sample` bit for
/// bit and the distribution is built once per call. Returns `Array(Boolean, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<BernoulliParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    samples_by_index(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&dist, rng)
    })
}

/// Element-wise multi-draw Bernoulli sampler: `size` draws per row in one call, the distribution
/// built once per row. Returns `Array(Boolean, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`bernoulli_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();
    P.check_column(proba.f64()?)?;

    let rows = binary_param_rows(proba.f64()?, index.u64()?, build_dist);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, draw)
}
