use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution;
use statrs::distribution::Bernoulli;

use crate::distributions::{
    align_inputs, validate_params_unary, value_keyed_derived_per_row, value_keyed_derived_scalar,
};
use crate::rng::{
    binary_param_rows, sample_by_index, sample_per_row_binary, samples_bool_output,
    samples_by_index, samples_per_row, SampleKwargs, SampleScalarKwargs, SamplesKwargs,
    SamplesScalarKwargs,
};

fn build_dist(proba: f64) -> PolarsResult<Bernoulli> {
    Bernoulli::new(proba).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in [0, 1], got {proba}: {e}").into())
    })
}

/// Bernoulli's constant success probability, deserialised once per call.
#[derive(serde::Deserialize)]
struct BernoulliParamsKwargs {
    p: f64,
}

impl BernoulliParamsKwargs {
    fn build(&self) -> PolarsResult<Bernoulli> {
        build_dist(self.p)
    }

    /// Binds the constant parameter and `build_dist` into [`value_keyed_derived_scalar`], which owns
    /// the validate-and-derive-once contract the fast path rests on.
    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(Option<f64>) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        value_keyed_derived_scalar(value, self.p, build_dist, derive, select)
    }
}

/// Element-wise validation of the success probability: returns `p` unchanged, raising
/// `InvalidOperation` if `p` is outside `[0, 1]`. `null` propagates.
///
/// The moments derive from this so they report an invalid `p` consistently with `bernoulli_sample`,
/// instead of silently computing a negative variance. The value-keyed methods validate inside their
/// own plugin instead.
#[polars_expr(output_type=Float64)]
fn bernoulli_proba(inputs: &[Series]) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;

    validate_params_unary(proba.f64()?, |proba| {
        build_dist(proba)?;
        Ok(proba)
    })
}

// Per-method bodies, shared by the per-row plugins and their `*_scalar` twins. Each method pairs a
// `derive` that turns `p` into its branch answers with an `at` that picks one by where the value
// sits. `derive` runs once per call on the constant path and once per row only when `p` is a column,
// which is what keeps the constant path from recomputing a logarithm 10M times.
//
// Every slot is an `Option<f64>`, so a null `p` nulls exactly the answers that read it. The slots
// holding a plain `Some(constant)` are the off-support answers (`pmf(2) = 0`, `cdf(-1) = 0`,
// `sf(1) = 0`), which are documented to survive a null `p`: writing them as constants leaves no `p`
// in scope to thread through them by accident. Pinned by
// `tests/distributions/bernoulli/null_param_test.py`.

/// `pmf` / `log_pmf`: the answer at each of the two support points, and off the support.
struct Mass {
    at_zero: Option<f64>,
    at_one: Option<f64>,
    off_support: Option<f64>,
}

impl Mass {
    /// `1 - p` at 0, `p` at 1, `0` off the support.
    fn pmf(p: Option<f64>) -> Self {
        Mass {
            at_zero: p.map(|p| 1.0 - p),
            at_one: p,
            off_support: Some(0.0),
        }
    }

    /// `ln_1p(-p)` at 0, `ln(p)` at 1, `-inf` off the support.
    ///
    /// `ln_1p`, not `ln(1 - p)`: the latter collapses to `0.0` below `p ~ 1.1e-16`. Both logarithms
    /// are derived where a row reads one, which costs the column path a spare `ln` and saves the
    /// constant path a per-row one.
    fn ln_pmf(p: Option<f64>) -> Self {
        Mass {
            at_zero: p.map(|p| (-p).ln_1p()),
            at_one: p.map(f64::ln),
            off_support: Some(f64::NEG_INFINITY),
        }
    }

    /// Exact `f64` equality, never a cast to an integer type, so a non-integral value is off the
    /// support rather than an error: `pmf(0.5)` is `0.0`.
    fn at(&self, value: f64) -> Option<f64> {
        if value == 0.0 {
            self.at_zero
        } else if value == 1.0 {
            self.at_one
        } else {
            self.off_support
        }
    }
}

/// `cdf` / `log_cdf` / `sf` / `log_sf`: the answer on each side of the two steps, at 0 and at 1.
struct Steps {
    below_zero: Option<f64>,
    below_one: Option<f64>,
    at_least_one: Option<f64>,
}

impl Steps {
    /// `0` below 0, `1 - p` on `[0, 1)`, `1` at or above 1.
    fn cdf(p: Option<f64>) -> Self {
        Steps {
            below_zero: Some(0.0),
            below_one: p.map(|p| 1.0 - p),
            at_least_one: Some(1.0),
        }
    }

    /// `-inf` below 0, `ln_1p(-p)` on `[0, 1)`, `0` at or above 1; same `ln_1p` reason as
    /// [`Mass::ln_pmf`].
    fn ln_cdf(p: Option<f64>) -> Self {
        Steps {
            below_zero: Some(f64::NEG_INFINITY),
            below_one: p.map(|p| (-p).ln_1p()),
            at_least_one: Some(0.0),
        }
    }

    /// `1` below 0, `p` on `[0, 1)`, `0` at or above 1.
    ///
    /// Read straight off `p`, never as `1 - cdf`: that recomputes `p` as `1 - (1 - p)` and so
    /// quantises it to the `1.1e-16` spacing of `1.0`, reaching `0.0` below that.
    fn sf(p: Option<f64>) -> Self {
        Steps {
            below_zero: Some(1.0),
            below_one: p,
            at_least_one: Some(0.0),
        }
    }

    /// The plain log of [`Steps::sf`], slot for slot: `ln(1) = 0` and `ln(0) = -inf` are exact, and
    /// the middle slot is `p` itself, so there is no complement for an `ln_1p` to rescue.
    fn ln_sf(p: Option<f64>) -> Self {
        Steps {
            below_zero: Some(0.0),
            below_one: p.map(f64::ln),
            at_least_one: Some(f64::NEG_INFINITY),
        }
    }

    fn at(&self, value: f64) -> Option<f64> {
        if value < 0.0 {
            self.below_zero
        } else if value < 1.0 {
            self.below_one
        } else {
            self.at_least_one
        }
    }
}

/// `ppf`: the cdf step the quantile is compared against, and the answer at `q == 1`.
struct PpfCutoffs {
    /// `1 - p`, the only step in the cdf.
    step: Option<f64>,
    /// `1.0` unless the mass at 1 is zero, in which case `0.0`.
    at_quantile_one: Option<f64>,
}

impl PpfCutoffs {
    fn derive(p: Option<f64>) -> Self {
        PpfCutoffs {
            step: p.map(|p| 1.0 - p),
            at_quantile_one: p.map(|p| f64::from(p > 0.0)),
        }
    }

    /// Smallest `x` with `cdf(x) >= q`: `0.0` if `q <= 1 - p` else `1.0`; `None` (null) outside
    /// `[0, 1]`. The support point comes back as `Float64` rather than a `Boolean` so the wrapper's
    /// `NaN -> NaN` contract stays representable.
    ///
    /// `q == 1` is a separate branch because `1 - p` rounds to exactly `1.0` below `p ~ 1.1e-16`,
    /// which made `q > 1 - p` false and answered `0.0` where `1.0` is the only correct answer. Every
    /// representable `q < 1` is safely below the true `1 - p` for such a `p`, so the branch is
    /// needed only at the endpoint; `p == 0` keeps `0.0`.
    fn at(&self, quantile: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&quantile) {
            return None;
        }
        if quantile == 1.0 {
            self.at_quantile_one
        } else {
            self.step.map(|step| f64::from(quantile > step))
        }
    }
}

/// `isf` compares against `p` itself, so there is nothing to derive.
fn isf_proba(p: Option<f64>) -> Option<f64> {
    p
}

/// Smallest `x` with `sf(x) <= q`: `1.0` if `p > q` else `0.0`; `None` (null) outside `[0, 1]`.
///
/// Tested against `q` itself, never as `ppf(1 - q)`: `sf(0)` *is* `p`, so this comparison forms no
/// complement, while `ppf(1 - q)` forms two and loses the answer whenever either saturates
/// (`Bernoulli(1e-17).isf(1e-20)` gave `0.0` against a true `1.0`).
fn isf_at(p: &Option<f64>, quantile: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&quantile) {
        return None;
    }
    p.map(|p| f64::from(p > quantile))
}

/// Element-wise pmf: `1 - p` at 0, `p` at 1, `0` off the support.
/// See [`value_keyed_derived_per_row`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn bernoulli_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, Mass::pmf, Mass::at)
}

/// Element-wise log-pmf; see [`Mass::ln_pmf`] for the `ln_1p` reason.
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, Mass::ln_pmf, Mass::at)
}

/// Element-wise cdf `P(X <= value)`; see [`Steps::cdf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, Steps::cdf, Steps::at)
}

/// Element-wise log-cdf; see [`Steps::ln_cdf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, Steps::ln_cdf, Steps::at)
}

/// Element-wise survival function `P(X > value)`; see [`Steps::sf`] for why it is not `1 - cdf`.
#[polars_expr(output_type=Float64)]
fn bernoulli_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, Steps::sf, Steps::at)
}

/// Element-wise log-sf; see [`Steps::ln_sf`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, Steps::ln_sf, Steps::at)
}

/// Element-wise ppf (inverse cdf), returning the support point as `f64`; see [`PpfCutoffs::at`].
#[polars_expr(output_type=Float64)]
fn bernoulli_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, PpfCutoffs::derive, PpfCutoffs::at)
}

/// Element-wise inverse survival function; see [`isf_at`] for why it never forms a complement.
#[polars_expr(output_type=Float64)]
fn bernoulli_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, isf_proba, isf_at)
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
    kwargs.value_keyed(&inputs[0], isf_proba, isf_at)
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
/// Per row, `null` propagates and an invalid `p` raises via [`build_dist`]. Seeding and
/// chunk-invariance follow [`sample_per_row_binary`].
#[polars_expr(output_type=Boolean)]
fn bernoulli_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

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

    let rows = binary_param_rows(proba.f64()?, index.u64()?, build_dist);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, draw)
}
