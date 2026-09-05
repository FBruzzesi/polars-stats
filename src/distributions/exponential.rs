use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Exp;

use crate::distributions::{
    align_inputs, validate_params_unary, value_keyed_derived_per_row, value_keyed_derived_scalar,
    Domain,
};
use crate::rng::{
    binary_param_rows, sample_by_index, sample_per_row_binary, samples_by_index,
    samples_f64_output, samples_per_row, SampleKwargs, SampleScalarKwargs, SamplesKwargs,
    SamplesScalarKwargs,
};

/// Construct a `statrs::Exp`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::Exp::new` rejects a `NaN` rate or `rate <= 0` (a positive-infinite rate is accepted, as a
/// degenerate point mass at 0). That surfaces as `InvalidOperation`, so an invalid rate fails the whole
/// evaluation rather than silently nulling the row.
fn build_dist(rate: f64) -> PolarsResult<Exp> {
    Exp::new(rate).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("rate must be strictly positive, got rate={rate}: {e}").into(),
        )
    })
}

/// Exponential's constant rate, deserialised once per call.
#[derive(serde::Deserialize)]
struct ExponentialParamsKwargs {
    rate: f64,
}

impl ExponentialParamsKwargs {
    fn build(&self) -> PolarsResult<Exp> {
        build_dist(self.rate)
    }

    /// Binds the constant rate and `build_dist` into [`value_keyed_derived_scalar`], which
    /// validates and derives once per call rather than per row.
    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(Option<f64>) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        value_keyed_derived_scalar(value, self.rate, build_dist, derive, select)
    }
}

/// Element-wise validation of the rate (λ): returns `rate` unchanged, raising `InvalidOperation`
/// if `rate` is `NaN` or `rate <= 0`. `null` propagates.
///
/// The moments derive from this so they report an invalid rate consistently with
/// `exponential_sample`, instead of silently computing with a non-positive rate. The value-keyed
/// methods validate inside their own plugin instead.
#[polars_expr(output_type=Float64)]
fn exponential_rate(inputs: &[Series]) -> PolarsResult<Series> {
    let rate = inputs[0].cast(&DataType::Float64)?;

    validate_params_unary(rate.f64()?, |rate| {
        build_dist(rate)?;
        Ok(rate)
    })
}

/// Crossover of [`derive_cdf`], in units of `t = rate * x`.
///
/// Above `t = 1` the plain `1 - exp(-t)` is already exact, and the [`expm1`] identity that replaces
/// it below would overflow past `t ~ 1420`. The two branches agree to `1e-16` here.
const CDF_SINH_MAX: f64 = 1.0;

/// `exp(t) - 1` through the identity `2 exp(t / 2) sinh(t / 2)`, which has no subtraction to cancel.
///
/// Not `f64::exp_m1`, which differs from this identity by one ulp on roughly a fifth of the left
/// tail. Every expected value in `tests/distributions/exponential/` is this identity's rounding, and
/// `_base.expm1` spells it the same way for the distributions still assembled in Polars.
///
/// `sinh(t / 2)` overflows above `|t| ~ 1420`; the only caller crosses over at [`CDF_SINH_MAX`],
/// long before that.
#[inline]
fn expm1(t: f64) -> f64 {
    let half = t / 2.0;
    2.0 * half.exp() * half.sinh()
}

// Exponential hoists less than Bernoulli: every on-support answer is a function of the rate *and*
// the evaluation point, so a `derive_*` captures the rate-only terms and the rest stays in the arm.
// Free functions rather than associated ones, because each returns a distinct opaque arm type.

/// The two sides of the support, for the six value-keyed methods.
///
/// Below `x = 0` every answer is the rate-free support constant, so a null rate must not null it.
/// `below_support` is an `f64` rather than an `Option<f64>`, leaving nothing to thread a rate
/// through by accident; all six constants are pinned by
/// `tests/distributions/exponential/null_params_test.py`. `on_support` is `None` exactly when the
/// rate is null.
struct Sides<Arm> {
    below_support: f64,
    on_support: Option<Arm>,
}

impl<Arm: Fn(f64) -> f64> Sides<Arm> {
    /// `x < 0` takes the support constant, everything else the arm, so `-0.0` is on the support.
    ///
    /// A `NaN` point never reaches here: both drivers short-circuit it. That is what lets this be a
    /// bare `<`; the `!(x >= 0)` a negated predicate would spell puts `NaN` below the support.
    fn at(&self, value: f64) -> Option<f64> {
        if value < 0.0 {
            Some(self.below_support)
        } else {
            self.on_support.as_ref().map(|arm| arm(value))
        }
    }
}

/// `rate * exp(-rate * x)` on `x >= 0`, `0` below, keeping the subnormal range exact.
///
/// Reassociated as `(rate * exp(-rate * x / 2)) * exp(-rate * x / 2)`: the same product with the
/// rounding moved to the end. Written literally, `exp(-rate * x)` rounds into the gradual-underflow
/// range while the scale is still to be applied, and the final multiply then magnifies what the
/// subnormal threw away. Halving the exponent keeps the intermediate normal, at the cost of one
/// multiply and no branch.
fn derive_pdf(rate: Option<f64>) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        below_support: 0.0,
        on_support: rate.map(|rate| {
            move |x: f64| {
                let half_exp = (-rate * x / 2.0).exp();
                (rate * half_exp) * half_exp
            }
        }),
    }
}

/// `ln(rate) - rate * x` on `x >= 0`, `-inf` below. `ln(rate)` is the only term the rate alone fixes,
/// so it is the only one [`value_keyed_derived_scalar`] can lift out of the loop.
fn derive_ln_pdf(rate: Option<f64>) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        below_support: f64::NEG_INFINITY,
        on_support: rate.map(|rate| {
            let ln_rate = rate.ln();
            move |x: f64| ln_rate - rate * x
        }),
    }
}

/// `1 - exp(-rate * x)` on `x >= 0`, `0` below, keeping the left tail exact.
///
/// `1 - exp(-t)` cancels to `0` below `t ~ 1.1e-16`, so the small branch reads it as `-expm1(-t)`;
/// see [`CDF_SINH_MAX`] for why the crossover is where it is.
fn derive_cdf(rate: Option<f64>) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        below_support: 0.0,
        on_support: rate.map(|rate| {
            move |x: f64| {
                let t = rate * x;
                if t < CDF_SINH_MAX {
                    -expm1(-t)
                } else {
                    1.0 - (-t).exp()
                }
            }
        }),
    }
}

/// `ln(cdf)` in the left tail, `ln_1p(-sf)` in the right; `-inf` below the support.
///
/// As `cdf -> 1` the cdf rounds to ~1 and its log to a tiny, inaccurate value, so that side goes
/// through `ln_1p` of the small `sf`. As `cdf -> 0` the log is well conditioned and [`derive_cdf`] is
/// exact there, so that side is its log.
///
/// The predicate `rate * x < 1` says where `ln(cdf)` is well conditioned; it is not a comparison
/// against the computed cdf. It coincides with [`CDF_SINH_MAX`] rather than deriving from it, which
/// is what lets the left arm inline [`derive_cdf`]'s `expm1` branch: on the support and below
/// `t = 1` that is the only branch it would take.
fn derive_ln_cdf(rate: Option<f64>) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        below_support: f64::NEG_INFINITY,
        on_support: rate.map(|rate| {
            move |x: f64| {
                let t = rate * x;
                if t < 1.0 {
                    (-expm1(-t)).ln()
                } else {
                    (-(-t).exp()).ln_1p()
                }
            }
        }),
    }
}

/// `exp(-rate * x)` on `x >= 0`, `1` below.
///
/// The closed form, never `1 - cdf`: the complement quantises the upper tail to the `1.1e-16`
/// spacing of `1.0` and reaches `0.0` below that.
fn derive_sf(rate: Option<f64>) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        below_support: 1.0,
        on_support: rate.map(|rate| move |x: f64| (-rate * x).exp()),
    }
}

/// `-rate * x` on `x >= 0`, `0` below: the plain log of [`derive_sf`].
fn derive_ln_sf(rate: Option<f64>) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        below_support: 0.0,
        on_support: rate.map(|rate| move |x: f64| -rate * x),
    }
}

/// `-ln_1p(-q) / rate`, the exact inverse cdf; null outside `[0, 1]`.
///
/// Through `ln_1p` rather than `ln(1 - q)`: the latter rounds `1 - q` to exactly `1` below
/// `q ~ 1.1e-16` and collapses to `-0.0`.
///
/// The rate is divided by, never reciprocated: `x * (1 / rate)` rounds twice where `x / rate` rounds
/// once, and at a subnormal rate the reciprocal reaches `inf` and `NaN` where the division stays
/// finite. So neither inverse hoists anything into its `derive`.
fn derive_ppf(rate: Option<f64>) -> Domain<impl Fn(f64) -> f64> {
    Domain {
        inside: rate.map(|rate| move |quantile: f64| (-((-quantile).ln_1p())) / rate),
    }
}

/// `-ln(q) / rate`, the exact inverse survival function; null outside `[0, 1]`.
///
/// Never `ppf(1 - q)`: that forms the complement and then undoes it, losing the answer whenever
/// either step saturates.
fn derive_isf(rate: Option<f64>) -> Domain<impl Fn(f64) -> f64> {
    Domain {
        inside: rate.map(|rate| move |quantile: f64| (-quantile.ln()) / rate),
    }
}

/// Element-wise pdf; see [`derive_pdf`] for the subnormal reassociation.
/// See [`value_keyed_derived_per_row`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn exponential_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_pdf, Sides::at)
}

/// Element-wise log-pdf; see [`derive_ln_pdf`].
#[polars_expr(output_type=Float64)]
fn exponential_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_ln_pdf, Sides::at)
}

/// Element-wise cdf `P(X <= value)`; see [`derive_cdf`] and [`CDF_SINH_MAX`].
#[polars_expr(output_type=Float64)]
fn exponential_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_cdf, Sides::at)
}

/// Element-wise log-cdf; see [`derive_ln_cdf`].
#[polars_expr(output_type=Float64)]
fn exponential_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_ln_cdf, Sides::at)
}

/// Element-wise survival function `P(X > value)`; see [`derive_sf`] for why it is not `1 - cdf`.
#[polars_expr(output_type=Float64)]
fn exponential_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_sf, Sides::at)
}

/// Element-wise log-sf; see [`derive_ln_sf`].
#[polars_expr(output_type=Float64)]
fn exponential_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_ln_sf, Sides::at)
}

/// Element-wise ppf (inverse cdf); see [`derive_ppf`].
#[polars_expr(output_type=Float64)]
fn exponential_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_ppf, Domain::at)
}

/// Element-wise inverse survival function; see [`derive_isf`] for why it never forms a complement.
#[polars_expr(output_type=Float64)]
fn exponential_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_isf, Domain::at)
}

/// Constant-rate fast path for [`exponential_pdf`].
#[polars_expr(output_type=Float64)]
fn exponential_pdf_scalar(
    inputs: &[Series],
    kwargs: ExponentialParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pdf, Sides::at)
}

/// Constant-rate fast path for [`exponential_ln_pdf`].
#[polars_expr(output_type=Float64)]
fn exponential_ln_pdf_scalar(
    inputs: &[Series],
    kwargs: ExponentialParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pdf, Sides::at)
}

/// Constant-rate fast path for [`exponential_cdf`].
#[polars_expr(output_type=Float64)]
fn exponential_cdf_scalar(
    inputs: &[Series],
    kwargs: ExponentialParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, Sides::at)
}

/// Constant-rate fast path for [`exponential_ln_cdf`].
#[polars_expr(output_type=Float64)]
fn exponential_ln_cdf_scalar(
    inputs: &[Series],
    kwargs: ExponentialParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, Sides::at)
}

/// Constant-rate fast path for [`exponential_sf`].
#[polars_expr(output_type=Float64)]
fn exponential_sf_scalar(
    inputs: &[Series],
    kwargs: ExponentialParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, Sides::at)
}

/// Constant-rate fast path for [`exponential_ln_sf`].
#[polars_expr(output_type=Float64)]
fn exponential_ln_sf_scalar(
    inputs: &[Series],
    kwargs: ExponentialParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, Sides::at)
}

/// Constant-rate fast path for [`exponential_ppf`].
#[polars_expr(output_type=Float64)]
fn exponential_ppf_scalar(
    inputs: &[Series],
    kwargs: ExponentialParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, Domain::at)
}

/// Constant-rate fast path for [`exponential_isf`].
#[polars_expr(output_type=Float64)]
fn exponential_isf_scalar(
    inputs: &[Series],
    kwargs: ExponentialParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, Domain::at)
}

/// One Exponential draw from a `&mut` per-row RNG already seeded from `(root_seed, index)`.
///
/// Every Exponential sampler draws through here, per-row and fast path alike, so their bit-equality is
/// structural rather than only sampled by `sample_test.py`.
#[inline]
fn draw(dist: &Exp, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

/// Element-wise Exponential sampler over `(rate, row_index)`, returning `Float64`.
///
/// Per row, `null` propagates and an invalid rate raises via [`build_dist`]. Seeding and
/// chunk-invariance follow [`sample_per_row_binary`].
///
/// The draw keeps `statrs` (`O(1)` ziggurat: `sample_exp_1(rng) / rate`); routing it through
/// `rand_distr` would buy nothing, since that is already the algorithm class `statrs` uses (unlike
/// the binomial draw, see docs/explanation/design.md).
#[polars_expr(output_type=Float64)]
fn exponential_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let rate = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    sample_per_row_binary(
        name,
        rate.f64()?,
        index.u64()?,
        kwargs.seed,
        build_dist,
        draw,
    )
}

/// Constant-rate fast path for [`exponential_sample`].
#[polars_expr(output_type=Float64)]
fn exponential_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<ExponentialParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    sample_by_index(name, &inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

/// Constant-parameter multi-draw fast path: the `samples` twin of [`exponential_sample_scalar`].
///
/// `size` consecutive draws from each row's stream, so `samples(size=1)` matches `sample` bit for
/// bit and the distribution is built once per call. Returns `Array(Float64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn exponential_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<ExponentialParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    samples_by_index(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&dist, rng)
    })
}

/// Element-wise multi-draw Exponential sampler: `size` draws per row in one call, the distribution
/// built once per row. Returns `Array(Float64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`exponential_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn exponential_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let rate = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = binary_param_rows(rate.f64()?, index.u64()?, build_dist);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, draw)
}
