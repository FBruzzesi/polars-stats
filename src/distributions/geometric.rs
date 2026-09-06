use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Geometric;

use crate::distributions::{
    align_inputs, expm1, ln_abs_expm1, validate_params_unary, value_keyed_derived_per_row,
    value_keyed_derived_scalar, Domain, Sides,
};
use crate::rng::{
    binary_param_rows, sample_by_index, sample_per_row_binary, samples_by_index, samples_per_row,
    samples_u64_output, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

/// `statrs::Geometric::new` rejects `NaN` and any `p` outside `(0, 1]`, so unlike `Bernoulli` the
/// degenerate `p = 0` point mass is not representable.
fn build_dist(proba: f64) -> PolarsResult<Geometric> {
    Geometric::new(proba).map_err(|e| {
        PolarsError::InvalidOperation(format!("p must be in (0, 1], got {proba}: {e}").into())
    })
}

/// `ln(1 - p)` as `ln_1p(-p)`: the literal `ln(1.0 - p)` inherits the rounding of `1 - p` (a few
/// parts in `1e9` at `p = 1e-8`) and collapses to `0.0` below `p ~ 1.1e-16`. `-inf` at `p = 1`.
#[inline]
fn ln_failure(proba: f64) -> f64 {
    (-proba).ln_1p()
}

/// The samplers' per-row state: [`ln_failure`], validated through [`build_dist`] so an invalid `p`
/// raises identically either way. Built once per row, not once per draw: the multi-draw and
/// constant-parameter paths take many draws per state.
fn build_sampler(proba: f64) -> PolarsResult<f64> {
    build_dist(proba)?;
    Ok(ln_failure(proba))
}

/// Geometric's constant success probability, deserialised once per call.
#[derive(serde::Deserialize)]
struct GeometricParamsKwargs {
    p: f64,
}

impl GeometricParamsKwargs {
    fn build_sampler(&self) -> PolarsResult<f64> {
        build_sampler(self.p)
    }

    /// Binds the constant `p` and `build_dist` into [`value_keyed_derived_scalar`], which validates
    /// and derives once per call rather than per row.
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
/// `InvalidOperation` if `p` is outside `(0, 1]`. `null` propagates.
///
/// The moments derive from this so they report an invalid `p` consistently with `geometric_sample`,
/// instead of silently computing with a non-positive probability. The value-keyed methods validate
/// inside their own plugin instead.
#[polars_expr(output_type=Float64)]
fn geometric_p(inputs: &[Series]) -> PolarsResult<Series> {
    let proba = inputs[0].cast(&DataType::Float64)?;

    validate_params_unary(proba.f64()?, |proba| {
        build_dist(proba)?;
        Ok(proba)
    })
}

/// Crossover of [`derive_cdf`], in units of `ln(sf)`.
///
/// Below it `exp(ln_sf)` is small enough that the direct `1 - exp(ln_sf)` rounds no worse than the
/// `sinh` identity and cannot round above `1`. The identity itself only breaks past `ln_sf ~ -1455`,
/// where `sinh` overflows, far inside this branch.
const CDF_DIRECT_COMPLEMENT_MAX: f64 = -20.0;

/// Crossover of [`derive_ln_cdf`], in units of `ln(sf)`.
///
/// Below it the cdf sits within `0.63` of `1` and `ln_1p(-exp(ln_sf))` carries that difference
/// exactly, down through the `cdf ~ 1 - 1e-16` zone where `cdf.ln()` reads the tail mass as `0`.
const LN_CDF_LN_1P_MAX: f64 = -1.0;

/// `pmf` / `log_pmf`: `off_support` is a plain `f64` because a point below `1` or a non-integral one
/// carries no mass whatever `p` is, so a null `p` must not null it; `on_support` is `None` exactly
/// when `p` is null. Pinned by `tests/distributions/geometric/null_param_test.py`.
struct Mass<Arm> {
    off_support: f64,
    on_support: Option<Arm>,
}

impl<Arm: Fn(f64) -> f64> Mass<Arm> {
    /// Only a positive integer takes the arm, so `pmf(2.5)` is `0`, not the mass at `2`.
    /// `floor(k) == k` rather than an integer cast keeps `1e300` and `+inf` on the support, answering
    /// `0` through the arm instead of saturating. A `NaN` point never reaches here: both drivers
    /// short-circuit it.
    fn at(&self, value: f64) -> Option<f64> {
        if value >= 1.0 && value.floor() == value {
            self.on_support.as_ref().map(|arm| arm(value))
        } else {
            Some(self.off_support)
        }
    }
}

/// `cdf` / `log_cdf` / `sf` / `log_sf`: the tail methods floor a non-integral point onto the support,
/// so `cdf(2.5)` is `cdf(2)` and needs `p` where [`Mass`] answers its constant; the two cannot share
/// a table.
type Tail<Arm> = Sides<Arm, 1>;

/// `(1 - p)^(k - 1) * p` on the positive integers, `0` elsewhere.
///
/// `k = 1` short-circuits the power: its exponent `(k - 1) * ln(1 - p)` is `0 * -inf = NaN` at
/// `p = 1`, where the whole mass sits on `k = 1`.
fn derive_pmf(p: Option<f64>) -> Mass<impl Fn(f64) -> f64> {
    Mass {
        off_support: 0.0,
        on_support: p.map(|p| {
            let ln_failure = ln_failure(p);
            move |k: f64| {
                if k == 1.0 {
                    p
                } else {
                    p * ((k - 1.0) * ln_failure).exp()
                }
            }
        }),
    }
}

/// `(k - 1) * ln(1 - p) + ln(p)` on the positive integers, `-inf` elsewhere.
///
/// Not `ln(pmf)`, whose `ln(1 - p)` collapses to `0.0` below `p ~ 1.1e-16`. `k = 1` reads `ln(p)`
/// alone, for the same `0 * -inf` reason as [`derive_pmf`].
fn derive_ln_pmf(p: Option<f64>) -> Mass<impl Fn(f64) -> f64> {
    Mass {
        off_support: f64::NEG_INFINITY,
        on_support: p.map(|p| {
            let (ln_failure, ln_p) = (ln_failure(p), p.ln());
            move |k: f64| {
                if k == 1.0 {
                    ln_p
                } else {
                    (k - 1.0) * ln_failure + ln_p
                }
            }
        }),
    }
}

/// `1 - (1 - p)^floor(k)` from `1` up, `0` below, read as `-expm1(ln_sf)`.
///
/// Below [`CDF_DIRECT_COMPLEMENT_MAX`] it is the direct `1 - exp(ln_sf)` instead. The literal
/// `1 - (1 - p)^k` would inherit the rounding of both `1 - p` and the subtraction against `1`.
fn derive_cdf(p: Option<f64>) -> Tail<impl Fn(f64) -> f64> {
    Tail {
        below_support: 0.0,
        on_support: p.map(|p| {
            let ln_failure = ln_failure(p);
            move |k: f64| {
                let ln_sf = k.floor() * ln_failure;
                if ln_sf <= CDF_DIRECT_COMPLEMENT_MAX {
                    1.0 - ln_sf.exp()
                } else {
                    -expm1(ln_sf)
                }
            }
        }),
    }
}

/// `ln(1 - exp(ln_sf))` from `1` up, `-inf` below.
///
/// Split at [`LN_CDF_LN_1P_MAX`], where the answer's own magnitude stops dwarfing the absolute
/// granularity of the pieces: below it `ln_1p` carries the difference from `1` exactly, above it
/// [`ln_abs_expm1`] assembles the answer on the log scale so the rounding stays relative.
fn derive_ln_cdf(p: Option<f64>) -> Tail<impl Fn(f64) -> f64> {
    Tail {
        below_support: f64::NEG_INFINITY,
        on_support: p.map(|p| {
            let ln_failure = ln_failure(p);
            move |k: f64| {
                let ln_sf = k.floor() * ln_failure;
                if ln_sf <= LN_CDF_LN_1P_MAX {
                    (-ln_sf.exp()).ln_1p()
                } else {
                    ln_abs_expm1(ln_sf)
                }
            }
        }),
    }
}

/// `exp(ln_sf)` from `1` up, exactly `1` below.
///
/// Never `1 - cdf`, which recomputes `p` as `1 - (1 - p)` and so quantises it to the `1.1e-16`
/// spacing of `1.0`, reaching `0.0` below that.
fn derive_sf(p: Option<f64>) -> Tail<impl Fn(f64) -> f64> {
    Tail {
        below_support: 1.0,
        on_support: p.map(|p| {
            let ln_failure = ln_failure(p);
            move |k: f64| (k.floor() * ln_failure).exp()
        }),
    }
}

/// `ln(sf) = floor(k) * ln(1 - p)` from `1` up, `0` below.
fn derive_ln_sf(p: Option<f64>) -> Tail<impl Fn(f64) -> f64> {
    Tail {
        below_support: 0.0,
        on_support: p.map(|p| {
            let ln_failure = ln_failure(p);
            move |k: f64| k.floor() * ln_failure
        }),
    }
}

/// Smallest `k >= 1` with `k * ln_failure <= log_target`, the shape both inverses share: they differ
/// only in what they take the log of (`1 - q` against `q`).
///
/// A ratio sitting an ulp above an exact integer overshoots by one support point, so the ceiling
/// steps back down whenever `k - 1` already satisfies the inequality, compared in the same log domain
/// both sides entered through. The clip to `1.0` is also where `-0.0` lands at the degenerate
/// endpoint of either inverse.
///
/// `ln_failure` is divided by, never reciprocated: `x * (1 / d)` rounds twice where `x / d` rounds
/// once, and at a subnormal `ln(1 - p)` the reciprocal overflows to `inf`, turning `q = 0` into `NaN`
/// and every other quantile into `inf`. So neither inverse hoists it into its `derive`.
fn smallest_support_point(log_target: f64, ln_failure: f64) -> f64 {
    let ceiling = (log_target / ln_failure).ceil();
    let overshot = (ceiling - 1.0) * ln_failure <= log_target;
    let k = if overshot { ceiling - 1.0 } else { ceiling };
    k.max(1.0)
}

/// `ceil(ln_1p(-q) / ln(1 - p))`, the smallest `k` with `cdf(k) >= q`; null outside `[0, 1]`.
///
/// `ln_1p(-q)`, not `ln(1 - q)`, which collapses to `0.0` below `q ~ 1.1e-16` and answers `0` where
/// `1` is the only correct answer. `q = 1` runs the ratio off to `+inf`, right for an unbounded
/// support. `p = 1` short-circuits before [`smallest_support_point`], where the ratio would be
/// `-inf / -inf = NaN`.
fn derive_ppf(p: Option<f64>) -> Domain<impl Fn(f64) -> f64> {
    Domain {
        inside: p.map(|p| {
            let ln_failure = ln_failure(p);
            move |quantile: f64| {
                if p == 1.0 {
                    1.0
                } else {
                    smallest_support_point((-quantile).ln_1p(), ln_failure)
                }
            }
        }),
    }
}

/// `ceil(ln(q) / ln(1 - p))`, the smallest `k` with `sf(k) <= q`; null outside `[0, 1]`.
///
/// Entered against `q` itself, never as `ppf(1 - q)`, whose complement throws the tail mass away
/// before the inverse runs. `q = 1` degenerates to `-0.0` and `q = 0` flows through as `+inf`. At
/// `p = 1` every survival quantile inverts to `k = 1`, `q = 0` included, since `sf(1) = 0` already
/// satisfies the inequality.
fn derive_isf(p: Option<f64>) -> Domain<impl Fn(f64) -> f64> {
    Domain {
        inside: p.map(|p| {
            let ln_failure = ln_failure(p);
            move |quantile: f64| {
                if p == 1.0 {
                    1.0
                } else {
                    smallest_support_point(quantile.ln(), ln_failure)
                }
            }
        }),
    }
}

/// Element-wise pmf; see [`derive_pmf`], and [`Mass::at`] for the support rule.
/// See [`value_keyed_derived_per_row`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn geometric_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_pmf, Mass::at)
}

/// Element-wise log-pmf; see [`derive_ln_pmf`].
#[polars_expr(output_type=Float64)]
fn geometric_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_ln_pmf, Mass::at)
}

/// Element-wise cdf `P(X <= value)`; see [`derive_cdf`] and [`CDF_DIRECT_COMPLEMENT_MAX`].
#[polars_expr(output_type=Float64)]
fn geometric_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_cdf, Tail::at)
}

/// Element-wise log-cdf; see [`derive_ln_cdf`] and [`LN_CDF_LN_1P_MAX`].
#[polars_expr(output_type=Float64)]
fn geometric_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_ln_cdf, Tail::at)
}

/// Element-wise survival function `P(X > value)`; see [`derive_sf`] for why it is not `1 - cdf`.
#[polars_expr(output_type=Float64)]
fn geometric_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_sf, Tail::at)
}

/// Element-wise log-sf; see [`derive_ln_sf`].
#[polars_expr(output_type=Float64)]
fn geometric_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_ln_sf, Tail::at)
}

/// Element-wise ppf (inverse cdf); see [`derive_ppf`] and [`smallest_support_point`].
#[polars_expr(output_type=Float64)]
fn geometric_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_ppf, Domain::at)
}

/// Element-wise inverse survival function; see [`derive_isf`] for why it never forms a complement.
#[polars_expr(output_type=Float64)]
fn geometric_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_per_row(inputs, build_dist, derive_isf, Domain::at)
}

/// Constant-`p` fast path for [`geometric_pmf`].
#[polars_expr(output_type=Float64)]
fn geometric_pmf_scalar(inputs: &[Series], kwargs: GeometricParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pmf, Mass::at)
}

/// Constant-`p` fast path for [`geometric_ln_pmf`].
#[polars_expr(output_type=Float64)]
fn geometric_ln_pmf_scalar(
    inputs: &[Series],
    kwargs: GeometricParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pmf, Mass::at)
}

/// Constant-`p` fast path for [`geometric_cdf`].
#[polars_expr(output_type=Float64)]
fn geometric_cdf_scalar(inputs: &[Series], kwargs: GeometricParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, Tail::at)
}

/// Constant-`p` fast path for [`geometric_ln_cdf`].
#[polars_expr(output_type=Float64)]
fn geometric_ln_cdf_scalar(
    inputs: &[Series],
    kwargs: GeometricParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, Tail::at)
}

/// Constant-`p` fast path for [`geometric_sf`].
#[polars_expr(output_type=Float64)]
fn geometric_sf_scalar(inputs: &[Series], kwargs: GeometricParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, Tail::at)
}

/// Constant-`p` fast path for [`geometric_ln_sf`].
#[polars_expr(output_type=Float64)]
fn geometric_ln_sf_scalar(
    inputs: &[Series],
    kwargs: GeometricParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, Tail::at)
}

/// Constant-`p` fast path for [`geometric_ppf`].
#[polars_expr(output_type=Float64)]
fn geometric_ppf_scalar(inputs: &[Series], kwargs: GeometricParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, Domain::at)
}

/// Constant-`p` fast path for [`geometric_isf`].
#[polars_expr(output_type=Float64)]
fn geometric_isf_scalar(inputs: &[Series], kwargs: GeometricParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, Domain::at)
}

/// One Geometric draw from a `&mut` per-row RNG already seeded from `(root_seed, index)`.
///
/// Every Geometric sampler draws through here, per-row and fast path alike, so their bit-equality is
/// structural rather than only sampled by `sample_test.py`.
///
/// The inverse transform `ceil(ln u / ln(1 - p))` takes its log base from [`ln_failure`], not from
/// `statrs`' `Distribution<u64>`, which forms the literal `1.0 - p`: that rounds to `1.0` at
/// `p <= 2^-54`, so the base is `0.0`, the ratio `-inf`, and every draw below the threshold casts to
/// `0` where the true draws are around `1 / p`.
///
/// `u` comes from `(0, 1]`, so `u = 1` gives a raw draw of `0`; `.max(1)` lifts it to the support
/// floor, where that endpoint's mass belongs in the limit. At the other end the cast saturates once
/// `-ln u` outgrows `u64::MAX * p`, from around `p ~ 2e-18`: a dtype limit, not an algorithm one.
#[inline]
fn draw(ln_failure: &f64, rng: &mut impl rand::Rng) -> u64 {
    let uniform: f64 = RandDistribution::sample(&rand::distr::OpenClosed01, rng);
    ((uniform.ln() / ln_failure).ceil() as u64).max(1)
}

/// Element-wise Geometric sampler over `(p, row_index)`, returning `UInt64`.
///
/// Per row, `null` propagates and an invalid `p` raises via [`build_sampler`]. Seeding and
/// chunk-invariance follow [`sample_per_row_binary`].
#[polars_expr(output_type=UInt64)]
fn geometric_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    sample_per_row_binary(
        name,
        proba.f64()?,
        index.u64()?,
        kwargs.seed,
        build_sampler,
        draw,
    )
}

/// Constant-parameter fast path for [`geometric_sample`].
#[polars_expr(output_type=UInt64)]
fn geometric_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<GeometricParamsKwargs>,
) -> PolarsResult<Series> {
    let ln_failure = kwargs.params.build_sampler()?;
    let name = inputs[0].name().clone();

    sample_by_index(name, &inputs[0], kwargs.seed, |rng| draw(&ln_failure, rng))
}

/// Constant-parameter multi-draw fast path: the `samples` twin of [`geometric_sample_scalar`].
///
/// `size` consecutive draws from each row's stream, so `samples(size=1)` matches `sample` bit for
/// bit. Returns `Array(UInt64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn geometric_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<GeometricParamsKwargs>,
) -> PolarsResult<Series> {
    let ln_failure = kwargs.params.build_sampler()?;
    let name = inputs[0].name().clone();

    samples_by_index(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&ln_failure, rng)
    })
}

/// Element-wise multi-draw Geometric sampler: `size` draws per row in one call. Returns
/// `Array(UInt64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`geometric_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn geometric_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let proba = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = binary_param_rows(proba.f64()?, index.u64()?, build_sampler);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, draw)
}
