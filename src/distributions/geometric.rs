use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Geometric;

use crate::distributions::{
    expm1, ln_abs_expm1, on_unit_interval, validated_param, value_keyed_derived_binary,
    value_keyed_scalar, ParamDomain, Sides,
};
use crate::rng::{
    sample_by_index, sample_per_row_binary, samples_by_index, samples_per_row_binary,
    samples_u64_output, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

/// Unlike `Bernoulli`, the degenerate `p = 0` point mass is not representable.
const P: ParamDomain = ParamDomain {
    name: "p",
    rule: "in (0, 1]",
    accepts: |p| p.is_finite() && p > 0.0 && p <= 1.0,
};

fn build_dist(p: f64) -> PolarsResult<Geometric> {
    Geometric::new(p).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// `ln(1 - p)` as `ln_1p(-p)`: the literal `ln(1.0 - p)` inherits the rounding of `1 - p` and
/// collapses to `0.0` below `p ~ 1.1e-16`. `-inf` at `p = 1`.
#[inline]
fn ln_failure(p: f64) -> f64 {
    (-p).ln_1p()
}

/// The samplers' per-row state, behind [`build_dist`]'s check.
fn build_sampler(p: f64) -> PolarsResult<f64> {
    build_dist(p)?;
    Ok(ln_failure(p))
}

/// Constant parameter, deserialised once per call; every `_scalar` twin checks it once here.
#[derive(serde::Deserialize)]
struct GeometricParams {
    p: f64,
}

impl GeometricParams {
    fn build_sampler(&self) -> PolarsResult<f64> {
        P.check(self.p)?;
        build_sampler(self.p)
    }

    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(f64) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        P.check(self.p)?;
        value_keyed_scalar(value, &derive(self.p), select)
    }
}

#[polars_expr(output_type=Float64)]
fn geometric_p(inputs: &[Series]) -> PolarsResult<Series> {
    validated_param(inputs, &P)
}

/// Crossover of [`derive_cdf`] in units of `ln(sf)`: below it `exp(ln_sf)` is small enough that the
/// direct `1 - exp(ln_sf)` rounds no worse than the `sinh` identity and cannot round above `1`.
const CDF_DIRECT_COMPLEMENT_MAX: f64 = -20.0;

/// Crossover of [`derive_ln_cdf`] in units of `ln(sf)`: below it the cdf sits within `0.63` of `1`
/// and `ln_1p(-exp(ln_sf))` carries that difference exactly, where `cdf.ln()` reads the tail as `0`.
const LN_CDF_LN_1P_MAX: f64 = -1.0;

/// `pmf` / `log_pmf`: a constant off the support (below `1`, or a non-integral point), and the arm
/// `p` derives on it.
struct Mass<Arm> {
    off_support: f64,
    on_support: Arm,
}

impl<Arm: Fn(f64) -> f64> Mass<Arm> {
    /// `floor(k) == k` rather than an integer cast keeps `1e300` and `+inf` on the support, answering
    /// `0` through the arm instead of saturating.
    fn at(&self, value: f64) -> Option<f64> {
        Some(if value >= 1.0 && value.floor() == value {
            (self.on_support)(value)
        } else {
            self.off_support
        })
    }
}

/// The tail methods floor a non-integral point onto the support, so `cdf(2.5)` is `cdf(2)`.
type Tail<Arm> = Sides<Arm, 1>;

/// `(1 - p)^(k - 1) * p` on the positive integers, `0` elsewhere. `k = 1` short-circuits the power,
/// whose exponent `(k - 1) * ln(1 - p)` is `0 * -inf = NaN` at `p = 1`.
fn derive_pmf(p: f64) -> Mass<impl Fn(f64) -> f64> {
    Mass {
        off_support: 0.0,
        on_support: {
            let ln_failure = ln_failure(p);
            move |k: f64| {
                if k == 1.0 {
                    p
                } else {
                    p * ((k - 1.0) * ln_failure).exp()
                }
            }
        },
    }
}

/// `(k - 1) * ln(1 - p) + ln(p)` on the positive integers, `-inf` elsewhere; same `k = 1` guard as
/// [`derive_pmf`].
fn derive_ln_pmf(p: f64) -> Mass<impl Fn(f64) -> f64> {
    Mass {
        off_support: f64::NEG_INFINITY,
        on_support: {
            let (ln_failure, ln_p) = (ln_failure(p), p.ln());
            move |k: f64| {
                if k == 1.0 {
                    ln_p
                } else {
                    (k - 1.0) * ln_failure + ln_p
                }
            }
        },
    }
}

/// `1 - (1 - p)^floor(k)` from `1` up, `0` below, as `-expm1(ln_sf)`; the direct complement below
/// [`CDF_DIRECT_COMPLEMENT_MAX`].
fn derive_cdf(p: f64) -> Tail<impl Fn(f64) -> f64> {
    Tail {
        below_support: 0.0,
        on_support: {
            let ln_failure = ln_failure(p);
            move |k: f64| {
                let ln_sf = k.floor() * ln_failure;
                if ln_sf <= CDF_DIRECT_COMPLEMENT_MAX {
                    1.0 - ln_sf.exp()
                } else {
                    -expm1(ln_sf)
                }
            }
        },
    }
}

/// `ln(1 - exp(ln_sf))` from `1` up, `-inf` below: `ln_1p` below [`LN_CDF_LN_1P_MAX`], where the
/// difference from `1` is what carries the answer, [`ln_abs_expm1`] above it.
fn derive_ln_cdf(p: f64) -> Tail<impl Fn(f64) -> f64> {
    Tail {
        below_support: f64::NEG_INFINITY,
        on_support: {
            let ln_failure = ln_failure(p);
            move |k: f64| {
                let ln_sf = k.floor() * ln_failure;
                if ln_sf <= LN_CDF_LN_1P_MAX {
                    (-ln_sf.exp()).ln_1p()
                } else {
                    ln_abs_expm1(ln_sf)
                }
            }
        },
    }
}

/// `exp(ln_sf)` from `1` up, exactly `1` below; never `1 - cdf`, which quantises `p` to the
/// `1.1e-16` spacing of `1.0`.
fn derive_sf(p: f64) -> Tail<impl Fn(f64) -> f64> {
    Tail {
        below_support: 1.0,
        on_support: {
            let ln_failure = ln_failure(p);
            move |k: f64| (k.floor() * ln_failure).exp()
        },
    }
}

fn derive_ln_sf(p: f64) -> Tail<impl Fn(f64) -> f64> {
    Tail {
        below_support: 0.0,
        on_support: {
            let ln_failure = ln_failure(p);
            move |k: f64| k.floor() * ln_failure
        },
    }
}

/// Smallest `k >= 1` with `k * ln_failure <= log_target`, shared by both inverses.
///
/// A ratio one ulp above an exact integer overshoots by one support point, so the ceiling steps back
/// whenever `k - 1` already satisfies the inequality, compared in the same log domain. The clip to
/// `1.0` is also where `-0.0` lands at the degenerate endpoint. `ln_failure` is divided by, never
/// reciprocated: at a subnormal `ln(1 - p)` the reciprocal overflows to `inf`.
fn smallest_support_point(log_target: f64, ln_failure: f64) -> f64 {
    let ceiling = (log_target / ln_failure).ceil();
    let overshot = (ceiling - 1.0) * ln_failure <= log_target;
    let k = if overshot { ceiling - 1.0 } else { ceiling };
    k.max(1.0)
}

/// `ceil(ln_1p(-q) / ln(1 - p))`, the smallest `k` with `cdf(k) >= q`; null outside `[0, 1]`.
/// `ln_1p(-q)`, not `ln(1 - q)`, which collapses below `q ~ 1.1e-16`. `p = 1` short-circuits the
/// `-inf / -inf` ratio.
fn derive_ppf(p: f64) -> impl Fn(f64) -> f64 {
    let ln_failure = ln_failure(p);
    move |quantile: f64| {
        if p == 1.0 {
            1.0
        } else {
            smallest_support_point((-quantile).ln_1p(), ln_failure)
        }
    }
}

/// `ceil(ln(q) / ln(1 - p))`, the smallest `k` with `sf(k) <= q`, solved on `q` itself rather than
/// as `ppf(1 - q)`; null outside `[0, 1]`.
fn derive_isf(p: f64) -> impl Fn(f64) -> f64 {
    let ln_failure = ln_failure(p);
    move |quantile: f64| {
        if p == 1.0 {
            1.0
        } else {
            smallest_support_point(quantile.ln(), ln_failure)
        }
    }
}

#[polars_expr(output_type=Float64)]
fn geometric_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_pmf, Mass::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_ln_pmf, Mass::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_cdf, Tail::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_ln_cdf, Tail::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_sf, Tail::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_ln_sf, Tail::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn geometric_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn geometric_pmf_scalar(inputs: &[Series], kwargs: GeometricParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pmf, Mass::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_ln_pmf_scalar(inputs: &[Series], kwargs: GeometricParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pmf, Mass::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_cdf_scalar(inputs: &[Series], kwargs: GeometricParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, Tail::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_ln_cdf_scalar(inputs: &[Series], kwargs: GeometricParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, Tail::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_sf_scalar(inputs: &[Series], kwargs: GeometricParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, Tail::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_ln_sf_scalar(inputs: &[Series], kwargs: GeometricParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, Tail::at)
}

#[polars_expr(output_type=Float64)]
fn geometric_ppf_scalar(inputs: &[Series], kwargs: GeometricParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn geometric_isf_scalar(inputs: &[Series], kwargs: GeometricParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

/// Inverse transform `ceil(ln u / ln(1 - p))` with the log base from [`ln_failure`] rather than
/// `statrs`' draw, whose literal `1.0 - p` rounds to `1.0` at `p <= 2^-54` and casts every draw to
/// `0`. `u` is in `(0, 1]`, so `u = 1` gives `0` and `.max(1)` lifts it to the support floor; the
/// cast saturates once `-ln u` outgrows `u64::MAX * p`, from around `p ~ 2e-18`.
#[inline]
fn draw(ln_failure: &f64, rng: &mut impl rand::Rng) -> u64 {
    let uniform: f64 = RandDistribution::sample(&rand::distr::OpenClosed01, rng);
    ((uniform.ln() / ln_failure).ceil() as u64).max(1)
}

#[polars_expr(output_type=UInt64)]
fn geometric_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_binary(inputs, kwargs, &P, build_sampler, draw)
}

#[polars_expr(output_type=UInt64)]
fn geometric_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<GeometricParams>,
) -> PolarsResult<Series> {
    let ln_failure = kwargs.params.build_sampler()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&ln_failure, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn geometric_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<GeometricParams>,
) -> PolarsResult<Series> {
    let ln_failure = kwargs.params.build_sampler()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&ln_failure, rng)
    })
}

#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn geometric_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_binary(inputs, kwargs, &P, build_sampler, draw)
}
