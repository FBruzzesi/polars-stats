//! `NegativeBinomial(r, p)` counts the failures before the `r`-th success. Every value-keyed method reads
//! `beta.rs`'s incomplete beta at `x = p`: `P(X <= k) = I_p(r, k + 1)`, `P(X > k)` its complement, and the mass
//! `Gamma(r + k) / (Gamma(r) k!) p^r (1 - p)^k` is its prefactor at shapes `(r, k)` over `k`. statrs' `sf` forms
//! `1 - p`, which loses the whole upper tail once that rounds to `1`, and its `ln_pmf` is an `ln_gamma` difference
//! that cancels at large `r` and `k`.

use std::sync::atomic::{AtomicBool, Ordering};

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use rand_distr::Poisson as PoissonSampler;
use statrs::distribution::Gamma;

use crate::distributions::beta::IncompleteBeta;
use crate::distributions::{
    coerce_f64, discrete_cdf, discrete_isf, discrete_ln_cdf, discrete_ln_pmf, discrete_ln_sf,
    discrete_pmf, discrete_ppf, discrete_quantile, discrete_sf, validated_pair,
    value_keyed_derived_ternary, value_keyed_scalar, DiscreteTails, ParamDomain, Rule, Tail, Tails,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_per_row_ternary,
    samples_u64_output, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const R: ParamDomain = ParamDomain::positive("r");
/// `p = 0` has no mass anywhere on the support; `p = 1` is the point mass at `0`.
const P: ParamDomain = ParamDomain {
    name: "p",
    rule: Rule::PositiveProbability,
};

fn check_params(r: &Float64Chunked, p: &Float64Chunked) -> PolarsResult<()> {
    R.check_column(r)?;
    P.check_column(p)
}

/// Constant parameters, deserialised once per call; every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct NegativeBinomialParams {
    r: f64,
    p: f64,
}

impl NegativeBinomialParams {
    fn check(&self) -> PolarsResult<()> {
        R.check(self.r)?;
        P.check(self.p)
    }

    fn build_sampler(&self) -> PolarsResult<Sampler> {
        self.check()?;
        build_sampler(self.r, self.p)
    }

    fn value_keyed(
        &self,
        value: &Series,
        select: impl Fn(&NegativeBinomial, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        self.check()?;
        value_keyed_scalar(value, &NegativeBinomial::new(self.r, self.p), select)
    }
}

/// The validated `p`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn negativebinomial_p(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

/// `r`, `p` and `q = 1 - p`, each logarithm read off the smaller of `p` and `q`: `ln q` is `ln_1p(-p)` below
/// `p = 1/2`, where `1 - p` has rounded.
struct NegativeBinomial {
    r: f64,
    p: f64,
    q: f64,
    ln_p: f64,
    ln_q: f64,
}

impl NegativeBinomial {
    fn new(r: f64, p: f64) -> Self {
        let q = 1.0 - p;
        Self {
            r,
            p,
            q,
            ln_p: if p <= q { p.ln() } else { (-q).ln_1p() },
            ln_q: if q <= p { q.ln() } else { (-p).ln_1p() },
        }
    }
}

impl DiscreteTails for NegativeBinomial {
    /// `P(X <= k)`, the lower incomplete beta `I_p(r, k + 1)`.
    const CDF: Tail = Tail::Lower;

    fn tails_at(&self, k: f64) -> Option<Tails> {
        k.is_finite()
            .then(|| IncompleteBeta::new(self.r, k.floor() + 1.0).tails(self.p, self.q))
    }

    /// `ln(Gamma(r + k) / (Gamma(r) k!) p^r q^k)` at a support point: `r ln p` at `k = 0`, and above it the
    /// prefactor `ln(p^r q^k / B(r, k))` less `ln k`.
    fn ln_mass(&self, k: f64) -> f64 {
        if k == 0.0 {
            return self.r * self.ln_p;
        }
        IncompleteBeta::new(self.r, k).ln_prefactor(self.p, self.q, self.ln_p, self.ln_q) - k.ln()
    }

    /// Seeded at the continuity-corrected Cornish-Fisher estimate `mean + sd (z + skew (z^2 - 1) / 6) - 1/2`, whose
    /// `sd skew` is `(1 + q) / p`. At `p = 1` every quantile is the point mass at `0`.
    fn quantile(&self, tail: Tail, level: f64) -> f64 {
        let (r, p, q) = (self.r, self.p, self.q);
        if q == 0.0 {
            return 0.0;
        }
        discrete_quantile(
            |k| self.tails_at(k),
            Self::CDF,
            tail,
            level,
            |z| r * q / p + (r * q).sqrt() / p * z + (1.0 + q) * (z * z - 1.0) / (6.0 * p) - 0.5,
        )
    }
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, NegativeBinomial::new, discrete_pmf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, NegativeBinomial::new, discrete_ln_pmf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, NegativeBinomial::new, discrete_cdf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, NegativeBinomial::new, discrete_ln_cdf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, NegativeBinomial::new, discrete_sf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, NegativeBinomial::new, discrete_ln_sf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, NegativeBinomial::new, discrete_ppf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, NegativeBinomial::new, discrete_isf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_pmf_scalar(
    inputs: &[Series],
    kwargs: NegativeBinomialParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_pmf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_ln_pmf_scalar(
    inputs: &[Series],
    kwargs: NegativeBinomialParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_ln_pmf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_cdf_scalar(
    inputs: &[Series],
    kwargs: NegativeBinomialParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_cdf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_ln_cdf_scalar(
    inputs: &[Series],
    kwargs: NegativeBinomialParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_ln_cdf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_sf_scalar(
    inputs: &[Series],
    kwargs: NegativeBinomialParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_sf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_ln_sf_scalar(
    inputs: &[Series],
    kwargs: NegativeBinomialParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_ln_sf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_ppf_scalar(
    inputs: &[Series],
    kwargs: NegativeBinomialParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_ppf)
}

#[polars_expr(output_type=Float64)]
fn negativebinomial_isf_scalar(
    inputs: &[Series],
    kwargs: NegativeBinomialParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_isf)
}

/// A draw's state: the point mass at `0` at `p = 1`, where the Gamma's scale `(1 - p) / p` is `0`, and the
/// Gamma-Poisson mixture otherwise.
enum Sampler {
    PointMass,
    Mixture(Gamma),
}

/// The mixture `Poisson(Gamma(r, scale = (1 - p) / p))`. The Poisson stage is `rand_distr`'s: statrs' accepts its
/// logistic proposal outright from a rate near `1e15`. A mean `r (1 - p) / p` past `MAX_LAMBDA` has no `UInt64`
/// count and is refused here; every other method still answers.
fn build_sampler(r: f64, p: f64) -> PolarsResult<Sampler> {
    if p == 1.0 {
        return Ok(Sampler::PointMass);
    }
    let q = 1.0 - p;
    let mean = r * q / p;
    polars_ensure!(
        mean <= PoissonSampler::<f64>::MAX_LAMBDA,
        ComputeError: "the mean r (1 - p) / p must be at most {:e} to sample a UInt64 count, got {mean}",
        PoissonSampler::<f64>::MAX_LAMBDA
    );
    Gamma::new(r, p / q)
        .map(Sampler::Mixture)
        .map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// One count. A Gamma draw of `0`, a rate `rand_distr` refuses, is a count of `0`. One past `MAX_LAMBDA` has no
/// `UInt64` count: it sets `overflowed` and answers a placeholder `0` that [`refusing_overflow`] never returns.
#[inline]
fn draw(sampler: &Sampler, rng: &mut impl rand::Rng, overflowed: &AtomicBool) -> u64 {
    let Sampler::Mixture(gamma) = sampler else {
        return 0;
    };
    let rate: f64 = RandDistribution::sample(gamma, rng);
    if rate == 0.0 {
        return 0;
    }
    match PoissonSampler::new(rate) {
        Ok(poisson) => {
            let count: f64 = RandDistribution::sample(&poisson, rng);
            count as u64
        },
        Err(_) => {
            overflowed.store(true, Ordering::Relaxed);
            0
        },
    }
}

/// Runs `sample` with the flag its draws set past `MAX_LAMBDA`, and refuses the call if any draw set it.
fn refusing_overflow(
    sample: impl FnOnce(&AtomicBool) -> PolarsResult<Series>,
) -> PolarsResult<Series> {
    let overflowed = AtomicBool::new(false);
    let out = sample(&overflowed)?;
    polars_ensure!(
        !overflowed.load(Ordering::Relaxed),
        ComputeError: "a draw's Gamma rate exceeded {:e}, past which its Poisson count does not fit a UInt64",
        PoissonSampler::<f64>::MAX_LAMBDA
    );
    Ok(out)
}

#[polars_expr(output_type=UInt64)]
fn negativebinomial_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    refusing_overflow(|overflowed| {
        sample_per_row_ternary(
            inputs,
            kwargs,
            coerce_f64,
            coerce_f64,
            check_params,
            build_sampler,
            |sampler, rng| draw(sampler, rng, overflowed),
        )
    })
}

#[polars_expr(output_type=UInt64)]
fn negativebinomial_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<NegativeBinomialParams>,
) -> PolarsResult<Series> {
    let sampler = kwargs.params.build_sampler()?;
    refusing_overflow(|overflowed| {
        sample_by_index(&inputs[0], kwargs.seed, |rng| {
            draw(&sampler, rng, overflowed)
        })
    })
}

#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn negativebinomial_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<NegativeBinomialParams>,
) -> PolarsResult<Series> {
    let sampler = kwargs.params.build_sampler()?;
    refusing_overflow(|overflowed| {
        samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| {
            draw(&sampler, rng, overflowed)
        })
    })
}

#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn negativebinomial_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    refusing_overflow(|overflowed| {
        samples_per_row_ternary(
            inputs,
            kwargs,
            coerce_f64,
            coerce_f64,
            check_params,
            build_sampler,
            |sampler, rng| draw(sampler, rng, overflowed),
        )
    })
}
