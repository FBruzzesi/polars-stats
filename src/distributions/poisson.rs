//! `Poisson(rate)` reads the incomplete gamma of `gamma.rs` at `t = rate`. `P(X <= k)` is the upper tail
//! `Q(floor(k) + 1, rate)` and `P(X > k)` the lower one, the reverse of the continuous case, and the mass
//! `rate^k e^-rate / k!` is the prefactor at shape `k`. statrs' `pmf` spells it `-rate + k ln(rate) - ln(k!)`, whose
//! `rate ln(rate)`-sized terms cancel to an error of `2.5e-7` in the log at `rate = 1e8`; Stirling's form of the
//! prefactor has nothing to cancel.

use std::f64::consts::LN_2;

use polars::prelude::arity::unary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use rand_distr::Poisson as PoissonSampler;

use crate::distributions::gamma::{IncompleteGamma, LN_2PI_E};
use crate::distributions::{
    coerce_f64, discrete_cdf, discrete_isf, discrete_ln_cdf, discrete_ln_pmf, discrete_ln_sf,
    discrete_pmf, discrete_ppf, discrete_quantile, discrete_sf, polynomial, validated_param,
    value_keyed_derived_binary, value_keyed_scalar, DiscreteTails, ParamDomain, Tail, Tails,
};
use crate::rng::{
    sample_by_index, sample_per_row_binary, samples_by_index, samples_per_row_binary,
    samples_u64_output, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const RATE: ParamDomain = ParamDomain::positive("rate");

/// Constant parameter, deserialised once per call; every `_scalar` twin checks it once here.
#[derive(serde::Deserialize)]
struct PoissonParams {
    rate: f64,
}

impl PoissonParams {
    fn build_sampler(&self) -> PolarsResult<PoissonSampler<f64>> {
        RATE.check(self.rate)?;
        build_sampler(self.rate)
    }

    fn value_keyed(
        &self,
        value: &Series,
        select: impl Fn(&Poisson, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        RATE.check(self.rate)?;
        value_keyed_scalar(value, &Poisson::new(self.rate), select)
    }
}

/// The validated `rate`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn poisson_rate(inputs: &[Series]) -> PolarsResult<Series> {
    validated_param(inputs, &RATE)
}

/// `ln(k!)` for `k = 0..=9`, correctly rounded: the shapes below `gamma.rs`'s Stirling switch, where
/// [`IncompleteGamma`] reads `ln Gamma(k + 1)` off statrs' `ln_gamma`, which is up to `3e-14` off.
const LN_FACTORIALS: [f64; 10] = [
    0.0,
    0.0,
    LN_2,
    1.791759469228055,
    3.1780538303479458,
    4.787491742782046,
    6.579251212010101,
    8.525161361065415,
    10.60460290274525,
    12.801827480081469,
];

/// `rate` and `ln(rate)`, the `(t, u)` pair every incomplete-gamma evaluation takes.
struct Poisson {
    rate: f64,
    ln_rate: f64,
}

impl Poisson {
    fn new(rate: f64) -> Self {
        Self {
            rate,
            ln_rate: rate.ln(),
        }
    }
}

impl DiscreteTails for Poisson {
    /// `P(X <= k)`, the upper incomplete gamma at shape `floor(k) + 1`.
    const CDF: Tail = Tail::Upper;

    /// No gamma shape stands for `k = +inf`, so it answers `None` there.
    fn tails_at(&self, k: f64) -> Option<Tails> {
        k.is_finite()
            .then(|| IncompleteGamma::new(k.floor() + 1.0).tails(self.rate, self.ln_rate))
    }

    /// `ln(rate^k e^-rate / k!)` at a support point, the prefactor at shape `k`: the density at shape
    /// `k + 1` would read a neighbouring float past `2^53`. Below `k = 10` the factorial comes from
    /// [`LN_FACTORIALS`], so no gamma of shape `0` is built. Forced inline: as a call, the entropy's
    /// support sum runs 1.5x to 2.4x slower.
    #[inline(always)]
    fn ln_mass(&self, k: f64) -> f64 {
        match LN_FACTORIALS.get(k as usize) {
            Some(ln_factorial) => k * self.ln_rate - self.rate - ln_factorial,
            None => IncompleteGamma::new(k).ln_prefactor(self.rate, self.ln_rate),
        }
    }

    /// Seeded at the continuity-corrected Cornish-Fisher estimate `rate + z sqrt(rate) + (z^2 - 1) / 6 - 1/2`: in
    /// the bulk the gallop brackets the answer in two or three evaluations, where a search from `0` takes
    /// `log2(rate)`.
    fn quantile(&self, tail: Tail, q: f64) -> f64 {
        let rate = self.rate;
        discrete_quantile(
            |k| self.tails_at(k),
            Self::CDF,
            tail,
            q,
            |z| rate + z * rate.sqrt() + (z * z - 1.0) / 6.0 - 0.5,
        )
    }
}

#[polars_expr(output_type=Float64)]
fn poisson_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, discrete_pmf)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, discrete_ln_pmf)
}

#[polars_expr(output_type=Float64)]
fn poisson_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, discrete_cdf)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, discrete_ln_cdf)
}

#[polars_expr(output_type=Float64)]
fn poisson_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, discrete_sf)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, discrete_ln_sf)
}

#[polars_expr(output_type=Float64)]
fn poisson_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, discrete_ppf)
}

#[polars_expr(output_type=Float64)]
fn poisson_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, discrete_isf)
}

#[polars_expr(output_type=Float64)]
fn poisson_pmf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_pmf)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_pmf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_ln_pmf)
}

#[polars_expr(output_type=Float64)]
fn poisson_cdf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_cdf)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_cdf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_ln_cdf)
}

#[polars_expr(output_type=Float64)]
fn poisson_sf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_sf)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_sf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_ln_sf)
}

#[polars_expr(output_type=Float64)]
fn poisson_ppf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_ppf)
}

#[polars_expr(output_type=Float64)]
fn poisson_isf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], discrete_isf)
}

/// From this `rate` the entropy is its asymptotic series, truncated `8e-17` relative here and falling
/// as `rate^-9`; below it the support sum, whose length grows as `sqrt(rate)`.
const ENTROPY_SERIES_FROM: f64 = 100.0;

/// The coefficients `c_j` of `rate^-j`, `j = 1..=8`, the asymptotic series subtracts from
/// `ln(2 pi e rate) / 2`: Stirling's series of `ln X!` expanded about `rate` and averaged over the
/// Poisson central moments of `X`. The first three are statrs' series.
const ENTROPY_SERIES: [f64; 8] = [
    1.0 / 12.0,
    1.0 / 24.0,
    19.0 / 360.0,
    9.0 / 80.0,
    863.0 / 2520.0,
    1375.0 / 1008.0,
    33953.0 / 5040.0,
    57281.0 / 1440.0,
];

/// A support-sum term below this fraction of the sum so far ends that direction of the sum.
const ENTROPY_TERM_FRACTION: f64 = f64::EPSILON / 2.0;

/// Shannon entropy (nats). From [`ENTROPY_SERIES_FROM`] it is the asymptotic series, its
/// `ln(2 pi e rate)` read as `ln(2 pi e) + ln(rate)`, which stays finite where the product overflows.
/// Below, `-sum p ln p` over the support from the mode outward, each mass through the `ln_pmf` body.
/// The terms fall with the masses away from the mode, except from `k = 0` to `1` below `rate = 1`,
/// where `pmf(0)` exceeds `1 / e`; each term is added before the stopping test, which steps over that
/// one rise.
fn entropy(rate: f64) -> f64 {
    if rate >= ENTROPY_SERIES_FROM {
        let inverse_rate = rate.recip();
        return 0.5 * (LN_2PI_E + rate.ln())
            - polynomial(&ENTROPY_SERIES, inverse_rate) * inverse_rate;
    }
    let poisson = Poisson::new(rate);
    let entropy_term = |k: f64| {
        let ln_mass = poisson.ln_mass(k);
        -ln_mass.exp() * ln_mass
    };
    let mode = rate.floor();
    let mut sum = entropy_term(mode);
    for direction in [-1.0, 1.0] {
        let mut k = mode + direction;
        while k >= 0.0 {
            let term = entropy_term(k);
            sum += term;
            if term <= ENTROPY_TERM_FRACTION * sum {
                break;
            }
            k += direction;
        }
    }
    sum
}

/// Shannon entropy (nats): the support sum below [`ENTROPY_SERIES_FROM`] has no Polars expression to
/// move to.
#[polars_expr(output_type=Float64)]
fn poisson_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let rate = coerce_f64(&inputs[0])?;
    RATE.check_column(&rate)?;
    let out: Float64Chunked = unary_elementwise(&rate, |rate| rate.map(entropy));
    Ok(out.into_series())
}

/// `rand_distr::Poisson` rather than statrs', whose rejection test cancels at a large `rate` (see the
/// design notes). A draw is a `UInt64` count, so a `rate` past `MAX_LAMBDA` is refused here, while
/// every other method still answers.
fn build_sampler(rate: f64) -> PolarsResult<PoissonSampler<f64>> {
    PoissonSampler::new(rate).map_err(|e| {
        polars_err!(
            ComputeError: "rate must be at most {:e} to sample a UInt64 count, got {rate}: {e}",
            PoissonSampler::<f64>::MAX_LAMBDA
        )
    })
}

#[inline]
fn draw(dist: &PoissonSampler<f64>, rng: &mut impl rand::Rng) -> u64 {
    let count: f64 = RandDistribution::sample(dist, rng);
    count as u64
}

#[polars_expr(output_type=UInt64)]
fn poisson_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_binary(inputs, kwargs, &RATE, build_sampler, draw)
}

#[polars_expr(output_type=UInt64)]
fn poisson_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<PoissonParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build_sampler()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn poisson_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<PoissonParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build_sampler()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_u64_output)]
fn poisson_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_binary(inputs, kwargs, &RATE, build_sampler, draw)
}
