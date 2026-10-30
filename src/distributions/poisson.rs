//! `Poisson(rate)` reads the incomplete gamma of `gamma.rs` at `t = rate`. `P(X <= k)` is the upper tail
//! `Q(floor(k) + 1, rate)` and `P(X > k)` the lower one, the reverse of the continuous case, and the mass
//! `rate^k e^-rate / k!` is the prefactor at shape `k`. statrs' `pmf` spells it `-rate + k ln(rate) - ln(k!)`, whose
//! `rate ln(rate)`-sized terms cancel to an error of `2.5e-7` in the log at `rate = 1e8`; Stirling's form of the
//! prefactor has nothing to cancel.

use std::f64::consts::SQRT_2;

use polars::prelude::arity::unary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use rand_distr::Poisson as PoissonSampler;
use statrs::function::erf::erfc_inv;

use crate::distributions::gamma::{IncompleteGamma, Tail, Tails, LN_2PI_E};
use crate::distributions::{
    coerce_f64, on_unit_interval, polynomial, validated_param, value_keyed_derived_binary,
    value_keyed_scalar, ParamDomain,
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

/// Relative slack on the inverses' step comparison, as `binomial.rs`'s `PPF_CDF_TOL`. The tails are
/// compared as logs, where a relative slack is an absolute one, so a quantile computed on a step,
/// `cdf(k)` itself, inverts to `k` whichever way its last ulp rounded.
const STEP_TOLERANCE: f64 = 1e-12;

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

    /// Both tails at a point `k >= 0`, or `None` at `k = +inf`, which no gamma shape stands for and
    /// which has the whole mass below it. `Tail::Upper` is `P(X <= k)`.
    fn tails_at(&self, k: f64) -> Option<Tails> {
        k.is_finite()
            .then(|| IncompleteGamma::new(k.floor() + 1.0).tails(self.rate, self.ln_rate))
    }

    /// `ln(rate^k e^-rate / k!)` at a support point: the prefactor at shape `k`, not the density at
    /// shape `k + 1`, which rounds to `k` past `2^53`. `k = 0` is `-rate` outright, so no gamma of
    /// shape `0` is built.
    fn ln_mass(&self, k: f64) -> f64 {
        if k == 0.0 {
            -self.rate
        } else {
            IncompleteGamma::new(k).ln_prefactor(self.rate, self.ln_rate)
        }
    }

    /// The smallest support point where `tail` has reached the level `exp(ln_target)`: `P(X <= k) >=`
    /// it for `Tail::Upper`, `P(X > k) <=` it for `Tail::Lower`, each within [`STEP_TOLERANCE`].
    ///
    /// `0` is tried first, before `seed` is computed: it is the answer for most levels at a small
    /// `rate`. Otherwise galloped outward from `seed` until `lo` has not reached the level and `hi`
    /// has, then bisected on the integers. The first step is one float spacing at `seed`, so no step
    /// vanishes above `2^53`. `+inf` reaches either level, so the gallop up ends, and the bisection
    /// ends where no float lies strictly inside the bracket.
    fn smallest_reaching(&self, tail: Tail, ln_target: f64, seed: impl FnOnce() -> f64) -> f64 {
        let reached = |k: f64| {
            self.tails_at(k).is_none_or(|tails| match tail {
                Tail::Upper => tails.ln_probability(tail) >= ln_target - STEP_TOLERANCE,
                Tail::Lower => tails.ln_probability(tail) <= ln_target + STEP_TOLERANCE,
            })
        };
        if reached(0.0) {
            return 0.0;
        }
        let seed = seed().clamp(1.0, f64::MAX);
        let mut step = (seed * f64::EPSILON).max(1.0);
        let (mut lo, mut hi) = if reached(seed) {
            let mut hi = seed;
            loop {
                let below = hi - step;
                if below <= 0.0 {
                    break (0.0, hi);
                }
                if !reached(below) {
                    break (below, hi);
                }
                hi = below;
                step *= 2.0;
            }
        } else {
            let mut lo = seed;
            loop {
                let above = lo + step;
                if reached(above) {
                    break (lo, above);
                }
                lo = above;
                step *= 2.0;
            }
        };
        loop {
            let mid = (lo + (hi - lo) / 2.0).floor();
            if mid <= lo || mid >= hi {
                return hi;
            }
            if reached(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    }

    /// The smallest `k` with `P(X <= k) >= q` for `Tail::Upper`, or with `P(X > k) <= q` for
    /// `Tail::Lower`, for `q` in `[0, 1]`. Past `q = 1/2` it compares the other tail at `ln(1 - q)`,
    /// exact by Sterbenz, so the comparison is always on the smaller tail. `P(X > k)` never reaches
    /// `0`, so that target is the open end of the support.
    ///
    /// The search is seeded at the continuity-corrected Cornish-Fisher estimate
    /// `rate + z sqrt(rate) + (z^2 - 1) / 6 - 1/2`, `z` the normal quantile of the cdf level: in the
    /// bulk the gallop brackets the answer in a few evaluations, where a search from `0` takes
    /// `log2(rate)`.
    fn quantile(&self, tail: Tail, q: f64) -> f64 {
        let seed = || {
            let z = match tail {
                Tail::Upper => -SQRT_2 * erfc_inv(2.0 * q),
                Tail::Lower => SQRT_2 * erfc_inv(2.0 * q),
            };
            (self.rate + z * self.rate.sqrt() + (z * z - 1.0) / 6.0 - 0.5).ceil()
        };
        let (tail, ln_target) = if q <= 0.5 {
            (tail, q.ln())
        } else {
            (tail.complement(), (-q).ln_1p())
        };
        if tail == Tail::Lower && ln_target == f64::NEG_INFINITY {
            return f64::INFINITY;
        }
        self.smallest_reaching(tail, ln_target, seed)
    }
}

/// `-inf` off the support: below `0`, a non-integral point, and `+inf`, which `floor` keeps integral.
fn ln_pmf_value(poisson: &Poisson, k: f64) -> Option<f64> {
    Some(if k >= 0.0 && k.floor() == k && k.is_finite() {
        poisson.ln_mass(k)
    } else {
        f64::NEG_INFINITY
    })
}

fn pmf_value(poisson: &Poisson, k: f64) -> Option<f64> {
    ln_pmf_value(poisson, k).map(f64::exp)
}

fn cdf_value(poisson: &Poisson, k: f64) -> Option<f64> {
    Some(if k < 0.0 {
        0.0
    } else {
        poisson
            .tails_at(k)
            .map_or(1.0, |tails| tails.probability(Tail::Upper))
    })
}

fn ln_cdf_value(poisson: &Poisson, k: f64) -> Option<f64> {
    Some(if k < 0.0 {
        f64::NEG_INFINITY
    } else {
        poisson
            .tails_at(k)
            .map_or(0.0, |tails| tails.ln_probability(Tail::Upper))
    })
}

fn sf_value(poisson: &Poisson, k: f64) -> Option<f64> {
    Some(if k < 0.0 {
        1.0
    } else {
        poisson
            .tails_at(k)
            .map_or(0.0, |tails| tails.probability(Tail::Lower))
    })
}

fn ln_sf_value(poisson: &Poisson, k: f64) -> Option<f64> {
    Some(if k < 0.0 {
        0.0
    } else {
        poisson
            .tails_at(k)
            .map_or(f64::NEG_INFINITY, |tails| tails.ln_probability(Tail::Lower))
    })
}

fn ppf_value(poisson: &Poisson, q: f64) -> Option<f64> {
    on_unit_interval(&|q| poisson.quantile(Tail::Upper, q), q)
}

fn isf_value(poisson: &Poisson, q: f64) -> Option<f64> {
    on_unit_interval(&|q| poisson.quantile(Tail::Lower, q), q)
}

#[polars_expr(output_type=Float64)]
fn poisson_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, pmf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, ln_pmf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, sf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, Poisson::new, isf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_pmf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pmf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_pmf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pmf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_cdf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_cdf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_sf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_ln_sf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_ppf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn poisson_isf_scalar(inputs: &[Series], kwargs: PoissonParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

/// From this `rate` the entropy is statrs' three-term asymptotic series, `2.3e-14` relative here and
/// falling as `rate^-4`; below it the support sum, whose length grows as `sqrt(rate)`.
const ENTROPY_SERIES_FROM: f64 = 1e3;

/// The coefficients of `rate^-1`, `rate^-2` and `rate^-3` the asymptotic series subtracts from
/// `ln(2 pi e rate) / 2`.
const ENTROPY_SERIES: [f64; 3] = [1.0 / 12.0, 1.0 / 24.0, 19.0 / 360.0];

/// A support-sum term below this fraction of the sum so far ends that direction of the sum.
const ENTROPY_TERM_FRACTION: f64 = f64::EPSILON / 2.0;

/// Shannon entropy (nats). From [`ENTROPY_SERIES_FROM`] it is the asymptotic series, its
/// `ln(2 pi e rate)` read as `ln(2 pi e) + ln(rate)`, which stays finite where the product overflows.
/// Below, `-sum p ln p` over the support from the mode outward, each mass through the `ln_pmf` body.
/// A direction ends once a term is below [`ENTROPY_TERM_FRACTION`] of the sum. Away from the mode
/// the terms fall with the masses, except from `k = 0` to `1` below `rate = 1`, where `pmf(0)`
/// exceeds `1 / e`; each term is added before the test, which steps over that one rise.
fn entropy(rate: f64) -> f64 {
    if rate >= ENTROPY_SERIES_FROM {
        let inverse = rate.recip();
        return 0.5 * (LN_2PI_E + rate.ln()) - polynomial(&ENTROPY_SERIES, inverse) * inverse;
    }
    let poisson = Poisson::new(rate);
    let term = |k: f64| {
        let ln_mass = poisson.ln_mass(k);
        -ln_mass.exp() * ln_mass
    };
    let mode = rate.floor();
    let mut sum = term(mode);
    for direction in [-1.0, 1.0] {
        let mut k = mode + direction;
        while k >= 0.0 {
            let added = term(k);
            sum += added;
            if added <= ENTROPY_TERM_FRACTION * sum {
                break;
            }
            k += direction;
        }
    }
    sum
}

/// Shannon entropy (nats): a support sum with no Polars expression to move to.
#[polars_expr(output_type=Float64)]
fn poisson_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let rate = coerce_f64(&inputs[0])?;
    RATE.check_column(&rate)?;
    let out: Float64Chunked = unary_elementwise(&rate, |rate| rate.map(entropy));
    Ok(out.into_series())
}

/// `rand_distr::Poisson` rather than statrs', whose rejection test forms `k ln(rate) - ln(k!)` and
/// cancels: from `rate ~ 1e15` it accepts its logistic proposal outright. A draw is a `UInt64` count,
/// so a `rate` past `MAX_LAMBDA` is refused here, while every other method still answers.
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
