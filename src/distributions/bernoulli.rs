use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution;
use statrs::distribution::Bernoulli;

use crate::distributions::{
    on_unit_interval, validated_param, value_keyed_derived_binary, value_keyed_scalar, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_binary, samples_bool_output, samples_by_index,
    samples_per_row_binary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const P: ParamDomain = ParamDomain::probability("p");

fn build_dist(p: f64) -> PolarsResult<Bernoulli> {
    Bernoulli::new(p).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameter, deserialised once per call; every `_scalar` twin checks it once here.
#[derive(serde::Deserialize)]
struct BernoulliParams {
    p: f64,
}

impl BernoulliParams {
    fn build(&self) -> PolarsResult<Bernoulli> {
        P.check(self.p)?;
        build_dist(self.p)
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
fn bernoulli_proba(inputs: &[Series]) -> PolarsResult<Series> {
    validated_param(inputs, &P)
}

// Each method pairs a `derive` that turns `p` into its branch answers with an `at` that picks one by
// where the point sits; `derive` runs once per call on a constant `p`, once per row on a column.

/// `pmf` / `log_pmf`: the answer at each support point, and off the support.
struct Mass {
    at_zero: f64,
    at_one: f64,
    off_support: f64,
}

impl Mass {
    fn pmf(p: f64) -> Self {
        Mass {
            at_zero: 1.0 - p,
            at_one: p,
            off_support: 0.0,
        }
    }

    /// `ln_1p(-p)`, not `ln(1 - p)`, which collapses to `0.0` below `p ~ 1.1e-16`.
    fn ln_pmf(p: f64) -> Self {
        Mass {
            at_zero: (-p).ln_1p(),
            at_one: p.ln(),
            off_support: f64::NEG_INFINITY,
        }
    }

    /// Exact `f64` equality, never an integer cast: `pmf(0.5)` is `0.0`, not an error.
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
    fn cdf(p: f64) -> Self {
        Steps {
            below_zero: 0.0,
            below_one: 1.0 - p,
            at_least_one: 1.0,
        }
    }

    fn ln_cdf(p: f64) -> Self {
        Steps {
            below_zero: f64::NEG_INFINITY,
            below_one: (-p).ln_1p(),
            at_least_one: 0.0,
        }
    }

    /// Read off `p` directly, never as `1 - cdf`, which quantises `p` to the `1.1e-16` spacing of
    /// `1.0` and reaches `0.0` below that.
    fn sf(p: f64) -> Self {
        Steps {
            below_zero: 1.0,
            below_one: p,
            at_least_one: 0.0,
        }
    }

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

/// `ppf`: the one cdf step, and the answer at `q == 1`.
struct PpfCutoffs {
    step: f64,
    /// `1.0` unless the mass at 1 is zero. `1 - p` rounds to exactly `1.0` below `p ~ 1.1e-16`, so
    /// `q > step` would answer `0.0` at `q = 1` where `1.0` is the only correct answer.
    at_quantile_one: f64,
}

impl PpfCutoffs {
    fn derive(p: f64) -> Self {
        PpfCutoffs {
            step: 1.0 - p,
            at_quantile_one: f64::from(p > 0.0),
        }
    }

    /// Smallest `x` with `cdf(x) >= q`, as `Float64` so `NaN` stays representable; null outside
    /// `[0, 1]`.
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

/// Smallest `x` with `sf(x) <= q`, compared against `q` itself: `sf(0)` is `p`, so no complement is
/// formed, where `ppf(1 - q)` forms two and loses the answer once either saturates.
fn derive_isf(p: f64) -> impl Fn(f64) -> f64 {
    move |quantile: f64| f64::from(p > quantile)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, Mass::pmf, Mass::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, Mass::ln_pmf, Mass::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, Steps::cdf, Steps::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, Steps::ln_cdf, Steps::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, Steps::sf, Steps::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, Steps::ln_sf, Steps::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, PpfCutoffs::derive, PpfCutoffs::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &P, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_pmf_scalar(inputs: &[Series], kwargs: BernoulliParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Mass::pmf, Mass::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_ln_pmf_scalar(inputs: &[Series], kwargs: BernoulliParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Mass::ln_pmf, Mass::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_cdf_scalar(inputs: &[Series], kwargs: BernoulliParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Steps::cdf, Steps::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_ln_cdf_scalar(inputs: &[Series], kwargs: BernoulliParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Steps::ln_cdf, Steps::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_sf_scalar(inputs: &[Series], kwargs: BernoulliParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Steps::sf, Steps::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_ln_sf_scalar(inputs: &[Series], kwargs: BernoulliParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Steps::ln_sf, Steps::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_ppf_scalar(inputs: &[Series], kwargs: BernoulliParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], PpfCutoffs::derive, PpfCutoffs::at)
}

#[polars_expr(output_type=Float64)]
fn bernoulli_isf_scalar(inputs: &[Series], kwargs: BernoulliParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

#[inline]
fn draw(dist: &Bernoulli, rng: &mut impl rand::Rng) -> bool {
    <Bernoulli as Distribution<bool>>::sample(dist, rng)
}

#[polars_expr(output_type=Boolean)]
fn bernoulli_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_binary(inputs, kwargs, &P, build_dist, draw)
}

#[polars_expr(output_type=Boolean)]
fn bernoulli_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<BernoulliParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<BernoulliParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_bool_output)]
fn bernoulli_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_binary(inputs, kwargs, &P, build_dist, draw)
}
