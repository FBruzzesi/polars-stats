#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution as RandDistribution;
use statrs::distribution::Exp;

use crate::distributions::param_validator;
use crate::rng::{
    binary_param_rows, sample_scalar_plugin, samples_f64_output, samples_per_row, SampleKwargs,
    SamplesKwargs,
};

/// Construct a `statrs::Exp`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::Exp::new` rejects a `NaN` rate or `rate <= 0` (a positive-infinite rate is accepted, as a
/// degenerate point mass at 0). We surface the rejection as `InvalidOperation` so an invalid rate
/// fails the whole evaluation rather than silently nulling the row. Validation lives here so every
/// method that builds a distribution reports an invalid rate identically.
fn build_dist(rate: f64) -> PolarsResult<Exp> {
    Exp::new(rate).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("rate must be strictly positive, got rate={rate}: {e}").into(),
        )
    })
}

param_validator! {
    /// Element-wise validation of the rate (λ): returns `rate` unchanged, raising `InvalidOperation`
    /// if `rate` is `NaN` or `rate <= 0`. `null` propagates.
    ///
    /// Exponential is an elementary closed-form distribution, so its pdf / cdf / ppf and moments are
    /// pure Polars expressions; routing the rate through this validator is what lets them report an
    /// invalid parameterisation consistently with `exponential_sample`, instead of silently computing
    /// with a non-positive rate. One of the two unary `param_validator!` users (with `bernoulli_proba`).
    fn exponential_rate;
    params = (rate: DataType::Float64 => f64);
    build = build_dist;
    returns = rate;
    output_name = inputs[0];
}

/// Element-wise Exponential sampler.
///
/// `inputs[0]` carries the rate (one per row) and `inputs[1]` a per-row index used to derive a
/// per-row sub-seed, so the function is genuinely element-wise: chunking and threading cannot change
/// the output. With `seed=None`, a fresh root seed is drawn once per call.
///
/// Per-row validation:
///   * `null` (in either input) propagates;
///   * a `NaN` or non-positive `rate` raises `InvalidOperation`.
///
/// The draw keeps `statrs` (`O(1)` ziggurat: `sample_exp_1(rng) / rate`); routing it through
/// `rand_distr` would buy nothing, since that is already the algorithm class `statrs` uses (unlike
/// the binomial draw, see DESIGN.md). Returns a `Float64` series.
#[polars_expr(output_type=Float64)]
fn exponential_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let rate = inputs[0].cast(&DataType::Float64)?;
    let rate_ca = rate.f64()?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs();

    let ca: Float64Chunked = try_binary_elementwise(
        rate_ca,
        index_ca,
        |rate_opt, i_opt| -> PolarsResult<Option<f64>> {
            match (rate_opt, i_opt) {
                (Some(r), Some(i)) => {
                    let dist = build_dist(r)?;
                    let mut rng = rngs.rng(i);
                    let draw: f64 = RandDistribution::sample(&dist, &mut rng);
                    Ok(Some(draw))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

sample_scalar_plugin! {
    /// Static parameter for the constant-rate sampler fast path: when `rate` is a Python scalar, it
    /// travels here as a kwarg instead of as a full-length `pl.repeat` column re-validated on every row.
    struct ExponentialScalarKwargs { rate: f64 }

    /// Constant-rate Exponential sampler: [`exponential_sample`] for the scalar-`rate` case. The
    /// distribution is validated and built once, and only the per-row index travels as an input;
    /// seeding and the draw are unchanged, so output matches `exponential_sample` for the same
    /// `(seed, index, rate)`.
    fn exponential_sample_scalar(output_type = Float64, physical = Float64Type);

    samples = exponential_samples_scalar as ExponentialSamplesScalarKwargs -> samples_f64_output;

    build = |kw| build_dist(kw.rate)?;
    draw = |dist, rng| RandDistribution::sample(&dist, rng);
}

/// Element-wise multi-draw Exponential sampler: `size` draws per row in one call.
///
/// The column-parameter counterpart of [`exponential_samples_scalar`]: the distribution is built once
/// per row instead of once per draw. Row `i`'s draws come from the one stream seeded `(seed, i)`, so
/// output is bit-identical to the scalar path for the same `rate` (the seeding is positional,
/// parameters never enter it, so equal-`rate` rows still draw independently). Null/error contract
/// follows [`exponential_sample`] per row; a null row yields a null array element. Returns
/// `Array(Float64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn exponential_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let rate = inputs[0].cast(&DataType::Float64)?;
    let index = inputs[1].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = binary_param_rows(rate.f64()?, index.u64()?, build_dist);

    samples_per_row::<Float64Type, _, _, _, _>(
        name,
        kwargs.seed,
        kwargs.size,
        rows,
        RandDistribution::sample,
    )
}
