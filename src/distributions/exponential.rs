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
    /// with a non-positive rate.
    fn exponential_rate;
    params = (rate: DataType::Float64 => f64);
    build = build_dist;
    returns = rate;
    output_name = inputs[0];
}

/// Element-wise Exponential sampler.
///
/// `inputs[0]` carries the rate (one per row), `inputs[1]` the per-row index each row's sub-seed
/// derives from, so chunking and threading cannot change the output; `seed=None` draws a fresh
/// root seed once per call. Per row, `null` propagates and an invalid rate raises via
/// [`build_dist`].
///
/// The draw keeps `statrs` (`O(1)` ziggurat: `sample_exp_1(rng) / rate`); routing it through
/// `rand_distr` would buy nothing, since that is already the algorithm class `statrs` uses (unlike
/// the binomial draw, see docs/explanation/design.md). Returns a `Float64` series.
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
    /// Static `rate` for the constant-parameter sampler fast path: validated once, passed as a
    /// kwarg instead of a full-length column.
    struct ExponentialScalarKwargs { rate: f64 }

    /// Constant-rate Exponential sampler: [`exponential_sample`] with the distribution built once;
    /// seeding and draw unchanged, so output is bit-identical for the same inputs.
    fn exponential_sample_scalar(output_type = Float64, physical = Float64Type);

    samples = exponential_samples_scalar as ExponentialSamplesScalarKwargs -> samples_f64_output;

    build = |kw| build_dist(kw.rate)?;
    draw = |dist, rng| RandDistribution::sample(&dist, rng);
}

/// Element-wise multi-draw Exponential sampler: `size` draws per row in one call, the distribution
/// built once per row.
///
/// Seeding is positional (see [`samples_per_row`]), so output is bit-identical to
/// [`exponential_samples_scalar`] for the same `rate`. Null/error contract follows
/// [`exponential_sample`] per row; a null row yields a null array element.
/// Returns `Array(Float64, size)`.
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
