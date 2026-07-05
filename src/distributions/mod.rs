pub mod bernoulli;
pub mod beta;
pub mod binomial;
pub mod exponential;
pub mod lognormal;
pub mod normal;
pub mod uniform;

use polars::prelude::arity::unary_elementwise;
use polars::prelude::*;

/// Shared driver for the constant-parameter value-keyed fast paths.
///
/// The constant-parameter counterpart of each distribution's `value_keyed` helper: when every
/// distribution parameter is a Python scalar, the caller validates them and builds the
/// distribution **once**, and only the evaluation-point column crosses FFI. This maps `f` over
/// that single column, where `f` is the same per-method body the per-row path uses (a named
/// function in each distribution file), so the two paths cannot drift and output is bit-identical.
///
/// Null contract: `value` is the only nullable input this driver sees. The distribution
/// parameters are not passed here at all; the caller validates them once (via `build_dist`,
/// which raises on an invalid or non-finite parameterisation before this runs) and bakes them
/// into `f` through a pre-built `dist`. So a null `value` propagates to null, and `f` returning
/// `None` nulls the row on the method's own terms (e.g. `ppf` outside `[0, 1]`), matching the
/// per-row path element for element.
pub(crate) fn value_keyed_scalar<F>(value: &Series, f: F) -> PolarsResult<Series>
where
    F: Fn(f64) -> Option<f64>,
{
    let value = value.cast(&DataType::Float64)?;
    let value_ca = value.f64()?;
    let name = value_ca.name().clone();

    let ca: Float64Chunked = unary_elementwise(value_ca, |opt| opt.and_then(&f));
    Ok(ca.with_name(name).into_series())
}

/// Generates a distribution's constant-parameter value-keyed fast paths: one `<method>_scalar`
/// plugin per Rust-bound method (`pdf`/`pmf`, `ln_*`, `cdf`, `sf`, `ppf`), all over the shared
/// [`value_keyed_scalar`] driver.
///
/// The constant-parameter counterpart of the per-row `value_keyed` helper, and the value-keyed
/// twin of [`sample_scalar_plugin!`](crate::rng::sample_scalar_plugin): when every distribution
/// parameter is a Python scalar, the parameters arrive as kwargs (validated and built into the
/// distribution **once** per call), and only the evaluation-point column crosses FFI. The three
/// things that vary between distributions are the macro's inputs:
///
/// * the kwargs fields (parameter names and types), one `<Name>ParamsKwargs` struct shared by
///   every method;
/// * `build`: validates the parameters and returns the `statrs` distribution, built **once** per
///   call (`?` is available);
/// * the `methods` list, each `fn <plugin_name> => <body>`, where `<body>` is the same named
///   per-method function the per-row `value_keyed` path applies (`pdf_value`, `ppf_value`, ...),
///   so the scalar and per-row paths share one body and cannot drift.
///
/// Every value-keyed method returns `Float64` (a density, probability, or quantile), so the output
/// dtype is fixed here; that is the structural difference from `sample_scalar_plugin!`, whose dtype
/// varies. Plugin and struct names stay literal in each invocation (greppable from the Python
/// layer), so `value_keyed_test.py`'s `test_value_keyed_scalar_fast_path_matches_per_row` keeps
/// pinning bit-equality with the per-row path. Call sites are the distribution modules, which all
/// have `polars::prelude::*` in scope.
macro_rules! value_keyed_scalar_plugins {
    (
        $(#[$kwargs_meta:meta])*
        struct $kwargs:ident { $($param:ident: $param_ty:ty),+ $(,)? }

        build = |$kw:ident| $build:expr;

        methods {
            $(
                $(#[$fn_meta:meta])*
                fn $fn_name:ident => $body:ident;
            )+
        }
    ) => {
        $(#[$kwargs_meta])*
        #[derive(serde::Deserialize)]
        struct $kwargs {
            $($param: $param_ty,)+
        }

        $(
            $(#[$fn_meta])*
            #[pyo3_polars::derive::polars_expr(output_type=Float64)]
            fn $fn_name(inputs: &[Series], kwargs: $kwargs) -> PolarsResult<Series> {
                let $kw = &kwargs;
                let dist = $build;
                $crate::distributions::value_keyed_scalar(&inputs[0], |v| $body(&dist, v))
            }
        )+
    };
}
pub(crate) use value_keyed_scalar_plugins;

/// Generates a distribution's per-row `value_keyed` helper: the column-parameter counterpart of
/// [`value_keyed_scalar`], building and validating the distribution **once per row** over
/// `(value, p1, p2)` and applying the per-method body `f`.
///
/// The general (non-fast-path) side of the value-keyed methods. Where the scalar fast path builds
/// the distribution once per call, this rebuilds it per row because at least one parameter is a
/// column. The three things that vary between distributions are the macro's inputs:
///
/// * the distribution type, used in the `F: Fn(&$dist, f64) -> Option<f64>` bound;
/// * each parameter's `<cast dtype> => <accessor>` pair (binomial's `n` is `Int64`/`.i64()`, the
///   continuous scales are `Float64`/`.f64()`); the evaluation point is always cast to `Float64`;
/// * `build`: the distribution constructor (`build_dist`), validating per row and raising on an
///   invalid parameterisation (`?` is available inside the closure).
///
/// `f` is the same named per-method body the scalar fast path applies, so the two paths share one
/// body and cannot drift. Only the 2-parameter (ternary) arm exists today, since every current
/// distribution takes two parameters; a 1-parameter distribution would add a `try_binary_elementwise`
/// arm here. Call sites are the distribution modules (`polars::prelude::*` in scope).
macro_rules! value_keyed_per_row {
    (
        $(#[$meta:meta])*
        fn $name:ident(&$dist:ty);
        params = ($p1_dt:expr => $p1_acc:ident, $p2_dt:expr => $p2_acc:ident);
        build = $build:path;
    ) => {
        $(#[$meta])*
        fn $name<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
        where
            F: Fn(&$dist, f64) -> Option<f64>,
        {
            let value = inputs[0].cast(&DataType::Float64)?;
            let value_ca = value.f64()?;
            let p1 = inputs[1].cast(&$p1_dt)?;
            let p1_ca = p1.$p1_acc()?;
            let p2 = inputs[2].cast(&$p2_dt)?;
            let p2_ca = p2.$p2_acc()?;
            let name = inputs[0].name().clone();

            let ca: Float64Chunked = polars::prelude::arity::try_ternary_elementwise(
                value_ca,
                p1_ca,
                p2_ca,
                |value_opt, p1_opt, p2_opt| -> PolarsResult<Option<f64>> {
                    match (value_opt, p1_opt, p2_opt) {
                        (Some(v), Some(a), Some(b)) => {
                            let dist = $build(a, b)?;
                            Ok(f(&dist, v))
                        },
                        _ => Ok(None),
                    }
                },
            )?;

            Ok(ca.with_name(name).into_series())
        }
    };
}
pub(crate) use value_keyed_per_row;

/// Generates a two-parameter validation plugin: validate `(p1, p2)` per row by constructing the
/// distribution, then return a derived `Float64` the closed-form Python methods can build on,
/// raising identically to the sampler on an invalid parameterisation.
///
/// These plugins exist so the closed-form moments and (for Uniform / Bernoulli) value-keyed
/// methods, which are pure Polars expressions, still report an invalid parameterisation through
/// the same Rust `build_dist` rather than silently producing a garbage result. The constant-
/// parameter fast path calls the very same plugin on length-1 `pl.lit` inputs, so it is built
/// once instead of per row. The four knobs that vary between distributions are the macro's inputs:
///
/// * each parameter's `<name>: <cast dtype> => <accessor>` (binomial's `n` is `Int64`/`.i64()`,
///   the continuous parameters are `Float64`/`.f64()`); the matched values bind to `<name>`;
/// * `build`: the constructor (`build_dist`), validating per row (`?` is available);
/// * `returns`: the `Float64` to emit, an expression over the parameter names (a parameter itself,
///   e.g. `std_dev`, or a derived width, e.g. `max - min`);
/// * `output_name = inputs[i]`: which input column names the output (most return the second
///   parameter and name after it; Uniform's width names after the first).
///
/// Two arms that differ only in arity: a binary one (`normal_std_dev`, `lognormal_sigma`,
/// `binomial_params`, `uniform_range`) and a unary one (`bernoulli_proba`, `exponential_rate`). The
/// unary arm landed with the second one-parameter distribution (Exponential), the trigger the
/// umbrella names for extracting it; before that `bernoulli_proba` was the lone hand-written unary
/// validator. A unary validator's `output_name` is always `inputs[0]` (its only input). Call sites
/// are the distribution modules (`polars::prelude::*` in scope).
macro_rules! param_validator {
    (
        $(#[$meta:meta])*
        fn $name:ident;
        params = ($p1:ident: $p1_dt:expr => $p1_acc:ident, $p2:ident: $p2_dt:expr => $p2_acc:ident);
        build = $build:path;
        returns = $ret:expr;
        output_name = inputs[$out_idx:literal];
    ) => {
        $(#[$meta])*
        #[pyo3_polars::derive::polars_expr(output_type=Float64)]
        fn $name(inputs: &[Series]) -> PolarsResult<Series> {
            let p1 = inputs[0].cast(&$p1_dt)?;
            let p1_ca = p1.$p1_acc()?;
            let p2 = inputs[1].cast(&$p2_dt)?;
            let p2_ca = p2.$p2_acc()?;
            let name = inputs[$out_idx].name().clone();

            let ca: Float64Chunked = polars::prelude::arity::try_binary_elementwise(
                p1_ca,
                p2_ca,
                |o1, o2| -> PolarsResult<Option<f64>> {
                    match (o1, o2) {
                        (Some($p1), Some($p2)) => {
                            $build($p1, $p2)?;
                            Ok(Some($ret))
                        },
                        _ => Ok(None),
                    }
                },
            )?;

            Ok(ca.with_name(name).into_series())
        }
    };
    (
        $(#[$meta:meta])*
        fn $name:ident;
        params = ($p1:ident: $p1_dt:expr => $p1_acc:ident);
        build = $build:path;
        returns = $ret:expr;
        output_name = inputs[$out_idx:literal];
    ) => {
        $(#[$meta])*
        #[pyo3_polars::derive::polars_expr(output_type=Float64)]
        fn $name(inputs: &[Series]) -> PolarsResult<Series> {
            let p1 = inputs[0].cast(&$p1_dt)?;
            let p1_ca = p1.$p1_acc()?;
            let name = inputs[$out_idx].name().clone();

            let ca: Float64Chunked = polars::prelude::arity::try_unary_elementwise(
                p1_ca,
                |o1| -> PolarsResult<Option<f64>> {
                    match o1 {
                        Some($p1) => {
                            $build($p1)?;
                            Ok(Some($ret))
                        },
                        None => Ok(None),
                    }
                },
            )?;

            Ok(ca.with_name(name).into_series())
        }
    };
}
pub(crate) use param_validator;
