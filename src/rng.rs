//! Shared per-row RNG foundation for all distribution samplers.
//!
//! Every sampler needs the same property: a deterministic, independent random stream
//! per row, derived only from `(root_seed, row_index)`. Because the seed is a function
//! of the row index (not of position within a chunk), the output is invariant to how
//! Polars chunks or threads the input.
//!
//! The generator is [`Pcg64Mcg`]: construction is a handful of integer ops (no key
//! schedule, no keystream block), it passes TestU01 BigCrush, and its output is stable
//! across `rand_pcg` releases and platforms (so seeded results stay reproducible). That
//! makes it a safe default for any distribution, including rejection/Ziggurat samplers
//! that consume an unbounded number of words per draw.
//!
//! Contrast with a one-shot hash-to-uniform: that only serves distributions needing a
//! single uniform per draw (e.g. Bernoulli) and cannot back the general case, so it is
//! deliberately not the foundation here.

use polars::prelude::*;
use rand::rngs::OsRng;
use rand::RngCore;
use rand_pcg::Pcg64Mcg;
use serde::Deserialize;

/// splitmix64 finalizer: full-avalanche mixing of a single 64-bit word.
///
/// Used to decorrelate adjacent `(root_seed, index)` pairs before they seed the
/// generator, so neighbouring rows get well-separated states rather than nearly
/// identical ones.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Resolve the root seed for a sampler call: the caller's seed if given, otherwise a
/// fresh OS-entropy draw. Called once per plugin invocation, never per row.
#[inline]
fn resolve_root_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| OsRng.next_u64())
}

/// Per-row RNG seeded by `(root_seed, index)`.
///
/// Identical inputs always yield identical streams, so a sampler built on this is
/// genuinely elementwise: chunking and thread scheduling cannot change its output.
#[inline]
fn row_rng(root_seed: u64, index: u64) -> Pcg64Mcg {
    // Fold both inputs into a 128-bit state via two splitmix64 draws. The low bit is
    // forced odd to give the MCG its full period.
    let lo = splitmix64(root_seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let hi = splitmix64(lo);
    let state = (((hi as u128) << 64) | lo as u128) | 1;
    Pcg64Mcg::new(state)
}

/// Kwargs shared by every per-row sampler plugin.
///
/// A sampler's only static input is an optional root seed: the per-row index travels as
/// a regular input `Series`, not a kwarg. So every distribution deserialises the *same*
/// shape, and they share this one struct rather than each declaring an identical copy.
///
/// Only add a field here if it is universal to all samplers. A knob specific to one
/// distribution belongs in that distribution's own kwargs type, not in this shared one.
#[derive(Deserialize)]
pub(crate) struct SampleKwargs {
    pub(crate) seed: Option<u64>,
}

impl SampleKwargs {
    /// Resolve the root seed **once** and hand back a per-row RNG source.
    ///
    /// Call this a single time per plugin invocation, outside the elementwise loop: it
    /// draws OS entropy at most once here, after which every [`RowRngs::rng`] is a few
    /// integer ops.
    #[inline]
    pub(crate) fn row_rngs(&self) -> RowRngs {
        row_rngs(self.seed)
    }
}

/// Resolve a root seed **once** and hand back a per-row RNG source.
///
/// The free-function counterpart to [`SampleKwargs::row_rngs`], for samplers whose static
/// inputs do not deserialise into a bare [`SampleKwargs`] (e.g. the scalar-parameter fast
/// paths, which carry the distribution's constant parameters alongside the seed). Both go
/// through the same resolve-once-then-derive-per-row path, so seeded output is identical
/// regardless of which entry point a distribution uses.
#[inline]
pub(crate) fn row_rngs(seed: Option<u64>) -> RowRngs {
    RowRngs {
        root_seed: resolve_root_seed(seed),
    }
}

/// Shared driver for the constant-parameter sampler fast paths.
///
/// When every distribution parameter is a Python scalar, the parameters are validated **once** by
/// the caller and passed as kwargs, so the only FFI input is the per-row index produced by
/// `pl.int_range(0, len)`. That index is dense and non-null by construction, which lets this skip
/// the per-row `Option`/validity bookkeeping the general `try_*_elementwise` paths must carry.
///
/// This factors out the boilerplate every fast path shares: cast the index to `UInt64`, resolve the
/// root seed once, then build the typed output by mapping each row index through `build`. The
/// builder owns the output element type (so float-, integer- and boolean-valued samplers all reuse
/// this), and the per-row draw seeds from `(root_seed, index)` exactly as the general path does, so
/// seeded output is identical between the two.
#[inline]
pub(crate) fn sample_by_index<T, F>(
    index: &Series,
    seed: Option<u64>,
    build: F,
) -> PolarsResult<Series>
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoSeries,
    F: FnOnce(&UInt64Chunked, &RowRngs) -> ChunkedArray<T>,
{
    let index = index.cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let rngs = row_rngs(seed);
    Ok(build(index_ca, &rngs).into_series())
}

/// Generates a distribution's constant-parameter sampler fast path: the kwargs struct
/// (the scalar parameters next to the optional root `seed`) and the `#[polars_expr]`
/// plugin that drives [`sample_by_index`].
///
/// The four things that vary between distributions are the macro's inputs:
///
/// * the kwargs fields (parameter names and types);
/// * the output dtype, as the `(logical, physical)` pair `output_type = Float64,
///   physical = Float64Type` (the two must agree; the logical name feeds
///   `#[polars_expr]`, the physical one the output `ChunkedArray`);
/// * `build`: validates the parameters and returns the per-call sampler state, built
///   **once** (`?` is available). Usually the built distribution; Uniform validates and
///   keeps the raw bounds instead;
/// * `draw`: one draw from that state, given a `&mut` per-row RNG already seeded from
///   `(root_seed, index)`.
///
/// `draw` must be the *same* draw the general per-row plugin performs, so the two paths
/// stay byte-identical for the same `(seed, index, params)`; that contract is pinned by
/// `test_sample_scalar_fast_path_matches_per_row`. Call sites are the distribution
/// modules, which all have `polars::prelude::*` in scope (the expansion relies on it).
macro_rules! sample_scalar_plugin {
    (
        $(#[$kwargs_meta:meta])*
        struct $kwargs:ident { $($param:ident: $param_ty:ty),+ $(,)? }

        $(#[$fn_meta:meta])*
        fn $fn_name:ident(output_type = $logical:ident, physical = $physical:ty);
        build = |$kw:ident| $build:expr;
        draw = |$state:pat_param, $rng:ident| $draw:expr;
    ) => {
        $(#[$kwargs_meta])*
        #[derive(serde::Deserialize)]
        struct $kwargs {
            seed: Option<u64>,
            $($param: $param_ty,)+
        }

        $(#[$fn_meta])*
        #[pyo3_polars::derive::polars_expr(output_type=$logical)]
        fn $fn_name(inputs: &[Series], kwargs: $kwargs) -> PolarsResult<Series> {
            let $kw = &kwargs;
            let $state = $build;
            let name = inputs[0].name().clone();

            $crate::rng::sample_by_index::<$physical, _>(&inputs[0], kwargs.seed, |index_ca, rngs| {
                ChunkedArray::<$physical>::from_iter_values(
                    name,
                    index_ca.into_no_null_iter().map(|i| {
                        let mut rng = rngs.rng(i);
                        let $rng = &mut rng;
                        $draw
                    }),
                )
            })
        }
    };
}
pub(crate) use sample_scalar_plugin;

/// Per-call source of per-row RNGs, all derived from one already-resolved root seed.
///
/// The resolve-once step happens when this is constructed (via [`SampleKwargs::row_rngs`]),
/// so holding it in a type makes the correct usage the only easy one: resolve per call,
/// then derive per row. Every elementwise sampler depends on that invariant, and a fresh
/// distribution gets it for free by writing `let rngs = kwargs.row_rngs();` once and
/// calling `rngs.rng(index)` inside its closure.
pub(crate) struct RowRngs {
    root_seed: u64,
}

impl RowRngs {
    /// The deterministic, independent RNG for `index`. Identical `(seed, index)` pairs
    /// always yield identical streams, so output is invariant to Polars chunking/threading.
    #[inline]
    pub(crate) fn rng(&self, index: u64) -> Pcg64Mcg {
        row_rng(self.root_seed, index)
    }
}
