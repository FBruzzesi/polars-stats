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
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::reshape::ReshapeDimension;
use polars_core::utils::rayon::prelude::*;
use polars_core::POOL;
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

/// Total draw count below which the multi-draw row fill runs serially: a fork-join dispatch
/// costs more than drawing this few values, and elementwise plugins run once per
/// `group_by` / `over` partition, so tiny calls are common.
const PARALLEL_FILL_MIN_DRAWS: usize = 4096;

/// Fill the row-major multi-draw buffer: `fill_row(i, slot)` writes row `i`'s `size` draws into
/// its own disjoint `size`-wide slice.
///
/// Rows fill in parallel on the polars thread pool when the total work justifies it. That is
/// deterministic because a row's draws depend only on `(root_seed, row_index)`, never on other
/// rows or on visit order, so the parallel fill is bit-identical to the serial one.
fn fill_rows<V, F>(flat: &mut [V], size: usize, fill_row: F)
where
    V: Send,
    F: Fn(usize, &mut [V]) + Sync,
{
    if flat.len() >= PARALLEL_FILL_MIN_DRAWS {
        POOL.install(|| {
            flat.par_chunks_mut(size)
                .enumerate()
                .for_each(|(row, slot)| fill_row(row, slot));
        });
    } else {
        for (row, slot) in flat.chunks_mut(size).enumerate() {
            fill_row(row, slot);
        }
    }
}

/// Shared driver for the constant-parameter multi-draw (`samples`) fast paths.
///
/// The multi-draw counterpart of [`sample_by_index`]: one plugin call returns the
/// `Array(width=size)` column directly, instead of `size` `sample` plugin calls glued by
/// `concat_arr` paying the fixed per-call expression and FFI cost `size` times.
///
/// Row `i`'s `size` draws are **consecutive values from one per-row stream** seeded
/// `(root_seed, i)`, the same stream `sample` takes its single draw from. So `samples(size=1)`
/// is bit-identical to `sample` for the same seed, and growing `size` extends each row's array
/// without changing the existing draws. The root seed resolves once per call (fresh OS entropy
/// when `None`); a row's stream depends only on `(root_seed, i)`, never on other rows, so
/// chunk/thread invariance is untouched and rows can fill in parallel (see [`fill_rows`]).
/// `draw` is one draw from a `&mut` per-row RNG already seeded `(root_seed, i)`, the same draw
/// the distribution's `sample` plugin performs.
#[inline]
pub(crate) fn samples_by_index<T, V, F>(
    name: PlSmallStr,
    index: &Series,
    seed: Option<u64>,
    size: usize,
    draw: F,
) -> PolarsResult<Series>
where
    T: PolarsDataType,
    ChunkedArray<T>: NewChunkedArray<T, V> + IntoSeries,
    V: Default + Clone + Send,
    F: Fn(&mut Pcg64Mcg) -> V + Sync,
{
    // The Python layer rejects `size <= 0` before registering the plugin; this guards the
    // zero-width `Array` reshape against any other caller.
    if size == 0 {
        return Err(PolarsError::InvalidOperation(
            "samples requires a positive size".into(),
        ));
    }
    let index = index.cast(&DataType::UInt64)?;
    let indices: Vec<u64> = index.u64()?.into_no_null_iter().collect();
    let rngs = row_rngs(seed);

    let mut flat = vec![V::default(); indices.len() * size];
    fill_rows(&mut flat, size, |row, slot| {
        let mut rng = rngs.rng(indices[row]);
        for value in slot {
            *value = draw(&mut rng);
        }
    });

    ChunkedArray::<T>::from_iter_values(name, flat.into_iter())
        .into_series()
        .reshape_array(&[ReshapeDimension::Infer, ReshapeDimension::new(size as i64)])
}

/// Shared driver for the column-parameter multi-draw (`samples`) per-row paths.
///
/// The column-parameter counterpart of [`samples_by_index`]. The caller iterates its parameter
/// columns and yields, per row, the index and a ready-to-draw state (typically the built
/// distribution, so it is constructed once per row rather than once per draw). A `None` row (any
/// null input) becomes a null `Array` element directly, with its inner slots also null (the same
/// two-layer shape as `pl.lit(None, dtype=Array(...))`); an invalid parameterisation `?`-raises
/// out of the row iterator.
///
/// Seeding is identical to [`samples_by_index`]: row `i`'s draws are consecutive values from one
/// stream keyed `(root_seed, i)`, a function of position only, never of the parameters. So the
/// scalar and column paths stay bit-identical for the same parameters, equal-parameter rows
/// still draw independently, and rows can fill in parallel (see [`fill_rows`]). Note the
/// per-draw *consumption* of that stream does depend on the row's parameters (rejection
/// samplers draw a variable number of words), which is fine: the stream is private to the row.
#[inline]
pub(crate) fn samples_per_row<T, V, S, I, F>(
    name: PlSmallStr,
    seed: Option<u64>,
    size: usize,
    rows: I,
    draw: F,
) -> PolarsResult<Series>
where
    T: PolarsDataType,
    ChunkedArray<T>: NewChunkedArray<T, V> + IntoSeries,
    V: Default + Clone + Send,
    S: Sync,
    I: Iterator<Item = PolarsResult<Option<(u64, S)>>>,
    F: Fn(&S, &mut Pcg64Mcg) -> V + Sync,
{
    if size == 0 {
        return Err(PolarsError::InvalidOperation(
            "samples requires a positive size".into(),
        ));
    }
    // Materialising the states up front separates the sequential part (parameter validation,
    // which can raise) from the draw loop, whose rows then fill independently. A null row keeps
    // its `V::default()` slice; it is masked by the validity bitmaps below, never read.
    let states: Vec<Option<(u64, S)>> = rows.collect::<PolarsResult<_>>()?;
    let rngs = row_rngs(seed);

    let mut flat = vec![V::default(); states.len() * size];
    fill_rows(&mut flat, size, |row, slot| {
        if let Some((index, state)) = &states[row] {
            let mut rng = rngs.rng(*index);
            for value in slot {
                *value = draw(state, &mut rng);
            }
        }
    });

    let flat = ChunkedArray::<T>::from_iter_values(name.clone(), flat.into_iter()).into_series();
    let shape = [ReshapeDimension::Infer, ReshapeDimension::new(size as i64)];

    let outer_validity: Bitmap = states.iter().map(Option::is_some).collect();
    if outer_validity.unset_bits() == 0 {
        return flat.reshape_array(&shape);
    }
    // A null row nulls both layers, matching `pl.lit(None, dtype=Array(...))`: the outer bit makes
    // the element null, and the inner bits keep value-level reads (`arr.first`, `explode`, ...)
    // from leaking the placeholder defaults behind it.
    let inner_validity: Bitmap = states
        .iter()
        .flat_map(|state| std::iter::repeat_n(state.is_some(), size))
        .collect();
    let inner = flat.rechunk().chunks()[0].with_validity(Some(inner_validity));
    let out = Series::from_arrow(name.clone(), inner)?.reshape_array(&shape)?;
    let masked = out.rechunk().chunks()[0].with_validity(Some(outer_validity));
    Series::from_arrow(name, masked)
}

/// Build the per-row state iterator a single-parameter column `samples` plugin feeds to
/// [`samples_per_row`]: zip the parameter column with the row index and, on a fully-non-null row,
/// run `build` once to make the row's draw state.
///
/// Centralises the null contract every per-row multi-draw plugin shares (any null input nulls the
/// whole row, matching the single-draw `try_*_elementwise` paths), so each distribution spells only
/// its cast and its `build`.
pub(crate) fn binary_param_rows<'a, A, S, F>(
    param: &'a ChunkedArray<A>,
    index: &'a UInt64Chunked,
    build: F,
) -> impl Iterator<Item = PolarsResult<Option<(u64, S)>>> + 'a
where
    A: PolarsNumericType,
    F: Fn(A::Native) -> PolarsResult<S> + 'a,
    S: 'a,
{
    param
        .iter()
        .zip(index.iter())
        .map(move |(p_opt, i_opt)| match (p_opt, i_opt) {
            (Some(p), Some(i)) => Ok(Some((i, build(p)?))),
            _ => Ok(None),
        })
}

/// Two-parameter counterpart of [`binary_param_rows`] (e.g. `(mean, std_dev)`, `(n, p)`): zip both
/// parameter columns with the row index and, on a fully-non-null row, run `build` once. The two
/// parameter dtypes are independent, so a mixed `(i64, f64)` parameterisation (Binomial) fits.
pub(crate) fn ternary_param_rows<'a, A, B, S, F>(
    a: &'a ChunkedArray<A>,
    b: &'a ChunkedArray<B>,
    index: &'a UInt64Chunked,
    build: F,
) -> impl Iterator<Item = PolarsResult<Option<(u64, S)>>> + 'a
where
    A: PolarsNumericType,
    B: PolarsNumericType,
    F: Fn(A::Native, B::Native) -> PolarsResult<S> + 'a,
    S: 'a,
{
    a.iter()
        .zip(b.iter())
        .zip(index.iter())
        .map(move |((a_opt, b_opt), i_opt)| match (a_opt, b_opt, i_opt) {
            (Some(a), Some(b), Some(i)) => Ok(Some((i, build(a, b)?))),
            _ => Ok(None),
        })
}

/// Kwargs shared by every column-parameter multi-draw plugin, and the slice the shared
/// output-dtype functions read from the scalar variants' kwargs (serde skips their extra
/// parameter fields): the optional root seed and the draw count, which is the output `Array`
/// width.
#[derive(Deserialize)]
pub(crate) struct SamplesKwargs {
    pub(crate) seed: Option<u64>,
    pub(crate) size: usize,
}

fn samples_output(fields: &[Field], width: usize, inner: DataType) -> PolarsResult<Field> {
    Ok(Field::new(
        fields[0].name().clone(),
        DataType::Array(Box::new(inner), width),
    ))
}

/// Output dtype of a float-valued multi-draw plugin: `Array(Float64, size)`.
pub(crate) fn samples_f64_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::Float64)
}

/// Output dtype of an integer-valued multi-draw plugin: `Array(UInt64, size)`.
pub(crate) fn samples_u64_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::UInt64)
}

/// Output dtype of a boolean-valued multi-draw plugin: `Array(Boolean, size)`.
pub(crate) fn samples_bool_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::Boolean)
}

/// Generates a distribution's constant-parameter fast paths: both the single-draw `sample`
/// plugin (driving [`sample_by_index`]) and the multi-draw `samples` plugin (driving
/// [`samples_by_index`]), from one shared `build` / `draw`.
///
/// Emitting the two from one definition is what keeps them from drifting: the multi-draw fast
/// path is just the single-draw one repeated `size` times on the same per-row stream, so they
/// must share the exact `build` and `draw`. The five things that vary between distributions are
/// the macro's inputs:
///
/// * the kwargs fields (parameter names and types), reused by both kwargs structs;
/// * the single-draw output dtype, as the `(logical, physical)` pair `output_type = Float64,
///   physical = Float64Type` (the two must agree; the logical name feeds `#[polars_expr]`, the
///   physical one the output `ChunkedArray`);
/// * the multi-draw twin, named by the `samples = <fn> as <kwargs> -> <output_fn>` clause: the
///   plugin fn, its kwargs struct (the same parameters plus `size`), and the `Array`-typed
///   output function (`samples_f64_output` / `_u64_` / `_bool_`);
/// * `build`: validates the parameters and returns the per-call sampler state, built **once**
///   (`?` is available). Usually the built distribution; Uniform validates and keeps the raw
///   bounds instead;
/// * `draw`: one draw from that state, given a `&mut` per-row RNG already seeded from
///   `(root_seed, index)`.
///
/// `draw` must be the *same* draw the general per-row plugins perform, so the fast paths stay
/// byte-identical to them for the same `(seed, index, params)`; pinned by
/// `test_sample_scalar_fast_path_matches_per_row` and `test_samples_scalar_fast_path_matches_per_row`.
/// Call sites are the distribution modules, which all have `polars::prelude::*` in scope and
/// import the `samples_*_output` function they name (the expansion relies on both).
macro_rules! sample_scalar_plugin {
    (
        $(#[$kwargs_meta:meta])*
        struct $kwargs:ident { $($param:ident: $param_ty:ty),+ $(,)? }

        $(#[$fn_meta:meta])*
        fn $fn_name:ident(output_type = $logical:ident, physical = $physical:ty);

        samples = $samples_fn:ident as $samples_kwargs:ident -> $samples_output:ident;

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

        /// Constant-parameter kwargs for the multi-draw fast path: the single-draw parameters
        /// plus the draw count `size` (the output `Array` width).
        #[derive(serde::Deserialize)]
        struct $samples_kwargs {
            seed: Option<u64>,
            size: usize,
            $($param: $param_ty,)+
        }

        #[doc = concat!(
            "Constant-parameter multi-draw fast path: the `samples` twin of [`", stringify!($fn_name),
            "`]. `size` draws per row in one call, taken as consecutive values from the same \
             `(seed, row_index)` per-row stream, so `samples(size=1)` matches `sample` bit for bit \
             and the state is built once per call rather than once per draw. Returns \
             `Array(inner, size)`."
        )]
        #[pyo3_polars::derive::polars_expr(output_type_func_with_kwargs=$samples_output)]
        fn $samples_fn(inputs: &[Series], kwargs: $samples_kwargs) -> PolarsResult<Series> {
            let $kw = &kwargs;
            let $state = $build;
            let name = inputs[0].name().clone();

            $crate::rng::samples_by_index::<$physical, _, _>(
                name,
                &inputs[0],
                kwargs.seed,
                kwargs.size,
                |$rng| $draw,
            )
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
