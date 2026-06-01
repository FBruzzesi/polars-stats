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
        RowRngs {
            root_seed: resolve_root_seed(self.seed),
        }
    }
}

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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::RngCore;

    use super::*;

    fn stream(root_seed: u64, index: u64, n: usize) -> Vec<u64> {
        let mut rng = row_rng(root_seed, index);
        (0..n).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn same_seed_and_index_give_identical_stream() {
        assert_eq!(stream(42, 7, 8), stream(42, 7, 8));
    }

    #[test]
    fn distinct_index_gives_distinct_stream() {
        assert_ne!(stream(42, 7, 8), stream(42, 8, 8));
    }

    #[test]
    fn distinct_root_seed_gives_distinct_stream() {
        assert_ne!(stream(1, 7, 8), stream(2, 7, 8));
    }

    #[test]
    fn adjacent_indices_do_not_alias() {
        // The first word of each per-row stream across a contiguous index block must be
        // unique. A collision would mean two rows share a stream, i.e. the per-row
        // seeding failed to decorrelate adjacent indices. Collisions in 100k draws from
        // a 64-bit space are otherwise astronomically unlikely.
        let n: u64 = 100_000;
        let mut seen = HashSet::with_capacity(n as usize);
        for index in 0..n {
            let first = row_rng(123, index).next_u64();
            assert!(seen.insert(first), "stream collision at index {index}");
        }
    }

    #[test]
    fn resolve_root_seed_passes_through_explicit_seed() {
        assert_eq!(resolve_root_seed(Some(99)), 99);
    }

    #[test]
    fn resolve_root_seed_draws_entropy_when_absent() {
        // Two OS-entropy draws coinciding is astronomically unlikely.
        assert_ne!(resolve_root_seed(None), resolve_root_seed(None));
    }

    #[test]
    fn row_rngs_with_explicit_seed_matches_primitive() {
        // The shared kwargs path must produce exactly the same per-row stream as calling
        // the primitive directly, so samplers stay reproducible regardless of which they use.
        let rngs = SampleKwargs { seed: Some(7) }.row_rngs();
        let mut via_source = rngs.rng(3);
        let from_source: Vec<u64> = (0..8).map(|_| via_source.next_u64()).collect();
        assert_eq!(from_source, stream(7, 3, 8));
    }

    #[test]
    fn splitmix64_diffuses_adjacent_inputs() {
        // Strong avalanche is what lets neighbouring (root_seed, index) pairs seed
        // well-separated states rather than near-identical ones.
        let diff_bits = (splitmix64(0) ^ splitmix64(1)).count_ones();
        assert!(
            diff_bits >= 16,
            "weak avalanche: only {diff_bits} bits differ"
        );
    }
}
