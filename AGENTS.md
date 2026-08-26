# AGENTS.md

## What this is

`polars-stats` exposes `scipy.stats`-style probability distributions as native Polars expressions, with column-valued
parameters. Rust plus `statrs` does the math, Python provides the distribution classes. It is a compiled extension
(`maturin`), so Python-only changes still need a rebuild when they cross the plugin boundary.

**`docs/` is canonical for everything it covers.** This file is the map, plus the handful of things an agent gets
wrong that reading the code does not prevent.

## Read before writing code

| You need | Read |
|---|---|
| to add a distribution: the steps, the test registries, the conventions | [Contributing](./docs/contributing.md) |
| which methods go in Rust, and how to parameterise the class | [Design notes](./docs/explanation/design.md#one-rust-file-per-distribution-one-plugin-function-per-method-that-needs-rust) |
| to touch any `log_*`, `ppf` or `isf` hook | [Contributing > Numerical stability](./docs/contributing.md#numerical-stability) |
| how the system is wired | [Architecture](./docs/explanation/architecture.md) |
| the public surface and the base-class defaults | [polars_stats/distributions/_base.py](polars_stats/distributions/_base.py) |
| a pattern to copy | [_uniform.py](polars_stats/distributions/_uniform.py) (closed form), [_normal.py](polars_stats/distributions/_normal.py) (statrs-backed), [_bernoulli.py](polars_stats/distributions/_bernoulli.py) (discrete), and their `.rs` twins |

**Work from those, not from the tree.** This repository is invariant-dense: understanding the fast-path, null and
parity contracts before you start beats debugging the property suite that encodes them.

## Build, test, verify

```bash
make install  # or make install-release, required for any benchmark
make lint     # prek (ruff, rumdl, ryl) + cargo fmt (nightly) + clippy
make typing   # pyrefly + pyright + mypy, all three as in CI
make test     # POLARS_MAX_THREADS=4 pytest tests
make audit    # mpmath tail-accuracy sweep, needed for any new distribution
```

Two things `make test` alone will not catch:

* **Both query engines.** CI runs each; they chunk a plugin's inputs differently, so a chunk-boundary or
  input-length bug passes under one and fails under the other.

    ```bash
    POLARS_ENGINE_AFFINITY=in-memory uv run --group testing pytest tests
    POLARS_ENGINE_AFFINITY=streaming uv run --group testing pytest tests
    ```

* **The strict docs build**, after any `docs/` edit: `uv run --group docs zensical build --strict`.

Fast loop after a Rust change: `make install-release && uv run pytest --no-cov tests/distributions/bernoulli/ -x`.
Bernoulli exercises the whole pipeline, and `--no-cov` keeps the 95% floor from failing a deliberately partial run.
Run `prek` through `make lint`, and do not pass `--no-verify` unless asked.

## Non-negotiables

* **Row-alignment belongs to Rust.** Any new `#[polars_expr]` with more than one input calls `align_inputs` first,
  before the casts. Polars broadcasts nothing into a plugin and `try_*_elementwise` truncates to its shortest input,
  so skipping it silently drops rows. `tests/distributions/broadcast_test.py` catches a missed one.
* **Every distribution needs a row in all five shared test registries.** Miss one and that suite silently skips your
  distribution while CI stays green. It is the only failure mode here with no signal at all. The five are listed in
  [Contributing > Adding a distribution](./docs/contributing.md#adding-a-distribution), step 3.
* **Write the scipy-parity test first**, then the implementation, then iterate until it passes within a tolerance you
  can justify.
* **Override the private `_x` hook, never the public method.** The public methods add the null and `NaN` contract, and
  the composing defaults call the hooks, so a public override is bypassed by everything built on it.
* **Priorities, in order: correctness, ergonomics, maintainability, performance.** Performance is last on purpose:
  the polars engine carries large frames, so a clear formula beats a fast one. Reject a choice on performance grounds
  only when it is clearly suboptimal (an `O(n)` draw per row, a per-draw rebuild), never to shave constants.

## For agents specifically

* **Do not create planning, decision, or analysis documents** unless asked. Work from conversation context.
* **Do not extend the scope.** Found work that needs doing? Say so; do not fold it into the PR.
* **Comments never narrate code**, and never reference a task, a fix, a PR or a caller. A docstring that pins a
  contract (null propagation, seeding, fast-path bit-equality) is the house style; everything else gets no comment.
