SHELL=/bin/bash

lint:
	uvx ruff version
	uvx ruff format .
	uvx ruff check . --fix
	uvx ruff clean
	uvx rumdl check .
	cargo +nightly fmt --all --check
	cargo clippy --all-features

test:
	POLARS_MAX_THREADS=4 uv run --group testing pytest tests

# Routine 1: CI regression guard, our crate only (tests/benchmark/). pytest-codspeed everywhere: without
# the CodSpeed runner (locally) it measures walltime and prints a timing table; the CI `bench-guard` job
# wraps the same command in the runner for deterministic instruction counts. `-o addopts=...` drops the
# project's `--cov` and the default `-m "not benchmark"` deselection; 1M rows give stable local numbers.
bench-guard:
	POLARS_STATS_BENCH_ROWS=10_000 uv run --group testing --group bench-guard pytest tests/benchmark \
		--codspeed -o "addopts=--import-mode=importlib" -o filterwarnings=default

# Routine 2: manual scipy/numpy comparison report (benchmarks/). Wall-clock + peak RSS, not gated in CI.
bench-compare:
	uv run --group bench-compare benchmarks/run.py

typing:
	uv run --group typing pyrefly check . --min-severity info
	uv run --group typing pyright .
	uv run --group typing mypy .

install:
	uvx maturin develop

install-release:
	uvx maturin develop --release
