SHELL=/bin/bash

lint:
	uvx ruff version
	uvx ruff format polars_stats tests
	uvx ruff check polars_stats tests --fix
	uvx ruff clean
	uvx rumdl check .
	cargo +nightly fmt --all --check
	cargo clippy --all-features

test:
	POLARS_MAX_THREADS=4 uv run --group testing pytest tests

benchmark:
	uv run --group benchmark pytest benchmarks -o filterwarnings=default --benchmark-only --benchmark-autosave

typing:
	uv run --group typing pyrefly check polars_stats tests
	uv run --group typing pyright polars_stats tests
	uv run --group typing mypy polars_stats tests

install:
	uvx maturin develop

install-release:
	uvx maturin develop --release
