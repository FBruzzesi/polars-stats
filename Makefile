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

benchmark:
	uv run --group benchmarks benchmarks/run.py

typing:
	uv run --group typing pyrefly check .
	uv run --group typing pyright .
	uv run --group typing mypy .

install:
	uvx maturin develop

install-release:
	uvx maturin develop --release
