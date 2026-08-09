SHELL=/bin/bash

.PHONY: audit lint test benchmark typing install install-release

audit:
	uv run --group audit tools/accuracy_audit.py

lint:
	uvx prek run --all-files ruff-format ruff-check rumdl ryl
	cargo +nightly fmt --all --check
	cargo clippy --all-features

test:
	POLARS_MAX_THREADS=4 uv run --group testing pytest tests

benchmark:
	uv run --group benchmarks benchmarks/run.py

typing:
	uv run --group typing pyrefly check . --min-severity info
	uv run --group typing pyright .
	uv run --group typing mypy .

install:
	uvx maturin develop

install-release:
	uvx maturin develop --release
