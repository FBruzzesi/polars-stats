SHELL=/bin/bash

lint:
	cargo +nightly fmt --all 
	cargo clippy --all-features
	uvx ruff version
	uvx ruff format .
	uvx ruff check . --fix
	uvx ruff clean
	uvx rumdl check .

test:
	uv run --group testing pytest tests

typing:
	uv run --group typing pyrefly check polars_stats tests
	uv run --group typing pyright polars_stats tests
	uv run --group typing mypy polars_stats tests

install:
	uvx maturin develop

install-release:
	uvx maturin develop --release
