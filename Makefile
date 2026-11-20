SHELL=/bin/bash

.PHONY: audit lint test benchmark typing install install-release talks docs docs-serve

audit:
	uv run --group tools -m tools.accuracy.audit

lint:
	uvx prek run --all-files ruff-format ruff-check rumdl ryl
	cargo +nightly fmt --all --check
	cargo clippy --all-features

test:
	POLARS_MAX_THREADS=4 uv run --group testing pytest tests

benchmark:
	uv run --group tools -m tools.benchmarks.run

typing:
	uv run --group typing pyrefly check . --min-severity info
	uv run --group typing pyright .
	uv run --group typing mypy .

install:
	uvx maturin develop

install-release:
	uvx maturin develop --release

# Each docs/talks/<name>/ deck builds into its own dist/, published at talks/<name>/dist/.
# The base path repeats site_url's /polars-stats/ prefix from zensical.toml.
talks:
	for slides in docs/talks/*/slides.md; do \
		deck=$$(dirname "$$slides"); \
		(cd "$$deck" && npx -y @slidev/cli@53.0.0 build slides.md \
			--base "/polars-stats/talks/$$(basename "$$deck")/dist/" --out dist --without-notes) || exit 1; \
	done

docs: talks
	uv run --group docs zensical build --strict

docs-serve: talks
	uv run --group docs zensical serve
