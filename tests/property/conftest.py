from __future__ import annotations

from hypothesis import HealthCheck, settings

# A single capped profile keeps the whole property suite well under the 60s CI budget, while staying high enough to
# exercise the parameter space. `deadline=None` removes per-example timing assertions: a plugin call's first invocation
# pays a one-off cost that would otherwise flake the deadline.
# `function_scoped_fixture` is suppressed because every property combines `@pytest.mark.parametrize("spec", ...)`
# (a function-scoped argument) with `@given`, which is intentional and not the bug that health check guards against.
settings.register_profile(
    "polars_stats",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("polars_stats")
