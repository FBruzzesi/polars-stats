"""The Zen of polars-stats, printed on `import polars_stats.this`.

In memory of Tim Peters, whose Zen of Python gave this its form.
"""

from __future__ import annotations

ZEN = """The Zen of polars-stats

Correct is better than fast.
Clear is better than clever.
Every parameter is a column. A scalar is just a short one.
Null in, null out.
Rows should never vanish silently.
Same seed, same draws.
In the face of a tiny tail, refuse the temptation to subtract.
Log space is a safe space.
scipy is the reference, not the referee.
Rust does the math. Python does the talking.
There should be one, and preferably only one, name for λ: it is `rate`.
Expressions are one honking great idea: let's compose more of those!"""

print(ZEN)  # noqa: T201
