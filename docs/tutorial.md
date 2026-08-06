---
icon: lucide/graduation-cap
---

# Tutorial: score telemetry against per-row baselines

In this tutorial you build a working anomaly detector: one lazy Polars query that scores every sensor reading against
the baseline distribution of *its own sensor*, ranks the outliers even when their probabilities underflow to zero, and
turns an alarm rate into a physical reading limit.

Along the way you use the four methods that carry most statistical work (`pdf`, `sf`, `log_sf`, `isf`) and parameters
that are columns rather than scalars.

You need Python `>=3.10` and:

```bash
pip install polars-stats
```

Every snippet below is executed when this page is built, and each one continues from the previous. Follow them in
order and you will see exactly the output shown.

## Step 1: get some readings

Two temperature sensors report on very different scales. Copy this into a Python session:

```python exec="yes" source="above" session="tutorial" result="python"
import polars as pl
import polars_stats as ps

readings = pl.DataFrame(
    {
        "sensor": ["temp-a", "temp-a", "temp-a", "temp-b", "temp-b", "temp-b"],
        "reading": [9.8, 10.4, 40.0, 99.4, 101.2, 250.0],
    }
)
print(readings)
```

Two of those six readings are faults: `40.0` on `temp-a` and `250.0` on `temp-b`. The task is to find them without
hard-coding either number.

## Step 2: score against a single baseline

Build a Normal distribution and ask, for every reading, how much probability lies above it. That upper-tail
probability is the survival function, `sf`:

```python exec="yes" source="above" session="tutorial" result="python"
shared = ps.Normal(mu=10.0, sigma=0.5)

print(readings.with_columns(upper_tail=shared.sf("reading")))
```

`sf("reading")` reads the column named `reading`. A bare string is always a column reference.

Sensor `temp-a` is scored correctly, but every `temp-b` reading looks impossible, including the healthy ones, because
they were scored against `temp-a`'s baseline. One distribution cannot serve both sensors.

## Step 3: give every row its own baseline

Put each sensor's baseline in its own frame and join it on:

```python exec="yes" source="above" session="tutorial" result="python"
baselines = pl.DataFrame(
    {
        "sensor": ["temp-a", "temp-b"],
        "mu": [10.0, 100.0],
        "sigma": [0.5, 2.0],
    }
)

joined = readings.join(baselines, on="sensor", maintain_order="left")
print(joined)
```

Now pass those columns as the distribution's parameters:

```python exec="yes" source="above" session="tutorial" result="python"
baseline = ps.Normal(mu="mu", sigma="sigma")

scored = joined.with_columns(upper_tail=baseline.sf("reading"))
print(scored)
```

That single `baseline` object described a different `Normal` on every row: `Normal(10.0, 0.5)` on the `temp-a` rows and
`Normal(100.0, 2.0)` on the `temp-b` rows. All four healthy readings now score between `0.2` and `0.7`, and only the two
faults sit near zero.

## Step 4: flag the faults

An upper tail below `0.001` means "less than a 1-in-1000 reading for this sensor". Filter on it:

```python exec="yes" source="above" session="tutorial" result="python"
alarm_rate = 0.001

print(scored.filter(pl.col("upper_tail") < alarm_rate))
```

Two rows, exactly the two faults, and no threshold in reading units anywhere in the code.

## Step 5: rank faults whose probability underflows

Both faults report `upper_tail` of exactly `0.0`. Their true probabilities are around `10^-784` and `10^-1224`, far
below the smallest number a `Float64` can hold, so they collapse to zero and become indistinguishable. Ask for the
logarithm instead:

```python exec="yes" source="above" session="tutorial" result="python"
ranked = scored.with_columns(log_tail=baseline.log_sf("reading")).sort("log_tail")
print(ranked)
```

`log_sf` is computed in log space throughout, so it stays finite where `sf` has already rounded to zero. The `temp-b`
fault at `-2817` is now clearly more extreme than the `temp-a` one at `-1805`, and sorting on it puts the worst fault
first. Use `log_sf` whenever you need to rank deep tails; use `sf` when you need a readable probability.

## Step 6: turn the alarm rate into a reading limit

Operators want a number in degrees, not a probability. `isf` inverts the survival function: give it a tail
probability, and it returns the reading value that leaves exactly that much probability above it.

```python exec="yes" source="above" session="tutorial" result="python"
print(baselines.with_columns(limit=ps.Normal(mu="mu", sigma="sigma").isf(alarm_rate)))
```

One 1-in-1000 limit per sensor, derived from that sensor's own baseline: `11.5` for `temp-a`, `106.2` for `temp-b`.
Change `alarm_rate` and both limits move with it.

## Step 7: assemble the detector

Nothing above needed a materialised `DataFrame`. Every method returned a `pl.Expr`, so the whole detector composes
into one lazy query that Polars optimises and evaluates in a single pass:

```python exec="yes" source="above" session="tutorial" result="python"
baseline = ps.Normal(mu="mu", sigma="sigma")

alerts = (
    readings.lazy()
    .join(baselines.lazy(), on="sensor", maintain_order="left")
    .with_columns(
        upper_tail=baseline.sf("reading"),
        log_tail=baseline.log_sf("reading"),
        limit=baseline.isf(alarm_rate),
    )
    .filter(pl.col("upper_tail") < alarm_rate)
    .select("sensor", "reading", "limit", "log_tail")
    .sort("log_tail")
    .collect()
)
print(alerts)
```

Two faults, each with the limit it broke and a severity score that ranks them, from a query that never left the Polars
engine.

## What you built

* A distribution parameterised from columns, so one object scores every row against its own baseline.
* Tail probabilities with `sf`, tail *ranking* with `log_sf`, and a reading-unit threshold with `isf`.
* A single lazy query, with the statistics inline rather than in a post-processing step.

## Where to next

* Solve a specific problem: the [How-to guides](how-to/index.md) cover deriving parameters from data, sampling,
    nulls, and porting `scipy.stats` code.
* Look something up: the [API reference](reference/index.md) has the distribution catalogue and the full method
    surface.
* Understand the design: [Why polars-stats](explanation/index.md) explains what this approach buys and what it costs.
