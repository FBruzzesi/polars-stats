# ruff: noqa: INP001
"""Draw the column-regime benchmark slide from `tools.benchmarks.run --format json` output.

Usage: python bench_chart.py RESULTS_DIR [OUT_DIR]

Writes `bench-column.svg` (every cell) and `bench-column-10k.svg` (the overlay ringing the 10k-row cells
where scipy is ahead) into OUT_DIR, `public/` by default, and prints the ranges the slide's notes quote.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

MAIN = [
    ("normal", "Normal"),
    ("lognormal", "LogNormal"),
    ("cauchy", "Cauchy"),
    ("pareto", "Pareto"),
    ("weibull", "Weibull"),
    ("uniform", "Uniform"),
    ("exponential", "Exponential"),
    ("gamma", "Gamma"),
    ("beta", "Beta"),
    ("binomial", "Binomial"),
    ("geometric", "Geometric"),
    ("poisson", "Poisson"),
]
# scipy takes a slow path for these two (np.vectorize, a general binomial), so they sit apart and unquoted.
STARRED = [("bernoulli", "Bernoulli*"), ("discrete_uniform", "DiscreteUniform*")]
PANELS = [("sample", "one draw per row"), ("samples", "10 draws per row"), ("sf", "tail probability")]
ROWS = {10_000: "10k", 1_000_000: "1M", 10_000_000: "10M"}
DOTS = {
    10_000: 'r="4" fill="#f8f8ff" stroke="#a59fd6" stroke-width="2.5"',
    1_000_000: 'r="4.5" fill="#6f66b8" stroke="#f8f8ff" stroke-width="2"',
    10_000_000: 'r="4.5" fill="#201a38" stroke="#f8f8ff" stroke-width="2"',
}

WIDTH = 860
PANEL_X = [122.0, 373.3, 624.7]
DECADE = 54.33  # 0.1x to 1000x is four decades
ROW_STEP = 17
FIRST_ROW = 78.5
GAP = 12
TEN = 10
INK, MUTED, GRID, PAPER = "#201a38", "#8a87a6", "#e4e3ee", "#f8f8ff"
FONT = "system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, Menlo, monospace"

Ratios = dict[tuple[str, str, int], float]


def text(x: float, y: float, body: str, *, size: int = 12, anchor: str = "middle", fill: str = INK) -> str:
    """One SVG text element in the deck's sans font."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
        f'font-family="{FONT}">{body}</text>'
    )


def svg_open(height: float) -> str:
    """The opening tag of a chart-width SVG of the given height."""
    size = f'viewBox="0 0 {WIDTH} {height:.0f}" width="{WIDTH}" height="{height:.0f}"'
    return f'<svg xmlns="http://www.w3.org/2000/svg" {size}>'


def x_of(panel: int, ratio: float) -> float:
    """Horizontal position of a ratio on a panel's log axis, clamped to 0.1x to 1000x."""
    clamped = min(max(ratio, 0.1), 1000.0)
    return PANEL_X[panel] + (math.log10(clamped) + 1) * DECADE


def fmt(ratio: float) -> str:
    """A ratio as the slide prints it: one decimal below 10x, none above."""
    return f"{ratio:.1f}x" if ratio < TEN else f"{ratio:.0f}x"


def load(results_dir: Path) -> Ratios:
    """The scipy / polars-stats ratio of every column-regime cell, keyed by distribution, method and rows."""
    ratios: Ratios = {}
    for path in results_dir.glob("*.json"):
        report = json.loads(path.read_text())
        for cell in report["results"]:
            if cell["regime"] == "column" and cell["method"] in dict(PANELS):
                ratios[report["distribution"], cell["method"], cell["rows"]] = cell["ratios"]["scipy"]
    return ratios


def row_y() -> dict[str, float]:
    """Vertical centre of each distribution's row; the starred rows sit apart below the rest."""
    ys = {key: FIRST_ROW + i * ROW_STEP for i, (key, _) in enumerate(MAIN)}
    start = FIRST_ROW + len(MAIN) * ROW_STEP + GAP
    ys.update({key: start + i * ROW_STEP for i, (key, _) in enumerate(STARRED)})
    return ys


def header(left: float, right: float, label: str) -> list[str]:
    """A group heading over one or more panels, with its rule."""
    rule = f'<line x1="{left:.1f}" y1="20" x2="{right:.1f}" y2="20" stroke="{INK}"/>'
    return [rule, text((left + right) / 2, 14, f"<tspan font-weight='700'>{label}</tspan>", size=13)]


def panel(p: int, ratios: Ratios, ys: dict[str, float], bottom: float) -> list[str]:
    """One method's panel: title, 1M range, decade grid and the dots of every distribution."""
    method, gloss = PANELS[p]
    mid = PANEL_X[p] + 2 * DECADE
    title = f'<tspan font-family="{MONO}" font-weight="700">{method}</tspan> {gloss}'
    out = [text(mid, 36, title)]
    at_1m = [ratios[key, method, 1_000_000] for key, _ in MAIN if (key, method, 1_000_000) in ratios]
    if at_1m:
        out.append(text(mid, 51, f"1M rows: {fmt(min(at_1m))} to {fmt(max(at_1m))}", size=11, fill=MUTED))
    for d, label in enumerate(["0.1x", "1x", "10x", "100x", "1000x"]):
        x = PANEL_X[p] + d * DECADE
        stroke, width, fill = (INK, 1.5, INK) if d == 1 else (GRID, 1, MUTED)
        out.append(
            f'<line x1="{x:.1f}" y1="66" x2="{x:.1f}" y2="{bottom:.1f}" stroke="{stroke}" stroke-width="{width}"/>'
        )
        out.append(text(x, bottom + 13, label, size=10, fill=fill))
    for key, _ in MAIN + STARRED:
        out += [
            f'<circle cx="{x_of(p, ratios[key, method, rows]):.1f}" cy="{ys[key]:.1f}" {style}/>'
            for rows, style in DOTS.items()
            if (key, method, rows) in ratios
        ]
    return out


def chart(ratios: Ratios) -> str:
    """Every cell, one panel per method."""
    ys = row_y()
    bottom = max(ys.values()) + 12
    height = bottom + 44
    out = [svg_open(height), f'<rect width="{WIDTH}" height="{height:.0f}" fill="{PAPER}"/>']
    out += header(PANEL_X[0], PANEL_X[1] + 4 * DECADE, "simulate")
    out += header(PANEL_X[2], PANEL_X[2] + 4 * DECADE, "score")
    for p in range(len(PANELS)):
        out += panel(p, ratios, ys, bottom)
    out += [
        text(112, ys[key] + 4, label, anchor="end", fill=MUTED if key in dict(STARRED) else INK)
        for key, label in MAIN + STARRED
    ]
    footer = bottom + 32
    note = "ratio = scipy time / polars-stats time, log scale  ·  right of 1x: polars-stats faster"
    out += [
        text(PANEL_X[0], footer, note, size=11, anchor="start", fill=MUTED),
        text(642, footer, "rows:", size=11, anchor="end", fill=MUTED),
    ]
    for x, (rows, label) in zip((660, 712, 764), ROWS.items(), strict=True):
        out += [
            f'<circle cx="{x}" cy="{footer - 4:.1f}" {DOTS[rows]}/>',
            text(x + 10, footer, label, size=11, anchor="start"),
        ]
    return "\n".join([*out, "</svg>"]) + "\n"


def overlay(ratios: Ratios) -> str:
    """The 10k-row reading: polars-stats' side dimmed, and a ring round every 10k cell scipy wins."""
    ys = row_y()
    bottom = max(ys.values()) + 12
    height = bottom + 44
    out = [svg_open(height)]
    for p, (method, _) in enumerate(PANELS):
        one = PANEL_X[p] + DECADE
        dim = f'<rect x="{one + 1:.1f}" y="64" width="{3 * DECADE - 1:.1f}" height="{bottom - 64:.1f}"'
        out += [
            f'{dim} fill="{PAPER}" fill-opacity="0.72"/>',
            text(one - 6, 62, "<tspan font-weight='700'>scipy ahead</tspan>", size=10, anchor="end"),
        ]
        for key, _ in MAIN + STARRED:
            ratio = ratios.get((key, method, 10_000))
            if ratio is not None and ratio < 1:
                ring = f'cx="{x_of(p, ratio):.1f}" cy="{ys[key]:.1f}" r="7.5"'
                out.append(f'<circle {ring} fill="none" stroke="{INK}" stroke-width="2"/>')
    return "\n".join([*out, "</svg>"]) + "\n"


def summary(ratios: Ratios) -> str:
    """The ranges the notes quote: per method and row count, and every cell scipy wins."""
    lines = []
    for method, _ in PANELS:
        for rows, label in ROWS.items():
            main = [ratios[key, method, rows] for key, _ in MAIN if (key, method, rows) in ratios]
            if not main:
                continue
            ahead = sorted(
                f"{key} {1 / ratio:.1f}x"
                for key, _ in MAIN + STARRED
                if (ratio := ratios.get((key, method, rows), 1.0)) < 1
            )
            span = f"{fmt(min(main))} to {fmt(max(main))}"
            lines.append(f"{method:8} {label:>4}: {span}; scipy ahead: {', '.join(ahead) or 'none'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Read the results, write both SVGs, print the summary."""
    results_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "public"  # noqa: PLR2004
    ratios = load(results_dir)
    if not ratios:
        msg = f"no column-regime sample / samples / sf cells in {results_dir}"
        raise SystemExit(msg)
    (out_dir / "bench-column.svg").write_text(chart(ratios))
    (out_dir / "bench-column-10k.svg").write_text(overlay(ratios))
    sys.stdout.write(summary(ratios))


if __name__ == "__main__":
    main()
