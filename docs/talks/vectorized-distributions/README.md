# Vectorized statistical distributions, built on the Polars engine

The slides for a 25-minute talk (plus 5 of Q&A) about `polars-stats`: how it puts `scipy.stats`-style distributions
inside Polars expressions, and the four contracts the plugin has to enforce because neither Polars nor statrs promises
them: reproducibility, broadcast, parameter validation and accuracy.

The deck is built with [Slidev](https://sli.dev), which turns one Markdown file into a slide deck that runs in the
browser.

One deck serves every event. The event's name is on the cover and in the footer, and the `EVENT` variable picks it:

| Event | `EVENT` | Shown as |
|---|---|---|
| PyData Milan meetup | `pydata-milan` | PyData Milan · 19 November 2026 |
| compute! Paris | `compute-paris` (the default) | compute! Paris 2026 |

The labels live in `vite.config.mjs`; add a row there for a new event. An unknown `EVENT` stops Slidev with the list of
valid keys.

| File | What it is |
|---|---|
| `slides.md` | The deck. Slides are separated by `---`; each slide's presenter notes, the spoken script, sit in its `<!-- ... -->` block |
| `talk-script.md` | The talk's outline: the question chain, the takeaways, the cut list and a source for every number on a slide |
| `style.css` | The whole theme. The deck uses `theme: none`, so every style lives here |
| `layouts/default.vue` | The one layout: eyebrow over the title (the event's name on the cover), footer, page number and progress bar |
| `components/` | `Frame` (the running sensor table) and `Stack` (Polars, polars-stats, statrs and the four contracts) |
| `setup/shiki.ts` | Keeps code windows dark on every slide |
| `vite.config.mjs` | The event labels, and lets Slidev's dev server serve KaTeX's fonts when Slidev runs through `npx` |
| `public/` | Images, QR codes, the benchmark charts and the bundled Inter and JetBrains Mono fonts (SIL OFL), served from `/` |
| `bench_chart.py` | Draws the benchmark charts from the benchmark harness's JSON output (see below) |

## Run the deck

You need Node.js 22.12 or newer (`node --version`). Nothing is installed in this folder: `npx` fetches Slidev on first
use, which takes a minute and a network connection. From this folder:

```bash
npx @slidev/cli@latest slides.md                      # compute! Paris
EVENT=pydata-milan npx @slidev/cli@latest slides.md   # PyData Milan
```

Open <http://localhost:3030>. Space or the right arrow advances one click, the left arrow goes back, the down and up
arrows skip whole slides, `o` shows every slide and `g` jumps to a slide by number. Saving `slides.md` or `style.css`
reloads the browser.

For presenting, open <http://localhost:3030/presenter> on the laptop and <http://localhost:3030> on the projector. The
presenter view shows the current and next slide, a timer and the notes. In the notes, `[click]` marks where to press the
next key and `[pause]` means stop and let the room think. The cover's notes carry the timing marks and the cut list.

Tested with Node 26.10.0, npm 11.19.1 and Slidev 53.0.0.

## Build or export

```bash
npx @slidev/cli@latest build slides.md  # a static site in dist/, keeps every click and transition
npx -p @slidev/cli@latest -p playwright-chromium slidev export slides.md  # slides-export.pdf
```

Prefix either with `EVENT=pydata-milan` for the Milan version; `--output pydata-milan` on the export writes
`pydata-milan.pdf`, so both PDFs can sit side by side.

The static site can be hosted anywhere that serves files; add `--base /some/path/` when it lives under a sub-path. The
PDF has one page per slide with every click revealed; add `--with-clicks` for one page per click. If the export reports
a missing browser, run `npx playwright install chromium` once. `dist/` and PDFs are gitignored.

## Regenerate the benchmark charts

The benchmark slides come from the repository's own harness. On a quiet machine, from the repository root:

```bash
uv run --no-sync python -m tools.benchmarks.run --methods sample samples sf --regimes column \
  --rows 10_000 1_000_000 10_000_000 --max-seconds 10 --format json \
  --output-dir docs/talks/vectorized-distributions/bench-results-after
uv run --no-project python docs/talks/vectorized-distributions/bench_chart.py \
  docs/talks/vectorized-distributions/bench-results-after docs/talks/vectorized-distributions/public/after
```

The second command rewrites `public/after/bench-column.svg` and its 10k-row overlay, and prints the ranges the
benchmark slide quotes. Update the disclaimer slide's versions and date to match, then restart the dev server: Vite
inlines small images and keeps serving the old copy otherwise.

The benchmark slide and the appendix both show this run (polars-stats 0.2.0). The appendix keeps the Nim comparison
for the Q&A, after the closing slide.

## Editing notes

* Library names are plain text (Polars, statrs, SciPy, polars-stats); code is in backticks, also in titles.
* Every output on a slide was re-run against the library; `talk-script.md` lists where each number comes from. The
  sensor table's values live in `components/Frame.vue`.
* Layout lives in `style.css`: the title sits under its eyebrow, a final `footnote` pins to the bottom, and colours come
  from the `:root` tokens.
