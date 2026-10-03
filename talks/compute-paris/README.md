# compute! Paris 2026 deck

Slides for "Vectorized statistical distributions, built on the Polars engine" (25 minutes plus 5 of Q&A), built with
[Slidev](https://sli.dev): a tool that turns one Markdown file into a slide deck that runs in the browser.

| File | What it is |
|---|---|
| `slides.md` | The deck. Slides are separated by `---`; each slide's presenter notes sit in the `<!-- ... -->` block at its end |
| `style.css` | The whole theme. The deck uses `theme: none`, so every style, including layout padding, lives here |
| `public/` | The static SVGs (bell curves, benchmark dot plot and its 10k-row overlay), served from `/` |
| `talk-script.md` | The canonical script. The notes are its Say text, verbatim |

## Prerequisites

* **Node.js 22.12 or newer** (Slidev 53 requires it). Check with `node --version`. npm comes with Node.
* Nothing to install in this folder. Every command below runs Slidev through `npx`, which downloads it into npm's
  cache on first use (that first run takes a minute and needs a network connection).
* There is deliberately no `package.json`: a global gitignore pattern for `**/*.json` would silently drop it from git. If
  you ever add one, track it with `git add -f package.json`.

Tested with Node 26.10.0, npm 11.19.1 and Slidev 53.0.0.

## Run the deck

From this folder:

```bash
npx @slidev/cli@latest slides.md
```

Open <http://localhost:3030>. Keys:

| Key | Action |
|---|---|
| Space or right arrow | next click (or next slide when the slide has no clicks left) |
| Left arrow | previous click |
| Down / up arrow | next / previous slide, skipping clicks |
| `o` | overview of every slide |
| `g` | go to a slide by number |

The server reloads the browser whenever `slides.md` or `style.css` is saved.

## Presenter mode

Open <http://localhost:3030/presenter> on the laptop and <http://localhost:3030> on the projector (drag that window to
the second screen and make it full screen). The two stay in sync: presenter mode shows the current slide, the next one,
a timer and the notes.

The notes are the script's Say text, split so each slide carries what is said while it is on screen. Inside a slide,
`[click]` marks where to press the next key, and presenter mode highlights the part of the notes for the current click.
`[pause]` means stop talking and let the room think. The cover and the first section slide have a short
"(Not spoken ...)" line instead, because nothing is said over them.

## Build a static copy

```bash
npx @slidev/cli@latest build slides.md
```

This writes a self-contained single-page site to `dist/` (gitignored). Host it anywhere that serves static files.

## Export a PDF (the backup for the venue)

Export drives a headless Chromium, so it needs `playwright-chromium` next to Slidev:

```bash
npx -p @slidev/cli@latest -p playwright-chromium slidev export slides.md
```

This writes `slides-export.pdf` (gitignored), one page per slide with every click revealed. Add `--with-clicks` for
one page per click step instead. If the command reports a missing browser, run `npx playwright install chromium` once.

Export a fresh PDF before travelling and keep it on a USB stick: it is what you present from if the laptop, the adapter
or the browser lets you down.

## Editing notes

* Library names are plain text (Polars, statrs, scipy, polars-stats); code is in backticks; sources and dates use the
  `footnote` class.
* Every number on a slide comes from `talk-script.md`. The simulated column of the sensor frame is still a
  `<TODO>` until a recorded run fills it in.
* The SVGs in `public/` are static: to change a benchmark number, edit the SVG. Then restart the dev server, because
  Vite inlines small images into the compiled slide and keeps serving the old copy.
* Layout lives in `style.css`: content slides keep the title at the top, centre the body in the space below it, and pin
  a final `footnote` to the bottom. Text sizes come from the `--fs-*` variables at the top of the deck section.
