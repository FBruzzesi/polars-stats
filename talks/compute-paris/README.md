# compute! Paris 2026 talk

`submission.md` is the accepted proposal. `talk-script.md` is the script (thesis, outline, spoken text per beat,
evidence, cut list, pre-talk TODOs): three caveats from the first commit (2026-04-26), each followed to where it
lives at 0.1.0, under the slide title *Fifty-five lines of caveats*. Dates are evidence, not structure; the word
"seam" and any count of distributions are banned. `slides.md` is the Slidev deck built from it; each slide's
presenter notes carry the script for that beat. `style.css` holds the palette
(Interstellar `#211C4F`, Oxygen `#9DF249`) over the `seriph` theme.

## Run locally

```bash
cd talks/compute-paris
npm install          # once
npm run dev          # http://localhost:3030, hot reload on slides.md
```

* Presenter view with notes and timer: <http://localhost:3030/presenter>
* All slides with notes on one page: <http://localhost:3030/overview>
* Keys: `space` / arrows step through clicks and slides, `o` overview, `g` go to slide.

## Export

```bash
npm run export                              # slides-export.pdf, one page per slide
npx slidev export slides.md --with-clicks   # one page per click step, for a handout
npm run build                               # static site in dist/
```

The PDF is the fallback for the venue machine. Export it before the talk.

## Before the talk

The pre-talk TODOs at the bottom of `talk-script.md`: re-run the Normal benchmark and `make audit` on the release you
present and refresh the two numbers slides, re-run the `explain()` pipeline, re-check the two upstream issues, confirm or soften the "no known
production user" line, and re-check the line counts on the tree-shape slide.
