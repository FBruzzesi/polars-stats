---
icon: lucide/presentation
---

# Talks

## Vectorized statistical distributions, built on the Polars engine

How `polars-stats` puts `scipy.stats`-style distributions inside Polars expressions, and the four contracts the plugin
enforces because neither Polars nor statrs promises them: reproducibility, broadcast, parameter validation and
accuracy. Given at compute! Paris.

The benchmark numbers and outputs on the slides are those of the release the talk was written against; the rest of
these docs track the current release.

Click the deck, then use the arrow keys to step through it, `o` for the overview and `f` for full screen.
[Open the deck in its own tab](vectorized-distributions/dist/index.html){ target="_blank" }.

<!-- rumdl-disable-next-line MD033 -->
<iframe
  src="vectorized-distributions/dist/index.html"
  title="Vectorized statistical distributions, built on the Polars engine"
  style="width: 100%; aspect-ratio: 16 / 9; border: 0;"
  allow="fullscreen"
  loading="lazy"></iframe>
