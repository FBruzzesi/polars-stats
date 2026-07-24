---
icon: lucide/graduation-cap
---

# Explanation

Understanding-oriented pages: how the plugin is put together and why it is put together that way. Nothing here is
needed to *use* the library; read these to contribute, to evaluate the design, or out of curiosity.

* [Architecture](architecture.md): what the system is and how it is wired, from the Python distribution classes down
    to the Rust plugins: the two-layer split, plugin granularity, the constant-parameter fast paths, and sampling.
* [Design notes](design.md): the *why* behind those choices and the questions still open, one decision per section,
    with the trade-offs spelled out.

If instead you want to build, test, or extend the project, the practical entry point is [Contributing](../contributing.md).
