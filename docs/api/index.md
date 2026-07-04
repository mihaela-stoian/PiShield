# API Reference

This reference is generated automatically from the docstrings in the source
code. It is organised as follows:

- **[Shield Layer & Memory-efficient Loss](core.md)** — the two top-level entry
  points, `build_shield_layer` and `build_shield_loss`. Start here.
- **[Hierarchical requirements](hierarchical.md)** — the backend for
  `hierarchical` requirements (class hierarchies), including the ARFF
  dataset loader.
- **[Propositional requirements](propositional.md)** — the backend for
  `propositional` requirements (boolean logic), including the Memory-efficient Loss.
- **[Linear requirements](linear.md)** — the backend for `linear` requirements
  (linear arithmetic inequalities).
- **[QFLRA requirements](qflra.md)** — the backend for `qflra` requirements
  (quantifier-free linear real arithmetic).


Most users only need the [top-level entry points](core.md); the per-backend
pages document the internals.
