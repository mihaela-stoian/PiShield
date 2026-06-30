"""Linear requirements subpackage of PiShield.

This subpackage implements the Shield Layer for *linear* requirements: linear
arithmetic inequality constraints over a set of variables (labels or features).
It provides the constraint data model, a parser for requirements files, the
machinery that precomputes per-variable bounds from the requirements, and the
differentiable :class:`~pishield.linear_requirements.shield_layer.ShieldLayer`
that corrects predictions so they satisfy all requirements.
"""
