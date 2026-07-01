"""PiShield: a neuro-symbolic framework for learning with requirements.

PiShield integrates logical requirements (constraints) directly into the
topology of neural networks, yielding models that are guaranteed to be
compliant with the given requirements regardless of the input.

The two main entry points are:

- :func:`pishield.shield_layer.build_shield_layer` -- builds a **Shield Layer**,
  a differentiable layer that *corrects* a model's outputs so that they are
  guaranteed to satisfy the requirements. Usable at inference and training time.
- :func:`pishield.shield_loss.build_shield_loss` -- builds the **Memory-efficient
  Loss**, an additional loss term that *encourages* requirement satisfaction at
  training time via t-norms. It is a memory-efficient t-norm loss inspired by
  Logic Tensor Networks (LTN).

Three requirement types are supported: ``linear`` (linear arithmetic
inequalities), ``qflra`` (quantifier-free linear real arithmetic), and
``propositional`` (boolean logic).
"""
