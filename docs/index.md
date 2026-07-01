# PiShield

**A NeSy (neuro-symbolic) framework for learning with requirements.**

PiShield is the first framework to allow the integration of requirements
(constraints) directly into a neural network's topology. The integration is
straightforward and efficient, and produces deep learning models that are
**guaranteed to be compliant** with the given requirements — no matter the
input. Requirements can be integrated at inference time and/or training time,
depending on your needs.

## Two entry points

PiShield exposes two main building blocks:

- **[`build_shield_layer`](api/core.md#pishield.shield_layer.build_shield_layer)**
  builds a **Shield Layer**: a differentiable layer that *corrects* a model's
  outputs so they are *guaranteed* to satisfy the requirements. Use it at
  inference and/or training time.
- **[`build_shield_loss`](api/core.md#pishield.shield_loss.build_shield_loss)**
  builds the **Memory-efficient Loss**: an additional loss term that *encourages*
  (but does not guarantee) requirement satisfaction at training time, using
  t-norms. It is a memory-efficient reimplementation of Logic Tensor Networks (LTN).

## Supported requirement types

| Type | Description |
| --- | --- |
| `linear` | Linear arithmetic inequalities over the variables. |
| `qflra` | Quantifier-free linear real arithmetic (inequalities combined with boolean operators). |
| `propositional` | Boolean logic, written as Horn rules (`head :- body`) or disjunctive clauses (`y_0 or not y_1`). |

## Installation

```bash
pip install .
```

PiShield requires Python 3.8 or later and PyTorch.

## Quick start

```python
from pishield.shield_layer import build_shield_layer

# num_variables matches the dimension of the tensors to be corrected
layer = build_shield_layer(
    num_variables=5,
    requirements_filepath="requirements.txt",
)

corrected = layer(model_output)  # guaranteed to satisfy the requirements
```

## Learn more

- **[API Reference](api/index.md)** — full documentation generated from the source.
- **[GitHub repository](https://github.com/mihaela-stoian/PiShield)** — source code, examples, and runnable notebooks.
- **[Project website](https://sites.google.com/view/pishield)** — overview, demos, and performance results.
