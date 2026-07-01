<h1 style="margin-top: 0;">
  <img src="assets/logo.png" alt="PiShield" style="max-width: 420px; width: 100%;">
</h1>

**A PyTorch package for learning with requirements.**

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
  t-norms. It is a memory-efficient t-norm loss [5] inspired by
  Logic Tensor Networks (LTN) [6].

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

## Citing PiShield

If you use PiShield, please cite:

```bibtex
@inproceedings{ijcai2024p1037,
  title     = {PiShield: A PyTorch Package for Learning with Requirements},
  author    = {Stoian, Mihaela C. and Tatomir, Alex and Lukasiewicz, Thomas and Giunchiglia, Eleonora},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {8805--8809},
  year      = {2024},
  month     = {8},
  note      = {Demo Track},
  doi       = {10.24963/ijcai.2024/1037},
  url       = {https://doi.org/10.24963/ijcai.2024/1037},
}
```

Depending on which feature you use, please additionally cite: the Shield Layer with linear requirements [1], with QFLRA requirements [4], or with propositional requirements [2]; and the Memory-efficient Loss with propositional requirements [5] (in addition to LTN [6]).

## References

[1] Mihaela Catalina Stoian, Salijona Dyrmishi, Maxime Cordy, Thomas Lukasiewicz, Eleonora Giunchiglia. How Realistic Is Your Synthetic Data? Constraining Deep Generative Models for Tabular Data. arXiv:2402.04823. In Proc. of International Conference on Learning Representations (ICLR), 2024.

[2] Eleonora Giunchiglia, Alex Tatomir, Mihaela Catalina Stoian, Thomas Lukasiewicz. CCN+: A neuro-symbolic framework for deep learning with requirements. International Journal of Approximate Reasoning, 2024.

[3] Eleonora Giunchiglia and Thomas Lukasiewicz. Coherent Hierarchical Multi-Label Classification Networks. In Proceedings of Neural Information Processing Systems, 2020.

[4] Mihaela Catalina Stoian and Eleonora Giunchiglia. Beyond the Convexity Assumption: Realistic Tabular Data Generation under Quantifier-Free Real Linear Constraints. In Proc. of International Conference on Learning Representations (ICLR) 2025.

[5] Mihaela Catalina Stoian, Eleonora Giunchiglia, Thomas Lukasiewicz. Exploiting T-norms for Deep Learning in Autonomous Driving. arXiv:2402.11362. In Proc. of the International Workshop on Neural-Symbolic Learning and Reasoning (NeSy), 2023.

[6] Samy Badreddine, Artur d'Avila Garcez, Luciano Serafini, Michael Spranger. Logic Tensor Networks. arXiv:2012.13635. Artificial Intelligence, 303, 2022.
