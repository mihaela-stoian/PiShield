"""The Shield Layer for hierarchical requirements.

Defines :class:`ShieldLayer`, a differentiable PyTorch layer that corrects a
model's predictions so they provably satisfy a set of hierarchical (subsumption)
requirements: whenever a class is predicted, all of its ancestors must be
predicted too. This is the Max Constraint Module (MCM) of C-HMCNN [3]: the
corrected score of a class is the maximum of the predicted scores over the class
and all of its descendants, computed in closed form from the descendants matrix
``R``. Because the correction is a single ``max`` per class (rather than the
stratified clause-by-clause correction of the propositional backend), it needs no
variable ordering.
"""

from typing import List

import torch
from torch import nn

from pishield.hierarchical_requirements.parser import parse_hierarchy_file


def get_constr_out(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply the MCM correction: each class score becomes its subtree max.

    Faithful port of C-HMCNN's ``get_constr_out``. For every class ``i`` the
    corrected score is ``max_j R[i, j] * x[j]``, i.e. the maximum predicted score
    over ``i`` and its descendants. Since a descendant's corrected score can only
    raise its ancestors, the output satisfies ``score(child) <= score(parent)``
    for every edge, so thresholding at any level yields a hierarchically coherent
    prediction.

    Args:
        x: Predicted scores (probabilities), shape (batch, num_classes).
        R: The descendants matrix, shape (num_classes, num_classes), with
            ``R[i, j] = 1`` iff ``j`` is ``i`` or a descendant of ``i``.

    Returns:
        The corrected scores, shape (batch, num_classes).
    """
    c_out = x.double().unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out, dim=2)
    return final_out


class ShieldLayer(nn.Module):
    """
    Differentiable layer that corrects predictions so they satisfy a set of hierarchical
    requirements over binary variables (the `num_classes` outputs, interpreted as probabilities).

    The requirements form a class hierarchy (a DAG of subsumptions ``child -> parent``). The layer
    implements C-HMCNN's Max Constraint Module: it precomputes the descendants matrix `R` and, in
    `forward`, replaces each class score with the maximum score over that class and its descendants.
    This guarantees ``score(child) <= score(parent)`` on the output for every edge, no matter the
    input.

    At training time a `goal` (the ground-truth labels) can be supplied to reproduce C-HMCNN's
    max-constraint loss behaviour: the correction is taken from the true subtree where the label is
    positive and from the predicted subtree elsewhere, so the model is pushed to raise the relevant
    descendant rather than the ancestor directly.

    Attributes:
        num_classes: The number of output variables (classes).
        hierarchy: The parsed :class:`Hierarchy`.
        class_names: The class names in variable-index order (row/column order of `R`).
        R: The (num_classes, num_classes) descendants matrix, held as a non-trainable buffer.

    Args:
        num_variables: The number of output variables. For ``.txt`` requirements this must match the
            hierarchy; for ``.arff`` requirements, where the class count (including the implicit
            root) is fixed by the file, it may be left as ``None`` to adopt the parsed count.
        requirements_filepath: Path to the requirements file (``.txt`` Horn rules or ``.arff``).
        ordering_choice: Accepted for signature compatibility with the other Shield Layers; the MCM
            correction is ordering-independent, so this argument is ignored.
        arff_hierarchy_style: For ``.arff`` requirements, ``'auto'`` (the default) detects the
            format from the file; ``'paths'`` (FUN-style) or ``'edges'`` (GO-style) force it.
            See :func:`pishield.hierarchical_requirements.parser.parse_arff_hierarchy`.

    Example:
        >>> layer = ShieldLayer(num_variables=4, requirements_filepath='hierarchy.txt')
        >>> corrected = layer(predictions)  # corrected satisfies score(child) <= score(parent)
    """

    def __init__(self, num_variables: int, requirements_filepath: str,
                 ordering_choice: str = 'given', arff_hierarchy_style: str = 'auto'):
        """Build the layer: parse the hierarchy and precompute the descendants matrix `R`."""
        super().__init__()
        self.ordering_choice = ordering_choice

        hierarchy = parse_hierarchy_file(requirements_filepath, arff_hierarchy_style=arff_hierarchy_style)
        self.hierarchy = hierarchy
        self.class_names = hierarchy.class_names

        if num_variables is not None and num_variables != hierarchy.num_classes:
            raise Exception(
                f'num_variables={num_variables} does not match the {hierarchy.num_classes} classes parsed from '
                f'{requirements_filepath}. Note that ARFF path-style hierarchies add an implicit root class; '
                f'pass num_variables={hierarchy.num_classes} (or None to adopt the parsed count).')
        self.num_classes = hierarchy.num_classes

        R = torch.tensor(hierarchy.descendants_matrix(), dtype=torch.float64)
        # Register R as a buffer so .to(device)/.state_dict() move it with the module but it is not trained.
        self.register_buffer('R', R)

    def forward(self, preds: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """Correct predictions so they satisfy the hierarchical requirements.

        Args:
            preds: The predicted probabilities, shape (batch, num_classes).
            goal: Optional ground-truth labels, shape (batch, num_classes). When
                supplied, the MCM correction is combined with the labels as in
                C-HMCNN's training loss: ``(1 - goal) * MCM(preds) + goal * MCM(goal * preds)``.
                When ``None`` (inference), the plain MCM correction ``MCM(preds)``
                is returned.

        Returns:
            The corrected predictions, same shape and dtype as ``preds``.

        Example:
            >>> corrected = layer(predictions)               # inference
            >>> train_out = layer(predictions, goal=labels)  # training (feed to BCELoss)
        """
        constrained = get_constr_out(preds, self.R)
        if goal is None:
            return constrained.to(preds.dtype)

        goal = goal.to(constrained.dtype)
        true_subtree = get_constr_out(goal * preds.double(), self.R)
        corrected = (1 - goal) * constrained + goal * true_subtree
        return corrected.to(preds.dtype)

    def satisfied(self, preds: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return, per sample, whether the thresholded predictions are coherent.

        A prediction is coherent when, for every edge ``child -> parent``, the
        child is not predicted without the parent.

        Args:
            preds: The predicted probabilities, shape (batch, num_classes).
            threshold: The decision threshold applied before checking coherence.

        Returns:
            A boolean tensor of shape (batch,), True where all edges hold.
        """
        labels = (preds > threshold)
        ok = torch.ones(preds.shape[0], dtype=torch.bool, device=preds.device)
        for child, parent in self.hierarchy.edges:
            ok &= ~labels[:, child] | labels[:, parent]
        return ok
