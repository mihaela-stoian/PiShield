"""The propositional Shield Layer.

Defines :class:`ShieldLayer`, a differentiable PyTorch layer that corrects a model's
predictions so they provably satisfy a set of propositional requirements. The
requirements are normalised into clauses, stratified into ordered layers, and applied
in order by per-stratum :class:`ConstraintsModule` modules.
"""

import math
from typing import Union, List

import torch
from torch import nn
import numpy as np

from pishield.propositional_requirements.clauses_group import ClausesGroup
from pishield.propositional_requirements.constraints_group import ConstraintsGroup
from pishield.propositional_requirements.constraints_module import ConstraintsModule
from pishield.propositional_requirements.slicer import Slicer
from pishield.propositional_requirements.util import get_order_and_centrality


class ShieldLayer(nn.Module):
    """
    Differentiable layer that corrects predictions so they satisfy a set of propositional
    requirements over binary variables (the `num_classes` outputs, interpreted as probabilities).

    The requirements are normalised into clauses and then *stratified*: each clause is assigned to a
    layer (stratum) such that a clause's head is only corrected after the variables in its body, with
    the `centrality`/ordering deciding which variable plays the head when there is a choice. Each
    stratum becomes a `ConstraintsModule`, and `forward` applies the modules in order, so correcting
    one stratum never violates an earlier one — guaranteeing all requirements hold on the output.

    `requirements` may be either a path to a constraints file or a precomputed list of strata.
    """

    def __init__(self, num_classes: int,
                 requirements: Union[str, List[ConstraintsGroup]] = None,
                 ordering_choice: str = None,
                 custom_ordering: str = None):
        """Build the layer by stratifying the requirements into correction modules.

        Args:
            num_classes: The number of output variables (labels) the layer operates on.
            requirements: Either a path to a constraints file or a precomputed list of
                strata (:class:`ConstraintsGroup` objects).
            ordering_choice: The variable-ordering / centrality strategy used to decide
                which variable plays the head during stratification.
            custom_ordering: An optional explicit comma-separated variable order; with a
                ``'given'`` ordering choice and no value, defaults to ``0,1,...,n-1``.

        Raises:
            Exception: If ``requirements`` is neither a str nor a list.

        Example:
            >>> layer = ShieldLayer(num_classes=5,
            ...                     requirements='constraints.txt',
            ...                     ordering_choice='given')
            >>> corrected = layer(predictions)  # predictions: (batch, 5)
        """
        super(ShieldLayer, self).__init__()

        self.num_classes = num_classes
        self.ordering_choice = ordering_choice
        # With the 'given' ordering and no explicit ordering, default to the ascending order 0,1,...,n-1.
        if 'given' in ordering_choice and custom_ordering is None:
            custom_ordering = ",".join([str(e) for e in np.arange(0,num_classes)])
        self.custom_ordering = custom_ordering
        self.constraints = requirements

        if type(requirements) == str:
            # Load constraints from file and convert them to a normalised set of clauses.
            constraints_filepath = requirements
            constraints_group = ConstraintsGroup(constraints_filepath)
            clauses_group = ClausesGroup.from_constraints_group(constraints_group)

            # forced = False  # TODO: what's this for?
            # clauses = clauses.add_detection_label(forced)
            # print(f"Shifted atoms and added n0 to all clauses (forced {forced})")

            # Stratify the clauses into ordered layers according to the chosen variable centrality.
            centrality = get_order_and_centrality(ordering_choice, custom_ordering)
            strata = clauses_group.stratify(centrality)
            self.stratified_constraints = strata

            print(f"Generated {len(strata)} strata of constraints with {centrality} centrality")
        elif type(requirements) == list:
            # Strata were supplied directly (e.g. from `from_clauses_group`), so use them as-is.
            strata = requirements
            centrality = None
        else:
            raise Exception(
                'constraints argument should be either str (i.e. filepath of the constraints) or List (i.e. strata)')

        # ConstraintsLayer([ConstraintsGroup], int)
        self.atoms = nn.Parameter(torch.tensor(list(range(num_classes))), requires_grad=False)

        # One correction module per stratum; forward() applies them in order.
        modules = [ConstraintsModule(stratum, num_classes) for stratum in strata]
        self.module_list = nn.ModuleList(modules)

        # The "core" is the set of variables that are never a clause head, i.e. never corrected;
        # they are passed through unchanged and seed the corrections of the strata above them.
        core = set(range(num_classes))
        strata = [stratum.heads() for stratum in strata]

        for stratum in strata:
            core = core.difference(stratum)

        assert len(core) > 0
        self.core = core
        self.strata = strata
        self.centrality = centrality

    @classmethod
    def from_clauses_group(cls, num_classes, clauses_group, centrality):
        """Build a Shield Layer from a clauses group and centrality ordering.

        Args:
            num_classes: The number of output variables.
            clauses_group: The :class:`ClausesGroup` of requirements to stratify.
            centrality: The centrality / ordering used to stratify the clauses.

        Returns:
            A configured ShieldLayer.
        """
        cls.centrality = centrality
        return cls(num_classes=num_classes, requirements=clauses_group.stratify(centrality))

    def gradual_prefix(self, ratio):
        """Select the atoms and number of strata covered by a fraction of variables.

        Grows the never-corrected core with whole strata until roughly ``ratio`` of all
        variables are covered; used to gradually enable the requirements in training.

        Args:
            ratio: The target fraction of variables to cover, in [0, 1].

        Returns:
            A tuple ``(atoms, num_modules)`` of the covered atom set and the number of
            leading strata included.
        """
        atoms = self.core
        remaining = math.floor(ratio * self.num_classes) - len(atoms)
        if (remaining <= 0): return atoms, 0

        for i, stratum in enumerate(self.strata):
            remaining -= len(stratum)
            if (remaining < 0): return atoms, i
            atoms = atoms.union(stratum)

        return atoms, len(self.strata)

    def slicer(self, ratio):
        """Build a :class:`Slicer` covering a fraction of the requirements.

        Args:
            ratio: The target fraction of variables to cover, in [0, 1].

        Returns:
            A Slicer over the corresponding atoms and leading modules.
        """
        atoms, modules = self.gradual_prefix(ratio)
        return Slicer(atoms, modules)

    def to_minimal(self, tensor):
        """Restrict a tensor to the layer's atom columns.

        Args:
            tensor: A tensor of shape (batch, num_classes).

        Returns:
            The tensor restricted to the layer's atoms.
        """
        return tensor[:, self.atoms].reshape(tensor.shape[0], len(self.atoms))

    def from_minimal(self, tensor, init):
        """Scatter a minimal-atom tensor back into a full tensor.

        Args:
            tensor: The minimal tensor over the layer's atoms.
            init: The full tensor to write into.

        Returns:
            ``init`` with the atom columns overwritten by ``tensor``.
        """
        return init.index_copy(1, self.atoms, tensor)

    def forward(self, preds, goal=None, iterative=True, slicer=None):
        """Correct predictions so they satisfy all requirements.

        Restricts the predictions to the constrained atoms, applies each stratum's
        correction module in order, and scatters the corrected values back into the
        full prediction tensor.

        Args:
            preds: The model predictions, shape (batch, num_classes).
            goal: Optional goal (ground-truth) assignment to keep corrections
                consistent with during training.
            iterative: If True use the iterative correction implementation.
            slicer: Optional :class:`Slicer` restricting how many strata are applied.

        Returns:
            The corrected predictions, same shape as ``preds``.

        Example:
            >>> corrected = layer(predictions)
            >>> # every requirement now holds on `corrected`
        """
        # Restrict to the atoms involved in the constraints, apply each stratum's correction in order,
        # then scatter the corrected values back into the full prediction tensor.
        updated = self.to_minimal(preds)
        goal = None if goal is None else self.to_minimal(goal)

        modules = self.module_list if slicer is None else slicer.slice_modules(self.module_list)
        for module in modules:
            updated = module(updated, goal=goal, iterative=iterative)

        return self.from_minimal(updated, preds)


def run_layer(layer, preds, backward=False):
    """Run a Shield Layer both ways and assert the results agree.

    Runs the layer with the iterative and tensor implementations, checks they are
    numerically close, and optionally exercises a backward pass; mainly a
    testing/debugging helper.

    Args:
        layer: The :class:`ShieldLayer` to run.
        preds: The prediction tensor.
        backward: If True, add a random differentiable perturbation and backpropagate.

    Returns:
        The corrected predictions (from the iterative implementation), detached.

    Raises:
        AssertionError: If the two implementations disagree.
    """
    if backward:
        extra = torch.rand_like(preds, requires_grad=True)
        preds = preds + extra

    iter = layer(preds, iterative=True)
    tens = layer(preds, iterative=False)
    assert torch.isclose(iter, tens).all()

    if backward:
        sum = iter.sum() + tens.sum()
        sum.backward()

    return iter.detach()
