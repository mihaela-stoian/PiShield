import math
from typing import Union, List

import torch
from torch import nn

from pishield.propositional_requirements.clauses_group import ClausesGroup
from pishield.propositional_requirements.constraints_group import ConstraintsGroup
from pishield.propositional_requirements.constraints_module import ConstraintsModule
from pishield.propositional_requirements.slicer import Slicer
from pishield.propositional_requirements.util import get_order_and_centrality


class ShieldLayer(nn.Module):

    def __init__(self, num_classes: int,
                 constraints: Union[str, List[ConstraintsGroup]] = None,
                 ordering_choice: str = None,
                 custom_ordering: str = None):
        super(ShieldLayer, self).__init__()

        self.num_classes = num_classes
        self.ordering_choice = ordering_choice
        self.custom_ordering = custom_ordering
        self.constraints = constraints

        if type(constraints) == str:
            constraints_filepath = constraints
            constraints_group = ConstraintsGroup(constraints_filepath)
            clauses_group = ClausesGroup.from_constraints_group(constraints_group)

            # forced = False  # TODO: what's this for?
            # clauses = clauses.add_detection_label(forced)
            # print(f"Shifted atoms and added n0 to all clauses (forced {forced})")

            centrality = get_order_and_centrality(ordering_choice, custom_ordering)
            strata = clauses_group.stratify(centrality)
            self.stratified_constraints = strata

            print(f"Generated {len(strata)} strata of constraints with {centrality} centrality")
        elif type(constraints) == list:
            strata = constraints
            centrality = None
        else:
            raise Exception(
                'constraints argument should be either str (i.e. filepath of the constraints) or List (i.e. strata)')

        # ConstraintsLayer([ConstraintsGroup], int)
        self.atoms = nn.Parameter(torch.tensor(list(range(num_classes))), requires_grad=False)

        modules = [ConstraintsModule(stratum, num_classes) for stratum in strata]
        self.module_list = nn.ModuleList(modules)

        # Compute all strata & core 
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
        cls.centrality = centrality
        return cls(num_classes=num_classes, constraints=clauses_group.stratify(centrality))

    def gradual_prefix(self, ratio):
        atoms = self.core
        remaining = math.floor(ratio * self.num_classes) - len(atoms)
        if (remaining <= 0): return atoms, 0

        for i, stratum in enumerate(self.strata):
            remaining -= len(stratum)
            if (remaining < 0): return atoms, i
            atoms = atoms.union(stratum)

        return atoms, len(self.strata)

    def slicer(self, ratio):
        atoms, modules = self.gradual_prefix(ratio)
        return Slicer(atoms, modules)

    def to_minimal(self, tensor):
        return tensor[:, self.atoms].reshape(tensor.shape[0], len(self.atoms))

    def from_minimal(self, tensor, init):
        return init.index_copy(1, self.atoms, tensor)

    def forward(self, preds, goal=None, iterative=True, slicer=None):
        updated = self.to_minimal(preds)
        goal = None if goal is None else self.to_minimal(goal)

        modules = self.module_list if slicer is None else slicer.slice_modules(self.module_list)
        for module in modules:
            updated = module(updated, goal=goal, iterative=iterative)

        return self.from_minimal(updated, preds)


def run_layer(layer, preds, backward=False):
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
