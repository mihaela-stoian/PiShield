import math
import torch
from torch import nn

from cloverd.propositional_constraints.constraints_module import ConstraintsModule
from cloverd.propositional_constraints.slicer import Slicer


class ConstraintsLayer(nn.Module):
    def __init__(self, strata, num_classes):
        super(ConstraintsLayer, self).__init__()

        # ConstraintsLayer([ConstraintsGroup], int)
        self.num_classes = num_classes
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

    @classmethod
    def from_clauses_group(cls, group, num_classes, centrality):
        return cls(group.stratify(centrality), num_classes)

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
