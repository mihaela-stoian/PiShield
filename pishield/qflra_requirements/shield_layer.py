from typing import List, Union
import torch

from pishield.qflra_requirements.classes import Variable
from pishield.qflra_requirements.compute_sets_of_constraints import compute_sets_of_constraints
from pishield.qflra_requirements.correct_predictions import get_constr_at_level_x, correct_preds
from pishield.qflra_requirements.feature_orderings import set_ordering
from pishield.qflra_requirements.parser import parse_constraints_file

INFINITY = torch.inf
EPSILON = 1e-12


class ShieldLayer(torch.nn.Module):
    def __init__(self, num_variables: int, requirements_filepath: str, ordering_choice: str = 'given'):
        super().__init__()
        self.num_variables = num_variables
        ordering, constraints = parse_constraints_file(requirements_filepath)
        self.ordering = set_ordering(ordering, ordering_choice)
        self.constraints = constraints
        self.sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    def get_dense_ordering(self) -> List[Variable]:
        dense_ordering = []
        for x in self.ordering:
            x_constr = get_constr_at_level_x(x, self.sets_of_constr)
            if len(x_constr) == 0:
                continue
            else:
                dense_ordering.append(x)
        return dense_ordering

    # def __call__(self, preds, *args, **kwargs):
    def __call__(self, preds: torch.Tensor):
        return correct_preds(preds, self.ordering, self.sets_of_constr)