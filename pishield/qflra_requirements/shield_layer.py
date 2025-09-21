from typing import List, Union
import torch

from pishield.qflra_requirements.classes import Variable, Constraint, Atom
from pishield.qflra_requirements.compute_sets_of_constraints import get_pos_neg_x_constr, compute_sets_of_constraints
from pishield.qflra_requirements.correct_predictions import get_constr_at_level_x, get_final_x_correction, correct_preds
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

    def __call__(self, preds: torch.Tensor):
        return correct_preds(preds, self.ordering, self.sets_of_constr)
