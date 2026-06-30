"""Evaluation and statistics helpers for QFLRA constraints.

Functions here evaluate inequality atoms on batched predictions, detect missing
values and disjunctions, and report constraint-satisfaction statistics.
"""

from typing import List

import pandas as pd
import torch

import pishield.qflra_requirements.classes as classes
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pishield.qflra_requirements.classes import Atom, Variable, Constraint, DisjunctInequality

TOLERANCE=1e-2


def eval_atoms_list(atoms_list: List['Atom'], preds: torch.Tensor, reduction='sum'):
    """Evaluate a list of atoms on batched predictions and reduce them.

    Each atom is evaluated at the predicted value of its variable, then the
    per-atom results are combined according to ``reduction``.

    Args:
        atoms_list: The :class:`Atom` objects forming an inequality body.
        preds: Predictions tensor of shape ``(B, D)``; column ``id`` holds the value
            of the variable with that id.
        reduction: How to combine atoms across the body. Only ``'sum'`` is supported.

    Returns:
        A tensor of shape ``(B,)`` with the reduced body value per sample; an
        all-zero tensor if ``atoms_list`` is empty.

    Raises:
        Exception: If ``reduction`` is not ``'sum'``.
    """
    evaluated_atoms = []
    for atom in atoms_list:
        atom_value = preds[:, atom.variable.id]
        evaluated_atoms.append(atom.eval(atom_value))

    if evaluated_atoms == []:
        return torch.zeros(preds.shape[0])

    evaluated_atoms = torch.stack(evaluated_atoms, dim=1)
    if reduction == 'sum':
        result = evaluated_atoms.sum(dim=1)
    else:
        raise Exception(f'{reduction} reduction not implemented!')
    return result



def any_disjunctions_in_constraint_set(constraints: List['Constraint']):
    """Report whether any constraint is a disjunction of inequalities.

    Args:
        constraints: The constraints to inspect.

    Returns:
        ``True`` if at least one constraint has more than one inequality (i.e. an
        'or'), ``False`` otherwise.
    """
    for constraint in constraints:
        if len(constraint.list_inequalities) > 1:
            return True
    return False


def get_missing_mask(ineq_atoms: List['Atom'], preds: torch.Tensor):
    """Identify samples whose involved variable values indicate missing data.

    A sample is flagged as missing when the product of the raw values of the
    relevant variables falls below ``-TOLERANCE`` (the convention used to encode
    missing values in this codebase).

    Args:
        ineq_atoms: Atoms (or list of atom lists) whose variables are checked.
        preds: Predictions tensor of shape ``(B, D)``.

    Returns:
        A boolean tensor of shape ``(B,)`` that is ``True`` for samples with missing
        values.
    """
    raw_variable_values = []
    if type(ineq_atoms[0]) != list:
        ineq_atoms = [ineq_atoms]
    for atoms_list in ineq_atoms:
        for atom in atoms_list:
            raw_variable_values.append(preds[:, atom.variable.id])

    raw_variable_values = torch.stack(raw_variable_values, dim=1)
    missing_values_mask = raw_variable_values.prod(dim=1) < -classes.TOLERANCE
    return missing_values_mask


def split_constr_atoms(y: 'Variable', constr: 'Constraint'):
    """Separate the atom of a variable from the rest of a constraint body.

    Args:
        y: The variable to extract.
        constr: The constraint whose body atoms are split.

    Returns:
        A tuple ``(red_coefficient, complementary_atoms)`` where ``red_coefficient``
        is the (positive) coefficient of ``y`` and ``complementary_atoms`` are the
        remaining body atoms.
    """
    complementary_atoms = []
    for atom in constr.get_body_atoms():
        if atom.variable.id == y.id:
            red_coefficient = atom.coefficient  # note this is a positive real constant
        else:
            complementary_atoms.append(atom)
    return red_coefficient, complementary_atoms

def get_samples_violating_constraints(constraints, preds):
    """Flag samples that violate at least one constraint.

    Args:
        constraints: The constraints to check.
        preds: Predictions tensor of shape ``(B, D)``.

    Returns:
        A boolean tensor of shape ``(B,)`` that is ``True`` for samples violating any
        constraint.
    """
    samples_violating_req = []
    for constr in constraints:
        samples_sat_req = constr.disjunctive_inequality.check_satisfaction(preds)  # shape Bx1
        samples_violating_req.append(~samples_sat_req)

    samples_violating_req = torch.stack(samples_violating_req, dim=1)
    mask_samples_violating_req = samples_violating_req.any(dim=1)
    return mask_samples_violating_req


def check_all_constraints_are_sat(constraints, preds, corrected_preds):
    """Print which constraints are violated before and after correction.

    Args:
        constraints: The constraints to check.
        preds: The original predictions tensor.
        corrected_preds: The predictions after correction.

    Returns:
        ``True`` if all constraints are satisfied by ``corrected_preds``, ``False``
        otherwise.
    """
    # print('sat req?:')
    for constr in constraints:
        sat = constr.check_satisfaction(preds)
        if not sat:
            print('Not satisfied!', constr.readable())

    # print('*' * 80)
    all_sat_after_correction = True
    for constr in constraints:
        sat = constr.check_satisfaction(corrected_preds)
        if not sat:
            all_sat_after_correction = False
            print('Not satisfied!', constr.readable())
    if all_sat_after_correction:
        print('All constraints are satisfied after correction!')
    else:
        print('There are still constraint violations!!!')
    return all_sat_after_correction


def compute_sat_stats(real_data, constraints, mask_out_missing_values=False):
    """Compute per-constraint and overall constraint-satisfaction statistics.

    Args:
        real_data: Tensor of data/predictions of shape ``(B, D)``.
        constraints: The constraints to evaluate.
        mask_out_missing_values: If ``True``, samples flagged as missing (see
            :func:`get_missing_mask`) are excluded from the satisfaction rates.

    Returns:
        A tuple ``(sat_rate_per_constr, percentage_of_samples_violating_constraints)``
        of pandas DataFrames: per-constraint satisfaction rates (as percentages) and
        the overall percentage of samples violating constraints.
    """
    real_data = pd.DataFrame(real_data.detach().numpy())
    sat_rate_per_constr = {i: [] for i in range(len(constraints))}
    percentage_of_samples_sat_constraints = []

    samples_sat_constr = torch.ones(real_data.shape[0]) == 1.
    # real_data = real_data.iloc[:, :-1].to_numpy()
    real_data = torch.tensor(real_data.to_numpy())

    for j, constr in enumerate(constraints):
        sat_per_datapoint = constr.disjunctive_inequality.check_satisfaction(real_data)
        if mask_out_missing_values:
            missing_values_mask = get_missing_mask(constr.disjunctive_inequality.body, real_data)
        else:
            missing_values_mask = torch.ones(real_data.shape[0]) == 0.
        sat_per_datapoint[missing_values_mask] = True
        sat_rate = sat_per_datapoint[~missing_values_mask].sum() / (~missing_values_mask).sum()
        # print('Real sat_rate is', sat_rate, sat_per_datapoint.sum(), len(sat_per_datapoint), sat_per_datapoint)
        sat_rate_per_constr[j].append(sat_rate)
        # sat_rate_per_constr[j].append(sat_per_datapoint.sum() / len(sat_per_datapoint))
        samples_sat_constr = samples_sat_constr & sat_per_datapoint

    percentage_of_samples_sat_constraints.append(sum(samples_sat_constr) / len(samples_sat_constr))
    sat_rate_per_constr = {i: [sum(sat_rate_per_constr[i]) / len(sat_rate_per_constr[i]) * 100.0] for i in
                           range(len(constraints))}
    percentage_of_samples_violating_constraints = 100.0 - sum(percentage_of_samples_sat_constraints) / len(
        percentage_of_samples_sat_constraints) * 100.0
    print('REAL', 'sat_rate_per_constr', sat_rate_per_constr)
    print('REAL', 'percentage_of_samples_violating_constraints', percentage_of_samples_violating_constraints)

    sat_rate_per_constr = pd.DataFrame(sat_rate_per_constr, columns=list(range(len(constraints))))

    percentage_of_samples_violating_constraints = pd.DataFrame(
        {'real_percentage_of_samples_violating_constraints': [percentage_of_samples_violating_constraints]},
        columns=['real_percentage_of_samples_violating_constraints'])

    return sat_rate_per_constr, percentage_of_samples_violating_constraints
