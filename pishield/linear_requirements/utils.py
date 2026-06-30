"""Utility functions for evaluating atoms and checking requirement satisfaction."""

from typing import List
import torch


def eval_atoms_list(atoms_list: List, preds: torch.Tensor, reduction='sum'):
    """Evaluate a list of atoms against a batch of predictions.

    Each atom is evaluated by multiplying its variable's predicted value by the
    atom's signed coefficient; the per-atom values are then reduced across atoms.

    Args:
        atoms_list: The :class:`Atom` objects to evaluate (the body of an
            inequality). An empty list evaluates to zeros.
        preds: Prediction tensor of shape ``(batch_size, num_variables)``.
        reduction: How to combine the per-atom values. Only ``'sum'`` is
            supported.

    Returns:
        A tensor of shape ``(batch_size,)`` with the reduced atom values.

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


def check_constraint_satisfaction(preds: torch.Tensor, constraints: List) -> bool:
    """Check whether predictions satisfy all requirements, printing the outcome.

    Args:
        preds: Prediction tensor of shape ``(batch_size, num_variables)``.
        constraints: The requirements to check.

    Returns:
        ``True`` if every requirement is satisfied by every sample, ``False``
        otherwise.
    """
    all_constr_sat = True
    for constr in constraints:
        sat = constr.check_satisfaction_per_sample(preds)
        if not sat.all():
            all_constr_sat = False
            # raise Exception('Not satisfied!', constr.readable())
            print('Not satisfied!', constr.readable())
    if all_constr_sat:
        print('All constraints are satisfied!')
    return all_constr_sat
