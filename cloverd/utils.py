from typing import List
import torch


def eval_atoms_list(atoms_list: List, preds: torch.Tensor, reduction='sum'):
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