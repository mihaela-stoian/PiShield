"""Helper functions for manipulating lists of linear inequality atoms.

These utilities merge, scale and negate the :class:`Atom` objects that make up the
body (left-hand side) of QFLRA inequalities.
"""

from typing import List
import numpy as np
from pishield.qflra_requirements.classes import Atom


def collapse_atoms(atom_list):
    """Merge duplicated atoms that refer to the same variable.

    Atoms that share a variable are combined by summing their signed coefficients.
    Any atom whose coefficients cancel out (resulting coefficient of zero) is dropped.

    Args:
        atom_list: List of :class:`Atom` objects, possibly with several atoms per
            variable.

    Returns:
        A list of :class:`Atom` objects with at most one atom per variable.
    """
    # merge any duplicated atoms in a atom list
    merged_atoms = {}
    merged_atoms: {int: Atom}
    for atom in atom_list:
        var = atom.variable.id
        if var not in merged_atoms.keys():
            merged_atoms[var] = atom
        else:
            variable, coefficient, positive_sign = merged_atoms[var].get_atom_attributes()
            existing_coeff = coefficient if positive_sign else -coefficient
            current_coeff = atom.coefficient if atom.positive_sign else -atom.coefficient
            new_coefficient = existing_coeff + current_coeff
            if new_coefficient != 0:
                new_atom = Atom(variable, float(np.abs(new_coefficient)), True if new_coefficient > 0 else False)
                merged_atoms[var] = new_atom
            else:
                # EG: Mihaela can you please check this? It seems to be the same problem as the lineaer case
                # The variable has cancelled out (net coefficient 0): drop it entirely,
                # otherwise a stale atom would survive and inject a spurious bound during
                # variable elimination (same failure as in the linear backend).
                del merged_atoms[var]
    return list(merged_atoms.values())


def multiply_coefficients_of_atoms(atoms: List[Atom], coeff: float):
    """Scale every atom's coefficient by a constant factor.

    Args:
        atoms: List of :class:`Atom` objects to scale.
        coeff: Multiplicative factor applied to each atom's coefficient.

    Returns:
        A new list of :class:`Atom` objects with scaled coefficients (the signs are
        preserved).
    """
    new_atoms = []
    for atom in atoms:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        new_atom = Atom(variable, coefficient*coeff, positive_sign)
        new_atoms.append(new_atom)
    return new_atoms


def negate_atoms(atoms: List[Atom]):
    """Flip the sign of every atom in a list.

    Args:
        atoms: List of :class:`Atom` objects to negate.

    Returns:
        A new list of :class:`Atom` objects with each positive sign toggled.
    """
    new_atoms = []
    for atom in atoms:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        new_atom = Atom(variable, coefficient, not positive_sign)
        new_atoms.append(new_atom)
    return new_atoms
