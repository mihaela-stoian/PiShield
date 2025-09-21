from typing import List
import numpy as np
from pishield.qflra_requirements.classes import Atom


def collapse_atoms(atom_list):
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
    return list(merged_atoms.values())


def multiply_coefficients_of_atoms(atoms: List[Atom], coeff: float):
    new_atoms = []
    for atom in atoms:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        new_atom = Atom(variable, coefficient*coeff, positive_sign)
        new_atoms.append(new_atom)
    return new_atoms


def negate_atoms(atoms: List[Atom]):
    new_atoms = []
    for atom in atoms:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        new_atom = Atom(variable, coefficient, not positive_sign)
        new_atoms.append(new_atom)
    return new_atoms
