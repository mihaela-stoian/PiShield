"""Derive, per variable, the set of linear requirements that bound it.

This module performs Fourier-Motzkin-style variable elimination over the linear
requirements: following the (reversed) variable ordering, it repeatedly
eliminates the highest-ranking remaining variable by combining the requirements
that contain it, producing for each variable the set of requirements that bound
it in terms of lower-ranked variables only.
"""

import numpy as np
from typing import List

from pishield.linear_requirements.classes import Variable, Constraint, Atom, Inequality


def collapse_atoms(atom_list):
    """Merge atoms that refer to the same variable into a single atom.

    Atoms over the same variable are combined by summing their signed
    coefficients; any atom whose resulting coefficient is zero is dropped.

    Args:
        atom_list: List of :class:`Atom` objects, possibly with duplicate
            variables.

    Returns:
        A list of :class:`Atom` objects with at most one atom per variable and
        no zero-coefficient atoms.
    """
    # merge any duplicated atoms in an atom list
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


def split_constr_atoms(y: Variable, constr: Constraint):
    """Separate the coefficient of ``y`` from the rest of a constraint's body.

    Args:
        y: The variable to extract.
        constr: The constraint whose body is split.

    Returns:
        A tuple ``(red_coefficient, complementary_atoms)`` where
        ``red_coefficient`` is the (positive) coefficient of ``y`` and
        ``complementary_atoms`` is the list of the remaining body atoms.
    """
    complementary_atoms = []
    for atom in constr.get_body_atoms():
        if atom.variable.id == y.id:
            red_coefficient = atom.coefficient  # note this is a positive real constant
        else:
            complementary_atoms.append(atom)
    return red_coefficient, complementary_atoms


def multiply_coefficients_of_atoms(atoms: List[Atom], coeff: float):
    """Scale every atom's coefficient by a constant factor.

    Args:
        atoms: The atoms to scale.
        coeff: The multiplicative factor applied to each coefficient.

    Returns:
        A new list of :class:`Atom` objects with scaled coefficients; the
        variables and signs are preserved.
    """
    new_atoms = []
    for atom in atoms:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        new_atom = Atom(variable, coefficient*coeff, positive_sign)
        new_atoms.append(new_atom)
    return new_atoms


def create_constr_by_reduction(y: Variable, constraints_with_y: List[Constraint]):
    """Eliminate a variable by pairwise combination of its requirements.

    Splits the requirements containing ``y`` by the sign of ``y`` and, for every
    pair of a positive and a negative occurrence, derives a new requirement in
    which ``y`` has been cancelled out (a Fourier-Motzkin elimination step).

    Args:
        y: The variable to eliminate.
        constraints_with_y: The requirements that contain ``y``.

    Returns:
        A list of new :class:`Constraint` objects that no longer mention ``y``.
    """
    red_constr = []
    # separate the constraints in two sets by the sign of y (pos or neg)
    pos_constr, neg_constr = get_pos_neg_x_constr(y, constraints_with_y)

    # then obtain new constr by reduction on y from pairs of constr (p,q)
    # where p is from pos_constr and q is from neg_constr
    for p in pos_constr:
        for q in neg_constr:
            p_coeff, p_complementary_body = split_constr_atoms(y, p)
            q_coeff, q_complementary_body = split_constr_atoms(y, q)
            # if p_complementary_body == []:
            #     break
            # if q_complementary_body == []:
            #     continue

            # multiply all coeff in p by q_coeff
            p_complementary_body = multiply_coefficients_of_atoms(p_complementary_body, q_coeff/p_coeff)

            # take the union of the p and q lists of atoms,
            # excluding the atom corresponding to y (whose coefficient is now 0)
            p_complementary_body.extend(q_complementary_body)

            # merge any atom duplicates
            p_complementary_body = collapse_atoms(p_complementary_body)

            _, ineq_sign_p, constant_p = p.single_inequality.get_ineq_attributes()
            _, ineq_sign_q, constant_q = q.single_inequality.get_ineq_attributes()
            new_ineq_sign = ineq_sign_p
            new_constant = constant_p + constant_q
            if p_complementary_body != []:
                new_inequality = Inequality(p_complementary_body, new_ineq_sign, new_constant)
                red_constr.append(Constraint([new_inequality]))

    return red_constr


def get_pos_neg_x_constr(y, constraints_with_y):
    """Partition requirements by the sign of a variable's occurrence.

    Args:
        y: The variable whose sign is inspected.
        constraints_with_y: Requirements that contain ``y``.

    Returns:
        A tuple ``(pos_constr, neg_constr)`` of the requirements in which ``y``
        occurs with a positive and a negative coefficient, respectively.
    """
    pos_constr, neg_constr = [], []
    for constr in constraints_with_y:
        for atom in constr.get_body_atoms():
            if atom.variable.id == y.id:
                if atom.positive_sign:
                    pos_constr.append(constr)
                else:
                    neg_constr.append(constr)
                break
    return pos_constr, neg_constr


def compute_set_of_constraints_for_variable(x: Variable, prev_x: Variable, constraints_at_previous_level: List[Constraint], verbose):
    """Derive the requirement set for one variable by eliminating the previous one.

    Takes the requirements available at the previous level (those that bound
    ``prev_x``), eliminates ``prev_x`` from the requirements that contain it, and
    returns the union of the requirements that did not contain ``prev_x`` with
    the newly reduced requirements.

    Args:
        x: The variable whose level is being computed.
        prev_x: The previously processed (higher-ranked) variable to eliminate.
        constraints_at_previous_level: Requirements available at ``prev_x``'s level.
        verbose: Whether to print diagnostic information.

    Returns:
        The list of :class:`Constraint` objects forming the requirement set at
        ``x``'s level (none of which mention ``prev_x``).
    """
    # create two sets starting from constraints_at_previous_level:
    # one containing only the constraints which variable prev_x appears in
    # and its complement
    constraints_without_prev = []
    constraints_with_prev = []

    for constr in constraints_at_previous_level:
        if constr.contains_variable(prev_x):
            constraints_with_prev.append(constr)
        else:
            constraints_without_prev.append(constr)

    # then compute a new set of constraints derived by algebraic manipulation on constraints containing prev_x
    # note that this new set of constraints will not have occurrences of prev_x by construction
    reduced_constr = create_constr_by_reduction(prev_x, constraints_with_prev)

    # finally, get the union of constraints which do not contain y (constraints_without_prev U reduced_constr)
    constraints_without_prev.extend(reduced_constr)

    # if verbose:
    #     print('\nLEVEL', x.readable())
    #     print('-------------------Constraints at this level:')
    #     for constr in constraints_without_prev:
    #         print(constr.readable())

    return constraints_without_prev


def compute_sets_of_constraints(ordering: List[Variable], constraints: List[Constraint], verbose) -> {Variable: List[Constraint]}:
    """Compute the requirement set bounding each variable along the ordering.

    Processes the variables from the highest-ranked to the lowest-ranked (the
    reversed ordering), at each step eliminating the previously processed
    variable, so that each variable is associated with the requirements that
    bound it in terms of lower-ranked variables only.

    Args:
        ordering: The ordering of the variables.
        constraints: All linear requirements.
        verbose: Whether to print diagnostic information.

    Returns:
        A dict mapping each :class:`Variable` to the list of :class:`Constraint`
        objects that bound it at its level.
    """
    # reverse the ordering:
    ordering = list(reversed(ordering))
    prev_x = ordering[0]

    # add all constraints for the highest ranking variable w.r.t. ordering
    ordered_constraints = {prev_x: constraints}
    print('All constraints')
    for constr in constraints:
        print(constr.readable())

    for x in ordering[1:]:
        constraints_at_previous_level = ordered_constraints[prev_x]
        set_of_constraints = compute_set_of_constraints_for_variable(x, prev_x, constraints_at_previous_level, verbose)
        ordered_constraints[x] = set_of_constraints
        prev_x = x

    # print('-'*80)
    # for x in ordering:
    #     print(f' *** Constraints for {x.readable()} ***')
    #     for i,constr in enumerate(ordered_constraints[x]):
    #         print(f'constr number {i}')
    #         print(constr.readable())
    #     if len(ordered_constraints[x]) == 0:
    #         print('empty set')
    #     print('***\n')
    # print('-'*80)

    return ordered_constraints
