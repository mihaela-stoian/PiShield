import random
from typing import List
import torch

from pishield.qflra_requirements.classes import Variable, Constraint, Inequality
from pishield.qflra_requirements.utils_atoms import collapse_atoms, multiply_coefficients_of_atoms, negate_atoms

TOLERANCE=1e-2


def normalise_constraint_wrt_selected_ineq(x: Variable, partially_normalised_inequality_list: List[Inequality], ineq_i: Inequality, signed_ineqs: list[Inequality], sign) -> List[Inequality]:

    new_partially_normalised_inequality_list = []
    new_partially_normalised_inequality_list.extend(partially_normalised_inequality_list)

    # we want to add ineq_j and ineq_k as they are to the set of inequalities that will form the new, normalised constraint
    new_partially_normalised_inequality_list.append(ineq_i)

    # STEP B:
    # once we picked inequalities j and k, we want to rewrite the other constraints where x appears pos, resp. neg, according to j and k
    complementary_atoms_in_i, x_atom_in_i, constant_of_i, _ = ineq_i.get_x_complement_body_atoms(x)

    assert type(x_atom_in_i) != list, "there should be only one x occurrence in ineq_i"

    if sign == 'positive':
        assert x_atom_in_i.positive_sign == True, "x_atom in ineq_i should have positive sign"
    elif sign == 'negative':
        assert x_atom_in_i.positive_sign == False, "x_atom in ineq_i should have negative sign"
    else:
        raise NotImplementedError

    complementary_body_of_i = multiply_coefficients_of_atoms(complementary_atoms_in_i, 1. / x_atom_in_i.coefficient)
    negated_complementary_body_of_i = negate_atoms(complementary_body_of_i)

    for ineq_p in signed_ineqs:
        # discard ineq_i from this process
        if ineq_p.readable() == ineq_i.readable():
            continue

        complementary_atoms_in_p, x_atom_in_p, constant_of_p, is_p_ineq_strict = ineq_p.get_x_complement_body_atoms(x)
        assert type(x_atom_in_p) != list, "there should be only one x occurrence in ineq_p"
        if sign == 'positive':
            assert x_atom_in_p.positive_sign == True, "x_atom in ineq_p should have positive sign"
        elif sign == 'negative':
            assert x_atom_in_p.positive_sign == False, "x_atom in ineq_p should have negative sign"
        else:
            raise NotImplementedError

        complementary_body_of_p = multiply_coefficients_of_atoms(complementary_atoms_in_p, 1. / x_atom_in_p.coefficient)

        if sign == 'positive':
            complementary_body = negate_atoms(complementary_body_of_p)
            complementary_body.extend(complementary_body_of_i)
            complementary_body = collapse_atoms(complementary_body)
            new_constant = ineq_i.constant / x_atom_in_i.coefficient - ineq_p.constant / x_atom_in_p.coefficient

        elif sign == 'negative':
            complementary_body = complementary_body_of_p
            complementary_body.extend(negated_complementary_body_of_i)
            complementary_body = collapse_atoms(complementary_body)
            new_constant = ineq_p.constant / x_atom_in_p.coefficient - ineq_i.constant / x_atom_in_i.coefficient
        else:
            raise NotImplementedError

        new_ineq_sign = ineq_p.ineq_sign if ineq_i.ineq_sign != ineq_p.ineq_sign else '>'
        new_inequality = Inequality(complementary_body, new_ineq_sign, new_constant)

        new_partially_normalised_inequality_list.append(new_inequality)

    return new_partially_normalised_inequality_list



def normalise_one_constraint(x: Variable, constraint: Constraint) -> Constraint | list[Constraint]:
    # FIRST STEP:
    # check if we need to normalise the constraint
    if len(constraint.list_inequalities) <= 1:
        return constraint

    positive_sign_x = {}
    count_positive_occurrences = 0
    count_negative_occurrences = 0
    for ineq in constraint.list_inequalities:
        if ineq.contains_variable(x):
            body_atoms = ineq.get_body_atoms()
            for atom in body_atoms:
                if atom.variable.id == x.id:
                    positive_sign_x[ineq] = atom.positive_sign
                    if atom.positive_sign:
                        count_positive_occurrences +=1
                    else:
                        count_negative_occurrences +=1
        else:
            positive_sign_x[ineq] = 'null'
    # 3 possible cases for which we don't need to normalise the constraint:
    #   x does not occur in constraint,
    #   x occurs neg/pos once,
    #   x occurs neg and pos once
    # anything else would need normalising
    if count_positive_occurrences <= 1 and count_negative_occurrences <= 1:
        return constraint

    # SECOND STEP:
    # from here we know there is a need to normalise the given constraint w.r.t. variable x
    totally_normalised_constraint = []  # list_of_partially_normalised_constraints

    # STEP A:
    # pick an inequality from those where x appears positively, call it j
    # and pick an inequality from those where x appears negatively, call it k
    pos_ineq = list([ineq for ineq in positive_sign_x.keys() if positive_sign_x[ineq] == True])
    neg_ineq = list([ineq for ineq in positive_sign_x.keys() if positive_sign_x[ineq] == False])
    null_ineq = list([ineq for ineq in positive_sign_x.keys() if positive_sign_x[ineq] == 'null'])

    if len(pos_ineq) > 0 and len(neg_ineq) > 0:
        for ineq_j in pos_ineq:
            pos_partially_normalised_inequality_list = normalise_constraint_wrt_selected_ineq(x, [], ineq_j, pos_ineq, sign='positive')
            for ineq_k in neg_ineq:
                partially_normalised_inequality_list = normalise_constraint_wrt_selected_ineq(x, pos_partially_normalised_inequality_list, ineq_k, neg_ineq, sign='negative')

                # extend the partially_normalised_inequality_list with the ineqs in which x did not appear originally
                partially_normalised_inequality_list.extend(null_ineq)

                # finally return the normalised constraint
                partially_normalised_constraint = Constraint(partially_normalised_inequality_list)
                totally_normalised_constraint.append(partially_normalised_constraint)

    elif len(pos_ineq) > 0 and len(neg_ineq) == 0:
        for ineq_j in pos_ineq:
            partially_normalised_inequality_list = normalise_constraint_wrt_selected_ineq(x, [], ineq_j, pos_ineq, sign='positive')

            # extend the partially_normalised_inequality_list with the ineqs in which x did not appear originally
            partially_normalised_inequality_list.extend(null_ineq)

            # finally return the normalised constraint
            partially_normalised_constraint = Constraint(partially_normalised_inequality_list)
            totally_normalised_constraint.append(partially_normalised_constraint)

    elif len(neg_ineq) > 0 and len(pos_ineq) == 0:
        for ineq_k in neg_ineq:
            partially_normalised_inequality_list = normalise_constraint_wrt_selected_ineq(x, [], ineq_k, neg_ineq, sign='negative')

            # extend the partially_normalised_inequality_list with the ineqs in which x did not appear originally
            partially_normalised_inequality_list.extend(null_ineq)

            # finally return the normalised constraint
            partially_normalised_constraint = Constraint(partially_normalised_inequality_list)
            totally_normalised_constraint.append(partially_normalised_constraint)

    return totally_normalised_constraint


def normalise(x: Variable, constraints: list[Constraint]) -> list[Constraint]:
    normalised_constraints: list[Constraint] = []

    for constraint in constraints:
        normalised_constraint = normalise_one_constraint(x, constraint)
        if type(normalised_constraint) == list:
            normalised_constraints.extend(normalised_constraint)
        else:
            normalised_constraints.append(normalised_constraint)
    return normalised_constraints


# def resolution():
#     normalise(Pi_j)
#     actual resolution()