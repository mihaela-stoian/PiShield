from typing import List, Tuple

from pishield.qflra_requirements.classes import Variable, Constraint, Inequality
from pishield.qflra_requirements.normalisation import normalise
from pishield.qflra_requirements.parser import parse_constraints_file
from pishield.qflra_requirements.utils_functions import split_constr_atoms
from pishield.qflra_requirements.utils_atoms import collapse_atoms, multiply_coefficients_of_atoms


def get_pos_pos_x_constr(y: Variable, pos_constr: List[Constraint], pos_neg_constr: List[Constraint]):
    pos_pos_constr: List[Constraint] = []

    for p in pos_constr:
        p: Constraint
        ineq_from_p_with_pos_y, list_ineqs_from_p_without_y = p.get_ineq_with_var_y_and_complement(y)
        for q in pos_neg_constr:
            q: Constraint
            ineq_from_q_with_pos_y, ineq_from_q_with_neg_y, list_ineqs_from_q_without_y = q.get_ineq_with_pos_and_neg_var_y_and_complement(y)

            # reduce(pos_p_ineq, neg_q_ineq)
            new_inequality = reduce_two_ineqs(y, ineq_from_p_with_pos_y, ineq_from_q_with_neg_y)
            if type(new_inequality) == bool:
                if new_inequality == True:
                    continue
                else:
                    new_list_of_ineqs = []
            else:
                new_list_of_ineqs = [new_inequality]
            new_list_of_ineqs.extend([ineq_from_q_with_pos_y])
            new_list_of_ineqs.extend(list_ineqs_from_p_without_y)
            new_list_of_ineqs.extend(list_ineqs_from_q_without_y)
            pos_pos_constr.append(Constraint(new_list_of_ineqs))

    # finally, extend pos_pos_constr with pos_constr
    pos_pos_constr.extend(pos_constr)
    return pos_pos_constr


def create_constr_by_reduction(y: Variable, constraints_with_y: List[Constraint]):
    red_constr = []
    # separate the constraints in two sets by the sign of y (pos or neg)
    pos_constr, neg_constr, pos_neg_constr = get_pos_neg_pn_x_constr(y, constraints_with_y)

    pos_pos_constr = get_pos_pos_x_constr(y, pos_constr, pos_neg_constr)

    # then obtain new constr by reduction on y from pairs of constr (p,q)
    # where p is from pos_constr and q is from neg_constr
    for p in pos_pos_constr:
        p: Constraint
        ineq_from_p_with_y, list_ineqs_from_p_without_y = p.get_ineq_with_var_y_and_complement(y)
        for q in neg_constr:
            q: Constraint
            ineq_from_q_with_y, list_ineqs_from_q_without_y = q.get_ineq_with_var_y_and_complement(y)

            new_inequality = reduce_two_ineqs(y, ineq_from_p_with_y, ineq_from_q_with_y)

            if type(new_inequality) == bool:
                if new_inequality == True:
                    continue
                else:
                    new_list_of_ineqs = []
            else:
                new_list_of_ineqs = [new_inequality]
            new_list_of_ineqs.extend(list_ineqs_from_p_without_y)
            new_list_of_ineqs.extend(list_ineqs_from_q_without_y)
            red_constr.append(Constraint(new_list_of_ineqs))

    return red_constr


def reduce_two_ineqs(y, ineq_with_pos_y, ineq_with_neg_y):
    p_coeff, p_complementary_body = split_constr_atoms(y, ineq_with_pos_y)
    q_coeff, q_complementary_body = split_constr_atoms(y, ineq_with_neg_y)
    # if p_complementary_body == []:
    #     break
    # if q_complementary_body == []:
    #     continue
    # multiply all coeff in p by q_coeff
    complementary_body = multiply_coefficients_of_atoms(p_complementary_body, q_coeff / p_coeff)
    # take the union of the p and q lists of atoms,
    # excluding the atom corresponding to y (whose coefficient is now 0)
    complementary_body.extend(q_complementary_body)
    # merge any atom duplicates
    complementary_body = collapse_atoms(complementary_body)
    _, ineq_sign_p, constant_p = ineq_with_pos_y.get_ineq_attributes()
    _, ineq_sign_q, constant_q = ineq_with_neg_y.get_ineq_attributes()
    new_ineq_sign = '>' if ineq_sign_p != ineq_sign_q else ineq_sign_p
    new_constant = (constant_p * q_coeff / p_coeff) + constant_q
    if complementary_body != []:
        new_inequality = Inequality(complementary_body, new_ineq_sign, new_constant)
    elif complementary_body == []:
        if new_ineq_sign == '>':
            sat_val = 0 > new_constant
        elif new_ineq_sign == '>=':
            sat_val = 0 >= new_constant
        else:
            raise NotImplementedError
        return sat_val
    else:
        raise Exception
    return new_inequality



def get_pos_neg_pn_x_constr(y: Variable, constraints: List[Constraint]):
    pos_constr, neg_constr, pos_neg_constr = [], [], []
    for constr in constraints:
        # determine if y appears: (i) only pos, (ii) only neg, (iii) both pos and neg in the constr
        if constr.contains_variable_only_positively(y):
            pos_constr.append(constr)
        if constr.contains_variable_only_negatively(y):
            neg_constr.append(constr)
        if constr.contains_variable_both_positively_and_negatively(y):
            pos_neg_constr.append(constr)
    return pos_constr, neg_constr, pos_neg_constr



def get_pos_neg_x_constr(y: Variable, constraints_with_y: List[Constraint]):
    pos_constr, neg_constr = [], []
    for constr in constraints_with_y:
        # first determine if y appears: (i) only pos, (ii) only neg, (iii) both pos and neg in the constr

        # if (iii), discard this constr
        if constr.contains_variable_both_positively_and_negatively(y):
            continue
        # if (i), then add to pos_constr; if (ii), add to neg_constr
        if constr.contains_variable_only_positively(y):
            pos_constr.append(constr)
        if constr.contains_variable_only_negatively(y):
            neg_constr.append(constr)
    return pos_constr, neg_constr


def compute_set_of_constraints_for_variable(x: Variable, prev_x: Variable, normalised_constraints_at_previous_level: List[Constraint], verbose):
    # create two sets starting from constraints_at_previous_level:
    # one containing only the constraints which variable prev_x appears in
    # and its complement
    unnormalised_constraints_without_prev = []
    normalised_constraints_with_prev = []

    for constr in normalised_constraints_at_previous_level:
        if constr.contains_variable(prev_x):
            normalised_constraints_with_prev.append(constr)
        else:
            unnormalised_constraints_without_prev.append(constr)

    # then compute a new set of constraints derived by algebraic manipulation on constraints containing prev_x
    # note that this new set of constraints will not have occurrences of prev_x by construction
    reduced_constr = create_constr_by_reduction(prev_x, normalised_constraints_with_prev)

    # finally, get the union of constraints which do not contain y (unnormalised_constraints_without_prev U reduced_constr)
    unnormalised_constraints_without_prev.extend(reduced_constr)

    # if verbose:
    #     print('\nLEVEL', x.readable())
    #     print('-------------------Constraints at this level:')
    #     for constr in unnormalised_constraints_without_prev:
    #         print(constr.readable())

    return unnormalised_constraints_without_prev


def compute_sets_of_constraints(ordering: List[Variable], constraints: List[Constraint], verbose) -> {Variable: List[Constraint]}:
    print(f' *** ALL CONSTRAINTS ***')
    for constr in constraints:
        print(constr.readable())
    # reverse the ordering:
    ordering = list(reversed(ordering))
    prev_x = ordering[0]

    # add all constraints for the highest ranking variable w.r.t. ordering
    ordered_normalised_constraints = {prev_x: normalise(prev_x, constraints)}
    print(f' *** Normalised Constraints for {prev_x.readable()} ***')
    for constr in constraints:
        print(constr.readable())

    for x in ordering[1:]:
        normalised_constraints_at_previous_level = ordered_normalised_constraints[prev_x]
        set_of_constraints = compute_set_of_constraints_for_variable(x, prev_x, normalised_constraints_at_previous_level, verbose)
        print(f' *** Unnormalised Constraints for {x.readable()} ***')
        for constr in set_of_constraints:
            print(constr.readable())
        ordered_normalised_constraints[x] = normalise(x, set_of_constraints)
        print(f'\n *** Normalised Constraints for {x.readable()} ***')
        for constr in ordered_normalised_constraints[x]:
            print(constr.readable())
        prev_x = x

    # print('-'*80)
    # for x in ordering:
    #     print(f' *** Constraints for {x.readable()} ***')
    #     for i,constr in enumerate(ordered_normalised_constraints[x]):
    #         print(f'constr number {i}')
    #         print(constr.readable())
    #     if len(ordered_normalised_constraints[x]) == 0:
    #         print('empty set')
    #     print('***\n')
    print('-'*80)

    return ordered_normalised_constraints


def main():
    # ordering, constraints = parser.parse_constraints_file('../data/tiny_constraints.txt')
    ordering, constraints = parse_constraints_file('../data/heloc/heloc_constraints.txt')
    # for constr in constraints:
    #     print(constr.readable())
    #     # for elem in constr.inequality_list[-1].body:
    #     #     print('id', elem.get_variable_id())

    print('verbose constr')
    for constr in constraints:
        print(constr.verbose_readable())
    # print(constraints)

    print('compute sets of constraints')
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
    # for var in ordering:
    #     print(var.readable())
    #     set_of_constr = sets_of_constr[var]
    #     for constr in set_of_constr:
    #         print(constr.verbose_readable())
    #     print()

if __name__ == '__main__':
    main()
