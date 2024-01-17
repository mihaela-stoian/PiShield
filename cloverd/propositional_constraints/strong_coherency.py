import numpy as np

from cloverd.propositional_constraints.constraints_group import ConstraintsGroup, Constraint
from cloverd.propositional_constraints.literal import Literal


def get_max_ranking_atom(atoms_list, ranking):
    ranks = [list(ranking).index(atom) for atom in atoms_list]
    return atoms_list[np.argmin(ranks)], np.min(ranks)


def create_new_rule(rule, extra_literal, positive):
    new_literal = Literal(extra_literal, positive)
    new_rule = Constraint(rule.head, rule.body.union([new_literal]))
    # rule.body = rule.body.union([new_literal])
    return new_rule


def get_max_ranking_eligible_atom_from_sets_of_rules(R, R_other, literal_ranking):
    all_literals = set([])

    for r in R:
        all_literals = all_literals.union([lit.atom for lit in r.body])
    for r in R_other:
        all_literals = all_literals.union([lit.atom for lit in r.body])

    max_ranking_atom, max_rank = get_max_ranking_atom(list(all_literals), literal_ranking)
    return max_ranking_atom


def extend_rules_set(R, max_ranking_body_literal):
    new_rules_R_hat = set([])
    print("head", R[0].head, "with old rules", len(R))

    for r in R:
        literals_in_r = [lit.atom for lit in r.body]

        # if l is not in body(r), add new rules:
        if max_ranking_body_literal not in literals_in_r:
            new_rules_R_hat.add(create_new_rule(r, max_ranking_body_literal, positive=True))
            new_rules_R_hat.add(create_new_rule(r, max_ranking_body_literal, positive=False))
        else:
            # do the following step for each r, to ensure no r is left out in case no new rules can be added for it!!!
            # so, if r already contains l (pos or neg), then return the old rule:
            new_rules_R_hat.add(r)

    print("head", R[0].head, "with new rules", len(new_rules_R_hat))
    return list(new_rules_R_hat)


def strong_coherency_constraint_preprocessing(R_atom, literal_ranking):
    literal_ranking = list(literal_ranking)

    R_atom_plus, R_atom_minus = [], []
    for constr in R_atom:
        if constr.head.positive:
            R_atom_plus.append(constr)
        else:
            R_atom_minus.append(constr)
    print(R_atom_minus, R_atom_plus)

    # l = max_lambda over literals in R+ and R-
    max_ranking_body_literal = get_max_ranking_eligible_atom_from_sets_of_rules(R_atom_plus, R_atom_minus,
                                                                                literal_ranking)

    if len(R_atom_plus):
        R_atom_plus = extend_rules_set(R=R_atom_plus, max_ranking_body_literal=max_ranking_body_literal)
    if len(R_atom_minus):
        R_atom_minus = extend_rules_set(R=R_atom_minus, max_ranking_body_literal=max_ranking_body_literal)

    new_R_atom = ConstraintsGroup(R_atom_plus.copy() + R_atom_minus.copy())
    return new_R_atom
