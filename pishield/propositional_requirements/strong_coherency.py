"""Strong-coherency preprocessing of constraints.

When several constraints share the same head (some asserting it positively, some
negatively), the Shield Layer must correct that head consistently. This module rewrites
such constraint sets so that the positive and negative rules branch on a common body
literal, which guarantees a strongly coherent correction. The shared literal is chosen
as the highest-ranking atom (per the variable ordering) appearing in the rule bodies.
"""

import numpy as np

from pishield.propositional_requirements.constraints_group import ConstraintsGroup, Constraint
from pishield.propositional_requirements.literal import Literal


def get_max_ranking_atom(atoms_list, ranking):
    """Find the atom with the smallest position in the ranking.

    Args:
        atoms_list: The candidate atom indices.
        ranking: An ordered sequence of atoms; earlier means higher priority.

    Returns:
        A tuple ``(atom, rank)`` of the highest-ranked atom and its index in
        ``ranking``, or ``(None, None)`` if ``atoms_list`` is empty.
    """
    ranks = [list(ranking).index(atom) for atom in atoms_list]
    if len(ranks) == 0:
        return None, None
    return atoms_list[np.argmin(ranks)], np.min(ranks)


def create_new_rule(rule, extra_literal, positive):
    """Return a copy of a rule with one extra literal added to its body.

    Args:
        rule: The :class:`Constraint` to extend.
        extra_literal: The atom index of the literal to add.
        positive: The polarity of the added literal.

    Returns:
        A new :class:`Constraint` with the same head and the extended body.
    """
    new_literal = Literal(extra_literal, positive)
    new_rule = Constraint(rule.head, rule.body.union([new_literal]))
    # rule.body = rule.body.union([new_literal])
    return new_rule


def get_max_ranking_eligible_atom_from_sets_of_rules(R, R_other, literal_ranking):
    """Pick the highest-ranked body atom across two rule sets.

    Args:
        R: A list of constraints (e.g. the positive-head rules).
        R_other: Another list of constraints (e.g. the negative-head rules).
        literal_ranking: The ordering used to rank atoms.

    Returns:
        The atom index of the highest-ranked body literal occurring in either set, or
        None if neither set has any body literals.
    """
    all_literals = set([])

    for r in R:
        all_literals = all_literals.union([lit.atom for lit in r.body])
    for r in R_other:
        all_literals = all_literals.union([lit.atom for lit in r.body])

    max_ranking_atom, max_rank = get_max_ranking_atom(list(all_literals), literal_ranking)
    return max_ranking_atom


def extend_rules_set(R, max_ranking_body_literal):
    """Branch every rule on the chosen literal unless it already contains it.

    For each rule that does not already mention ``max_ranking_body_literal``, two new
    rules are produced (with the literal added positively and negatively); rules that
    already contain the literal (in either polarity) are kept unchanged.

    Args:
        R: The list of constraints (all sharing the same head) to extend.
        max_ranking_body_literal: The atom index to branch the rules on.

    Returns:
        A list of the extended constraints.
    """
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
    """Rewrite a head's constraints to guarantee a strongly coherent correction.

    Splits the constraints for one head into those asserting it positively and those
    asserting it negatively, picks the highest-ranked shared body atom, and branches
    both groups on that atom so the head can be corrected consistently.

    Args:
        R_atom: The list of constraints sharing the same head atom.
        literal_ranking: The variable ordering used to choose the branching atom.

    Returns:
        A :class:`ConstraintsGroup` of the rewritten constraints, or None if no
        eligible shared body atom exists.
    """
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
    if max_ranking_body_literal is None:
        return None

    if len(R_atom_plus):
        R_atom_plus = extend_rules_set(R=R_atom_plus, max_ranking_body_literal=max_ranking_body_literal)
    if len(R_atom_minus):
        R_atom_minus = extend_rules_set(R=R_atom_minus, max_ranking_body_literal=max_ranking_body_literal)

    new_R_atom = ConstraintsGroup(R_atom_plus.copy() + R_atom_minus.copy())
    return new_R_atom
