"""Parser for linear requirements files.

Reads a requirements file into the constraint data model: the variable ordering
and a list of :class:`Constraint` objects built from the textual linear
inequalities (with support for ``or`` disjunctions and ``neg`` negation).
"""

from typing import List

from pishield.linear_requirements.classes import Variable, Atom, Inequality, Constraint

ALLOWED_BOOL_OPS = ['or', 'neg']
ALLOWED_OPS = ['+', '-', '*', '/']
ALLOWED_INEQ_SIGNS = ['>=', '>']


def neg_postprocess_ineq(ineq: Inequality) -> Inequality:
    """Negate an inequality (logical negation of the linear constraint).

    Negates every atom and flips the inequality sign between ``'>'`` and
    ``'>='``, e.g. ``neg (x1 + x2 - x3 >= 0)`` becomes ``-x1 - x2 + x3 > 0``.

    Args:
        ineq: The inequality to negate.

    Returns:
        A new :class:`Inequality` representing the negation.
    """
    # equivalent to negating the sign of the inequality and the multiplying by -1
    # e.g. neg x1+x2-x3 >=0 becomes -x1-x2+x3 > 0
    atoms_list, ineq_sign, constant = ineq.get_ineq_attributes()

    # negate all atoms in the inequality
    neg_atoms_list = []
    for atom in atoms_list:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        neg_atom = Atom(variable, coefficient, not positive_sign)
        neg_atoms_list.append(neg_atom)

    # change the sign of the inequality
    neg_ineq_sign = '>' if ineq_sign == '>=' else '>='
    neg_ineq = Inequality(neg_atoms_list, neg_ineq_sign, constant)
    return neg_ineq


def parse_atom(x):
    """Parse a single textual atom into an :class:`Atom`.

    Args:
        x: The atom string, e.g. ``'y1'``, ``'-2y2'``, ``'-y23'`` or ``'6y3'``.
            A missing coefficient defaults to 1; a leading ``'-'`` sets a negative
            sign.

    Returns:
        The parsed :class:`Atom`, or ``None`` if the string is empty.
    """
    x = x.strip().replace(" ", "")
    if x == '':
        return None
    positive_sign = False if '-' in x else True

    if '-' in x or '+' in x:
        x = x[1:]

    coefficient, var_id = x.split('y')
    coefficient = float(coefficient) if coefficient != '' else 1
    var = Variable('y' + var_id)

    atom = Atom(var, coefficient, positive_sign)
    return atom


def parse_inequality(inequality):
    """Parse a textual linear inequality into an :class:`Inequality`.

    Args:
        inequality: The inequality string, e.g. ``'y1-2y2>0'`` or ``'y1+y2>=0'``.
            The body atoms are split on the ``+``, ``-``, ``*`` and ``/``
            operators and the right-hand side is read as a constant.

    Returns:
        The parsed :class:`Inequality`.
    """
    ineq_sign = None
    inequality = inequality.strip()
    for sign in ALLOWED_INEQ_SIGNS:
        if sign in inequality:
            ineq_sign = sign
            break
    body, constant = inequality.split(ineq_sign)
    constant = float(constant)

    atoms_list = []
    constructed_atom = ''
    for elem in body:
        if elem in ALLOWED_OPS:
            if constructed_atom == '':
                constructed_atom += elem
            else:
                atom = parse_atom(constructed_atom)
                if atom is not None:
                    atoms_list.append(atom)
                constructed_atom = elem
        else:
            constructed_atom += elem
    atom = parse_atom(constructed_atom)
    if atom is not None:
        atoms_list.append(atom)

    ineq = Inequality(atoms_list, ineq_sign, constant)
    return ineq


def parse_constraint(constr) -> Constraint:
    """Parse one textual requirement line into a :class:`Constraint`.

    Splits the line on ``or`` into disjunct inequalities, applying ``neg``
    negation where present, and wraps the resulting inequalities in a constraint.

    Args:
        constr: The requirement line, e.g. ``'y1>0 or neg y2>=0'``.

    Returns:
        The parsed :class:`Constraint`.
    """
    ineqs = []
    neg_postprocess = False

    disjunct_inequalities = constr.split('or')
    for ineq in disjunct_inequalities:
        if 'neg' in ineq:
            ineq = ineq.split('neg')[-1]
            neg_postprocess = True
        ineq = parse_inequality(ineq)
        if neg_postprocess:
            ineq = neg_postprocess_ineq(ineq)
            neg_postprocess = False
        ineqs.append(ineq)

    constr = Constraint(ineqs)
    return constr


def parse_constraints_file(file: str) -> (List[Variable], List[Constraint]):
    """Parse a requirements file into a variable ordering and requirements.

    The file must contain an ``ordering`` line listing the variables, followed by
    one requirement per line.

    Args:
        file: Path to the requirements file.

    Returns:
        A tuple ``(ordering, constraints)`` of the parsed variable ordering and
        the list of :class:`Constraint` objects.

    Raises:
        Exception: If the file does not provide a variable ordering.
    """
    f = open(file, 'r')
    constraints = []
    ordering = []
    for line in f:
        line = line.strip()
        if 'ordering' in line:
            str_ordering = line.split('ordering ')[-1].split()
            for elem in str_ordering:
                ordering.append(Variable(elem.replace(" ", ""))) # remove all whitespace from variable string
            continue
        constraints.append(parse_constraint(line))
    if len(ordering) == 0:
        raise Exception('An ordering of the variables must be provided!')
    return ordering, constraints


def split_constraints(ordering: List[Variable], constraints: List[Constraint]):
    r"""Cluster requirements into variable-disjoint groups.

    Splits a list of requirements into groups ``Gi`` of requirements such that
    ``Vars(Gi) \intersect Vars(Gj) = null`` for distinct groups, i.e. requirements
    are grouped together whenever they share any variable (transitively).

    Args:
        ordering: The variable ordering, used to order each group's variables.
        constraints: The requirements to cluster.

    Returns:
        A tuple ``(clustered_orderings, clustered_constraints)`` of lists, where
        each entry is, respectively, the variable ordering and the requirement
        list of one variable-disjoint cluster.
    """
    constr_vars = {}
    for i,constr in enumerate(constraints):
        vars_in_constr = constr.single_inequality.get_body_variables()
        constr_vars[i] = set([var.id for var in vars_in_constr])

    clustered_constraints_ids = list(range(len(constraints)))
    for constr_i in range(len(constraints)):
        vars_in_constr_i = constr_vars[constr_i]
        for constr_j in range(constr_i+1, len(constraints)):

            vars_in_constr_j = constr_vars[constr_j]
            if vars_in_constr_i.intersection(vars_in_constr_j):
                clustered_constraints_ids[constr_j] = clustered_constraints_ids[constr_i]

    clustered_constraints = {id:[] for id in set(clustered_constraints_ids)}
    clustered_orderings = {id:[] for id in set(clustered_constraints_ids)}
    var_ordering = [var.id for var in ordering]
    for cluster_id in clustered_constraints:
        ids_cluster_components = [id for id, group_id in enumerate(clustered_constraints_ids) if group_id == cluster_id]
        clustered_constraints[cluster_id] = [constraints[cluster_component] for cluster_component in ids_cluster_components]

        cluster_ordering = set.union(*[constr_vars[constr_i] for constr_i in ids_cluster_components])
        clustered_orderings[cluster_id] = [var for var in var_ordering if var in cluster_ordering]

    clustered_constraints = list(clustered_constraints.values())
    clustered_orderings = list(clustered_orderings.values())
    return clustered_orderings, clustered_constraints

