"""Core data structures for QFLRA requirements.

Defines variables, linear inequality atoms, inequalities, disjunctions of
inequalities and constraints, together with methods to inspect, render and evaluate
them on batched predictions.
"""

from typing import List
import torch

from pishield.qflra_requirements.utils_functions import eval_atoms_list

TOLERANCE=1e-2

class Variable():
    """A variable (label or feature) identified by a string name.

    Attributes:
        variable: The string name of the variable, e.g. ``'y_3'``.
        id: The integer id parsed from the trailing part of the name.
    """
    def __init__(self, variable: str):
        """Initialise the variable from its string name.

        Args:
            variable: The variable name, whose trailing integer (after the last
                ``'_'``) becomes its id.
        """
        super().__init__()
        self.variable = variable
        self.id = self.get_variable_id()

    def readable(self):
        """Return the human-readable name of the variable.

        Returns:
            The variable's string name.
        """
        return self.variable

    def get_variable_id(self):
        """Parse the integer id from the variable name.

        Returns:
            The integer following the last ``'_'`` in the variable name.
        """
        id = int(self.variable.split('_')[-1])
        return id


class Atom():
    """A signed linear term (coefficient times a variable) in an inequality body.

    Attributes:
        variable: The :class:`Variable` this atom refers to.
        coefficient: The absolute value of the coefficient.
        positive_sign: Whether the term is added (``True``) or subtracted
            (``False``).
    """
    def __init__(self, variable: Variable, coefficient: float, positive_sign: bool):
        """Initialise the atom.

        Args:
            variable: The :class:`Variable` of this atom.
            coefficient: The coefficient; its absolute value is stored.
            positive_sign: ``True`` if the term is positive, ``False`` if negative.
        """
        super().__init__()
        self.variable = variable
        self.coefficient = abs(coefficient)
        self.positive_sign = positive_sign

    def get_variable_id(self):
        """Return the id of this atom's variable.

        Returns:
            The integer id of the underlying :class:`Variable`.
        """
        return self.variable.get_variable_id()

    def eval(self, x_value):
        """Evaluate the atom at a given variable value.

        Args:
            x_value: The value (possibly batched tensor) of the variable.

        Returns:
            ``x_value`` multiplied by the signed coefficient.
        """
        return x_value * self.get_signed_coefficient()

    def get_signed_coefficient(self):
        """Return the coefficient with its sign applied.

        Returns:
            ``+coefficient`` if the atom is positive, ``-coefficient`` otherwise.
        """
        return self.coefficient if self.positive_sign else -1 * self.coefficient

    def readable(self):
        """Return a human-readable string for the atom, e.g. ``' + 2y_3'``.

        Returns:
            The signed, formatted term as a string.
        """
        readable = ' + ' if self.positive_sign else ' - '
        readable += (f'{self.coefficient:.2f}' if self.coefficient != int(
            self.coefficient) else f'{self.coefficient:.0f}') if self.coefficient != 1 else ''
        readable += self.variable.readable()
        return readable

    def get_atom_attributes(self):
        """Return the atom's defining attributes.

        Returns:
            A tuple ``(variable, coefficient, positive_sign)``.
        """
        return self.variable, self.coefficient, self.positive_sign


class Inequality():
    """A single linear inequality ``body <sign> constant``.

    Attributes:
        ineq_sign: The inequality sign, either ``'>'`` or ``'>='``.
        constant: The right-hand side constant.
        body: List of :class:`Atom` objects forming the left-hand side.
    """
    def __init__(self, body: List[Atom], ineq_sign: str, constant: float):
        """Initialise the inequality.

        Args:
            body: List of :class:`Atom` objects forming the left-hand side.
            ineq_sign: Either ``'>'`` or ``'>='``.
            constant: The right-hand side constant.

        Raises:
            AssertionError: If ``ineq_sign`` is not ``'>'`` or ``'>='``.
        """
        super().__init__()
        assert ineq_sign in ['>', '>=']
        self.ineq_sign = ineq_sign
        self.constant = constant
        self.body = body

    def readable(self):
        """Return a human-readable string for the inequality.

        Returns:
            The rendered inequality, e.g. ``' + y_1 - 2y_2 > 0'``.
        """
        readable_ineq = ''
        for elem in self.body:
            readable_ineq += elem.readable()
        readable_ineq += ' ' + self.ineq_sign + ' ' + str(self.constant)
        return readable_ineq

    def get_body_variables(self):
        """Return the variables appearing in the inequality body.

        Returns:
            A list of :class:`Variable` objects, one per body atom.
        """
        var_list = []
        for atom in self.body:
            var_list.append(atom.variable)
        return var_list

    def get_body_atoms(self):
        """Return the atoms of the inequality body.

        Returns:
            A list of the body :class:`Atom` objects.
        """
        atom_list = []
        for atom in self.body:
            atom_list.append(atom)
        return atom_list

    def get_x_complement_body_atoms(self, x: Variable) -> (List[Atom], Atom, bool):
        """Split the body into the atom of ``x`` and the remaining atoms.

        Args:
            x: The variable to isolate.

        Returns:
            A tuple ``(complementary_atom_list, x_atom, constant, is_strict)`` where
            ``complementary_atom_list`` are the body atoms other than ``x``,
            ``x_atom`` is the single atom of ``x`` (or an empty list if absent),
            ``constant`` is the inequality constant, and ``is_strict`` is ``True``
            for a ``'>'`` inequality.

        Raises:
            AssertionError: If ``x`` appears more than once in the body.
        """
        # given a constraint constr in which variable x appears,
        # return the body of the constraint (i.e. the left-hand side of the inequality)
        # from which x occurrences have been removed
        complementary_atom_list = []
        x_atom_occurrences = []
        for atom in self.body:
            if atom.variable.id != x.id:
                complementary_atom_list.append(atom)
            else:
                x_atom_occurrences.append(atom)
        assert len(x_atom_occurrences) <= 1, "variable {x.id} appears more than one time, function collapse_atoms() from compute_sets_of_constraints should be applied"
        if len(x_atom_occurrences) == 1:
            x_atom_occurrences = x_atom_occurrences[0]
        is_strict_inequality = True if self.ineq_sign == '>' else False
        return complementary_atom_list, x_atom_occurrences, self.constant, is_strict_inequality

    def get_ineq_attributes(self):
        """Return the inequality's defining attributes.

        Returns:
            A tuple ``(body, ineq_sign, constant)``.
        """
        return self.body, self.ineq_sign, self.constant

    def contains_variable(self, x: Variable):
        """Report whether the variable appears anywhere in the body.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` occurs in the body, ``False`` otherwise.
        """
        body_variables = [elem.id for elem in self.get_body_variables()]
        return x.id in body_variables

    def contains_variable_only_positively(self, x: Variable):
        """Report whether the variable appears with a positive sign.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if a positive-sign atom of ``x`` is present.
        """
        body_variables_ids = [elem.variable.id for elem in self.get_body_atoms() if elem.positive_sign]
        return x.id in body_variables_ids

    def contains_variable_only_negatively(self, x: Variable):
        """Report whether the variable appears with a negative sign.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if a negative-sign atom of ``x`` is present.
        """
        body_variables_ids = [elem.variable.id for elem in self.get_body_atoms() if not elem.positive_sign]
        return x.id in body_variables_ids

    def contains_variable_both_positively_and_negatively(self, x: Variable):
        """Report mixed-sign occurrence of a variable.

        A single inequality never contains a variable both positively and
        negatively; this is only possible for a disjunctive inequality.

        Args:
            x: The variable to look for (unused).

        Returns:
            Always ``False``.
        """
        # this can only happen for a disjunctive inequality
        return False

    def get_ineq_with_pos_and_neg_var_y_and_complement(self, x: Variable):
        """Return positive/negative clauses of a variable (disjunction-only).

        Only meaningful for a disjunctive inequality; a single inequality has no such
        split.

        Args:
            x: The variable to look for (unused).

        Returns:
            Always ``None``.
        """
        # this can only happen for a disjunctive inequality: i.e. one clause with pos x, another clause with neg x
        return None

    def get_ineq_with_var_y_and_complement(self, x: Variable):
        """Return this inequality if it contains the variable, else ``None``.

        Args:
            x: The variable to look for.

        Returns:
            ``(self, [])`` if ``x`` occurs in the body (the complement is empty for a
            single inequality), otherwise ``None``.
        """
        if self.contains_variable(x):
            return self, []
        return None

    def check_satisfaction(self, preds: torch.Tensor) -> torch.Tensor:
        """Evaluate satisfaction of the inequality on batched predictions.

        Args:
            preds: Predictions tensor of shape ``(B, D)``.

        Returns:
            A boolean tensor of shape ``(B,)`` that is ``True`` for samples
            satisfying the inequality (within ``TOLERANCE``).

        Raises:
            NotImplementedError: If the inequality sign is unsupported.
        """
        eval_body_value = eval_atoms_list(self.body, preds)
        # note that preds might be batched
        # in general, if preds is of shape BxD, then the result will be of shape Bx1
        eval_body_value = eval_atoms_list(self.body, preds)
        if self.ineq_sign == '>':
            results = eval_body_value > self.constant - TOLERANCE
        elif self.ineq_sign == '>=':
            results = eval_body_value >= self.constant - TOLERANCE
        else:
            raise NotImplementedError
        return results

    def check_satisfaction_all(self, preds: torch.Tensor) -> bool:
        """Report whether all samples satisfy the inequality.

        Args:
            preds: Predictions tensor of shape ``(B, D)``.

        Returns:
            ``True`` if every sample satisfies the inequality.
        """
        return self.check_satisfaction(preds).all()

    def eval_to_bool(self, preds: torch.Tensor) -> bool:
        """Reduce per-sample satisfaction to a single boolean.

        Args:
            preds: Predictions tensor of shape ``(B, D)``.

        Returns:
            ``True`` if every sample satisfies the inequality.
        """
        sat_per_datapoint = self.check_satisfaction(preds)
        bool_eval_result = sat_per_datapoint.all()
        return bool_eval_result

    # def check_satisfaction(self, preds: torch.Tensor, tolerance=1e-4) -> torch.Tensor:
    #     eval_body_value = eval_atoms_list(self.body, preds)
    #     eval_body_value -= self.constant
    #     if self.ineq_sign == '>':
    #         results = eval('eval_body_value + tolerance > 0') | eval('eval_body_value - tolerance > 0')
    #     elif self.ineq_sign == '>=':
    #         results =  eval('eval_body_value + tolerance >= 0') | eval('eval_body_value - tolerance >= 0')
    #     return results #.all()


class DisjunctInequality():
    """A disjunction ('or') of linear inequalities.

    Attributes:
        list_inequalities: The :class:`Inequality` clauses; the disjunction is
            satisfied when at least one clause holds.
    """
    def __init__(self, inequality_list: List[Inequality]):
        """Initialise the disjunction.

        Args:
            inequality_list: The list of :class:`Inequality` clauses.
        """
        self.list_inequalities = inequality_list

    def readable(self):
        """Return a multi-line human-readable string listing each disjunct.

        Returns:
            A string with one ``disjunct i:`` line per clause.
        """
        readable_ineq = ''
        for i,ineq in enumerate(self.list_inequalities):
            readable_ineq += f'disjunct {i}: {ineq.readable()}\n'
        return readable_ineq

    def verbose_readable(self):
        """Return a single-line human-readable string joining clauses with 'or'.

        Returns:
            The clauses rendered and joined by ``' or'``.
        """
        readable_constr = self.list_inequalities[0].readable()
        for ineq in self.list_inequalities[1:]:
            readable_constr += ' or' + ineq.readable()
        return readable_constr

    def contains_variable(self, x: Variable):
        """Report whether the variable appears in any clause.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` occurs in at least one clause.
        """
        for ineq in self.list_inequalities:
            if ineq.contains_variable(x):
                return True
        return False

    def contains_variable_only_positively(self, x: Variable):
        """Report whether the variable occurs only positively across clauses.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` appears positively in some clause and never negatively.
        """
        pos_occurrence = False
        neg_occurrence = False
        for ineq in self.list_inequalities:
            if ineq.contains_variable_only_positively(x):
                pos_occurrence = True
            if ineq.contains_variable_only_negatively(x):
                neg_occurrence = True

        if pos_occurrence and not neg_occurrence:
            return True
        else:
            return False

    def contains_variable_only_negatively(self, x: Variable):
        """Report whether the variable occurs only negatively across clauses.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` appears negatively in some clause and never positively.
        """
        pos_occurrence = False
        neg_occurrence = False
        for ineq in self.list_inequalities:
            if ineq.contains_variable_only_positively(x):
                pos_occurrence = True
            if ineq.contains_variable_only_negatively(x):
                neg_occurrence = True

        if neg_occurrence and not pos_occurrence:
            return True
        else:
            return False

    def contains_variable_both_positively_and_negatively(self, x: Variable):
        """Report whether the variable occurs both positively and negatively.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` appears positively in some clause and negatively in
            another.
        """
        pos_occurrence = False
        neg_occurrence = False
        for ineq in self.list_inequalities:
            if ineq.contains_variable_only_positively(x):
                pos_occurrence = True
            if ineq.contains_variable_only_negatively(x):
                neg_occurrence = True

        if neg_occurrence and pos_occurrence:
            return True
        else:
            return False

    def get_ineq_with_var_y_and_complement(self, x: Variable):
        """Find the clause containing a variable and the remaining clauses.

        Args:
            x: The variable to look for.

        Returns:
            A tuple ``(ineq_with_y, complement)`` where ``ineq_with_y`` is the clause
            containing ``x`` and ``complement`` are the other clauses, or ``None`` if
            no clause contains ``x``.
        """
        complement = []
        ineq_with_y = None
        for ineq in self.list_inequalities:
            if ineq.contains_variable(x):
                ineq_with_y = ineq
            else:
                complement.append(ineq)
        if ineq_with_y is None:
            return None
        else:
            return ineq_with_y, complement

    def get_ineq_with_pos_and_neg_var_y_and_complement(self, x: Variable):
        """Find the positive-x and negative-x clauses and the remaining clauses.

        Args:
            x: The variable to look for.

        Returns:
            A tuple ``(ineq_with_pos_y, ineq_with_neg_y, complement)``: the clause
            where ``x`` appears positively, the clause where it appears negatively,
            and the other clauses.

        Raises:
            NotImplementedError: If either a positive-x or negative-x clause is
                missing.
        """
        complement = []
        ineq_with_pos_y = None
        ineq_with_neg_y = None

        for ineq in self.list_inequalities:
            if ineq.contains_variable(x):
                if ineq.contains_variable_only_positively(x):
                    ineq_with_pos_y = ineq
                elif ineq.contains_variable_only_negatively(x):
                    ineq_with_neg_y = ineq
            else:
                complement.append(ineq)
        if ineq_with_pos_y is None or ineq_with_neg_y is None:
            raise NotImplementedError
        else:
            return ineq_with_pos_y, ineq_with_neg_y, complement

    def get_body_variables(self):
        """Return the variables across all clause bodies.

        Returns:
            A list of :class:`Variable` objects gathered from every clause body.
        """
        var_list = []
        for body_elem in self.body:
            for atom in body_elem:
                var_list.append(atom.variable)
        return var_list

    def get_body_atoms(self):
        """Return the atoms across all clause bodies.

        Returns:
            A flat list of :class:`Atom` objects from every clause body.
        """
        atom_list = []
        for body_elem in self.body:
            for atom in body_elem:
                atom_list.append(atom)
        return atom_list

    def split_ineqs_with_and_without_x(self, x: Variable) -> (List[Inequality], List[Inequality]):
        """Partition the clauses by whether they contain a variable.

        Args:
            x: The variable to look for.

        Returns:
            A tuple ``(ineqs_with_x, ineqs_without_x)`` of clause lists.
        """
        # separate ineqs that contain x from those that do not contain x
        ineqs_with_x = []
        ineqs_without_x = []
        for ineq in self.list_inequalities:
            if ineq.contains_variable(x):
                ineqs_with_x.append(ineq)
            else:
                ineqs_without_x.append(ineq)

        return ineqs_with_x, ineqs_without_x

    def get_x_complement_body_atoms(self, x: Variable, sign_of_x: str) -> (List[Atom], Atom):
        """Select the clause with the requested sign of a variable and split it.

        Picks the clause in which ``x`` occurs with ``sign_of_x`` (handling both the
        mixed-sign and single-occurrence cases) and delegates to that inequality's
        :meth:`Inequality.get_x_complement_body_atoms`.

        Args:
            x: The variable to isolate.
            sign_of_x: Either ``'positive'`` or ``'negative'``.

        Returns:
            The tuple returned by :meth:`Inequality.get_x_complement_body_atoms`
            for the selected clause.

        Raises:
            NotImplementedError: If ``sign_of_x`` is neither ``'positive'`` nor
                ``'negative'``.
        """
        try:
            ineq_with_pos_x, ineq_with_neg_x, complement = self.get_ineq_with_pos_and_neg_var_y_and_complement(x)
        except:
            ineq_with_y, complement = self.get_ineq_with_var_y_and_complement(x)
            ineq_with_y: Inequality
            if sign_of_x == 'positive':
                ineq_with_pos_x = ineq_with_y
                assert ineq_with_pos_x.contains_variable_only_positively(x)
            elif sign_of_x == 'negative':
                ineq_with_neg_x = ineq_with_y
                assert ineq_with_neg_x.contains_variable_only_negatively(x)
            else:
                raise NotImplementedError
        selected_ineq = ineq_with_pos_x if sign_of_x == 'positive' else ineq_with_neg_x
        return selected_ineq.get_x_complement_body_atoms(x)

    def check_satisfaction(self, preds: torch.Tensor) -> torch.Tensor:
        """Evaluate the disjunction on batched predictions.

        A sample satisfies the disjunction if it satisfies at least one clause.

        Args:
            preds: Predictions tensor of shape ``(B, D)``.

        Returns:
            A boolean tensor of shape ``(B,)`` that is ``True`` for samples
            satisfying any clause.
        """
        # returns True if all preds (i.e. all samples) satisfy at least one of the clauses of this DI

        clause_sat = []
        for ineq in self.list_inequalities:
            clause_sat.append(ineq.check_satisfaction(preds).unsqueeze(1))
        # disj res: B x num_ineqs
        clause_sat = torch.cat(clause_sat, dim=1)

        # disj_results: B
        batched_sat_results = clause_sat.any(dim=1)
        # print(preds[~batched_sat_results], 'Samples that do not satisfy any of the clauses in this DI')
        return batched_sat_results

    def check_satisfaction_all(self, preds: torch.Tensor) -> bool:
        """Report whether all samples satisfy the disjunction.

        Args:
            preds: Predictions tensor of shape ``(B, D)``.

        Returns:
            ``True`` if every sample satisfies at least one clause.
        """
        return self.check_satisfaction(preds).all()


class Constraint():
    """A single QFLRA requirement: one inequality or a disjunction of them.

    Most methods delegate to the underlying inequality (or disjunction). When the
    constraint holds a single inequality, ``disjunctive_inequality`` is that
    :class:`Inequality`; otherwise it is a :class:`DisjunctInequality`.

    Attributes:
        list_inequalities: The inequalities making up the constraint.
        disjunctive_inequality: The :class:`DisjunctInequality` (if more than one
            inequality) or the single :class:`Inequality`.
    """
    def __init__(self, inequality_list: List[Inequality]):
        """Initialise the constraint from its inequalities.

        Args:
            inequality_list: The :class:`Inequality` objects; if more than one, they
                form a disjunction.
        """
        super().__init__()
        self.list_inequalities = inequality_list
        self.disjunctive_inequality = DisjunctInequality(inequality_list) if len(inequality_list)>1 else inequality_list[0]

    def readable(self):
        """Return a human-readable string for the constraint.

        Returns:
            The rendered constraint.
        """
        return self.disjunctive_inequality.readable()

    def verbose_readable(self):
        """Return a single-line human-readable string for the constraint.

        Returns:
            The constraint rendered on one line (clauses joined with 'or').
        """
        return self.disjunctive_inequality.verbose_readable()

    def contains_variable(self, x: Variable):
        """Report whether the constraint involves a variable.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` appears in the constraint.
        """
        return self.disjunctive_inequality.contains_variable(x)

    def get_body_atoms(self):
        """Return the atoms of the constraint body.

        Returns:
            A list of :class:`Atom` objects.
        """
        return self.disjunctive_inequality.get_body_atoms()

    def check_satisfaction(self, preds):
        """Report whether all samples satisfy the constraint.

        Args:
            preds: Predictions tensor of shape ``(B, D)``.

        Returns:
            ``True`` if every sample satisfies the constraint.
        """
        return self.disjunctive_inequality.check_satisfaction_all(preds)

    def contains_variable_only_positively(self, x: Variable):
        """Report whether the variable occurs only positively.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` appears only positively in the constraint.
        """
        return self.disjunctive_inequality.contains_variable_only_positively(x)

    def contains_variable_only_negatively(self, x: Variable):
        """Report whether the variable occurs only negatively.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` appears only negatively in the constraint.
        """
        return self.disjunctive_inequality.contains_variable_only_negatively(x)

    def contains_variable_both_positively_and_negatively(self, x: Variable):
        """Report whether the variable occurs both positively and negatively.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` appears both positively and negatively.
        """
        return self.disjunctive_inequality.contains_variable_both_positively_and_negatively(x)

    def get_ineq_with_var_y_and_complement(self, x: Variable):
        """Return the inequality containing a variable and the remaining ones.

        Args:
            x: The variable to look for.

        Returns:
            The result of the underlying
            :meth:`get_ineq_with_var_y_and_complement`.
        """
        return self.disjunctive_inequality.get_ineq_with_var_y_and_complement(x)

    def get_ineq_with_pos_and_neg_var_y_and_complement(self, x: Variable):
        """Return the positive-x and negative-x inequalities and the remainder.

        Args:
            x: The variable to look for.

        Returns:
            The result of the underlying
            :meth:`get_ineq_with_pos_and_neg_var_y_and_complement`.
        """
        return self.disjunctive_inequality.get_ineq_with_pos_and_neg_var_y_and_complement(x)

