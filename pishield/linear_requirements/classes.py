"""Data model for linear requirements.

Defines the building blocks of a linear requirement: a :class:`Variable`, a
signed scaled :class:`Atom` (a coefficient times a variable), an
:class:`Inequality` (a body of atoms compared against a constant), and a
:class:`Constraint` wrapping one inequality. These classes also provide
satisfaction checks against batches of predictions.
"""

from typing import List
import torch

from pishield.linear_requirements.utils import eval_atoms_list

TOLERANCE=1e-2


class Variable():
    """A variable (a label or feature) identified by a string name.

    Attributes:
        variable: The variable's string name (e.g. ``'y_3'``).
        id: The integer id parsed from the trailing part of the name.

    Args:
        variable: The variable's string name, whose trailing token after the
            final underscore is parsed as the integer id.
    """

    def __init__(self, variable: str):
        """Initialize the variable and parse its integer id from the name."""
        super().__init__()
        self.variable = variable
        self.id = self.get_variable_id()

    def readable(self):
        """Return the human-readable string name of the variable.

        Returns:
            The variable's string name.
        """
        return self.variable

    def get_variable_id(self):
        """Parse the integer id from the variable name.

        Returns:
            The integer following the final underscore in the variable name.
        """
        id = int(self.variable.split('_')[-1])
        return id


class Atom():
    """A signed, scaled occurrence of a variable in a linear requirement.

    An atom represents ``(+/-) coefficient * variable``.

    Attributes:
        variable: The :class:`Variable` the atom refers to.
        coefficient: The (unsigned) magnitude of the coefficient.
        positive_sign: ``True`` if the term is added, ``False`` if subtracted.

    Args:
        variable: The :class:`Variable` the atom refers to.
        coefficient: The unsigned coefficient magnitude.
        positive_sign: Whether the term carries a positive sign.
    """

    def __init__(self, variable: Variable, coefficient: float, positive_sign: bool):
        """Initialize the atom from a variable, coefficient and sign."""
        super().__init__()
        self.variable = variable
        self.coefficient = coefficient
        self.positive_sign = positive_sign

    def get_variable_id(self):
        """Return the integer id of the atom's variable.

        Returns:
            The variable's integer id.
        """
        return self.variable.get_variable_id()

    def eval(self, x_value):
        """Evaluate the atom for a given value of its variable.

        Args:
            x_value: The value(s) of the atom's variable.

        Returns:
            ``x_value`` multiplied by the atom's signed coefficient.
        """
        return x_value * self.get_signed_coefficient()

    def get_signed_coefficient(self):
        """Return the coefficient with its sign applied.

        Returns:
            ``+coefficient`` if the atom is positive, ``-coefficient`` otherwise.
        """
        return self.coefficient if self.positive_sign else -1 * self.coefficient

    def readable(self):
        """Return a human-readable string for the atom, e.g. ``' + 2.00y_3'``.

        Returns:
            A string with the sign, coefficient (omitted when 1) and variable.
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
    """A linear inequality: a body of atoms compared against a constant.

    Represents ``sum(body) ineq_sign constant`` (e.g. ``2y_1 - y_2 >= 0``).

    Attributes:
        ineq_sign: The inequality operator, ``'>'`` or ``'>='``.
        constant: The right-hand-side constant.
        body: The list of :class:`Atom` objects on the left-hand side.

    Args:
        body: The atoms forming the left-hand side.
        ineq_sign: The inequality operator (``'>'`` or ``'>='``).
        constant: The right-hand-side constant.
    """

    def __init__(self, body: List[Atom], ineq_sign: str, constant: float):
        """Initialize the inequality from its body, sign and constant."""
        super().__init__()
        self.ineq_sign = ineq_sign
        self.constant = constant
        self.body = body

    def readable(self):
        """Return a human-readable string for the inequality.

        Returns:
            A string combining the readable body atoms, the sign and the
            constant.
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
        """Return the atoms forming the inequality body.

        Returns:
            A list of the body :class:`Atom` objects.
        """
        atom_list = []
        for atom in self.body:
            atom_list.append(atom)
        return atom_list

    def get_x_complement_body_atoms(self, x: Variable) -> (List[Atom], Atom, bool):
        """Split the body into ``x``'s atom and the remaining atoms.

        Given an inequality in which ``x`` appears, separate the single atom over
        ``x`` from the rest of the body.

        Args:
            x: The variable to isolate.

        Returns:
            A tuple ``(complementary_atom_list, x_atom_occurrences, constant,
            is_strict_inequality)`` where ``complementary_atom_list`` are the body
            atoms other than ``x``, ``x_atom_occurrences`` is the single
            :class:`Atom` over ``x`` (or an empty list if ``x`` is absent),
            ``constant`` is the inequality constant and ``is_strict_inequality``
            is ``True`` when the sign is ``'>'``.

        Raises:
            AssertionError: If ``x`` appears in more than one atom (atoms over
                ``x`` should be merged first via ``collapse_atoms``).
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
        """Check whether a variable appears in the inequality body.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` (matched by id) appears in the body.
        """
        body_variables = [elem.id for elem in self.get_body_variables()]
        return x.id in body_variables

    def check_satisfaction(self, preds: torch.Tensor) -> torch.Tensor:
        """Check, per sample, whether predictions satisfy the inequality.

        The body is evaluated against the predictions and compared to the
        constant up to ``TOLERANCE``.

        Args:
            preds: Prediction tensor of shape ``(batch_size, num_variables)``.

        Returns:
            A boolean tensor of shape ``(batch_size,)``, one entry per sample.
        """
        eval_body_value = eval_atoms_list(self.body, preds)
        if self.ineq_sign == '>':
            results = eval('(eval_body_value > self.constant - TOLERANCE) | (eval_body_value > self.constant + TOLERANCE)')
        elif self.ineq_sign == '>=':
            results = eval('(eval_body_value >= self.constant - TOLERANCE) | (eval_body_value >= self.constant + TOLERANCE)')
        # if not results.all():
        #     print('Problem here:', eval_body_value[eval_body_value<=self.constant-TOLERANCE])
        return results #.all()

    def detailed_sat_check(self, preds: torch.Tensor) -> torch.Tensor:
        """Check satisfaction and also return the intermediate evaluation values.

        Args:
            preds: Prediction tensor of shape ``(batch_size, num_variables)``.

        Returns:
            A tuple ``(results, eval_body_value, constant, ineq_sign)`` where
            ``results`` is the per-sample boolean satisfaction tensor,
            ``eval_body_value`` is the evaluated body, and ``constant`` /
            ``ineq_sign`` are the inequality's constant and operator.
        """
        eval_body_value = eval_atoms_list(self.body, preds)
        if self.ineq_sign == '>':
            results = eval('(eval_body_value > self.constant - TOLERANCE) | (eval_body_value > self.constant + TOLERANCE)')
        elif self.ineq_sign == '>=':
            results = eval('(eval_body_value >= self.constant - TOLERANCE) | (eval_body_value >= self.constant + TOLERANCE)')
        return results, eval_body_value, self.constant, self.ineq_sign


class Constraint():
    """A linear requirement, wrapping its inequality (or disjunction thereof).

    Most methods delegate to the first inequality in the list (``single_inequality``).

    Attributes:
        inequality_list: The list of :class:`Inequality` objects making up the
            requirement.
        single_inequality: The first inequality, used as the primary inequality.

    Args:
        inequality_list: The list of inequalities forming the requirement; the
            first element becomes ``single_inequality``.
    """

    def __init__(self, inequality_list: List[Inequality]):
        """Initialize the requirement and select its primary inequality."""
        super().__init__()
        self.inequality_list = inequality_list
        self.single_inequality = self.inequality_list[0]

    def readable(self):
        """Return a human-readable string for the requirement.

        Returns:
            The readable form of the primary inequality.
        """
        readable_constr = self.single_inequality.readable()
        return readable_constr

    def verbose_readable(self):
        """Return a human-readable string for the requirement's first inequality.

        Returns:
            The readable form of the first inequality.
        """
        readable_constr = self.inequality_list[0].readable()
        return readable_constr

    def contains_variable(self, x: Variable):
        """Check whether the requirement involves a given variable.

        Args:
            x: The variable to look for.

        Returns:
            ``True`` if ``x`` appears in the primary inequality.
        """
        return self.single_inequality.contains_variable(x)

    def get_body_atoms(self):
        """Return the atoms of the requirement's primary inequality body.

        Returns:
            A list of :class:`Atom` objects.
        """
        return self.single_inequality.get_body_atoms()

    def check_satisfaction(self, preds):
        """Check whether the whole batch satisfies the requirement.

        Args:
            preds: Prediction tensor of shape ``(batch_size, num_variables)``.

        Returns:
            ``True`` if every sample in the batch satisfies the requirement.
        """
        return self.single_inequality.check_satisfaction(preds).all()  # for the whole batch

    def check_satisfaction_per_sample(self, preds):
        """Check requirement satisfaction for each sample individually.

        Args:
            preds: Prediction tensor of shape ``(batch_size, num_variables)``.

        Returns:
            A boolean tensor of shape ``(batch_size,)``.
        """
        return self.single_inequality.check_satisfaction(preds)

    def detailed_sample_sat_check(self, preds):
        """Return per-sample satisfaction along with intermediate values.

        Args:
            preds: Prediction tensor of shape ``(batch_size, num_variables)``.

        Returns:
            The tuple produced by :meth:`Inequality.detailed_sat_check`.
        """
        return self.single_inequality.detailed_sat_check(preds)
