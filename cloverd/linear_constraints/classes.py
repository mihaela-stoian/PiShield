from typing import List
import torch

from cloverd.linear_constraints.utils import eval_atoms_list

TOLERANCE=1e-2


class Variable():
    def __init__(self, variable: str):
        super().__init__()
        self.variable = variable
        self.id = self.get_variable_id()

    def readable(self):
        return self.variable

    def get_variable_id(self):
        id = int(self.variable.split('_')[-1])
        return id


class Atom():
    def __init__(self, variable: Variable, coefficient: float, positive_sign: bool):
        super().__init__()
        self.variable = variable
        self.coefficient = coefficient
        self.positive_sign = positive_sign

    def get_variable_id(self):
        return self.variable.get_variable_id()

    def eval(self, x_value):
        return x_value * self.get_signed_coefficient()

    def get_signed_coefficient(self):
        return self.coefficient if self.positive_sign else -1 * self.coefficient

    def readable(self):
        readable = ' + ' if self.positive_sign else ' - '
        readable += (f'{self.coefficient:.2f}' if self.coefficient != int(
            self.coefficient) else f'{self.coefficient:.0f}') if self.coefficient != 1 else ''
        readable += self.variable.readable()
        return readable

    def get_atom_attributes(self):
        return self.variable, self.coefficient, self.positive_sign


class Inequality():
    def __init__(self, body: List[Atom], ineq_sign: str, constant: float):
        super().__init__()
        self.ineq_sign = ineq_sign
        self.constant = constant
        self.body = body

    def readable(self):
        readable_ineq = ''
        for elem in self.body:
            readable_ineq += elem.readable()
        readable_ineq += ' ' + self.ineq_sign + ' ' + str(self.constant)
        return readable_ineq

    def get_body_variables(self):
        var_list = []
        for atom in self.body:
            var_list.append(atom.variable)
        return var_list

    def get_body_atoms(self):
        atom_list = []
        for atom in self.body:
            atom_list.append(atom)
        return atom_list

    def get_x_complement_body_atoms(self, x: Variable) -> (List[Atom], Atom, bool):
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
        return self.body, self.ineq_sign, self.constant

    def contains_variable(self, x: Variable):
        body_variables = [elem.id for elem in self.get_body_variables()]
        return x.id in body_variables

    def check_satisfaction(self, preds: torch.Tensor) -> torch.Tensor:
        eval_body_value = eval_atoms_list(self.body, preds)
        if self.ineq_sign == '>':
            results = eval('(eval_body_value > self.constant - TOLERANCE) | (eval_body_value > self.constant + TOLERANCE)')
        elif self.ineq_sign == '>=':
            results = eval('(eval_body_value >= self.constant - TOLERANCE) | (eval_body_value >= self.constant + TOLERANCE)')
        # if not results.all():
        #     print('Problem here:', eval_body_value[eval_body_value<=self.constant-TOLERANCE])
        return results #.all()

    def detailed_sat_check(self, preds: torch.Tensor) -> torch.Tensor:
        eval_body_value = eval_atoms_list(self.body, preds)
        if self.ineq_sign == '>':
            results = eval('(eval_body_value > self.constant - TOLERANCE) | (eval_body_value > self.constant + TOLERANCE)')
        elif self.ineq_sign == '>=':
            results = eval('(eval_body_value >= self.constant - TOLERANCE) | (eval_body_value >= self.constant + TOLERANCE)')
        return results, eval_body_value, self.constant, self.ineq_sign


class Constraint():
    def __init__(self, inequality_list: List[Inequality]):
        super().__init__()
        self.inequality_list = inequality_list
        self.single_inequality = self.inequality_list[0]

    def readable(self):
        readable_constr = self.single_inequality.readable()
        return readable_constr

    def verbose_readable(self):
        readable_constr = self.inequality_list[0].readable()
        return readable_constr

    def contains_variable(self, x: Variable):
        return self.single_inequality.contains_variable(x)

    def get_body_atoms(self):
        return self.single_inequality.get_body_atoms()

    def check_satisfaction(self, preds):
        return self.single_inequality.check_satisfaction(preds).all()  # for the whole batch

    def check_satisfaction_per_sample(self, preds):
        return self.single_inequality.check_satisfaction(preds)

    def detailed_sample_sat_check(self, preds):
        return self.single_inequality.detailed_sat_check(preds)
