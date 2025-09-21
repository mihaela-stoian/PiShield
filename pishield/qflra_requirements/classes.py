from typing import List
import torch

from pishield.qflra_requirements.utils_functions import eval_atoms_list

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
        self.coefficient = abs(coefficient)
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
        assert ineq_sign in ['>', '>=']
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

    def contains_variable_only_positively(self, x: Variable):
        body_variables_ids = [elem.variable.id for elem in self.get_body_atoms() if elem.positive_sign]
        return x.id in body_variables_ids

    def contains_variable_only_negatively(self, x: Variable):
        body_variables_ids = [elem.variable.id for elem in self.get_body_atoms() if not elem.positive_sign]
        return x.id in body_variables_ids

    def contains_variable_both_positively_and_negatively(self, x: Variable):
        # this can only happen for a disjunctive inequality
        return False

    def get_ineq_with_pos_and_neg_var_y_and_complement(self, x: Variable):
        # this can only happen for a disjunctive inequality: i.e. one clause with pos x, another clause with neg x
        return None

    def get_ineq_with_var_y_and_complement(self, x: Variable):
        if self.contains_variable(x):
            return self, []
        return None

    def check_satisfaction(self, preds: torch.Tensor) -> torch.Tensor:
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
        return self.check_satisfaction(preds).all()

    def eval_to_bool(self, preds: torch.Tensor) -> bool:
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
    def __init__(self, inequality_list: List[Inequality]):
        self.list_inequalities = inequality_list

    def readable(self):
        readable_ineq = ''
        for i,ineq in enumerate(self.list_inequalities):
            readable_ineq += f'disjunct {i}: {ineq.readable()}\n'
        return readable_ineq

    def verbose_readable(self):
        readable_constr = self.list_inequalities[0].readable()
        for ineq in self.list_inequalities[1:]:
            readable_constr += ' or' + ineq.readable()
        return readable_constr

    def contains_variable(self, x: Variable):
        for ineq in self.list_inequalities:
            if ineq.contains_variable(x):
                return True
        return False

    def contains_variable_only_positively(self, x: Variable):
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
        var_list = []
        for body_elem in self.body:
            for atom in body_elem:
                var_list.append(atom.variable)
        return var_list

    def get_body_atoms(self):
        atom_list = []
        for body_elem in self.body:
            for atom in body_elem:
                atom_list.append(atom)
        return atom_list

    def split_ineqs_with_and_without_x(self, x: Variable) -> (List[Inequality], List[Inequality]):
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
        return self.check_satisfaction(preds).all()


class Constraint():
    def __init__(self, inequality_list: List[Inequality]):
        super().__init__()
        self.list_inequalities = inequality_list
        self.disjunctive_inequality = DisjunctInequality(inequality_list) if len(inequality_list)>1 else inequality_list[0]

    def readable(self):
        return self.disjunctive_inequality.readable()

    def verbose_readable(self):
        return self.disjunctive_inequality.verbose_readable()

    def contains_variable(self, x: Variable):
        return self.disjunctive_inequality.contains_variable(x)

    def get_body_atoms(self):
        return self.disjunctive_inequality.get_body_atoms()

    def check_satisfaction(self, preds):
        return self.disjunctive_inequality.check_satisfaction_all(preds)

    def contains_variable_only_positively(self, x: Variable):
        return self.disjunctive_inequality.contains_variable_only_positively(x)

    def contains_variable_only_negatively(self, x: Variable):
        return self.disjunctive_inequality.contains_variable_only_negatively(x)

    def contains_variable_both_positively_and_negatively(self, x: Variable):
        return self.disjunctive_inequality.contains_variable_both_positively_and_negatively(x)

    def get_ineq_with_var_y_and_complement(self, x: Variable):
        return self.disjunctive_inequality.get_ineq_with_var_y_and_complement(x)

    def get_ineq_with_pos_and_neg_var_y_and_complement(self, x: Variable):
        return self.disjunctive_inequality.get_ineq_with_pos_and_neg_var_y_and_complement(x)

