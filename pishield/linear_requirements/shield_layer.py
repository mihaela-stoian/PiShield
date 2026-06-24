from typing import List, Union
import torch

from pishield.linear_requirements.classes import Variable, Constraint, Atom
from pishield.linear_requirements.compute_sets_of_constraints import get_pos_neg_x_constr, compute_sets_of_constraints
from pishield.linear_requirements.correct_predictions import get_constr_at_level_x, get_final_x_correction
from pishield.linear_requirements.feature_orderings import set_ordering
from pishield.linear_requirements.parser import parse_constraints_file, split_constraints

INFINITY = torch.inf
EPSILON = 1e-12


class ShieldLayer(torch.nn.Module):
    """
    Differentiable layer that corrects predictions so they satisfy a set of linear inequality
    requirements.

    The correction works variable by variable, following a fixed `ordering`. For each variable x,
    the requirements that bound x are rewritten as a lower bound and an upper bound, each expressed
    as a linear function of the *other* variables. At inference time the variables are corrected one
    at a time in this order: x is clipped into the [lower bound, upper bound] interval implied by the
    requirements, using the already-corrected values of the variables that precede it. Because every
    variable is only ever clipped using values that are already feasible, correcting one variable can
    never break a requirement that was already satisfied, so all requirements hold after the pass.
    """

    def __init__(self, num_variables: int, requirements_filepath: str, ordering_choice: str = 'given'):
        super().__init__()
        self.num_variables = num_variables
        # Read the variable ordering and the list of linear constraints from the requirements file.
        ordering, constraints = parse_constraints_file(requirements_filepath)
        # clustered_constraints = split_constraints(ordering, constraints)
        # Decide the order in which variables will be corrected ('given', 'random' or custom).
        self.ordering = set_ordering(ordering, ordering_choice)
        self.constraints = constraints
        # Group the constraints by the variable they bound (its "level" in the ordering).
        self.sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        # Precompute, per variable, the coefficient matrices encoding its lower/upper bounds.
        self.pos_matrices, self.neg_matrices = self.create_matrices()
        # Restrict the ordering to the variables that actually appear in some constraint.
        self.dense_ordering = self.get_dense_ordering()  # requires self.sets_of_constraints

    def create_matrices(self):
        # This function creates matrices C+ and C- for each variable x_i.
        # C+ (pos) holds the constraints that lower-bound x_i; C- (neg) those that upper-bound it.
        # Note that the column corresponding to x_i in the matrices will be 0s.
        pos_matrices: {Variable: torch.Tensor} = {}
        neg_matrices: {Variable: torch.Tensor} = {}
        for x in self.sets_of_constr:
            x: Variable
            # print(x.id)
            x_constr = get_constr_at_level_x(x, self.sets_of_constr)
            # Split x's constraints by the sign of x: positive occurrences give lower bounds on x,
            # negative occurrences give upper bounds.
            pos_x_constr, neg_x_constr = get_pos_neg_x_constr(x, x_constr)

            pos_matrices[x] = self.create_matrix(x, pos_x_constr, positive_x=True)
            neg_matrices[x] = self.create_matrix(x, neg_x_constr, positive_x=False)
        return pos_matrices, neg_matrices

    def get_dense_ordering(self) -> List[Variable]:
        dense_ordering = []
        for x in self.ordering:
            x_constr = get_constr_at_level_x(x, self.sets_of_constr)
            if len(x_constr) == 0:
                continue
            else:
                dense_ordering.append(x)
        return dense_ordering

    def create_matrix(self, x: Variable, x_constr: List[Constraint], positive_x: bool) -> Union[torch.Tensor, float]:
        # Build one row per constraint, expressing the bound it places on x as a linear function of
        # the other variables (plus a bias term for the constraint's constant). Evaluating a row
        # against a prediction vector yields the value x must be >= (positive_x) or <= (negative_x).
        # With no constraints, x is unbounded on that side: return -inf (lower) or +inf (upper).
        if len(x_constr) == 0:
            return -INFINITY if positive_x else INFINITY

        matrix = torch.zeros((len(x_constr), self.num_variables), dtype=torch.float)
        x_unsigned_coefficients = torch.ones((len(x_constr),), dtype=torch.float)  # bias (i.e. the constraint constant)
        bias = torch.zeros((len(x_constr),), dtype=torch.float)
        for constr_index, constr in enumerate(x_constr):
            constr: Constraint

            is_strict_inequality = True if constr.single_inequality.ineq_sign == '>' else False
            constant = constr.single_inequality.constant
            epsilon = EPSILON if is_strict_inequality else 0.
            bias[constr_index] = constant + epsilon
            complementary_atoms: List[Atom] = constr.get_body_atoms()
            for atom in complementary_atoms:
                atom_id = atom.variable.id
                if atom_id == x.id:
                    x_unsigned_coefficients[constr_index] = atom.coefficient
                    continue
                else:
                    signed_coefficient = atom.get_signed_coefficient()
                    matrix[constr_index, atom_id] = signed_coefficient

        # next, divide by the unsigned coefficients of x:
        matrix = matrix / x_unsigned_coefficients.unsqueeze(-1)  # num constraints that contain x x num variables

        # if x is positive, multiply by -1 the matrix
        if positive_x:
            matrix *= (-1.)

        # add bias (constraint constant)
        bias = bias / x_unsigned_coefficients
        if not positive_x:
            bias *= (-1.)

        matrix = torch.cat([matrix, bias.unsqueeze(1)], dim=1)
        return matrix

    # def __call__(self, preds, *args, **kwargs):
    def __call__(self, preds: torch.Tensor):
        device = preds.device
        N = preds.shape[-1]
        # Append a constant column of 1s so the bias term in each matrix is picked up by the dot product.
        corrected_preds = torch.cat([preds.clone(), torch.ones(preds.shape[0], 1, device=device)], dim=1)
        preds = corrected_preds.clone()

        # Correct one variable at a time, in order. Each variable sees the already-corrected values
        # of the variables processed before it.
        for x in self.dense_ordering:
            pos = x.id

            # Evaluate every bound on x against the current predictions, then reduce across constraints:
            # the tightest lower bound is the max over the positive constraints, the tightest upper
            # bound the min over the negative ones.
            # pos_matrix and neg_matrix have shape: num constraints that contain x x num variables
            pos_matrix = self.apply_matrix(preds.clone(), self.pos_matrices[x], reduction='amax')
            neg_matrix = self.apply_matrix(preds.clone(), self.neg_matrices[x], reduction='amin')

            # Clip x into [lower bound, upper bound]; this makes all of x's constraints satisfied.
            corrected_preds[:, pos] = get_final_x_correction(preds[:, pos], pos_matrix, neg_matrix)
            preds = corrected_preds.clone()
            corrected_preds = preds.clone()
        # Drop the bias column before returning.
        return corrected_preds[:, :N]

    def apply_matrix(self, preds: torch.Tensor, matrix: Union[torch.Tensor, float], reduction='none') -> torch.Tensor:
        if type(matrix) != torch.Tensor:
            return matrix
        else:
            matrix = matrix.to(preds.device)

        B = preds.shape[0]  # batch size
        C = matrix.shape[0]  # num constraints in the current set
        N = matrix.shape[1]  # num variables

        # expand tensors
        preds = preds.unsqueeze(1).expand((B, C, N))
        matrix = matrix.clone().unsqueeze(0).expand((B, C, N))

        result = (preds * matrix).sum(dim=2)
        if reduction == 'amax':
            result = result.amax(dim=1)
        elif reduction == 'amin':
            result = result.amin(dim=1)
        else:
            pass
        return result
