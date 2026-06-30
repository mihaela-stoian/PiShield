"""Differentiable correction of predictions against QFLRA requirements.

Given precomputed per-variable constraint sets, this module derives lower/upper
bounds for each variable and clips predictions into the feasible region implied by
the (possibly disjunctive) constraints, processing variables in the given ordering.
"""

import time
from typing import List, Tuple
import torch

from pishield.qflra_requirements.classes import Variable, Constraint
from pishield.qflra_requirements.utils_functions import eval_atoms_list, check_all_constraints_are_sat, \
    compute_sat_stats, any_disjunctions_in_constraint_set, get_samples_violating_constraints
from pishield.qflra_requirements.compute_sets_of_constraints import compute_sets_of_constraints, get_pos_neg_pn_x_constr
from pishield.qflra_requirements.parser import parse_constraints_file
from pishield.qflra_requirements.feature_orderings import set_random_ordering
import numpy as np

INFINITY = torch.inf
# A large finite value used to replace +/-inf in the corrected predictions (see get_final_x_correction).
# 1e16 does not fit in int32, so we use a float sentinel matching the magnitude of the predictions.
INFINITY_NP = torch.tensor(1e16)


def get_constr_at_level_x(x, sets_of_constr):
    """Look up the constraint set associated with a variable.

    Args:
        x: The variable whose constraints are requested.
        sets_of_constr: Mapping from :class:`Variable` to its list of constraints.

    Returns:
        The list of :class:`Constraint` objects for the variable with the same id as
        ``x``, or ``None`` if no matching variable is present.
    """
    for var in sets_of_constr:
        if var.id == x.id:
            return sets_of_constr[var]



def get_lb_from_one_constraint(x: Variable, constraint: Constraint, preds: torch.Tensor, epsilon=1e-12):
    """Compute the per-sample lower bound on a variable from one constraint.

    The constraint (in which ``x`` appears positively) is rearranged into the form
    ``x >= ...`` and evaluated on the batch. A small ``epsilon`` is added for strict
    inequalities.

    Args:
        x: The variable to bound.
        constraint: A constraint that bounds ``x`` from below.
        preds: Predictions tensor of shape ``(B, D)``.
        epsilon: Slack added for strict ('>') inequalities.

    Returns:
        A tensor of shape ``(B,)`` with the lower bound on ``x`` per sample.
    """
    mask_sat_batch_elem = None  # NOTE: the mask should be reset when a new constraint is considered, regardless of its type (ineq or disj of ineq!)
    if len(constraint.list_inequalities) == 1:
        complement_body_atoms, x_atom, constr_constant, is_strict_inequality = constraint.disjunctive_inequality.get_x_complement_body_atoms(x)  #   this gets the get_x_complement_body_atoms from Ineq class
    else:
        complement_body_atoms, x_atom, constr_constant, is_strict_inequality = constraint.disjunctive_inequality.get_x_complement_body_atoms(x, sign_of_x='positive')
        # if x_atom is None: # this is not possible as the method is called on constr that contain x: either pos, neg or both pos and neg
        #     continue

    # if x is y1 and inequality is -y1>0, then add 0+bias to dependency_complements
    if len(complement_body_atoms) == 0:
        eval_body = torch.zeros(preds.shape[0])  # shape (B,)
    else:
        # evaluate the body of constr, after eliminating x occurrences from it
        eval_body = eval_atoms_list(complement_body_atoms, preds)  # shape (B,)

    # rewrite   ax + B >= c
    # to        x >= -B/a + c/a
    # to get lb = c*(1/a) - B*(1/a)
    lb = constr_constant * (1. / x_atom.coefficient) - eval_body * (1. / x_atom.coefficient)

    # then add/subtract epsilon if the ineq is strict: +epsilon if positive x, -epsilon otherwise
    if is_strict_inequality:
        lb += epsilon

    # if the constr is a disj of inqs, mask out the batch elements for which mask_sat_batch_elem is True
    if mask_sat_batch_elem is not None:
        lb[mask_sat_batch_elem] += -INFINITY
    return lb


def get_ub_from_one_constraint(x: Variable, constraint: Constraint, preds: torch.Tensor, epsilon=1e-12):
    """Compute the per-sample upper bound on a variable from one constraint.

    The constraint (in which ``x`` appears negatively) is rearranged into the form
    ``x <= ...`` and evaluated on the batch. A small ``epsilon`` is subtracted for
    strict inequalities.

    Args:
        x: The variable to bound.
        constraint: A constraint that bounds ``x`` from above.
        preds: Predictions tensor of shape ``(B, D)``.
        epsilon: Slack subtracted for strict ('>') inequalities.

    Returns:
        A tensor of shape ``(B,)`` with the upper bound on ``x`` per sample, or
        ``None`` if ``x`` does not appear in the constraint with the expected sign.
    """
    mask_sat_batch_elem = None  # NOTE: the mask should be reset when a new constraint is considered, regardless of its type (ineq or disj of ineq!)
    if len(constraint.list_inequalities) == 1:
        complement_body_atoms, x_atom, constr_constant, is_strict_inequality = constraint.disjunctive_inequality.get_x_complement_body_atoms(x)  #  this gets the get_x_complement_body_atoms from Ineq class
    else:
        complement_body_atoms, x_atom, constr_constant, is_strict_inequality = constraint.disjunctive_inequality.get_x_complement_body_atoms(x, sign_of_x='negative')
        if x_atom is None: # this is not possible as the method is called on constr that contain x: either pos, neg or both pos and neg
            return None

    # if x is y1 and inequality is -y1>0, then add 0+bias to dependency_complements
    if len(complement_body_atoms) == 0:
        eval_body = torch.zeros(preds.shape[0])  # shape (B,)
    else:
        # evaluate the body of constr, after eliminating x occurrences from it
        eval_body = eval_atoms_list(complement_body_atoms, preds)  # shape (B,)

    # rewrite   -ax + B >= c        equiv. to ax - B <= -c
    # to        -x >= -B/a + c/a    equiv. to x <= B/a - c/a
    # to get ub = B*(1/a) - c*(1/a)
    ub = eval_body * (1. / x_atom.coefficient) - constr_constant * (1. / x_atom.coefficient)

    # then add/subtract epsilon if the ineq is strict: +epsilon if positive x, -epsilon otherwise
    if is_strict_inequality:
        ub -= epsilon

    # if the constr is a disj of inqs, mask out the batch elements for which mask_sat_batch_elem is True
    if mask_sat_batch_elem is not None:
        ub[mask_sat_batch_elem] += INFINITY
    return ub


def get_lower_bounds(x: Variable, x_constraints: List[Constraint], preds: torch.Tensor, epsilon=1e-12) -> torch.Tensor:
    """Collect the lower bounds on a variable from all relevant constraints.

    Args:
        x: The variable to bound.
        x_constraints: Constraints that bound ``x`` from below.
        preds: Predictions tensor of shape ``(B, D)``.
        epsilon: Slack passed through to :func:`get_lb_from_one_constraint`.

    Returns:
        A tensor of shape ``(B, M)`` stacking the ``M`` lower bounds per sample, or
        ``-INFINITY`` when there are no constraints.
    """
    if len(x_constraints) == 0:
        return -INFINITY

    lbs = [] # size: num_atom_dependencies x B (i.e. B=batch size)
    for constr in x_constraints:
        lb = get_lb_from_one_constraint(x, constr, preds, epsilon)
        lbs.append(lb)

    if len(lbs) > 1:
        lbs = torch.stack(lbs, dim=1)
    elif len(lbs) == 1:
        lbs = lbs[0].unsqueeze(1)
    else:
        # if no lb is found, return -inf as lower bound (i.e., x>-inf)
        return -INFINITY
    return lbs


def get_upper_bounds(x: Variable, x_constraints: List[Constraint], preds: torch.Tensor, epsilon=1e-12) -> torch.Tensor:
    """Collect the upper bounds on a variable from all relevant constraints.

    Args:
        x: The variable to bound.
        x_constraints: Constraints that bound ``x`` from above.
        preds: Predictions tensor of shape ``(B, D)``.
        epsilon: Slack passed through to :func:`get_ub_from_one_constraint`.

    Returns:
        A tensor of shape ``(B, M)`` stacking the ``M`` upper bounds per sample, or
        ``INFINITY`` when there are no constraints (or none yield a bound).
    """
    if len(x_constraints) == 0:
        return INFINITY

    ubs = []  # size: num_atom_dependencies x B (i.e. B=batch size)
    for constr in x_constraints:
        ub = get_ub_from_one_constraint(x, constr, preds, epsilon)
        if ub is None:
            continue
        ubs.append(ub)

    if len(ubs) > 1:
        ubs = torch.stack(ubs, dim=1)
    elif len(ubs) == 1:
        ubs = ubs[0].unsqueeze(1)
    else:
        # if no ub is found, return +inf as upper bound (i.e., x<inf equiv. to -x > -inf)
        return INFINITY
    return ubs



def get_final_x_correction(initial_x_val: torch.Tensor, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> torch.Tensor:
    """Clip a variable's values into the feasible interval given its bounds.

    The initial value is raised to the lower bound and then lowered to the upper
    bound (per sample). Infinite bounds are treated as no constraint. Used for the
    purely conjunctive (non-disjunctive) case.

    Args:
        initial_x_val: Tensor of shape ``(B,)`` with the current value of ``x``.
        lower_bounds: Greatest lower bound per sample, or a non-tensor sentinel when
            there is no lower bound.
        upper_bounds: Least upper bound per sample, or a non-tensor sentinel when
            there is no upper bound.

    Returns:
        A tensor of shape ``(B,)`` with the corrected, feasible values of ``x``.
    """

    if type(lower_bounds) is not torch.Tensor:
        lb_constrained_result = initial_x_val
    else:
        # print(initial_x_val, pos_x_corrected)
        lower_bounds = lower_bounds.where(~(lower_bounds == INFINITY), initial_x_val)
        # keep_initial_pos_mask = pos_x_corrected.isinf()
        # pos_x_corrected1 = pos_x_corrected.clone()
        # pos_x_corrected2 = pos_x_corrected.clone()
        # pos_x_corrected3 = pos_x_corrected.clone()
        # pos_x_corrected1[keep_initial_pos_mask] = initial_x_val[keep_initial_pos_mask]
        # pos_x_corrected2 = torch.where(pos_x_corrected == INFINITY, initial_x_val, pos_x_corrected)
        # pos_x_corrected3 = pos_x_corrected.where(~(pos_x_corrected == INFINITY), initial_x_val)
        #
        # assert (pos_x_corrected1 == pos_x_corrected2).all(), (pos_x_corrected1, pos_x_corrected2, pos_x_corrected)
        # assert (pos_x_corrected1 == pos_x_corrected3).all(), (pos_x_corrected1, pos_x_corrected3, pos_x_corrected)
        lb_constrained_result = torch.cat([initial_x_val.unsqueeze(1), lower_bounds.unsqueeze(1)], dim=1).amax(dim=1)

    if type(upper_bounds) is not torch.Tensor:
        ub_constrained_result = lb_constrained_result
    else:
        # keep_initial_neg_mask = neg_x_corrected.isinf()
        # neg_x_corrected[keep_initial_neg_mask] = initial_x_val[keep_initial_neg_mask]
        upper_bounds = upper_bounds.where(~(upper_bounds == INFINITY), initial_x_val)
        ub_constrained_result = torch.cat([lb_constrained_result.unsqueeze(1), upper_bounds.unsqueeze(1)], dim=1).amin(dim=1)

    final_constrained_result = ub_constrained_result
    return final_constrained_result


def get_right_bounds(x: Variable, partially_corrected_preds: torch.Tensor, constraints_at_level_x: List[Constraint]) -> torch.Tensor:
    """Compute the right (lower) bounds on a variable.

    Right bounds are the lower bounds on ``x`` (values ``x`` must lie to the right
    of), obtained from constraints where ``x`` appears positively.

    Args:
        x: The variable to bound.
        partially_corrected_preds: Predictions tensor of shape ``(B, D)`` with
            corrections from previously processed variables.
        constraints_at_level_x: Constraints bounding ``x`` from below.

    Returns:
        The lower bounds tensor (or sentinel) returned by :func:`get_lower_bounds`.
    """
    all_right_bounds = get_lower_bounds(x, constraints_at_level_x, partially_corrected_preds)
    return all_right_bounds


def get_left_bounds(x: Variable, partially_corrected_preds: torch.Tensor, constraints_at_level_x: List[Constraint]) -> torch.Tensor:
    """Compute the left (upper) bounds on a variable.

    Left bounds are the upper bounds on ``x`` (values ``x`` must lie to the left
    of), obtained from constraints where ``x`` appears negatively.

    Args:
        x: The variable to bound.
        partially_corrected_preds: Predictions tensor of shape ``(B, D)`` with
            corrections from previously processed variables.
        constraints_at_level_x: Constraints bounding ``x`` from above.

    Returns:
        The upper bounds tensor (or sentinel) returned by :func:`get_upper_bounds`.
    """
    all_left_bounds = get_upper_bounds(x, constraints_at_level_x, partially_corrected_preds)
    return all_left_bounds


def get_closest_left_bound(x, partially_corrected_preds_for_x, left_bounds, right_bounds, constraints_at_level_x):
    """Pick the feasible left (upper) bound closest to a variable's value.

    Among the candidate left bounds, those that would still violate some constraint
    are discarded; of the remaining ones, the bound closest to the current value of
    ``x`` is returned (per sample).

    Args:
        x: The variable to bound.
        partially_corrected_preds_for_x: Predictions tensor of shape ``(B, D)``.
        left_bounds: Candidate left bounds tensor of shape ``(B, M)``, or a non-tensor
            sentinel (returned as-is).
        right_bounds: Unused in this routine (kept for signature symmetry).
        constraints_at_level_x: Constraints used to test feasibility of each bound.

    Returns:
        A tensor of shape ``(B,)`` with the closest feasible left bound per sample,
        or the ``left_bounds`` sentinel when it is not a tensor.
    """
    if type(left_bounds) is not torch.Tensor:
        return left_bounds
    else:
        mask_samples_violating_req = []
        initial_x_val = partially_corrected_preds_for_x[:, x.id].clone()

        for b in range(0, left_bounds.shape[-1]):
            preds_with_left_bounds_for_x = partially_corrected_preds_for_x.clone()
            preds_with_left_bounds_for_x[:, x.id] = left_bounds[:, b]

            mask_samples_violating_req_for_bound_b = get_samples_violating_constraints(constraints_at_level_x, preds_with_left_bounds_for_x)
            mask_samples_violating_req.append(mask_samples_violating_req_for_bound_b)

        mask_samples_violating_req = torch.stack(mask_samples_violating_req, dim=1)
        valid_left_bounds = torch.where(~mask_samples_violating_req, left_bounds, torch.inf)
        distance_of_bounds_from_x = torch.abs(valid_left_bounds - initial_x_val.unsqueeze(1))
        mask_bounds_of_minimum_distance = distance_of_bounds_from_x == distance_of_bounds_from_x.amin(dim=1, keepdim=True)

        closest_left_bound_wrt_x = torch.where(mask_bounds_of_minimum_distance, valid_left_bounds, torch.inf).amin(dim=1)
        return closest_left_bound_wrt_x


def get_closest_right_bound(x, partially_corrected_preds_for_x, left_bounds, right_bounds, constraints_at_level_x):
    """Pick the feasible right (lower) bound closest to a variable's value.

    Among the candidate right bounds, those that would still violate some constraint
    are discarded; of the remaining ones, the bound closest to the current value of
    ``x`` is returned (per sample).

    Args:
        x: The variable to bound.
        partially_corrected_preds_for_x: Predictions tensor of shape ``(B, D)``.
        left_bounds: Unused in this routine (kept for signature symmetry).
        right_bounds: Candidate right bounds tensor of shape ``(B, M)``, or a
            non-tensor sentinel (returned as-is).
        constraints_at_level_x: Constraints used to test feasibility of each bound.

    Returns:
        A tensor of shape ``(B,)`` with the closest feasible right bound per sample,
        or the ``right_bounds`` sentinel when it is not a tensor.
    """
    if type(right_bounds) is not torch.Tensor:
        return right_bounds
    else:
        mask_samples_violating_req = []
        initial_x_val = partially_corrected_preds_for_x[:, x.id].clone()

        for b in range(0, right_bounds.shape[-1]):
            preds_with_right_bounds_for_x = partially_corrected_preds_for_x.clone()
            preds_with_right_bounds_for_x[:, x.id] = right_bounds[:, b]

            mask_samples_violating_req_for_bound_b = get_samples_violating_constraints(constraints_at_level_x, preds_with_right_bounds_for_x)
            mask_samples_violating_req.append(mask_samples_violating_req_for_bound_b)

        mask_samples_violating_req = torch.stack(mask_samples_violating_req, dim=1)
        valid_right_bounds = torch.where(~mask_samples_violating_req, right_bounds, torch.inf)
        distance_of_bounds_from_x = torch.abs(valid_right_bounds - initial_x_val.unsqueeze(1))
        mask_bounds_of_minimum_distance = distance_of_bounds_from_x == distance_of_bounds_from_x.amin(dim=1, keepdim=True)

        closest_right_bound_wrt_x = torch.where(mask_bounds_of_minimum_distance, valid_right_bounds, torch.inf).amin(dim=1) # shape Bx1
        return closest_right_bound_wrt_x


def get_closest_left_bound_wv(x, partially_corrected_preds_for_x, left_bounds, right_bounds, constraints_at_level_x):
    """Pick the smallest feasible left bound greater than the closest right bound.

    Alternative bound-selection variant that first finds the greatest feasible right
    bound not exceeding ``x``, then returns the smallest left bound strictly above
    it. Note: this variant does not treat vacuous constraints.

    Args:
        x: The variable to bound.
        partially_corrected_preds_for_x: Predictions tensor of shape ``(B, D)``.
        left_bounds: Candidate left bounds tensor of shape ``(B, M)``, or a non-tensor
            sentinel.
        right_bounds: Candidate right bounds tensor of shape ``(B, M)``, or a
            non-tensor sentinel.
        constraints_at_level_x: Constraints used to test feasibility of each bound.

    Returns:
        A tensor of shape ``(B,)`` (or the ``left_bounds`` sentinel) with the
        selected left bound per sample.
    """
    # does not treat vacuous constraints!
    initial_x_val = partially_corrected_preds_for_x[:, x.id].clone()

    if type(right_bounds) is not torch.Tensor:
        greatest_right_bound_le_x = right_bounds
    else:

        mask_samples_violating_req = []
        for b in range(0, right_bounds.shape[-1]):
            preds_with_right_bounds_for_x = partially_corrected_preds_for_x.clone()
            preds_with_right_bounds_for_x[:, x.id] = right_bounds[:, b]

            mask_samples_violating_req_for_bound_b = get_samples_violating_constraints(constraints_at_level_x, preds_with_right_bounds_for_x)
            mask_samples_violating_req.append(mask_samples_violating_req_for_bound_b)

        mask_samples_violating_req = torch.stack(mask_samples_violating_req, dim=1)
        greatest_right_bound_le_x = torch.where((right_bounds < initial_x_val.unsqueeze(1)) * ~mask_samples_violating_req, right_bounds, -torch.inf).amax(dim=1)  # shape Bx1

    if type(greatest_right_bound_le_x) is torch.Tensor:
        greatest_right_bound_le_x = greatest_right_bound_le_x.unsqueeze(dim=1)
    smallest_left_bound_gt_r = left_bounds if type(left_bounds) is not torch.Tensor else torch.where(left_bounds > greatest_right_bound_le_x, left_bounds, torch.inf).amin(dim=1)  # shape Bx1
    return smallest_left_bound_gt_r


def get_closest_right_bound_wv(x, partially_corrected_preds_for_x, left_bounds, right_bounds, constraints_at_level_x):
    """Pick the greatest feasible right bound less than the closest left bound.

    Alternative bound-selection variant that first finds the smallest feasible left
    bound strictly above ``x``, then returns the greatest right bound below it. Note:
    this variant does not treat vacuous constraints.

    Args:
        x: The variable to bound.
        partially_corrected_preds_for_x: Predictions tensor of shape ``(B, D)``.
        left_bounds: Candidate left bounds tensor of shape ``(B, M)``, or a non-tensor
            sentinel.
        right_bounds: Candidate right bounds tensor of shape ``(B, M)``, or a
            non-tensor sentinel.
        constraints_at_level_x: Constraints used to test feasibility of each bound.

    Returns:
        A tensor of shape ``(B,)`` (or the ``right_bounds`` sentinel) with the
        selected right bound per sample.
    """
    # does not treat vacuous constraints!
    initial_x_val = partially_corrected_preds_for_x[:, x.id].clone()

    if type(left_bounds) is not torch.Tensor:
        smallest_left_bound_gt_x = left_bounds
    else:

        mask_samples_violating_req = []
        for b in range(0, left_bounds.shape[-1]):
            preds_with_left_bounds_for_x = partially_corrected_preds_for_x.clone()
            preds_with_left_bounds_for_x[:, x.id] = left_bounds[:, b]

            mask_samples_violating_req_for_bound_b = get_samples_violating_constraints(constraints_at_level_x, preds_with_left_bounds_for_x)
            mask_samples_violating_req.append(mask_samples_violating_req_for_bound_b)

        mask_samples_violating_req = torch.stack(mask_samples_violating_req, dim=1)
        smallest_left_bound_gt_x = torch.where((left_bounds > initial_x_val.unsqueeze(1)) * ~mask_samples_violating_req, left_bounds, torch.inf).amin(dim=1)  # shape Bx1

    if type(smallest_left_bound_gt_x) is torch.Tensor:
        smallest_left_bound_gt_x = smallest_left_bound_gt_x.unsqueeze(dim=1)
    greatest_right_bound_le_l = right_bounds if type(right_bounds) is not torch.Tensor else torch.where(right_bounds < smallest_left_bound_gt_x, right_bounds, -torch.inf).amax(dim=1)  # shape Bx1
    return greatest_right_bound_le_l


def get_closest_left_and_right_bounds(x: Variable, partially_corrected_preds_for_x: torch.Tensor, constraints_at_level_x: List[Constraint]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the closest feasible left and right bounds for a variable.

    Splits the constraints by the sign of ``x``, computes the candidate left and
    right bounds, then selects the feasible bound closest to ``x`` on each side.

    Args:
        x: The variable to bound.
        partially_corrected_preds_for_x: Predictions tensor of shape ``(B, D)``.
        constraints_at_level_x: Constraints bounding ``x`` at its level.

    Returns:
        A tuple ``(closest_left_bound, closest_right_bound)`` of per-sample bounds.
    """
    pos_x_constr, neg_x_constr, pos_neg_x_constr = get_pos_neg_pn_x_constr(x, constraints_at_level_x)
    constr_for_right_bounds = pos_x_constr + pos_neg_x_constr
    constr_for_left_bounds = neg_x_constr + pos_neg_x_constr

    left_bounds = get_left_bounds(x, partially_corrected_preds_for_x, constr_for_left_bounds)  # shape BxM, M is num of constraints
    right_bounds = get_right_bounds(x, partially_corrected_preds_for_x, constr_for_right_bounds) # shape BxM

    # get the closest left boundary
    # closest_left_bound = get_closest_left_bound_wv(x, partially_corrected_preds_for_x, left_bounds, right_bounds, constraints_at_level_x)
    closest_left_bound = get_closest_left_bound(x, partially_corrected_preds_for_x, left_bounds, None, constraints_at_level_x)
    # get the closest right boundary
    # closest_right_bound = get_closest_right_bound_wv(x, partially_corrected_preds_for_x, left_bounds, right_bounds, constraints_at_level_x)
    closest_right_bound = get_closest_right_bound(x, partially_corrected_preds_for_x, None, right_bounds, constraints_at_level_x)

    return closest_left_bound, closest_right_bound


def get_closest_bound(x_vals_violating_req: torch.Tensor, left_bounds: torch.Tensor, right_bounds: torch.Tensor) -> torch.Tensor:
    """Snap violating values to whichever of two bounds is nearer.

    Args:
        x_vals_violating_req: Tensor of current ``x`` values that violate constraints.
        left_bounds: Per-sample left bound candidates.
        right_bounds: Per-sample right bound candidates.

    Returns:
        A tensor with each value replaced by the closer of its left or right bound.
    """
    # METHOD 1
    dist_from_right = torch.abs(x_vals_violating_req - right_bounds)
    dist_from_left = torch.abs(x_vals_violating_req - left_bounds)
    mask_right_smaller_than_left_dist = dist_from_right < dist_from_left
    corrected_x_vals = torch.where(mask_right_smaller_than_left_dist, right_bounds, left_bounds)

    # METHOD 2
    # corrected_x_vals = torch.min(x_vals_violating_req - right_bounds, left_bounds - x_vals_violating_req)
    return corrected_x_vals


def compute_DRL(x: Variable, partially_corrected_preds: torch.Tensor, constraints_at_level_x: List[Constraint]) -> torch.Tensor:
    """Correct a variable against disjunctive constraints.

    Identifies samples violating the (disjunctive) constraints at ``x``'s level and
    moves only those values to the nearest feasible bound, leaving satisfying samples
    unchanged.

    Args:
        x: The variable being corrected.
        partially_corrected_preds: Predictions tensor of shape ``(B, D)`` including
            corrections to previously processed variables.
        constraints_at_level_x: Constraints bounding ``x`` at its level (at least one
            of which is disjunctive).

    Returns:
        A tensor of shape ``(B,)`` with the corrected values of ``x``.
    """
    # the partially_corrected_preds should contain the corrections made at the prev analysed vars!

    x_id = x.id
    initial_x_val = partially_corrected_preds[:, x_id].clone()

    # STEP 1
    # check if the partially_corrected_preds satisfy the requirements
    mask_samples_violating_req = get_samples_violating_constraints(constraints_at_level_x, partially_corrected_preds)
    if not mask_samples_violating_req.any():
        return initial_x_val
    partially_corrected_preds_for_wrong_x = partially_corrected_preds[mask_samples_violating_req]
    x_vals_violating_req = initial_x_val.clone()[mask_samples_violating_req]
    # x_vals_sat_req = initial_x_val.clone()[~mask_samples_violating_req]

    # correct x where the constraints are violated
    # compute right and left bounds
    closest_left_bounds, closest_right_bounds = get_closest_left_and_right_bounds(x, partially_corrected_preds_for_wrong_x, constraints_at_level_x)
    corrected_x_vals = get_closest_bound(x_vals_violating_req, closest_left_bounds, closest_right_bounds)

    # refill initial_x_val where values violate the constr with the corrected values
    initial_x_val[mask_samples_violating_req] = corrected_x_vals

    return initial_x_val



def correct_preds(preds: torch.Tensor, ordering: List[Variable], sets_of_constr: {Variable: List[Constraint]}):
    """Correct predictions so they satisfy all QFLRA requirements.

    Processes the variables in the given ``ordering``, correcting each variable into
    the feasible region implied by its constraint set given the already-corrected
    values of the earlier variables. Purely conjunctive levels are handled by simple
    interval clipping; levels with disjunctions are handled by :func:`compute_DRL`.
    Infinite corrected values are replaced by a large finite sentinel.

    Args:
        preds: Predictions tensor of shape ``(B, D)`` (one column per variable).
        ordering: The variable ordering driving the correction sweep.
        sets_of_constr: Mapping from each :class:`Variable` to its constraint set, as
            produced by :func:`compute_sets_of_constraints`.

    Returns:
        A tensor of shape ``(B, D)`` with predictions corrected to satisfy the
        constraints.

    Example:
        >>> sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=False)
        >>> corrected = correct_preds(preds, ordering, sets_of_constr)
    """
    # given a NN's preds [h1, h2, .., hn],
    # an ordering of the n variables and
    # a set of constraints computed at each variable w.r.t. descending order of the provided ordering
    # correct the preds according to the constraints in ascending order w.r.t. provided ordering
    corrected_preds = preds.clone()

    for x in ordering:
        pos = x.id
        x_constr = get_constr_at_level_x(x, sets_of_constr)
        if len(x_constr) == 0:
            continue

        if not any_disjunctions_in_constraint_set(x_constr):
            # print(x.id, [e.readable() for e in x_constr], 'KKKK')  # .readable()],'@@@')
            pos_x_constr, neg_x_constr, pos_neg_x_constr = get_pos_neg_pn_x_constr(x, x_constr)

            constr_for_lbs = pos_x_constr + pos_neg_x_constr
            constr_for_ubs = neg_x_constr + pos_neg_x_constr

            lower_bounds = get_lower_bounds(x, constr_for_lbs, preds)
            greatest_lbs = lower_bounds if type(lower_bounds) is not torch.Tensor else lower_bounds.amax(dim=1)

            upper_bounds = get_upper_bounds(x, constr_for_ubs, preds)
            least_ubs = upper_bounds if type(upper_bounds) is not torch.Tensor else upper_bounds.amin(dim=1)

            # print('pos', [e.readable() for e in pos_x_constr], preds[pos], pos_x_corrected)
            # print('neg', [e.readable() for e in neg_x_constr], preds[pos], neg_x_corrected)

            corrected_preds[:,pos] = get_final_x_correction(preds[:, pos], greatest_lbs, least_ubs)

        else:
            corrected_preds[:,pos] = compute_DRL(x, preds, x_constr)

        preds = corrected_preds.clone()
        corrected_preds = preds.clone()

    corrected_preds = torch.where(corrected_preds == torch.inf, INFINITY_NP, corrected_preds)
    corrected_preds = torch.where(corrected_preds == -torch.inf, -INFINITY_NP, corrected_preds)

    return corrected_preds

def check_all_constraints_sat(corrected_preds, constraints, error_raise=True):
    """Check that corrected predictions satisfy every constraint.

    Args:
        corrected_preds: The corrected predictions tensor.
        constraints: The constraints to verify.
        error_raise: If ``True``, raise an exception on the first violated
            constraint instead of merely flagging it.

    Returns:
        ``True`` if all constraints are satisfied, ``False`` otherwise (when
        ``error_raise`` is ``False``).

    Raises:
        Exception: If ``error_raise`` is ``True`` and a constraint is violated.
    """
    all_sat_after_correction = True
    for constr in constraints:
        sat = constr.check_satisfaction(corrected_preds)
        if not sat:
            all_sat_after_correction = False
            if error_raise:
                constr: Constraint
                batched_sat_results = constr.disjunctive_inequality.check_satisfaction(corrected_preds)
                samples_violating_constr = corrected_preds[~batched_sat_results]
                raise Exception('Not satisfied!', constr.readable(), samples_violating_constr[0][3], samples_violating_constr[0][10], samples_violating_constr[0][11])
    return all_sat_after_correction


def example_predictions_custom():
    """Build a small example batch of predictions for demonstration.

    Returns:
        A tensor of shape ``(2, 2)`` with two hand-crafted prediction samples.
    """
    y_0= -2
    y_1 = 2
    p1 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(2)])+']')).unsqueeze(0)   # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True

    y_0 = 2
    y_1 = -2.
    p2 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(2)])+']')).unsqueeze(0)

    predictions = torch.cat([p1,p2],dim=0)
    return predictions


def main():
    """Run an end-to-end demo: parse constraints, correct example preds, report."""
    # ordering, constraints = parse_constraints_file('../data/constraints_disj1.txt')
    # ordering, constraints = parse_constraints_file('../data/url_constraints.txt')
    # ordering, constraints = parse_constraints_file('../data/lcld/lcld_constraints.txt')
    ordering, constraints = parse_constraints_file('../custom_constr.txt')

    # set ordering to random
    ordering = set_random_ordering(ordering)

    print('verbose constr')
    for constr in constraints:
        print(constr.verbose_readable())

    print('compute sets of constraints')
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    # preds = example_predictions()
    # preds = example_predictions_lcld()
    preds = example_predictions_custom()
    print('\n\nPreds', preds)

    preds.requires_grad = True
    t1 = time.time()
    corrected_preds = correct_preds(preds, ordering, sets_of_constr)
    print('Original predictions', preds[0])

    print('Corrected predictions', corrected_preds[0], corrected_preds.requires_grad)

    check_all_constraints_are_sat(constraints, preds, corrected_preds)

    print('Time to correct preds', time.time() - t1)

    compute_sat_stats(preds, constraints, mask_out_missing_values=True)
    print(corrected_preds, 'corrected')


if __name__ == '__main__':
    main()
