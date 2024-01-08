import pickle as pkl
import time
from typing import List

import torch

from cloverd.classes import Variable, Constraint
from cloverd.compute_sets_of_constraints import get_pos_neg_x_constr
from cloverd.correct_predictions import get_constr_at_level_x, get_final_x_correction
from cloverd.utils import eval_atoms_list

INFINITY = torch.inf




def get_partial_x_correction(x: Variable, x_positive: bool, x_constraints: List[Constraint],
                             preds: torch.Tensor, epsilon=1e-12) -> torch.Tensor:
    # if x.id == 25:
    #     print('DEBUG!!!')
    if len(x_constraints) == 0:
        if x_positive:
            return -INFINITY
        else:
            return INFINITY

    # dependency_complements = [preds[:, x.id]] # Note: the original prediction cannot be here, as this function gets called twice: for pos and neg occurences of x!
    # so using the original value of x here can undo the partial correction for pos occurrences of x!
    dependency_complements = [] # size: num_atom_dependencies x B (i.e. B=batch size)
    mask_sat_batch_elem = None
    for constr in x_constraints:
        # print(constr.readable(), 'LLLLLLLLLLLLLLLL')
        mask_sat_batch_elem = None  # NOTE: the mask should be reset when a new constraint is considered, regardless of its type (ineq or disj of ineq!)
        # print(constr.single_inequality.readable(), 'AAAAAA')
        if len(constr.inequality_list) == 1:
            complement_body_atoms, x_atom, constr_constant, is_strict_inequality = constr.single_inequality.get_x_complement_body_atoms(x)
            # print('first branch', [e.readable() for e in complement_body_atoms], x_atom, constr_constant, is_strict_inequality)
        else:
            (complement_body_atoms, x_atom, constr_constant, is_strict_inequality), mask_sat_batch_elem = constr.single_inequality.get_x_complement_body_atoms(x, preds)
            # print('second branch', preds[:,2])
            if x_atom is None:
                # print('atom is none')
                # print(constr.readable(), 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaa!!!!!!!!!!!!!!1')
                continue
            # else:
                # print('first branch', [e.readable() for e in complement_body_atoms], x_atom, constr_constant, is_strict_inequality)

                # print(constr.readable(), 'MMMMMMMMMMMMMMMMMMMM!!!!!!!!!!!!!!1')

        # get the coefficient of variable x in constr
        x_coeff = x_atom.coefficient  # do not use self.get_signed_coefficient() here!
        # print(x_coeff,'x coeff')
        # print(constr_constant, '@')
        # if x is y1 and inequality is -y1>0, then add 0+bias to dependency_complements
        if len(complement_body_atoms) == 0:  # TODO: assumes that right hand side of ineq is 0, need to consider cases where the constant/bias is non-zero
            evaluated_complement_body = torch.zeros(preds.shape[0])  # shape (B,)
            # print('first br', evaluated_complement_body)
            # continue
        else:
            # evaluate the body of constr, after eliminating x occurrences from it
            evaluated_complement_body = eval_atoms_list(complement_body_atoms, preds)  # shape (B,)
            # print('second br', evaluated_complement_body)

        # print('x_coeff', x_coeff)
        # print('evaluated_complement_body', evaluated_complement_body)
        # weigh the evaluated complement body by the -1/(coefficient of x) if x occurs positively in constr
        # and by 1/(coefficient of x) if x occurs negatively in constr
        x_weight = -1. / x_coeff if x_positive else 1. / x_coeff
        evaluated_complement_body *= x_weight
        # print('evaluated_complement_body after weighing', evaluated_complement_body)

        # now add the weighted bias:
        evaluated_complement_body += constr_constant * (-x_weight)

        # then add/subtract epsilon if the ineq is strict: +epsilon if positive x, -epsilon otherwise
        if is_strict_inequality:
            # print(is_strict_inequality)
            evaluated_complement_body += epsilon if x_positive else -epsilon

        # if the constr is a disj of inqs, mask out the batch elements for which mask_sat_batch_elem is True
        if mask_sat_batch_elem is not None:
            evaluated_complement_body[mask_sat_batch_elem] += INFINITY * x_weight

        dependency_complements.append(evaluated_complement_body)

        # print('evaluated_complement_body after adding bias', evaluated_complement_body, dependency_complements)

    # if x.id == 25:
    #     print('END DEBUG!!!')

    if len(dependency_complements) > 1:
        dependency_complements = torch.stack(dependency_complements, dim=1)
    elif len(dependency_complements) == 1:
        dependency_complements = dependency_complements[0].unsqueeze(1)
    else:
        return -INFINITY if x_positive else INFINITY

    # print('@@@@@@@@@@@', complement_body_atoms, dependency_complements, len(dependency_complements), constr_constant)

    if x_positive:
        partially_corrected_val = dependency_complements.amax(dim=1)
        # print(partially_corrected_val, x.id, 'AAA')
    else:
        partially_corrected_val = dependency_complements.amin(dim=1)  # TODO: be careful here! the original value of h' to be corrected should not be added multiple times!! it shouldn't be added to the dependecy_complements
        # print(partially_corrected_val, 'BBB')

    # print('partially corrected val', partially_corrected_val)
    return partially_corrected_val



def correct_preds(preds: torch.Tensor, ordering: List[Variable], sets_of_constr: {Variable: List[Constraint]}):
    t1=time.time()
    # num_variables = preds.shape[-1]
    # from constraints_code.constraint_layer import ConstraintLayer
    # CL = ConstraintLayer(num_variables, ordering, sets_of_constr)
    # CL_corrected_preds = CL(preds.clone())
    # t2 = time.time()
    # print(t2-t1, 'timed')
    # return CL_corrected_preds

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
        # print(x.id, [e.readable() for e in x_constr], 'KKKK')  # .readable()],'@@@')
        pos_x_constr, neg_x_constr = get_pos_neg_x_constr(x, x_constr)

        pos_x_corrected = get_partial_x_correction(x, True, pos_x_constr, preds)
        neg_x_corrected = get_partial_x_correction(x, False, neg_x_constr, preds)

        # print('pos', [e.readable() for e in pos_x_constr], preds[pos], pos_x_corrected)
        # print('neg', [e.readable() for e in neg_x_constr], preds[pos], neg_x_corrected)

        corrected_preds[:,pos] = get_final_x_correction(preds[:,pos], pos_x_corrected, neg_x_corrected)
        preds = corrected_preds.clone()
        corrected_preds = preds.clone()

    t2 = time.time()
    print(t2-t1, 'timed')
    return corrected_preds


