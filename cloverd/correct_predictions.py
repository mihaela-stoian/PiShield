import pickle as pkl
import time

import torch

from cloverd.compute_sets_of_constraints import compute_sets_of_constraints
from cloverd.feature_orderings import set_ordering
from cloverd.parser import parse_constraints_file

INFINITY = torch.inf


def get_constr_at_level_x(x, sets_of_constr):
    for var in sets_of_constr:
        if var.id == x.id:
            return sets_of_constr[var]




def get_final_x_correction(initial_x_val: torch.Tensor, pos_x_corrected: torch.Tensor,
                           neg_x_corrected: torch.Tensor) -> torch.Tensor:
    # print(initial_x_val, pos_x_corrected, neg_x_corrected, 'VAR25!!!')

    if type(pos_x_corrected) is not torch.Tensor:
        result_1 = initial_x_val
    else:
        # print(initial_x_val, pos_x_corrected)
        pos_x_corrected = pos_x_corrected.where(~(pos_x_corrected == INFINITY), initial_x_val)
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
        result_1 = torch.cat([initial_x_val.unsqueeze(1), pos_x_corrected.unsqueeze(1)],dim=1).amax(dim=1)

    # print('result_1', result_1)
    if type(neg_x_corrected) is not torch.Tensor:
        result_2 = result_1
    else:
        # keep_initial_neg_mask = neg_x_corrected.isinf()
        # neg_x_corrected[keep_initial_neg_mask] = initial_x_val[keep_initial_neg_mask]
        neg_x_corrected = neg_x_corrected.where(~(neg_x_corrected == INFINITY), initial_x_val)
        result_2 = torch.cat([result_1.unsqueeze(1), neg_x_corrected.unsqueeze(1)],dim=1).amin(dim=1)
    # print('result_2', result_2)
    # print()
    # print(result_1, 'CCC')
    # print(result_2, 'CCC')

    return result_2





def example_predictions():
    # predictions = torch.tensor([-10.0, 5.0, -2.0, -9, 2, 20, -1]).unsqueeze(0)  # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True
    # y_38 y_37 y_21 y_3 y_31 y_26 y_28 y_2 y_19 y_25 y_23 y_13 y_20 y_1 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_14 y_15 y_16 y_17 y_0
    y_18 = y_22 = y_24 = y_27 = y_29 = y_30 = y_32 = y_33 = y_34 = y_35 = y_36 = -100
    y_38 = 0
    y_37 = 0
    y_21 = -1
    y_3 = 1
    y_31 = -3
    y_26 = 5
    y_28 = -3
    y_2 = 2
    y_19 = -4
    y_25 = -111 # change to -1 , value should be corrected to >0
    y_23 = 0
    y_13 = 0
    y_20 = 0
    y_1 = 0
    y_4 = 0
    y_5 = 0
    y_6 = 0
    y_7 = 0
    y_8 = 0
    y_9 = 0
    y_10 = 0
    y_11 = 0
    y_12 = 0
    y_14 = 0
    y_15 = 0
    y_16 = 0
    y_17 = 0
    y_0 = -14.0
    p1 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(39)])+']')).unsqueeze(0)   # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True

    y_2 = -18
    p2 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(39)])+']')).unsqueeze(0)
    predictions = torch.cat([p1,p1,p2,p1,p2],dim=0)
    return predictions

def example_predictions_heloc():
    # data = pd.read_csv(f"../data/heloc/test_data.csv")
    # data = data.to_numpy().astype(float)
    # return torch.tensor(data)

    data = pkl.load(open('/home/mihian/DEL_unsat/TEMP_uncons.pkl', 'rb'))
    return data


def check_all_constraints_are_sat(constraints, preds, corrected_preds, verbose=False):
    # print('sat req?:')
    for constr in constraints:
        sat = constr.check_satisfaction(preds)
        if not sat and verbose:
            print('Not satisfied!', constr.readable())

    # print('*' * 80)
    all_sat_after_correction = True
    for constr in constraints:
        sat = constr.check_satisfaction(corrected_preds)
        if not sat:
            all_sat_after_correction = False
            # print('Not satisfied!', constr.readable())
    if all_sat_after_correction and verbose:
        print('All constraints are satisfied after correction!')
    if not all_sat_after_correction:
        print('There are still constraint violations!!!')
        with open('./TEMP_uncons.pkl', 'wb') as f:
            pkl.dump(preds, f, -1)
        with open('./TEMP_cons.pkl', 'wb') as f:
            pkl.dump(corrected_preds, f, -1)
    return all_sat_after_correction


def check_all_constraints_sat(corrected_preds, constraints, error_raise=True):
    all_sat_after_correction = True
    for constr in constraints:
        sat = constr.check_satisfaction_per_sample(corrected_preds)
        if not sat.all():
            # all_sat_after_correction = False
            print(corrected_preds[~sat], 'aaa')
            sample_sat, eval_body_value, constant, ineq_sign = constr.detailed_sample_sat_check(corrected_preds)
            print(sample_sat.all(), eval_body_value[~sample_sat], constant, ineq_sign)
            raise Exception('Not satisfied!', constr.readable())
    return all_sat_after_correction


def main():
    use_case = 'heloc'
    # ordering, constraints = parse_constraints_file('../data/constraints_disj1.txt')
    # ordering, constraints = parse_constraints_file('../data/url_constraints.txt')
    ordering, constraints = parse_constraints_file(f'../data/{use_case}/{use_case}_constraints.txt')

    ordering = set_ordering(use_case, ordering, 'kde')

    print('verbose constr')
    for constr in constraints:
        print(constr.verbose_readable())

    print('compute sets of constraints')
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    # preds = example_predictions()
    preds = example_predictions_heloc()
    print('\n\nPreds', preds)

    preds.requires_grad = True
    t1 = time.time()
    # corrected_preds = correct_preds(preds, ordering, sets_of_constr)
    # print('Original predictions', preds[0])
    #
    # print('Corrected predictions', corrected_preds[0], corrected_preds.requires_grad)
    #
    # check_all_constraints_are_sat(constraints, preds, corrected_preds, verbose=True)
    # check_all_constraints_sat(corrected_preds, constraints)
    #
    # print('Time to correct preds', time.time() - t1)





if __name__ == '__main__':
    main()
