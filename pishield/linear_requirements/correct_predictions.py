"""Helpers for clipping predictions into the interval allowed by requirements.

Provides utilities to look up the requirement set bounding a given variable, to
clip a variable's predicted value into the feasible interval implied by its
lower and upper bounds, and to check that corrected predictions satisfy all
requirements.
"""

import pickle as pkl
import torch

INFINITY = torch.inf


def get_constr_at_level_x(x, sets_of_constr):
    """Look up the requirement set associated with a given variable.

    Args:
        x: The :class:`Variable` whose requirement set is requested.
        sets_of_constr: Mapping from variables to their requirement sets, as
            produced by :func:`compute_sets_of_constraints`.

    Returns:
        The list of requirements bounding ``x`` (matched by variable id), or
        ``None`` if ``x`` is not present in the mapping.
    """
    for var in sets_of_constr:
        if var.id == x.id:
            return sets_of_constr[var]


def get_final_x_correction(initial_x_val: torch.Tensor, pos_x_corrected: torch.Tensor,
                           neg_x_corrected: torch.Tensor) -> torch.Tensor:
    """
    Clip a variable's value into the interval allowed by its constraints.

    `initial_x_val` is the model's (per-sample) prediction for x; `pos_x_corrected` is the tightest
    lower bound implied by x's positive constraints, and `neg_x_corrected` the tightest upper bound
    from its negative ones. The corrected value is therefore min(max(x, lower bound), upper bound),
    i.e. x is only moved if it falls outside the feasible interval. A non-tensor bound (or an inf
    entry) means "unbounded on that side", in which case the initial value is kept.
    """

    # Enforce the lower bound: result_1 = max(initial value, lower bound).
    if type(pos_x_corrected) is not torch.Tensor:
        result_1 = initial_x_val
    else:
        pos_x_corrected = pos_x_corrected.where(~(pos_x_corrected.isinf()), initial_x_val)
        result_1 = torch.cat([initial_x_val.unsqueeze(1), pos_x_corrected.unsqueeze(1)], dim=1).amax(dim=1)

    # Enforce the upper bound: result_2 = min(result_1, upper bound).
    if type(neg_x_corrected) is not torch.Tensor:
        result_2 = result_1
    else:
        # keep_initial_neg_mask = neg_x_corrected.isinf()
        # neg_x_corrected[keep_initial_neg_mask] = initial_x_val[keep_initial_neg_mask]
        neg_x_corrected = neg_x_corrected.where(~(neg_x_corrected.isinf()), initial_x_val)
        result_2 = torch.cat([result_1.unsqueeze(1), neg_x_corrected.unsqueeze(1)], dim=1).amin(dim=1)

    return result_2



def example_predictions_heloc():
    """Load a pickled tensor of example HELOC predictions for debugging.

    Returns:
        The unpickled predictions object loaded from a hard-coded local path.
    """
    # data = pd.read_csv(f"../data/heloc/test_data.csv")
    # data = data.to_numpy().astype(float)
    # return torch.tensor(data)

    data = pkl.load(open('/home/mihian/DEL_unsat/TEMP_uncons.pkl', 'rb'))
    return data


def check_all_constraints_are_sat(constraints, preds, corrected_preds, verbose=False):
    """Check whether corrected predictions satisfy all requirements (batch-wise).

    Args:
        constraints: The requirements to check.
        preds: The original (uncorrected) predictions.
        corrected_preds: The predictions after correction.
        verbose: Whether to print which requirements are violated/satisfied.

    Returns:
        ``True`` if every requirement is satisfied across the whole batch after
        correction, ``False`` otherwise.
    """
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
        # with open('./TEMP_uncons.pkl', 'wb') as f:
        #     pkl.dump(preds, f, -1)
        # with open('./TEMP_cons.pkl', 'wb') as f:
        #     pkl.dump(corrected_preds, f, -1)
    return all_sat_after_correction


def check_all_constraints_sat(corrected_preds, constraints, error_raise=True):
    """Assert that corrected predictions satisfy all requirements per sample.

    Args:
        corrected_preds: The predictions after correction.
        constraints: The requirements to check.
        error_raise: Retained for backward compatibility; the function raises on
            violation regardless of this flag.

    Returns:
        ``True`` if no violation is found.

    Raises:
        Exception: If any requirement is violated by any sample.
    """
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

