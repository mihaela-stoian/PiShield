from typing import List

from pishield.shield_loss.shield_loss import ShieldLoss

def build_shield_loss(num_variables: int,
                       requirements_filepath: str,
                       tnorm_choice: str = 'godel',
                       requirements_type='propositional'):
    """
    Build a Shield Loss using the given requirements.
    Inputs:
        - num_variables: the total number of variables (e.g. labels or features, depending on the task) matching the dimension of the tensors which are to be corrected by the layer.
        - requirements_filepath: the path to a txt file containing the requirements.
        - tnorm_choice: can be 'product', 'godel' or 'lukasiewicz'
        - requirements_type: 'propositional.
    """

    if requirements_type == 'propositional':
        return ShieldLoss(num_variables, requirements_filepath, tnorm_choice)
    else:
        raise Exception('Unknown requirements type!')
