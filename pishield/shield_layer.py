from typing import List

from pishield.linear_requirements.shield_layer import ShieldLayer as LinearConstraintLayer
from pishield.qflra_requirements.shield_layer import ShieldLayer as QFLRAConstraintLayer
from pishield.propositional_requirements.shield_layer import ShieldLayer as PropositionalConstraintLayer


def build_shield_layer(num_variables: int,
                       requirements_filepath: str,
                       ordering_choice: str = 'given',
                       custom_ordering: List = None,
                       requirements_type='auto'):
    """
    Build a Shield Layer using the given requirements.
    Inputs:
        - num_variables: the total number of variables (e.g. labels or features, depending on the task) matching the dimension of the tensors which are to be corrected by the layer.
        - requirements_filepath: the path to a txt file containing the requirements.
        - ordering_choice: can be 'given', 'random' or a custom-made ordering implemented by the user.
            if ordering_choice is 'given', the ordering will be picked from requirements_filepath, if available, otherwise it will be the ascending order of the variables;
            if ordering_choice is 'random', the ordering will be a random ordering of the variables.
        - requirements_type: can be 'auto', 'linear', 'propositional, 'qflra'
            if requirements_type is 'auto', then the appropriate layer class will be selected
            based on the requirements provided in requirements_filepath.
    """

    if requirements_type == 'linear':
        return LinearConstraintLayer(num_variables, requirements_filepath, ordering_choice)
    elif requirements_type == 'qflra':
        return QFLRAConstraintLayer(num_variables, requirements_filepath, ordering_choice)
    elif requirements_type == 'propositional':
        return PropositionalConstraintLayer(num_variables, requirements_filepath, ordering_choice, custom_ordering=custom_ordering)
    elif requirements_type == 'auto':
        detected_requirements_type = detect_requirements_type(requirements_filepath)
        return build_shield_layer(num_variables, requirements_filepath, ordering_choice, custom_ordering=custom_ordering,
                                  requirements_type=detected_requirements_type)
    else:
        raise Exception('Unknown requirements type!')


def detect_requirements_type(requirements_filepath: str) -> str:
    f = open(requirements_filepath, 'r')
    linear_keywords = ['>', '>=', '<', '<=']
    qflra_keywords = ['neg', 'or']
    propositional_keywords = [':-']
    for line in f:
        line = line.strip()
        if 'ordering' in line:
            continue
        for keyword in linear_keywords:
            if keyword in line:
                print('Using auto mode ::: Detected linear requirements!')
                return 'linear'
        for keyword in qflra_keywords:
            if keyword in line:
                print('Using auto mode ::: Detected QFLRA requirements!')
                return 'qflra'
        for keyword in propositional_keywords:
            if keyword in line:
                print('Using auto mode ::: Detected propositional requirements!')
                return 'propositional'
    return None

