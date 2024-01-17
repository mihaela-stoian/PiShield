from typing import List

from cloverd.linear_constraints.constraint_layer import ConstraintLayer as LinearConstraintLayer
from cloverd.propositional_constraints.constraints_layer import ConstraintsLayer as PropositionalConstraintLayer


def build_constraint_layer(num_variables: int,
                           constraints_filepath: str,
                           ordering_choice: str = 'given',
                           custom_ordering: List = None,
                           constraints_type='auto'):
    """
    Build a constraint layer using the given constraints.
    Inputs:
        - num_variables: the total number of variables (e.g. labels or features, depending on the task) matching the dimension of the tensors which are to be corrected by the layer.
        - constraints_filepath: the path to a txt file containing the constraints.
        - ordering_choice: can be 'given', 'random' or a custom-made ordering implemented by the user.
            if ordering_choice is 'given', the ordering will be picked from constraints_filepath, if available, otherwise it will be the ascending order of the variables;
            if ordering_choice is 'random', the ordering will be a random ordering of the variables.
        - constraints_type: can be 'auto', 'linear', 'propositional.
            if constraints_type is 'auto', then the appropriate layer class will be selected
            based on the constraints provided in constraints_filepath.
    """

    if constraints_type == 'linear':
        return LinearConstraintLayer(num_variables, constraints_filepath, ordering_choice)
    elif constraints_type == 'propositional':
        return PropositionalConstraintLayer(num_variables, constraints_filepath, ordering_choice, custom_ordering=custom_ordering)
    elif constraints_type == 'auto':
        detected_constraints_type = detect_constraints_type(constraints_filepath)
        return build_constraint_layer(num_variables, constraints_filepath, ordering_choice,
                                      constraints_type=detected_constraints_type)
    else:
        raise Exception('Unknown constraints type!')


def detect_constraints_type(constraints_filepath: str) -> str:
    f = open(constraints_filepath, 'r')
    linear_keywords = ['>', '>=', '<', '<=']
    propositional_keywords = [':-']
    for line in f:
        line = line.strip()
        if 'ordering' in line:
            continue
        for keyword in linear_keywords:
            if keyword in line:
                print('Using auto mode ::: Detected linear constraints!')
                return 'linear'
        for keyword in propositional_keywords:
            if keyword in line:
                print('Using auto mode ::: Detected propositional constraints!')
                return 'propositional'
    return None

