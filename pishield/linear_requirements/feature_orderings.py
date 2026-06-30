"""Selection of the variable ordering used to correct predictions.

Provides helpers to choose the ordering in which variables are corrected by the
Shield Layer: either the ordering given in the requirements file or a random
permutation of it.
"""

import random
from typing import List

from pishield.linear_requirements.classes import Variable

random.seed(0)


def set_random_ordering(ordering: List[Variable]):
    """Shuffle a variable ordering in place into a random permutation.

    Args:
        ordering: The list of variables to shuffle (modified in place).

    Returns:
        The same list, randomly permuted.
    """
    random.shuffle(ordering) # in-place shuffling
    return ordering


def set_ordering(ordering: List[Variable], label_ordering_choice: str):
    """Select the variable ordering according to the chosen strategy.

    Args:
        ordering: The variable ordering parsed from the requirements file.
        label_ordering_choice: Either ``'random'`` (return a random permutation)
            or ``'given'`` (return the ordering unchanged).

    Returns:
        The chosen ordering of variables, or ``None`` if ``label_ordering_choice``
        is neither ``'random'`` nor ``'given'``.
    """
    if label_ordering_choice == 'random':
        ordering = set_random_ordering(ordering)
        return ordering
    elif label_ordering_choice == 'given':
        return ordering