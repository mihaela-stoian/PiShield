import random
from typing import List

from pishield.linear_requirements.classes import Variable

random.seed(0)


def set_random_ordering(ordering: List[Variable]):
    random.shuffle(ordering) # in-place shuffling
    return ordering


def set_ordering(ordering: List[Variable], label_ordering_choice: str):
    if label_ordering_choice == 'random':
        ordering = set_random_ordering(ordering)
        return ordering
    elif label_ordering_choice == 'given':
        return ordering