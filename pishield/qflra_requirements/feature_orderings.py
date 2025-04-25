import json
import random
from typing import List
from pishield.qflra_requirements.classes import Variable

random.seed(0)


def set_random_ordering(ordering: List[Variable]):
    random.shuffle(ordering) # in-place shuffling
    return ordering


def set_ordering(ordering: List[Variable] | List[str], label_ordering_choice: str):
    if label_ordering_choice == 'random':
        ordering = set_random_ordering(ordering)
    elif label_ordering_choice == 'predefined':
        ordering = list(map(lambda x: Variable(x), ordering.split()))

    readable_ordering = [e.readable() for e in ordering]
    print(f'Using *{label_ordering_choice}* feature ordering:\n', readable_ordering, 'len:', len(readable_ordering))
    return ordering