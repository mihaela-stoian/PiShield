import json
import random
from typing import List
from pishield.qflra_requirements.classes import Variable

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

# def set_ordering(use_case, ordering: List[Variable] | List[str], label_ordering_choice: str, model_type: str, data_partition='test'):
#     if label_ordering_choice == 'random':
#         ordering = set_random_ordering(ordering)
#     elif label_ordering_choice == 'predefined':
#         ordering = list(map(lambda x: Variable(x), ordering.split()))
#     else:
#         json_filename = f'feature_ordering/feature_orderings.json'
#         with open(json_filename, "r") as f:
#             data = json.load(f)
#
#         if label_ordering_choice == 'causal':
#             ordering = data["feature_orderings"][use_case]['general'][label_ordering_choice]['train']
#             ordering = list(map(lambda x: Variable(x), ordering.split()))
#         else:
#             model_type = model_type.lower()
#             ordering = data["feature_orderings"][use_case][model_type][label_ordering_choice][data_partition]
#             ordering = list(map(lambda x: Variable(x), ordering.split()))

    readable_ordering = [e.readable() for e in ordering]
    print(f'Using *{label_ordering_choice}* feature ordering:\n', readable_ordering, 'len:', len(readable_ordering))
    return ordering