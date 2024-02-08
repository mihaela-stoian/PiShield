import numpy as np
import torch

from pishield.propositional_requirements.clauses_group import ClausesGroup
from pishield.propositional_requirements.constraints_group import ConstraintsGroup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def custom_order():
    return np.array(
        [7, 40, 23, 5, 27, 28, 39, 18, 19, 21, 38, 22, 16, 11, 20, 33, 9, 30, 3, 24, 4, 15, 34, 31, 25, 26, 13, 17, 29,
         37, 14, 36, 12, 35, 6, 0, 10, 32, 2, 1, 8]
    )


def custom_order2():
    return np.array(
        [7, 40, 38, 23, 39, 27, 11, 33, 28, 5, 22, 21, 24, 20, 31, 16, 26, 15, 9, 6, 18, 34, 19, 30, 25, 3, 12, 2, 10,
         36, 8, 35, 4, 32, 1, 37, 29, 14, 13, 17, 0]
    )

def test_constraints():
    # centrality = 'katz'
    centrality = custom_order2()
    centrality = centrality[::-1]

    constraints = ConstraintsGroup(
        '../../data/propositional_requirements/custom_constraints/constraints_full_example.txt')
    print(len(constraints))

    clauses = ClausesGroup.from_constraints_group(constraints)
    print(len(clauses))

    constraints = clauses.stratify(centrality)

    lens = [len(group) for group in constraints]
    assert len(lens) == 16
    assert sum(lens) == 485

