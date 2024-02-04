import json

import numpy as np
import pytest
import torch

from pishield.propositional_constraints.constraint import Constraint
from pishield.propositional_constraints.constraints_group import ConstraintsGroup
from pishield.propositional_constraints.constraints_module import ConstraintsModule, run_cm


def test_symmetric():
    pos = torch.from_numpy(np.arange(0., 1., 0.1))
    symm = torch.from_numpy(np.arange(-1., 1., 0.2))
    assert torch.isclose(ConstraintsModule.to_symmetric(pos), symm).all()
    assert torch.isclose(ConstraintsModule.from_symmetric(symm), pos).all()


def _test_no_goal(device):
    group = ConstraintsGroup([
        Constraint('1 :- 0'),
        Constraint('2 :- n3 4'),
        Constraint('n5 :- 6 n7 8'),
        Constraint('2 :- 9 n10'),
        Constraint('n5 :- 11 n12 n13'),
    ])
    cm = ConstraintsModule(group, 14)
    preds = torch.rand((5000, 14))
    updated = run_cm(cm, preds, device=device).numpy()
    assert group.coherent_with(updated).all()


def _test_positive_goal(device):
    group = ConstraintsGroup([
        Constraint('0 :- 1 n2'),
        Constraint('3 :- 4 n5'),
        Constraint('n7 :- 7 n8'),
        Constraint('n9 :- 10 n11')
    ])

    cm = ConstraintsModule(group, 12)
    preds = torch.rand((5000, 12))
    goal = torch.tensor([1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0.]).unsqueeze(0).expand(5000, 12)
    updated = run_cm(cm, preds, goal=goal, device=device).numpy()
    assert (group.coherent_with(updated).all(axis=0) == [True, False, True, False]).all()


def _test_negative_goal(device):
    group = ConstraintsGroup([
        Constraint('0 :- 1 n2 3 n4'),
        Constraint('n5 :- 6 n7 8 n9')
    ])
    reduced_group = ConstraintsGroup([
        Constraint('0 :- 1 n2'),
        Constraint('n5 :- 6 n7')
    ])

    cm = ConstraintsModule(group, 10)
    preds = torch.rand((5000, 10))
    goal = torch.tensor([0., 0., 1., 1., 0., 1., 0., 1., 1., 0.]).unsqueeze(0).expand(5000, 10)
    updated = run_cm(cm, preds, goal=goal, device=device).numpy()
    assert reduced_group.coherent_with(updated).all()


def test_goal_cpu():
    _test_no_goal('cpu')
    _test_negative_goal('cpu')
    _test_positive_goal('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_goal_cuda():
    _test_no_goal('cuda')
    _test_negative_goal('cuda')
    _test_positive_goal('cuda')
    print(json.dumps(ConstraintsModule.profiler.combined(), indent=4, sort_keys=True))


def _test_empty_preds(device):
    group = ConstraintsGroup([
        Constraint('0 :- 1')
    ])

    cm = ConstraintsModule(group, 2)
    preds = torch.rand((0, 2))
    goal = torch.rand((0, 2))
    updated = run_cm(cm, preds, goal=goal, device=device)
    assert updated.shape == torch.Size([0, 2])


def _test_no_constraints(device):
    group = ConstraintsGroup([])
    cm = ConstraintsModule(group, 10)
    preds = torch.rand((500, 10))
    goal = torch.rand((500, 10))

    updated = run_cm(cm, preds, device=device)
    assert (updated == preds).all()
    updated = run_cm(cm, preds, goal=goal, device=device)
    assert (updated == preds).all()


def test_empty_preds_constraints_cpu():
    _test_empty_preds('cpu')
    _test_no_constraints('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empty_preds_constraints_cuda():
    _test_empty_preds('cuda')
    _test_no_constraints('cuda')


def test_lb_ub():
    group = ConstraintsGroup([
        Constraint('0 :- 1'),
        Constraint('n0 :- 2')
    ])
    cm = ConstraintsModule(group, 3)
    preds = torch.tensor([
        [0.5, 0.6, 0.3],
        [0.65, 0.6, 0.3],
        [0.8, 0.6, 0.3],
        [0.5, 0.7, 0.4],
        [0.65, 0.7, 0.4],
        [0.8, 0.7, 0.4],
    ])

    updated = run_cm(cm, preds)
    assert (updated[:, 0] == torch.tensor([0.6, 0.65, 0.7] * 2)).all()


def _test_time(iterative, device):
    group = ConstraintsGroup('../../data/propositional_constraints/custom_constraints/constraints_full_example.txt')
    cm = ConstraintsModule(group, 41).to(device)
    preds = torch.rand(5000, 41, device=device)
    cm(preds, iterative=iterative)


def test_time_iterative_cpu():
    for i in range(10):
        _test_time(True, 'cpu')


def test_time_tensor_cpu():
    for i in range(10):
        _test_time(False, 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_time_iterative_cuda():
    for i in range(10):
        _test_time(True, 'cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_time_tensor_cuda():
    for i in range(10):
        _test_time(False, 'cuda')
