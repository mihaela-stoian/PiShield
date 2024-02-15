import numpy as np

from pishield.propositional_requirements.constraint import Constraint
from pishield.propositional_requirements.constraints_group import ConstraintsGroup


def test_str():
    cons0 = Constraint('0 :- 1 n2')
    cons1 = Constraint('n0 :- 1')
    cons2 = Constraint('1 :- n2')
    group = ConstraintsGroup([cons0, cons1, cons2])
    assert str(group) == "n0 :- 1\n0 :- 1 n2\n1 :- n2"


def test_from_file():
    group = ConstraintsGroup('../../data/propositional_requirements/custom_constraints/constraints_simple_example.txt')
    assert str(group) == "n0 :- 1\n0 :- 1 n2\n1 :- n2"


def test_coherent_with():
    group = ConstraintsGroup('../../data/propositional_requirements/custom_constraints/constraints_simple_example.txt')
    assert (group.coherent_with(np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.2, 0.3, 0.4],
        [0.8, 0.2, 0.9, 0.4]
    ])) == np.array(
        [[False, True, False],
         [True, True, False],
         [True, False, True]])).all()


def test_add():
    c1 = Constraint('n0 :- 1 n2 3')
    c2 = Constraint('0 :- n1 n2 4')
    group0 = ConstraintsGroup([c1])
    group1 = ConstraintsGroup([c2])
    group = group0 + group1
    assert group == ConstraintsGroup([c1, c2])


def test_atoms():
    group = ConstraintsGroup('../../data/propositional_requirements/custom_constraints/constraints_full_example.txt')
    assert group.atoms() == set(range(41))


def test_graph():
    group = ConstraintsGroup('../../data/propositional_requirements/custom_constraints/constraints_simple_example.txt')
    graph = group.graph()
    assert set(graph.nodes()) == {0, 1, 2}
    assert set(graph.edges()) == {(1, 0), (2, 1), (2, 0)}


def test_duograph():
    group = ConstraintsGroup('../../data/propositional_requirements/custom_constraints/constraints_simple_example.txt')
    graph = group.duograph()
    print(graph)
    print(graph.nodes())
    print(graph.edges())
    assert set(graph.nodes()) == {'0', '1', '2', 'n0', 'n1', 'n2'}
    assert set(graph.edges()) == {('1', '0'), ('1', 'n0'), ('n2', '1'), ('n2', '0')}


def test_heads():
    group = ConstraintsGroup('../../data/propositional_requirements/custom_constraints/constraints_simple_example.txt')
    assert group.heads() == {0, 1}


def test_stratify():
    group = ConstraintsGroup([
        Constraint('1 :- 0'),
        Constraint('n2 :- n0 4'),
        Constraint('3 :- n1 2')
    ])
    groups = group.stratify()
    assert len(groups) == 2
    assert groups[0].heads() == {1, 2}
    assert groups[1].heads() == {3}
