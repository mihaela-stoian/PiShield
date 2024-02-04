import networkx as nx
import numpy as np
import pytest

from pishield.propositional_constraints.clause import Clause
from pishield.propositional_constraints.clauses_group import ClausesGroup
from pishield.propositional_constraints.constraint import Constraint
from pishield.propositional_constraints.constraints_group import ConstraintsGroup


def test_eq():
    c1 = Clause('1 n2 3')
    c2 = Clause('1 n2 n3 n2')
    c3 = Clause('1 3 n3')
    assert ClausesGroup([c1, c2, c3]) == ClausesGroup([c3, c2, c1, c1])
    assert ClausesGroup([c1, c2]) != ClausesGroup([c3, c2, c1, c1])


def test_add_detection_label():
    before = ClausesGroup([
        Clause('0 n1 2 n3'),
        Clause('0'),
        Clause('n1'),
        Clause('n2 4 5')
    ])

    after = ClausesGroup([
        Clause('n0 1 n2 3 n4'),
        Clause('n0 1'),
        Clause('n0 n2'),
        Clause('n0 n3 5 6'),
    ])

    after2 = ClausesGroup([
        Clause('n0 1 n2 3 n4'),
        Clause('n0 1'),
        Clause('n0 n2'),
        Clause('n0 n3 5 6'),
        Clause('n1 0'),
        Clause('n2 0'),
        Clause('n3 0'),
        Clause('n4 0'),
        Clause('n5 0'),
        Clause('n6 0')
    ])

    assert before.add_detection_label() == after
    print(before.add_detection_label(True))
    assert before.add_detection_label(True) == after2


def test_resolution():
    c1 = Clause('1 2 3')
    c2 = Clause('1 n2 4')
    c3 = Clause('n1 4 n5')
    c4 = Clause('n1 2 6')
    c5 = Clause('2 n3 4')
    constraints, clauses = ClausesGroup([c1, c2, c3, c4, c5]).resolution(1)
    print(clauses)

    assert constraints == ConstraintsGroup([
        Constraint('1 :- n2 n3'),
        Constraint('1 :- 2 n4'),
        Constraint('n1 :- n4 5'),
        Constraint('n1 :- n2 n6')
    ])

    assert clauses == ClausesGroup([
        Clause('2 3 4 n5'),
        Clause('2 3 6'),
        Clause('n2 4 n5'),
        Clause('4 2 n3')
    ])


def test_stratify():
    constraints = ClausesGroup([
        Clause('n0 n1'),
        Clause('n1 2'),
        Clause('1 n2')
    ]).stratify([0, 1, 2])
    assert len(constraints) == 2
    assert constraints[0] == ConstraintsGroup([
        Constraint('n1 :- 0')
    ])
    assert constraints[1] == ConstraintsGroup([
        Constraint('2 :- 1'),
        Constraint('n2 :- n1')
    ])


def test_coherent_with():
    clauses = ClausesGroup([
        Clause('0 1 n2 n3'),
        Clause('n0 1'),
        Clause('0 n1'),
        Clause('3 n3'),
        Clause('n2 n3')
    ])

    preds = np.array([
        [0.1, 0.2, 0.6, 0.7],
        [0.4, 0.7, 0.2, 0.3],
        [0.7, 0.2, 0.9, 0.8]
    ])

    assert (clauses.coherent_with(preds) == [
        [False, True, True, True, False],
        [True, True, False, True, True],
        [True, False, True, True, False]
    ]).all()


def test_empty_resolution():
    clauses = ClausesGroup([
        Clause('0 2'),
        Clause('n0 2'),
        Clause('1 n2'),
        Clause('n1 n2')
    ])

    with pytest.raises(Exception):
        clauses.stratify([0, 1, 2])


def test_random():
    num_classes = 10
    max_clauses = 30

    requirements = np.random.randint(low=0, high=2, size=(3, num_classes))
    clauses = ClausesGroup.random(max_clauses=max_clauses, num_classes=num_classes, coherent_with=requirements)
    assert len(clauses) <= max_clauses
    assert clauses.coherent_with(requirements).all()


def test_compacted():
    clauses = ClausesGroup([
        Clause('n1 n3'),
        Clause('2 n3 5'),
        Clause('1 n3'),
        Clause('1 2 n3 4'),
        Clause('n3 4'),
        Clause('2 5')
    ])

    correct = ClausesGroup([
        Clause('n1 n3'),
        Clause('1 n3'),
        Clause('n3 4'),
        Clause('2 5')
    ])

    assert clauses.compacted() == correct


def test_atoms():
    clauses = ClausesGroup([
        Clause('1 2 n3 4'),
        Clause('3 4 5 n6'),
        Clause('n6 n7 n8 9')
    ])

    assert clauses.atoms() == set(range(1, 10))


def test_graph():
    clauses = ClausesGroup([
        Clause('0 1 n2'),
        Clause('n1 2 n3'),
        Clause('n0 2')
    ])

    G = clauses.graph()
    assert len(G.nodes()) == 7
    assert nx.algorithms.is_bipartite(G)
    assert len(G.edges()) == 8


def test_centrality():
    clauses = ClausesGroup([
        Clause('0 2'),
        Clause('1 2'),
        Clause('2 3 4')
    ])

    assert np.allclose(list(clauses.centrality('degree').items())[0:5], [
        (0, 1 / 7),
        (1, 1 / 7),
        (2, 3 / 7),
        (3, 1 / 7),
        (4, 1 / 7)
    ])
    assert np.allclose(list(clauses.centrality('eigenvector').items())[0:5], [
        (0, 0.16827838529538847),
        (1, 0.16827838529538852),
        (2, 0.5745383453297614),
        (3, 0.23798157473898407),
        (4, 0.23798157473898351)
    ])
    assert np.allclose(list(clauses.centrality('katz').items())[0:5], [
        (0, 0.3243108798877643),
        (1, 0.3243108798877643),
        (2, 0.39975054223505035),
        (3, 0.3276201745804966),
        (4, 0.3276201745804966)
    ])
    assert np.allclose(list(clauses.centrality('closeness').items())[0:5], [
        (0, 0.3333333333333333),
        (1, 0.3333333333333333),
        (2, 0.6363636363636364),
        (3, 0.3684210526315789),
        (4, 0.3684210526315789)
    ])
    assert np.allclose(list(clauses.centrality('betweenness').items())[0:5], [
        (0, 0.),
        (1, 0.),
        (2, 0.7619047619047619),
        (3, 0.),
        (4, 0.)
    ])
