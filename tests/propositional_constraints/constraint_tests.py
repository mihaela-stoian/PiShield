import numpy as np

from cloverd.propositional_constraints.constraint import Constraint
from cloverd.propositional_constraints.literal import Literal


def test_str():
    assert str(Constraint(Literal('1'), [Literal('n0'), Literal("2")])) == "1 :- n0 2"
    assert str(Constraint('n0 :- 1 n2 n3')) == "n0 :- 1 n2 n3"
    assert str(Constraint('0.0 n0 :- 1 n2 n3')) == "n0 :- 1 n2 n3"


def test_eq():
    assert Constraint('0 :- n1 2 n1') == Constraint('0 :- 2 n1')
    assert Constraint('0 :- n1 2 n2') != Constraint('0 :- 2 n1')


def test_coherent_with():
    cons = Constraint('1 :- 0')
    assert (cons.coherent_with(np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.1, 0.1]
    ])) == [True, False, True]).all()

    cons = Constraint('n0 :- n1 2 3')
    assert (cons.coherent_with(np.array([
        [0.7, 0.8, 0.3, 0.4],
        [0.8, 0.8, 0.3, 0.4],
        [0.9, 0.8, 0.3, 0.4],
    ])) == [True, True, False]).all()


def test_atoms():
    c = Constraint('n2 :- 2 3 n4 5')
    assert c.atoms() == {2, 3, 4, 5}
