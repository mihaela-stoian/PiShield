import numpy as np

from pishield.propositional_requirements.clause import Clause
from pishield.propositional_requirements.constraint import Constraint
from pishield.propositional_requirements.literal import Literal


def test_eq():
    assert Clause('1 n2 1 2') == Clause('2 1 n2')
    assert Clause('1 n2 3 n4') != Clause('1 n2 3 4')


def test_str():
    assert str(Clause('1 n2 1 2')) == '1 n2 2'
    assert str(Clause([Literal('1'), Literal('n2'), Literal('1'), Literal('2')])) == '1 n2 2'


def test_shift_add_n0():
    assert Clause('0 n1 2 n3').shift_add_n0() == Clause('n0 1 n2 3 n4')


def test_always_true():
    assert not Clause('1 2 n3').always_true()
    assert Clause('1 2 n3 n1').always_true()


def test_constraint():
    assert Clause('1 2 n3').fix_head(Literal('1')) == Constraint('1 :- n2 3')
    assert Clause('1 2 n3').fix_head(Literal('1')) != Constraint('n1 :- n2 3')
    assert Clause.from_constraint(Constraint('2 :- 1 n0')) == Clause('2 n1 0')
    assert Clause.from_constraint(Constraint('n2 :- 1 n0')) != Clause('2 n1 0')


def test_resolution():
    c1 = Clause('1 n2 3')
    c2 = Clause('2 4 n5')
    assert c1.resolution(c2) == Clause('1 3 4 n5')
    c1 = Clause('1 2 n3')
    c2 = Clause('n1 2 3')
    assert c1.resolution(c2) == None
    c1 = Clause('1 2 n3')
    c2 = Clause('n3 n4 5 6')
    assert c1.resolution(c2) == None


def test_coherent_with():
    c = Clause('0 1 n2 n3')
    preds = np.array([
        [.1, .2, .8, .9, .1],
        [.2, .6, .6, .7, .2],
        [.2, .3, .2, .8, .3],
        [.6, .3, .3, .6, .4],
        [.9, .9, .1, .1, .5],
        [.4, .5, .6, .7, .6]
    ])

    assert (c.coherent_with(preds) == [False, True, True, True, True, False]).all()


def test_random():
    c = [Clause.random(10) for i in range(10)]
    assert not (np.array(c) == c[0]).all()


def test_is_subset():
    c1 = Clause('1 2 n3 4')
    c2 = Clause('1 n3')
    c3 = Clause('1 3')
    assert not c1.is_subset(c2)
    assert c2.is_subset(c1)
    assert not c3.is_subset(c1)


def test_atoms():
    assert Clause('1 3 n5 17').atoms() == {1, 3, 5, 17}
