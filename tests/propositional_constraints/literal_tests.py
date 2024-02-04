from pishield.propositional_constraints.literal import Literal


def test_init_and_str():
    assert str(Literal('13')) == "13"
    assert str(Literal('n0')) == "n0"
    assert str(Literal('0')) == "0"


def test_neg():
    assert str(Literal('13').neg()) == 'n13'
    assert str(Literal('n0').neg()) == '0'
    assert str(Literal('0').neg()) == 'n0'


def test_comparison():
    assert Literal('13').neg() == Literal('n13')
    assert Literal('13').neg() != Literal('n14')
    assert Literal('n13') < Literal('13')
    assert Literal('13') < Literal('15')
    assert hash(Literal('14')) == hash(Literal('n14').neg())
    assert hash(Literal('14')) != hash(Literal('15'))
    assert hash(Literal('14')) != hash(Literal('n14'))
