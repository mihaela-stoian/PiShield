"""Tests for :func:`pishield.shield_layer.detect_requirements_type`.

These focus on the auto-detection of the requirement type (linear / qflra /
propositional) from a requirements file, including the whole-file nature of the
linear-vs-QFLRA distinction (a QFLRA file may start with plain inequalities and
only later contain a disjunction).
"""

import os

from pishield.shield_layer import detect_requirements_type

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')


def _write(tmp_path, name, contents):
    p = tmp_path / name
    p.write_text(contents)
    return str(p)


def test_detect_linear():
    fp = os.path.join(DATA, 'linear_requirements', 'custom_constraints', 'tiny_constraints1.txt')
    assert detect_requirements_type(fp) == 'linear'


def test_detect_propositional_clauses():
    fp = os.path.join(DATA, 'propositional_requirements', 'custom_constraints',
                      'constraints_simple_example.txt')
    assert detect_requirements_type(fp) == 'propositional'


def test_detect_qflra_disjunction_first(tmp_path):
    fp = _write(tmp_path, 'qflra_a.txt',
                'ordering y_0 y_1 y_2\n'
                'y_1 > 0 or neg y_2 >= 1\n'
                'y_0 >= 0\n')
    assert detect_requirements_type(fp) == 'qflra'


def test_detect_qflra_plain_inequality_first(tmp_path):
    # Regression: a QFLRA file whose first constraint is a plain inequality must
    # still be detected as QFLRA, not misclassified as linear because of the
    # early inequality line.
    fp = _write(tmp_path, 'qflra_b.txt',
                'ordering y_0 y_1 y_2\n'
                'y_0 >= 0\n'
                'y_1 > 0 or neg y_2 >= 1\n')
    assert detect_requirements_type(fp) == 'qflra'


def test_detection_is_order_independent(tmp_path):
    a = _write(tmp_path, 'order_a.txt',
               'ordering y_0 y_1\ny_0 >= 0\ny_1 > 0 or y_0 < 5\n')
    b = _write(tmp_path, 'order_b.txt',
               'ordering y_0 y_1\ny_1 > 0 or y_0 < 5\ny_0 >= 0\n')
    assert detect_requirements_type(a) == detect_requirements_type(b) == 'qflra'


def test_ordering_line_ignored(tmp_path):
    fp = _write(tmp_path, 'lin.txt', 'ordering y_0 y_1\ny_0 - y_1 >= 0\n')
    assert detect_requirements_type(fp) == 'linear'


def test_no_constraints_returns_none(tmp_path):
    fp = _write(tmp_path, 'empty.txt', 'ordering y_0 y_1\n\n')
    assert detect_requirements_type(fp) is None
