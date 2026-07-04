import numpy as np
import pytest

from pishield.hierarchical_requirements.classes import Hierarchy
from pishield.hierarchical_requirements.parser import (
    arff_to_requirements,
    detect_arff_hierarchy_style,
    parse_arff_hierarchy,
    parse_hierarchy_file,
    parse_horn_requirements_file,
)
from pishield.propositional_requirements.constraints_group import ConstraintsGroup

DATA_DIR = 'data/hierarchical_requirements/custom_constraints'
SIMPLE_TXT = f'{DATA_DIR}/hierarchy_simple_example.txt'
PATHS_ARFF = f'{DATA_DIR}/hierarchy_paths_example.arff'
EDGES_ARFF = f'{DATA_DIR}/hierarchy_edges_example.arff'


def test_parse_horn_file():
    # Diamond: 3 has parents 1 and 2, both of which are under 0.
    hierarchy = parse_horn_requirements_file(SIMPLE_TXT)
    assert hierarchy.class_names == ['y_0', 'y_1', 'y_2', 'y_3']
    assert hierarchy.num_classes == 4
    assert hierarchy.edges == [(1, 0), (2, 0), (3, 1), (3, 2)]


def test_parse_horn_file_without_ordering(tmp_path):
    filepath = tmp_path / 'hierarchy.txt'
    filepath.write_text('y_0 :- y_2\n1 :- 2\n')
    hierarchy = parse_horn_requirements_file(str(filepath))
    assert hierarchy.class_names == ['0', '1', '2']
    assert hierarchy.edges == [(2, 0), (2, 1)]


def test_parse_horn_file_rejects_negated_literal(tmp_path):
    filepath = tmp_path / 'hierarchy.txt'
    filepath.write_text('0 :- n1\n')
    with pytest.raises(Exception, match='negated'):
        parse_horn_requirements_file(str(filepath))


def test_parse_horn_file_rejects_multiple_body_literals(tmp_path):
    filepath = tmp_path / 'hierarchy.txt'
    filepath.write_text('0 :- 1 2\n')
    with pytest.raises(Exception, match='hierarchy rule'):
        parse_horn_requirements_file(str(filepath))


def test_parse_horn_file_rejects_cycle(tmp_path):
    filepath = tmp_path / 'hierarchy.txt'
    filepath.write_text('0 :- 1\n1 :- 2\n2 :- 0\n')
    with pytest.raises(Exception, match='cycle'):
        parse_horn_requirements_file(str(filepath))


def test_parse_horn_file_rejects_index_outside_ordering(tmp_path):
    filepath = tmp_path / 'hierarchy.txt'
    filepath.write_text('ordering y_0 y_1\n0 :- 5\n')
    with pytest.raises(Exception, match='ordering'):
        parse_horn_requirements_file(str(filepath))


def test_descendants_matrix():
    hierarchy = parse_horn_requirements_file(SIMPLE_TXT)
    expected = np.array([
        [1, 1, 1, 1],  # 0: itself and every class below it
        [0, 1, 0, 1],  # 1: itself and 3
        [0, 0, 1, 1],  # 2: itself and 3
        [0, 0, 0, 1],  # 3: a leaf
    ])
    assert (hierarchy.descendants_matrix() == expected).all()


def test_adjacency_matrix():
    hierarchy = parse_horn_requirements_file(SIMPLE_TXT)
    adjacency = hierarchy.adjacency_matrix()
    assert adjacency.sum() == 4
    assert adjacency[1, 0] == 1 and adjacency[2, 0] == 1
    assert adjacency[3, 1] == 1 and adjacency[3, 2] == 1


def test_parse_arff_paths_format():
    hierarchy = parse_arff_hierarchy(PATHS_ARFF, arff_hierarchy_style='paths')
    # C-HMCNN ordering: by path depth, then name; 'root' is added implicitly.
    assert hierarchy.class_names == ['1', '2', 'root', '1.1', '1.2', '2.1', '1.2.1']
    name_of = hierarchy.class_names
    edges_by_name = {(name_of[child], name_of[parent]) for child, parent in hierarchy.edges}
    assert edges_by_name == {('1', 'root'), ('2', 'root'), ('1.1', '1'),
                             ('1.2', '1'), ('2.1', '2'), ('1.2.1', '1.2')}


def test_parse_arff_edges_format():
    hierarchy = parse_arff_hierarchy(EDGES_ARFF, arff_hierarchy_style='edges')
    assert hierarchy.class_names == ['root', 'GO0001', 'GO0002', 'GO0003', 'GO0004']
    name_of = hierarchy.class_names
    edges_by_name = {(name_of[child], name_of[parent]) for child, parent in hierarchy.edges}
    assert edges_by_name == {('GO0001', 'root'), ('GO0002', 'root'), ('GO0003', 'GO0001'),
                             ('GO0003', 'GO0002'), ('GO0004', 'GO0003')}


def test_detect_format_from_branches():
    # Any branch with != 2 terms => paths; all branches exactly 2 terms => edges.
    assert detect_arff_hierarchy_style(['1', '1/2', '1/2/3']) == 'paths'
    assert detect_arff_hierarchy_style(['1/2', '1/2/3']) == 'paths'
    assert detect_arff_hierarchy_style(['a/b', 'a/c', 'b/d']) == 'edges'


def test_auto_detection_matches_forced_format():
    # The paths fixture has varying-depth branches (incl. single-term) => detected as paths.
    assert parse_arff_hierarchy(PATHS_ARFF, 'auto') == parse_arff_hierarchy(PATHS_ARFF, 'paths')
    # The edges fixture has only 2-term branches => detected as edges.
    assert parse_arff_hierarchy(EDGES_ARFF, 'auto') == parse_arff_hierarchy(EDGES_ARFF, 'edges')


def test_edges_format_allows_multiparent_dag():
    # GO-style: GO0003 is a child of both GO0001 and GO0002 (a genuine DAG, not a tree).
    hierarchy = parse_arff_hierarchy(EDGES_ARFF, 'auto')
    go0003 = hierarchy.class_names.index('GO0003')
    parents = [parent for child, parent in hierarchy.edges if child == go0003]
    assert len(parents) == 2


def test_paths_format_gives_each_node_one_parent():
    # Path-encoded hierarchies always give each node exactly one parent (a tree).
    hierarchy = parse_arff_hierarchy(PATHS_ARFF, 'auto')
    parent_counts = {}
    for child, parent in hierarchy.edges:
        parent_counts[child] = parent_counts.get(child, 0) + 1
    assert all(count == 1 for count in parent_counts.values())


def test_ambiguous_all_two_term_paths_file_does_not_crash(tmp_path):
    # A path-encoded, all-depth-2 file that omits its top-level classes is ambiguous:
    # auto-detection reads it as 'edges', which makes '01/01' a self-loop. This must
    # fail cleanly (a validated error), never crash with a cryptic min()/empty error.
    filepath = tmp_path / 'ambiguous.arff'
    filepath.write_text('@RELATION t\n@ATTRIBUTE f numeric\n'
                        '@ATTRIBUTE class hierarchical 01/01,01/02\n@DATA\n0.1,01/01\n')
    with pytest.raises(Exception, match='self-loop'):
        parse_arff_hierarchy(str(filepath))
    # Forcing 'paths' parses it correctly.
    hierarchy = parse_arff_hierarchy(str(filepath), 'paths')
    assert hierarchy.num_classes == 3  # 01, 01.01, 01.02


def test_parse_hierarchy_file_dispatches_on_extension():
    assert parse_hierarchy_file(SIMPLE_TXT) == parse_horn_requirements_file(SIMPLE_TXT)
    assert parse_hierarchy_file(PATHS_ARFF) == parse_arff_hierarchy(PATHS_ARFF, 'paths')


def test_parse_arff_without_hierarchical_attribute(tmp_path):
    filepath = tmp_path / 'flat.arff'
    filepath.write_text('@RELATION flat\n@ATTRIBUTE feature1 numeric\n@DATA\n0.1\n')
    with pytest.raises(Exception, match='hierarchical'):
        parse_arff_hierarchy(str(filepath))


def test_arff_to_requirements_roundtrip(tmp_path):
    output = tmp_path / 'converted.txt'
    written = arff_to_requirements(PATHS_ARFF, str(output), arff_hierarchy_style='paths')
    reparsed = parse_horn_requirements_file(str(output))
    assert reparsed.edges == written.edges
    assert reparsed.num_classes == written.num_classes

    # The converted file must also be a valid propositional requirements file.
    constraints_group = ConstraintsGroup(str(output))
    assert len(constraints_group) == len(written.edges)


def test_hierarchy_validation():
    with pytest.raises(Exception, match='self-loop'):
        Hierarchy(['a', 'b'], [(0, 0)])
    with pytest.raises(Exception, match='outside'):
        Hierarchy(['a', 'b'], [(0, 5)])
