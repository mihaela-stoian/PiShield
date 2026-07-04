import numpy as np
import torch

from pishield.hierarchical_requirements.datasets import load_arff_dataset
from pishield.hierarchical_requirements.shield_layer import ShieldLayer

DATA_DIR = 'data/hierarchical_requirements/custom_constraints'
PATHS_ARFF = f'{DATA_DIR}/hierarchy_paths_example.arff'
EDGES_ARFF = f'{DATA_DIR}/hierarchy_edges_example.arff'
CELLCYCLE_FUN = 'data/hierarchical_requirements/cellcycle_FUN/cellcycle_FUN.train.arff'
CELLCYCLE_GO = 'data/hierarchical_requirements/cellcycle_GO/cellcycle_GO.train.arff'


def test_paths_fixture_exact_features_and_labels():
    data = load_arff_dataset(PATHS_ARFF)
    # class order: ['1','2','root','1.1','1.2','2.1','1.2.1'] -> indices 0..6
    assert data.class_names == ['1', '2', 'root', '1.1', '1.2', '2.1', '1.2.1']
    assert np.allclose(data.X, [[0.1, 0.2], [0.3, 0.4]])

    # Row 0 label '1/2/1@2/1' -> leaves 1.2.1(6) and 2.1(5), plus all ancestors.
    row0 = np.zeros(7)
    row0[[0, 1, 2, 4, 5, 6]] = 1  # 1, 2, root, 1.2, 2.1, 1.2.1
    # Row 1 label '1/1' -> leaf 1.1(3), ancestors 1(0), root(2).
    row1 = np.zeros(7)
    row1[[0, 2, 3]] = 1
    assert np.array_equal(data.Y, np.stack([row0, row1]))


def test_edges_fixture_multiparent_ancestor_closure():
    data = load_arff_dataset(EDGES_ARFF)
    # class order: ['root','GO0001','GO0002','GO0003','GO0004'] -> indices 0..4
    assert data.class_names == ['root', 'GO0001', 'GO0002', 'GO0003', 'GO0004']
    # Only row: label 'GO0004'(4); GO0003(3) has TWO parents GO0001(1) and GO0002(2),
    # both must be set, up to root(0). So the whole column set is 1.
    assert np.array_equal(data.Y, np.ones((1, 5)))
    assert np.allclose(data.X, [[0.3]])


def test_labels_are_ancestor_closed():
    # Every parent of a set child must also be set (Y is hierarchically coherent).
    data = load_arff_dataset(CELLCYCLE_FUN)
    for child, parent in data.hierarchy.edges:
        violated = (data.Y[:, child] == 1) & (data.Y[:, parent] == 0)
        assert not violated.any()


def test_shield_layer_leaves_coherent_labels_unchanged():
    # A coherent binary Y is a fixed point of the MCM correction.
    data = load_arff_dataset(PATHS_ARFF)
    layer = ShieldLayer(num_variables=None, requirements_filepath=PATHS_ARFF)
    corrected = layer(torch.tensor(data.Y, dtype=torch.float64))
    assert torch.allclose(corrected, torch.tensor(data.Y, dtype=torch.float64))


def test_cellcycle_fun_shapes_and_alignment():
    data = load_arff_dataset(CELLCYCLE_FUN)
    assert data.X.shape == (1628, 77)
    assert data.Y.shape == (1628, 500)
    assert not np.isnan(data.X).any()  # imputed by default
    # to_eval excludes exactly the 'root' class.
    assert data.to_eval.sum() == 499
    assert not data.to_eval[data.class_names.index('root')]
    # Column order matches the Shield Layer built from the same file.
    layer = ShieldLayer(num_variables=None, requirements_filepath=CELLCYCLE_FUN)
    assert layer.class_names == data.class_names


def test_cellcycle_go_shapes():
    data = load_arff_dataset(CELLCYCLE_GO)
    assert data.X.shape == (1625, 77)
    assert data.num_classes == 4126
    assert not data.to_eval[data.class_names.index('root')]
    assert data.to_eval.sum() < data.num_classes


def test_impute_missing_toggle():
    # cellcycle contains '?' missing values; without imputation they remain NaN.
    raw = load_arff_dataset(CELLCYCLE_FUN, impute_missing=False)
    assert np.isnan(raw.X).any()
    imputed = load_arff_dataset(CELLCYCLE_FUN, impute_missing=True)
    assert not np.isnan(imputed.X).any()
