import numpy as np
import torch

from pishield.hierarchical_requirements.shield_layer import ShieldLayer, get_constr_out
from pishield.shield_layer import build_shield_layer, detect_requirements_type

DATA_DIR = 'data/hierarchical_requirements/custom_constraints'
SIMPLE_TXT = f'{DATA_DIR}/hierarchy_simple_example.txt'
PATHS_ARFF = f'{DATA_DIR}/hierarchy_paths_example.arff'
CELLCYCLE_FUN = 'data/hierarchical_requirements/cellcycle_FUN/cellcycle_FUN.train.arff'
CELLCYCLE_GO = 'data/hierarchical_requirements/cellcycle_GO/cellcycle_GO.train.arff'


def reference_get_constr_out(x, R):
    """Independent reimplementation of C-HMCNN's get_constr_out (main.py)."""
    c_out = x.double().unsqueeze(1).expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out


def test_inference_is_coherent():
    layer = ShieldLayer(num_variables=4, requirements_filepath=SIMPLE_TXT)
    torch.manual_seed(0)
    preds = torch.rand(64, 4)
    corrected = layer(preds)
    # Every edge child -> parent must hold: score(child) <= score(parent).
    for child, parent in layer.hierarchy.edges:
        assert (corrected[:, child] <= corrected[:, parent] + 1e-9).all()
    assert layer.satisfied(corrected).all()


def test_matches_reference_mcm():
    layer = ShieldLayer(num_variables=4, requirements_filepath=SIMPLE_TXT)
    torch.manual_seed(1)
    preds = torch.rand(16, 4)
    ours = layer(preds)
    reference = reference_get_constr_out(preds, layer.R)
    assert torch.allclose(ours.double(), reference)


def test_goal_masked_training_matches_chmcnn():
    """Reproduce C-HMCNN's exact train_output expression from main.py."""
    layer = ShieldLayer(num_variables=4, requirements_filepath=SIMPLE_TXT)
    R = layer.R
    torch.manual_seed(2)
    output = torch.rand(8, 4)
    labels = (torch.rand(8, 4) > 0.5).double()

    # C-HMCNN main.py training block, verbatim.
    constr_output = get_constr_out(output, R)
    train_output = labels * output.double()
    train_output = get_constr_out(train_output, R)
    expected = (1 - labels) * constr_output.double() + labels * train_output

    ours = layer(output, goal=labels)
    assert torch.allclose(ours.double(), expected)


def test_gradients_flow():
    layer = ShieldLayer(num_variables=4, requirements_filepath=SIMPLE_TXT)
    preds = torch.rand(8, 4, requires_grad=True)
    labels = (torch.rand(8, 4) > 0.5).float()
    out = layer(preds, goal=labels)
    loss = torch.nn.BCELoss()(out.clamp(1e-6, 1 - 1e-6), labels)
    loss.backward()
    assert preds.grad is not None and torch.isfinite(preds.grad).all()


def test_build_via_dispatcher_txt():
    layer = build_shield_layer(4, SIMPLE_TXT, requirements_type='hierarchical')
    assert isinstance(layer, ShieldLayer)
    preds = torch.rand(4, 4)
    assert layer.satisfied(layer(preds)).all()


def test_arff_auto_detection():
    assert detect_requirements_type(PATHS_ARFF) == 'hierarchical'
    # num_variables=None adopts the parsed class count (paths format adds an implicit root).
    layer = build_shield_layer(None, PATHS_ARFF, requirements_type='auto')
    assert isinstance(layer, ShieldLayer)
    assert 'root' in layer.class_names


def test_buffer_moves_and_serialises():
    layer = ShieldLayer(num_variables=4, requirements_filepath=SIMPLE_TXT)
    assert 'R' in dict(layer.named_buffers())
    assert not layer.R.requires_grad
    state = layer.state_dict()
    assert 'R' in state


def test_cellcycle_fun_loads_and_is_coherent():
    # The real FUN benchmark (a tree): 499 classes + implicit root.
    layer = ShieldLayer(num_variables=None, requirements_filepath=CELLCYCLE_FUN)
    assert layer.num_classes == 500
    torch.manual_seed(0)
    preds = torch.rand(32, layer.num_classes)
    corrected = layer(preds)
    assert layer.satisfied(corrected).all()


def test_cellcycle_go_loads_and_is_coherent():
    # The real GO benchmark (a DAG): multi-parent classes, auto-detected as 'edges'.
    layer = ShieldLayer(num_variables=None, requirements_filepath=CELLCYCLE_GO)
    # A DAG: at least one class has more than one parent.
    parent_counts = {}
    for child, parent in layer.hierarchy.edges:
        parent_counts[child] = parent_counts.get(child, 0) + 1
    assert max(parent_counts.values()) > 1
    torch.manual_seed(0)
    preds = torch.rand(16, layer.num_classes)
    corrected = layer(preds)
    assert layer.satisfied(corrected).all()
