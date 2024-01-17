import torch

from cloverd.propositional_constraints.slicer import Slicer


def test_slicer():
    slicer = Slicer({0, 2, 3}, 2)
    preds = torch.rand((100, 5))

    assert (slicer.slice_atoms(preds) == preds[:, [0, 2, 3]]).all()
    assert slicer.slice_modules([0, 2, 4, 6, 8]) == [0, 2]
