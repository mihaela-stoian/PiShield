import unittest
from typing import List
import torch

from cloverd.compute_sets_of_constraints import compute_sets_of_constraints
from cloverd.constraint_layer import ConstraintLayer
from cloverd.correct_predictions import check_all_constraints_are_sat
from cloverd.manual_correct import correct_preds
from cloverd.parser import parse_constraints_file
from example_predictions import example_predictions_url, example_predictions_botnet


def make_constraints(filename: str, preds: List[float]):
    ordering, constraints = parse_constraints_file(filename)

    # set ordering to random
    # ordering = set_random_ordering(ordering)

    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    predictions = torch.tensor(preds)
    predictions.requires_grad = True
    predictions = torch.stack([predictions, predictions, predictions, predictions, predictions], dim=0)

    corrected_preds = correct_preds(predictions, ordering, sets_of_constr)
    all_sat = check_all_constraints_are_sat(constraints, predictions, corrected_preds)
    print(predictions[0])
    print(corrected_preds[0])
    return predictions, corrected_preds, all_sat

class TestConstraintCorrection(unittest.TestCase):

    def test_tiny1(self):
        predictions = [-1.0, 5.0, 2.0]
        # expected_correction = torch.tensor([-1.0, 5.0, 2.0])
        # expected_correction = torch.stack([expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/tiny_constraints1.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_tiny2(self):
        predictions = [-6.0, 15.0, 1.0]
        # expected_correction = torch.tensor([-6.0, 15.0, 1.0])
        # expected_correction = torch.stack([expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/tiny_constraints2.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_tiny3(self):
        predictions = [-1.0, 5.0, -2.0]
        # expected_correction = torch.tensor([-1., -0.75, -2.])
        # expected_correction = torch.stack([expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/tiny_constraints3.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)


    def test_constraints(self):
        predictions = [-10.0, 5.0, -2.0, -9, 2, 20]
        # expected_correction = torch.tensor([-10.,  10.,  -2.,  -8.,   2., -10.])
        # expected_correction = torch.stack([expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/constraints.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_disj1(self):
        predictions = [-10.0, 5.0, -2.0, -9, 2, 20, -1]
        # expected_correction = torch.tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.])
        # expected_correction = torch.stack([expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/constraints_disj1.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_disj2(self):
        predictions = [-10.0, 5.0, -2.0, -9, 2, 20, -1]
        # expected_correction = torch.tensor([-10.,  10.,  -2.,  -8.,   2., -10.,  -1.])
        # expected_correction = torch.stack([expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/constraints_disj2.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_constants1(self):
        predictions = [-21.]
        # expected_correction = torch.tensor([3.])
        # expected_correction = torch.stack([expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/constraints_constants1.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_constants2(self):
        predictions = [-5., 3.]
        # expected_correction = torch.tensor([4., 3.])
        # expected_correction = torch.stack([expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/constraints_constants2.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_constants3(self):
        predictions = [-5., 3.]
        # expected_correction = torch.tensor([0.5, 3.])
        # expected_correction = torch.stack([expected_correction, expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/constraints_constants3.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_equality1(self):
        predictions = [-5., -2., -1.]
        # expected_correction = torch.tensor([0.5, 3.])
        # expected_correction = torch.stack([expected_correction, expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/equality_constraints1.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_equality2(self):
        predictions = [-5., -2., -1.]
        # expected_correction = torch.tensor([0.5, 3.])
        # expected_correction = torch.stack([expected_correction, expected_correction, expected_correction],dim=0)
        predictions, corrected_preds, all_sat = make_constraints('../data/equality_constraints2.txt', predictions)
        # self.assertTrue(expected_correction.eq(corrected_preds).all(), f'Expected {expected_correction} but got {corrected_preds}')
        self.assertTrue(all_sat)

    def test_url_predictions(self):
        predictions = example_predictions_url()
        constraints_path = '../data/url/url_constraints.txt'
        ordering_choice = 'random'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)


    def test_botnet_predictions(self):
        predictions = example_predictions_botnet()
        predictions.requires_grad = True

        ordering, constraints = parse_constraints_file(('../data/botnet/botnet_constraints.txt'))
        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        print('\n\n Start correcting predictions')
        predictions = torch.tensor(predictions)
        predictions.requires_grad = True
        # predictions = predictions.clamp(-1000,1000)

        corrected_preds = correct_preds(predictions, ordering, sets_of_constr)
        all_sat = check_all_constraints_are_sat(constraints, predictions, corrected_preds)
        print(predictions)
        print(corrected_preds)
        self.assertTrue(all_sat)
        self.assertFalse(corrected_preds.sum().abs().isinf())

    def apply_test_CL(self, predictions, constraints_path, ordering_choice):
        ordering, constraints = parse_constraints_file((constraints_path))
        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        print('\n\n Start correcting predictions')

        predictions = torch.tensor(predictions)
        predictions.requires_grad = True
        corrected_preds = correct_preds(predictions.clone(), ordering, sets_of_constr)
        all_sat = check_all_constraints_are_sat(constraints, predictions, corrected_preds)
        print(predictions)
        print(corrected_preds)
        self.assertTrue(all_sat)
        self.assertFalse(corrected_preds.sum().abs().isinf())

        num_variables = predictions.shape[-1]
        CL = ConstraintLayer(ordering_choice, constraints_path, num_variables)
        CL_corrected_preds = CL(predictions.clone())
        CL_all_sat = check_all_constraints_are_sat(constraints, predictions, CL_corrected_preds)
        self.assertTrue(CL_all_sat)
        self.assertFalse(CL_corrected_preds.sum().abs().isinf())
        print('CL correction same as manual correction:', (corrected_preds-CL_corrected_preds).abs().sum() == 0)


if __name__ == '__main__':
    unittest.main()
