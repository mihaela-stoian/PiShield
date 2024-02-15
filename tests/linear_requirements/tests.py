import unittest

import torch
from pishield.linear_requirements.compute_sets_of_constraints import compute_sets_of_constraints
from pishield.linear_requirements.shield_layer import ShieldLayer
from pishield.linear_requirements.correct_predictions import check_all_constraints_are_sat
from pishield.linear_requirements.manual_correct import correct_preds
from pishield.linear_requirements.parser import parse_constraints_file
from example_predictions import example_predictions_url, example_predictions_botnet


class TestConstraintCorrection(unittest.TestCase):

    def test_tiny1(self):
        predictions = [-1.0, 5.0, 2.0]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/tiny_constraints1.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_tiny2(self):
        predictions = [-6.0, 15.0, 1.0]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/tiny_constraints2.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_tiny3(self):
        predictions = [-1.0, 5.0, -2.0]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/tiny_constraints3.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_tiny4(self):
        predictions = [-10.0, 5.0, -2.0, -9, 2, 20]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/tiny_constraints4.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_constants1(self):
        predictions = [-21.]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/constraints_constants1.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_constants2(self):
        predictions = [-5., 3.]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/constraints_constants2.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_constants3(self):
        predictions = [-5., 3.]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/constraints_constants3.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_equality1(self):
        predictions = [-5., -2., -1.]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/equality_constraints1.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_equality2(self):
        predictions = [-5., -2., -1.]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/equality_constraints2.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_equality3(self):
        predictions = [-5., -2., -1.]
        predictions = torch.tensor(predictions).unsqueeze(0)
        constraints_path = '../../data/linear_requirements/custom_constraints/equality_constraints3.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_url_predictions(self):
        predictions = example_predictions_url()
        constraints_path = '../../data/linear_requirements/url/url_constraints.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

    def test_botnet_predictions(self):
        predictions = example_predictions_botnet()
        constraints_path = '../../data/linear_requirements/botnet/botnet_constraints.txt'
        ordering_choice = 'given'
        self.apply_test_CL(predictions, constraints_path, ordering_choice)

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
        CL = ShieldLayer(num_variables, constraints_path, ordering_choice=ordering_choice)
        CL_corrected_preds = CL(predictions.clone())
        CL_all_sat = check_all_constraints_are_sat(constraints, predictions, CL_corrected_preds)
        print(CL_corrected_preds)
        self.assertTrue(CL_all_sat)
        self.assertFalse(CL_corrected_preds.sum().abs().isinf())
        diff = corrected_preds - CL_corrected_preds
        print('CL correction same as manual correction:', diff.abs().sum() == 0)
        print('differences:', diff[diff != 0])


if __name__ == '__main__':
    unittest.main()
