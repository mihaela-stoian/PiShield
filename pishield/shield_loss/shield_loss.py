from typing import List, Union
import torch
import numpy as np

from pishield.linear_requirements.classes import Variable, Constraint, Atom
from pishield.linear_requirements.compute_sets_of_constraints import get_pos_neg_x_constr, compute_sets_of_constraints
from pishield.linear_requirements.correct_predictions import get_constr_at_level_x, get_final_x_correction
from pishield.linear_requirements.feature_orderings import set_ordering
from pishield.linear_requirements.parser import parse_constraints_file, split_constraints

INFINITY = torch.inf
EPSILON = 1e-12


class ShieldLoss(torch.nn.Module):
    def __init__(self, num_variables: int, requirements_filepath: str, tnorm_choice: str = 'godel'):
        super().__init__()
        self.num_variables = num_variables
        self.requirements_filepath = requirements_filepath
        self.tnorm_choice = tnorm_choice
        self.create_matrices()


    def create_matrices(self):
        Iplus_np, Iminus_np = self.createIs(self.requirements_filepath, self.num_variables)
        Mplus_np, Mminus_np = self.createMs(self.requirements_filepath, self.num_variables)

        Iplus, Iminus = torch.from_numpy(Iplus_np).float(), torch.from_numpy(Iminus_np).float()
        Mplus, Mminus = torch.from_numpy(Mplus_np).float(), torch.from_numpy(Mminus_np).float()

        if self.tnorm_choice == "product":
            # These are already the negated literals
            # matrix of negative appearances in the conjunction
            Cminus = Iminus + torch.transpose(Mplus, 0, 1)
            # matrix of positive appearances in the conjunction
            Cplus = Iplus + torch.transpose(Mminus, 0, 1)
        else:  # elif args.LOGIC == "Godel" or args.LOGIC == "Lukasiewicz":
            # These are the literals as they appear in the disjunction
            # Matrix of the positive appearances in the disjunction
            Cplus = Iminus + torch.transpose(Mplus, 0, 1)
            # matrix of negative appearances in the conjunction
            Cminus = Iplus + torch.transpose(Mminus, 0, 1)

        self.Cplus = Cplus
        self.Cminus = Cminus
        self.NUM_REQ = Iplus.shape[0]


    def createIs(self):
        # Matrix with indices for positive literals
        Iplus = []
        # Matrix with indeces for negative literals
        Iminus = []
        with open(self.requirements_filepath, 'r') as f:
            for line in f:
                split_line = line.split()
                assert split_line[2] == ':-', "Instead of :- found: %s" % split_line[2]
                iplus = np.zeros(self.num_variables)
                iminus = np.zeros(self.num_variables)
                for item in split_line[3:]:
                    if 'n' in item:
                        index = int(item[1:])
                        iminus[index] = 1
                    else:
                        index = int(item)
                        iplus[index] = 1
                Iplus.append(iplus)
                Iminus.append(iminus)
        Iplus = np.array(Iplus)
        Iminus = np.array(Iminus)
        return Iplus, Iminus


    # createMs returns two matrices: Mplus: shape [num_labels, num_constraints] --> each column corresponds to a
    # constraint and it has a one if the constraint has positive head at the column number of the label of the head
    # Mminus: shape[num_labels, num_constraints] --> each column corresponds to a constraint and it has a one if the
    # constraint has negative head at the column number of the label of the head
    def createMs(self):
        Mplus, Mminus = [], []
        with open(self.requirements_filepath, 'r') as f:
            for line in f:
                split_line = line.split()
                assert split_line[2] == ':-'
                mplus = np.zeros(self.num_variables)
                mminus = np.zeros(self.num_variables)
                if 'n' in split_line[1]:
                    # one indentified that is negative, ignore the 'n' to get the index
                    index = int(split_line[1][1:])
                    mminus[index] = 1
                else:
                    index = int(split_line[1])
                    mplus[index] = 1
                Mplus.append(mplus)
                Mminus.append(mminus)
        Mplus = np.array(Mplus).transpose()
        Mminus = np.array(Mminus).transpose()

        return Mplus, Mminus

    def get_sparse_representation(self, req_matrix):
        req_matrix = req_matrix.to_sparse()
        return req_matrix.indices(), req_matrix.values()


    def godel_disjunctions_sparse(self, preds, weighted_literals=False):
        constr_values = torch.zeros(preds.shape[0], self.NUM_REQ).to(preds.device)

        indices_nnz_plus, values_nnz_plus = self.get_sparse_representation(self.Cplus)
        indices_nnz_minus, values_nnz_minus = self.get_sparse_representation(self.Cminus)

        # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
        # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
        predictions_at_nnz_values_plus = preds[:, indices_nnz_plus[1, :]]
        predictions_at_nnz_values_minus = (1. - preds[:, indices_nnz_minus[1, :]])
        if weighted_literals:
            predictions_at_nnz_values_plus *= values_nnz_plus
            predictions_at_nnz_values_minus *= values_nnz_minus

        # the line inside the loop below essentially means that:
        # the constraints containing label k are each multiplied by the value of the prediction for label k
        for k in range(self.num_variables):
            # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
            # positively in the conjunction
            # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
            # indexes is equal to k
            constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] = torch.maximum(
                constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]],
                predictions_at_nnz_values_plus[:, indices_nnz_plus[1] == k])
            constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] = torch.maximum(
                constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]],
                predictions_at_nnz_values_minus[:, indices_nnz_minus[1] == k])

        req_loss = torch.mean(constr_values)
        # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements,
        # and hence we want to minimize the 1-p
        return 1 - req_loss

    def lukasiewicz_disjunctions_sparse(self, preds, weighted_literals=False):
        constr_values_unbounded = torch.zeros(preds.shape[0], self.NUM_REQ).to(preds.device)

        # pred = sH.clone()  # grads propagated through the cloned pred tensor are propagated through the
        # original sH tensor as well (so grads are updated through sH, which is what we want)
        indices_nnz_plus, values_nnz_plus = self.get_sparse_representation(self.Cplus)
        indices_nnz_minus, values_nnz_minus = self.get_sparse_representation(self.Cminus)

        # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
        # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
        predictions_at_nnz_values_plus = preds[:, indices_nnz_plus[1, :]]
        predictions_at_nnz_values_minus = (1. - preds[:, indices_nnz_minus[1, :]])
        if weighted_literals:
            predictions_at_nnz_values_plus *= values_nnz_plus
            predictions_at_nnz_values_minus *= values_nnz_minus

        # the line inside the loop below essentially means that:
        # the constraints containing label k are each multiplied by the value of the prediction for label k
        for k in range(self.num_variables):
            # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
            # positively in the conjunction
            # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
            # indexes is equal to k
            constr_values_unbounded[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] += predictions_at_nnz_values_plus[
                                                                                         :,
                                                                                         indices_nnz_plus[1] == k]
            constr_values_unbounded[:,
            indices_nnz_minus[0, indices_nnz_minus[1] == k]] += predictions_at_nnz_values_minus[
                                                                :,
                                                                indices_nnz_minus[1] == k]

        constr_values = torch.min(torch.ones_like(constr_values_unbounded), constr_values_unbounded)
        req_loss = torch.mean(constr_values)

        # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements, and hence we want to minimize the 1-p
        return 1 - req_loss

    def product_disjunctions_sparse(self, preds, weighted_literals=False):
        # The disjunction is more complex to implement thant the conjunction
        # e.g., A and B --> A*B while A or B --> A + B - A*B
        # Thus we see the disjunction as the negation of the conjunction of the negations of all its
        # literals (i.e., A or B = neg (neg A and neg B))

        constr_values = torch.ones(preds.shape[0], self.NUM_REQ).to(preds.device)

        # pred = sH.clone()  # grads propagated through the cloned pred tensor are propagated through the
        # original sH tensor as well (so grads are updated through sH, which is what we want)
        indices_nnz_plus, values_nnz_plus = self.get_sparse_representation(self.Cplus)
        indices_nnz_minus, values_nnz_minus = self.get_sparse_representation(self.Cminus)

        # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
        # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
        predictions_at_nnz_values_plus = preds[:, indices_nnz_plus[1, :]]
        predictions_at_nnz_values_minus = (1. - preds[:, indices_nnz_minus[1, :]])
        if weighted_literals:
            predictions_at_nnz_values_plus *= values_nnz_plus
            predictions_at_nnz_values_minus *= values_nnz_minus

        # the line inside the loop below essentially means that:
        # the constraints containing label k are each multiplied by the value of the prediction for label k
        for k in range(self.num_variables):
            # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
            # positively in the conjunction
            # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
            # indexes is equal to k
            constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] *= predictions_at_nnz_values_plus[:,
                                                                               indices_nnz_plus[1] == k]
            constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] *= predictions_at_nnz_values_minus[:,
                                                                                 indices_nnz_minus[1] == k]

        # Negate the value of the conjunction
        req_loss = torch.mean(1. - constr_values)

        # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements,
        # and hence we want to minimize the 1-p
        return 1 - req_loss


    def __call__(self, preds: torch.Tensor):
        # Discard all the labels that should not be affecting the tnorm based loss before this call!
        # e.g., preds = original_preds[:, 1:self.num_variables + 1]  # e.g.,

        if len(preds) == 0:
            tnorm_loss = torch.zeros(1).cuda().squeeze()
            return tnorm_loss

        Cplus, Cminus = self.Cplus.squeeze(), self.Cminus.squeeze()
        tnorm_loss = torch.zeros([1]).cuda()

        if self.tnorm_choice == "godel":
            tnorm_loss = self.godel_disjunctions_sparse(preds, Cplus, Cminus)
        elif self.tnorm_choice == "lukasiewicz":
            tnorm_loss = self.lukasiewicz_disjunctions_sparse(preds, Cplus, Cminus)
        elif self.tnorm_choice == "product":
            tnorm_loss = self.product_disjunctions_sparse(preds, Cplus, Cminus)
        else:
            print("tnorm {:} not defined".format(self.tnorm_choice))
            exit(1)

        return tnorm_loss

