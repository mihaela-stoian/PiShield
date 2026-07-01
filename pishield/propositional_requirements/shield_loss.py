"""The propositional Memory-efficient Loss.

Defines :class:`ShieldLoss`, a t-norm based penalty term that encourages (but does not
enforce) the satisfaction of propositional requirements. It is a memory-efficient
t-norm loss inspired by Logic Tensor Networks (LTN). Each requirement is read as a
disjunction (clause) and its degree of satisfaction is computed under one of three
t-norms - Goedel, Lukasiewicz or product - using sparse matrix representations of the
requirements.
"""

import numpy as np
import torch


class ShieldLoss(torch.nn.Module):
    """
    The Memory-efficient Loss: a t-norm based loss term that encourages the satisfaction of
    propositional requirements. It is a memory-efficient t-norm loss inspired by Logic Tensor
    Networks (LTN).

    Unlike the Shield Layer, the Memory-efficient Loss does not correct the predictions; it
    returns a scalar penalty (in [0, 1]) which is minimised when the requirements are satisfied.
    The penalty is computed using one of three t-norms: 'godel', 'product' or 'lukasiewicz'.

    The requirements are read from a file whose lines have the form ``head :- body``, where
    ``head`` is a single literal and ``body`` is a (possibly empty) list of literals. A literal
    is the index of a variable (e.g. ``3``) for a positive literal, or that index prefixed with
    ``n`` (e.g. ``n3``) for a negative literal.
    """

    def __init__(self, num_variables: int, requirements_filepath: str, tnorm_choice: str = 'godel'):
        """Load the requirements and precompute the t-norm matrices.

        Args:
            num_variables: The number of variables (labels) the predictions cover.
            requirements_filepath: Path to the requirements file (one ``head :- body``
                rule per line).
            tnorm_choice: The t-norm to use: ``'godel'``, ``'lukasiewicz'`` or
                ``'product'``.

        Example:
            >>> loss_fn = ShieldLoss(num_variables=10,
            ...                      requirements_filepath='constraints.txt',
            ...                      tnorm_choice='product')
            >>> penalty = loss_fn(predictions)  # predictions: (batch, 10)
        """
        super().__init__()
        self.num_variables = num_variables
        self.requirements_filepath = requirements_filepath
        self.tnorm_choice = tnorm_choice
        self.create_matrices()

    def create_matrices(self):
        """Build the positive/negative literal matrices used by the t-norm losses.

        Combines the body (``I``) and head (``M``) encodings into ``Cplus`` and
        ``Cminus`` matrices marking the positive and negative literal appearances in
        each requirement's disjunction (the exact combination depends on the chosen
        t-norm), and records the number of requirements.
        """
        Iplus_np, Iminus_np = self.createIs()
        Mplus_np, Mminus_np = self.createMs()

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
        """Encode the body literals of each requirement into indicator matrices.

        Reads the requirements file and, for each rule's body, marks which variables
        appear as positive (``Iplus``) and negative (``Iminus``) literals.

        Returns:
            A tuple ``(Iplus, Iminus)`` of arrays of shape (num_requirements,
            num_variables).
        """
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
        """Encode the head literal of each requirement into indicator matrices.

        Reads the requirements file and marks, per requirement, whether its head is a
        positive (``Mplus``) or negative (``Mminus``) literal at the head's variable.

        Returns:
            A tuple ``(Mplus, Mminus)`` of arrays of shape (num_variables,
            num_requirements); each column corresponds to a requirement and carries a 1
            at the head variable's row for the matching polarity.
        """
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
        """Return the sparse indices and values of a requirement matrix.

        Args:
            req_matrix: A dense requirement matrix.

        Returns:
            A tuple ``(indices, values)`` of the matrix's non-zero coordinates and
            their values.
        """
        req_matrix = req_matrix.to_sparse()
        return req_matrix.indices(), req_matrix.values()

    def godel_disjunctions_sparse(self, preds, weighted_literals=False):
        """Compute the Goedel-t-norm requirement penalty.

        Each requirement's satisfaction degree is the maximum over its literals' truth
        values; the penalty is one minus the mean satisfaction degree.

        Args:
            preds: The predicted probabilities, shape (batch, num_variables).
            weighted_literals: If True, weight each literal by its matrix value.

        Returns:
            A scalar penalty in [0, 1], minimised when the requirements are satisfied.
        """
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
        """Compute the Lukasiewicz-t-norm requirement penalty.

        Each requirement's satisfaction degree is the sum of its literals' truth
        values clamped to 1; the penalty is one minus the mean satisfaction degree.

        Args:
            preds: The predicted probabilities, shape (batch, num_variables).
            weighted_literals: If True, weight each literal by its matrix value.

        Returns:
            A scalar penalty in [0, 1], minimised when the requirements are satisfied.
        """
        constr_values_unbounded = torch.zeros(preds.shape[0], self.NUM_REQ).to(preds.device)

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
            constr_values_unbounded[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] += \
                predictions_at_nnz_values_plus[:, indices_nnz_plus[1] == k]
            constr_values_unbounded[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] += \
                predictions_at_nnz_values_minus[:, indices_nnz_minus[1] == k]

        constr_values = torch.min(torch.ones_like(constr_values_unbounded), constr_values_unbounded)
        req_loss = torch.mean(constr_values)

        # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements,
        # and hence we want to minimize the 1-p
        return 1 - req_loss

    def product_disjunctions_sparse(self, preds, weighted_literals=False):
        """Compute the product-t-norm requirement penalty.

        The disjunction is computed as the negation of the product of the negated
        literals; the penalty is one minus the mean satisfaction degree.

        Args:
            preds: The predicted probabilities, shape (batch, num_variables).
            weighted_literals: If True, weight each literal by its matrix value.

        Returns:
            A scalar penalty in [0, 1], minimised when the requirements are satisfied.
        """
        # The disjunction is more complex to implement than the conjunction
        # e.g., A and B --> A*B while A or B --> A + B - A*B
        # Thus we see the disjunction as the negation of the conjunction of the negations of all its
        # literals (i.e., A or B = neg (neg A and neg B))

        constr_values = torch.ones(preds.shape[0], self.NUM_REQ).to(preds.device)

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
            constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] *= \
                predictions_at_nnz_values_plus[:, indices_nnz_plus[1] == k]
            constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] *= \
                predictions_at_nnz_values_minus[:, indices_nnz_minus[1] == k]

        # Negate the value of the conjunction
        req_loss = torch.mean(1. - constr_values)

        # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements,
        # and hence we want to minimize the 1-p
        return 1 - req_loss

    def __call__(self, preds: torch.Tensor):
        """Compute the requirement-satisfaction penalty for a batch of predictions.

        Dispatches to the configured t-norm. Note that the caller must first slice
        ``preds`` down to exactly the ``num_variables`` columns the requirements refer
        to. Returns a zero scalar for an empty batch.

        Args:
            preds: The predicted probabilities, shape (batch, num_variables).

        Returns:
            A scalar penalty in [0, 1], minimised when the requirements are satisfied.

        Raises:
            ValueError: If the configured t-norm is not recognised.

        Example:
            >>> penalty = loss_fn(predictions[:, 1:num_variables + 1])
            >>> total_loss = task_loss + penalty
        """
        # Discard all the labels that should not be affecting the tnorm based loss before this call!
        # e.g., preds = original_preds[:, 1:self.num_variables + 1]

        if len(preds) == 0:
            return torch.zeros(1, device=preds.device).squeeze()

        if self.tnorm_choice == "godel":
            return self.godel_disjunctions_sparse(preds)
        elif self.tnorm_choice == "lukasiewicz":
            return self.lukasiewicz_disjunctions_sparse(preds)
        elif self.tnorm_choice == "product":
            return self.product_disjunctions_sparse(preds)
        else:
            raise ValueError("tnorm {:} not defined".format(self.tnorm_choice))
