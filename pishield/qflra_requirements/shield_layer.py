"""The QFLRA Shield Layer, the main public entry point of this subpackage.

Defines :class:`ShieldLayer`, a differentiable PyTorch module that corrects neural
network predictions so they satisfy a set of QFLRA requirements.
"""

from typing import List, Union
import torch

from pishield.qflra_requirements.classes import Variable, Constraint, Atom
from pishield.qflra_requirements.compute_sets_of_constraints import get_pos_neg_x_constr, compute_sets_of_constraints
from pishield.qflra_requirements.correct_predictions import get_constr_at_level_x, get_final_x_correction, correct_preds
from pishield.qflra_requirements.feature_orderings import set_ordering
from pishield.qflra_requirements.parser import parse_constraints_file

INFINITY = torch.inf
EPSILON = 1e-12


class ShieldLayer(torch.nn.Module):
    """
    Differentiable layer that corrects predictions so they satisfy a set of QFLRA requirements:
    quantifier-free linear real arithmetic, i.e. disjunctions ('or') and negations ('neg') of linear
    inequalities.

    Like the linear ShieldLayer, the correction proceeds variable by variable following a fixed
    `ordering`, clipping each variable into the feasible region implied by its constraints given the
    already-corrected values of the preceding variables. The difference is that disjunctions make
    that feasible region a union of intervals rather than a single interval, so the per-variable
    correction (delegated to `correct_preds`) is more involved than in the purely conjunctive
    linear case.

    Attributes:
        num_variables: The number of variables (prediction dimensions).
        ordering: The ordering of variables used to drive the correction.
        constraints: The parsed list of QFLRA :class:`Constraint` objects.
        sets_of_constr: Mapping from each :class:`Variable` to its constraint set.

    Example:
        >>> layer = ShieldLayer(num_variables=2, requirements_filepath='constraints.txt')
        >>> corrected = layer(preds)  # preds is a (B, 2) tensor
    """

    def __init__(self, num_variables: int, requirements_filepath: str, ordering_choice: str = 'given'):
        """Build the Shield Layer from a requirements file.

        Args:
            num_variables: The number of variables (prediction dimensions).
            requirements_filepath: Path to the file holding the variable ordering and
                the QFLRA constraints.
            ordering_choice: How to order variables for correction; ``'given'`` keeps
                the file ordering, ``'random'`` shuffles it.
        """
        super().__init__()
        self.num_variables = num_variables
        # Read the variable ordering and the QFLRA constraints from the requirements file.
        ordering, constraints = parse_constraints_file(requirements_filepath)
        # Decide the order in which variables will be corrected ('given', 'random' or custom).
        self.ordering = set_ordering(ordering, ordering_choice)
        self.constraints = constraints
        # Group the constraints by the variable they bound (its "level" in the ordering).
        self.sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    def __call__(self, preds: torch.Tensor):
        """Correct a batch of predictions so they satisfy the requirements.

        Args:
            preds: Predictions tensor of shape ``(B, num_variables)``.

        Returns:
            A tensor of the same shape with predictions corrected to satisfy the
            QFLRA constraints.

        Example:
            >>> corrected = layer(preds)
        """
        # Run the per-variable correction pass over the ordering and return the corrected predictions.
        return correct_preds(preds, self.ordering, self.sets_of_constr)
