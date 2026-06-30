"""Differentiable correction module for one stratum of constraints.

A :class:`ConstraintsModule` holds the (precomputed, non-trainable) tensor encoding of a
single :class:`ConstraintsGroup` and corrects predictions so they satisfy that group.
Each constraint contributes a lower bound (for positive heads) or an upper bound (for
negative heads) on its head variable, derived from the truth value of its body; the
prediction is then clamped into those bounds. Two equivalent implementations are
provided: a vectorised 3D-tensor version (:meth:`apply_tensor`) and an iterative 2D
version (:meth:`apply_iter`). The Shield Layer stacks one such module per stratum.
"""

import torch

from torch import nn
from pishield.propositional_requirements.literal import Literal
from pishield.propositional_requirements.profiler import Profiler


class ConstraintsModule(nn.Module):
    """Corrects predictions to satisfy a single stratum of constraints.

    The module restricts attention to the atoms occurring in its constraints (re-indexed
    to a minimal range) and stores the encoded heads/bodies as buffers. Bodies are
    evaluated with a Goedel (min) semantics and used to bound the head predictions.

    Attributes:
        atoms: The original variable indices covered by this module's constraints.
        heads: The re-indexed head :class:`Literal` of each constraint.
        pos_head, neg_head: Encoded positive/negative head indicators per constraint.
        pos_body, neg_body: Encoded positive/negative body indicators per constraint.
        symm_body, symm_head: Precomputed symmetric (-1/+1) body/head encodings.
        literals_count: The number of literals in each constraint's body.
    """

    profiler = Profiler.shared().branch('cm')

    @profiler.wrap
    def __init__(self, constraints_group, num_classes):
        """Encode a constraints group into reusable correction tensors.

        Args:
            constraints_group: The :class:`ConstraintsGroup` (one stratum) to encode.
            num_classes: The total number of variables in the full prediction space.
        """
        super(ConstraintsModule, self).__init__()
        head, body = constraints_group.encoded(num_classes)
        pos_head, neg_head = head
        pos_body, neg_body = body

        # Compute necessary atoms
        self.atoms = nn.Parameter(torch.tensor(list(constraints_group.atoms())), requires_grad=False)
        reindexed = {float(atom): i for i, atom in enumerate(self.atoms)}
        if len(self.atoms) == 0: return

        # Reduce tensors to minimal size and reindex heads
        pos_head, neg_head = self.to_minimal(pos_head), self.to_minimal(neg_head)
        pos_body, neg_body = self.to_minimal(pos_body), self.to_minimal(neg_body)

        heads = [constraint.head for constraint in constraints_group]
        self.heads = [Literal(reindexed[head.atom], head.positive) for head in heads]

        # Module parameters
        self.pos_head = nn.Parameter(torch.from_numpy(pos_head).float(), requires_grad=False)
        self.neg_head = nn.Parameter(torch.from_numpy(neg_head).float(), requires_grad=False)
        self.pos_body = nn.Parameter(torch.from_numpy(pos_body).float(), requires_grad=False)
        self.neg_body = nn.Parameter(torch.from_numpy(neg_body).float(), requires_grad=False)

        # Precomputed parameters
        self.symm_body = nn.Parameter((self.pos_body - self.neg_body).t(), requires_grad=False)
        self.symm_head = nn.Parameter((self.pos_head - self.neg_head).t(), requires_grad=False)
        self.literals_count = nn.Parameter(self.pos_body.sum(dim=1) + self.neg_body.sum(dim=1), requires_grad=False)

    def dimensions(self, pred):
        """Return the (batch, num_atoms, num_constraints) dimensions for a tensor.

        Args:
            pred: A prediction tensor of shape (batch, num_atoms).

        Returns:
            A tuple ``(batch, num, cons)``.
        """
        batch, num = pred.shape[0], pred.shape[1]
        cons = self.pos_head.shape[0]
        return batch, num, cons

    @staticmethod
    @profiler.wrap
    def from_symmetric(preds):
        """Map symmetric values in [-1, 1] back to probabilities in [0, 1].

        Args:
            preds: A tensor of symmetric values.

        Returns:
            The corresponding probabilities.
        """
        return (preds + 1) / 2

    @staticmethod
    @profiler.wrap
    def to_symmetric(preds):
        """Map probabilities in [0, 1] to symmetric values in [-1, 1].

        Args:
            preds: A tensor of probabilities.

        Returns:
            The corresponding symmetric values.
        """
        return 2 * preds - 1

    @profiler.wrap
    def to_minimal(self, tensor, atoms=None):
        """Restrict a full tensor to the module's atom columns.

        Args:
            tensor: A tensor of shape (batch, num_classes).
            atoms: The atom indices to keep; defaults to ``self.atoms``.

        Returns:
            The tensor restricted to the selected atom columns.
        """
        if atoms is None: atoms = self.atoms
        return tensor[:, atoms].reshape(tensor.shape[0], len(atoms))

    @profiler.wrap
    def from_minimal(self, tensor, init, atoms=None):
        """Scatter a minimal-atom tensor back into a full tensor.

        Args:
            tensor: The minimal tensor over the module's atoms.
            init: The full tensor to copy the values into.
            atoms: The atom indices the values belong to; defaults to ``self.atoms``.

        Returns:
            ``init`` with the atom columns overwritten by ``tensor``.
        """
        if atoms is None: atoms = self.atoms
        return init.index_copy(1, atoms, tensor)

    # Get constraints with full sat body and those with unsat head
    @profiler.wrap
    def active_constraints(self, goal):
        """Identify which constraints are activated by a goal assignment.

        Args:
            goal: A ground-truth assignment over the module's atoms.

        Returns:
            A tuple ``(full_body, unsat_head)`` of boolean masks marking the
            constraints whose body is fully satisfied by ``goal`` and those whose head
            is unsatisfied by ``goal``.
        """
        symm_goal = ConstraintsModule.to_symmetric(goal)
        full_body = torch.matmul(symm_goal, self.symm_body) == self.literals_count
        unsat_head = torch.matmul(symm_goal, self.symm_head) == -1
        return full_body, unsat_head

        # Apply constraints together with 3D tensors

    @profiler.wrap
    def apply_tensor(self, preds, active_constraints=None, body_mask=None):
        """Correct predictions with a vectorised 3D-tensor computation.

        Computes, for every constraint, the Goedel (min) truth value of its body and
        uses it as a lower bound on positive heads and an upper bound on negative heads,
        then clamps each prediction into the resulting bounds.

        Args:
            preds: The predictions over the module's atoms, shape (batch, num_atoms).
            active_constraints: Optional per-constraint mask zeroing inactive
                constraints' contributions.
            body_mask: Optional per-atom mask used to drop body literals that the goal
                already determines.

        Returns:
            The corrected predictions, same shape as ``preds``.
        """
        batch, num, cons = self.dimensions(preds)

        # batch x cons x num: prepare (preds x body)
        exp_preds = preds.unsqueeze(1).expand(batch, cons, num)
        pos_body = self.pos_body.unsqueeze(0).expand(batch, cons, num)
        neg_body = self.neg_body.unsqueeze(0).expand(batch, cons, num)

        # batch x cons x num: ignore literals from constraints
        if body_mask != None:
            body_mask = body_mask.unsqueeze(1).expand(batch, cons, num)
            pos_body = pos_body * (1 - body_mask)
            neg_body = neg_body * body_mask

        # batch x cons: compute body minima
        body_rev = pos_body + exp_preds * (neg_body - pos_body)
        body_min = 1. - torch.max(body_rev, dim=2).values

        # batch x cons: ignore constraints
        if active_constraints != None:
            body_min = body_min * active_constraints.float()

        # batch x cons x num: prepare (body_min x head)
        body_min = body_min.unsqueeze(2).expand(batch, cons, num)
        pos_head = self.pos_head.unsqueeze(0).expand(batch, cons, num)
        neg_head = self.neg_head.unsqueeze(0).expand(batch, cons, num)

        # batch x num: compute head lower and upper bounds
        lb = torch.max(body_min * pos_head, dim=1).values
        ub = 1 - torch.max(body_min * neg_head, dim=1).values
        lb, ub = torch.minimum(lb, ub), torch.maximum(lb, ub)

        preds = torch.maximum(lb, torch.minimum(ub, preds))
        return preds

    # Apply constraints iteratively with 2D matrices
    @profiler.wrap
    def apply_iter(self, preds, active_constraints=None, body_mask=None, in_bounds=None, out_bounds=False):
        """Correct predictions by iterating constraint by constraint with 2D matrices.

        Equivalent to :meth:`apply_tensor` but loops over constraints, accumulating per
        atom lower/upper bounds; this avoids materialising the large 3D tensors. Can
        accept incoming bounds and/or return the raw bounds instead of corrected
        predictions, which the goal-conditioned path uses to chain two passes.

        Args:
            preds: The predictions over the module's atoms, shape (batch, num_atoms).
            active_constraints: Optional per-constraint mask zeroing inactive
                constraints' contributions.
            body_mask: Optional per-atom mask used to drop body literals.
            in_bounds: Optional ``(lb, ub)`` lists of incoming per-atom bounds to start
                from; defaults to ``[0, 1]`` per atom.
            out_bounds: If True, return the raw ``(lb, ub)`` bounds rather than the
                clamped predictions.

        Returns:
            Either the corrected predictions, or the ``(lb, ub)`` bounds when
            ``out_bounds`` is True.
        """
        batch, num, cons = self.dimensions(preds)
        device = preds.device

        if not active_constraints is None: active_constraints = active_constraints.float()
        zeros = torch.zeros(batch, 1, device=device)

        profiler = ConstraintsModule.profiler.branch('iter')

        with profiler.watch('init'):
            if in_bounds is None:
                lb = [torch.zeros(preds.shape[0], device=device) for i in range(preds.shape[1])]
                ub = [torch.ones(preds.shape[0], device=device) for i in range(preds.shape[1])]
            else:
                lb, ub = in_bounds

        with profiler.watch('precompute'):
            bool_pos_body = self.pos_body.bool()
            bool_neg_body = self.neg_body.bool()

            full_pos_body = 1 - preds
            full_neg_body = preds

            if not body_mask is None:
                full_pos_body = (1 - preds) * (1 - body_mask)
                full_neg_body = preds * body_mask

        for c, lit in enumerate(self.heads):
            # slice positive and negative body preds
            with profiler.watch('where'):
                pos_where = bool_pos_body[c]
                neg_where = bool_neg_body[c]

            # body predictions (possibly masked) 
            with profiler.watch('body'):
                pos_body = full_pos_body[:, pos_where]
                neg_body = full_neg_body[:, neg_where]

            # compute maximal inverted values
            with profiler.watch('candidate'):
                candidate = torch.cat((zeros, pos_body, neg_body), dim=1)
                candidate = 1 - candidate.max(dim=1).values

            # clear inactive constraints
            with profiler.watch('active_cons'):
                if not active_constraints is None:
                    candidate = candidate * active_constraints[:, c]

            # update preds
            with profiler.watch('min_max'):
                if lit.positive:
                    lb[lit.atom] = torch.maximum(lb[lit.atom], candidate)
                else:
                    ub[lit.atom] = torch.minimum(ub[lit.atom], 1 - candidate)

        with profiler.watch('lb_ub'):
            if out_bounds:
                return lb, ub

            lb, ub = torch.stack(lb, dim=1), torch.stack(ub, dim=1)
            lb, ub = torch.minimum(lb, ub), torch.maximum(lb, ub)
            updated = torch.maximum(lb, torch.minimum(ub, preds))

            return updated

    @profiler.wrap
    def apply(self, preds, iterative):
        """Correct predictions using the chosen implementation.

        Args:
            preds: The predictions over the module's atoms.
            iterative: If True use :meth:`apply_iter`, else :meth:`apply_tensor`.

        Returns:
            The corrected predictions.
        """
        if iterative:
            return self.apply_iter(preds)
        else:
            return self.apply_tensor(preds)

    @profiler.wrap
    def apply_goal(self, preds, goal, iterative):
        """Correct predictions consistently with a known goal assignment.

        Applies the constraints in two passes: first using only constraints whose body
        is fully satisfied by the goal (to propagate firm consequences), then using the
        constraints whose head is unsatisfied with the goal-determined body literals
        masked out.

        Args:
            preds: The predictions over the module's atoms.
            goal: The goal (ground-truth) assignment over the module's atoms.
            iterative: If True use the iterative implementation, else the tensor one.

        Returns:
            The corrected predictions.
        """
        full_body, unsat_head = self.active_constraints(goal)
        body_mask = goal

        if iterative:
            bounds = self.apply_iter(preds, active_constraints=full_body, out_bounds=True)
            updated = self.apply_iter(preds, active_constraints=unsat_head, body_mask=body_mask, in_bounds=bounds)
        else:
            updated = self.apply_tensor(preds, active_constraints=full_body)
            updated = self.apply_tensor(updated, active_constraints=unsat_head, body_mask=body_mask)

        return updated

    @profiler.wrap
    def forward(self, preds, goal=None, iterative=True):
        """Correct a full prediction tensor against this stratum's constraints.

        Restricts the predictions to the module's atoms, applies the correction
        (optionally goal-conditioned), and scatters the result back into the full
        tensor. Returns the input unchanged when there are no predictions or no atoms.

        Args:
            preds: The full prediction tensor, shape (batch, num_classes).
            goal: Optional goal assignment over the full variable space; when given,
                corrections are made consistent with it.
            iterative: If True use the iterative implementation, else the tensor one.

        Returns:
            The full prediction tensor with this stratum's atoms corrected.
        """
        if len(preds) == 0 or len(self.atoms) == 0:
            return preds

        updated = self.to_minimal(preds)

        if goal is None:
            updated = self.apply(updated, iterative=iterative)
            return self.from_minimal(updated, preds)
        else:
            goal = self.to_minimal(goal)
            updated = self.apply_goal(updated, goal=goal, iterative=iterative)
            return self.from_minimal(updated, preds)


def run_cm(cm, preds, goal=None, device='cpu'):
    """Run a constraints module both ways and assert they agree.

    Runs the module with the iterative and tensor implementations and checks the
    results are numerically close; mainly a testing/debugging helper.

    Args:
        cm: The :class:`ConstraintsModule` to run.
        preds: The prediction tensor.
        goal: Optional goal assignment.
        device: The torch device to run on.

    Returns:
        The corrected predictions (from the iterative implementation), on CPU.

    Raises:
        AssertionError: If the two implementations disagree.
    """
    cm, preds = cm.to(device), preds.to(device)
    if not goal is None: goal = goal.to(device)

    iter = cm(preds, goal=goal, iterative=True)
    tens = cm(preds, goal=goal, iterative=False)
    assert torch.isclose(iter, tens).all()
    return iter.cpu()
