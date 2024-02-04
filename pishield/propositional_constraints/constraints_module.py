import torch

from torch import nn
from pishield.propositional_constraints.literal import Literal
from pishield.propositional_constraints.profiler import Profiler


class ConstraintsModule(nn.Module):
    profiler = Profiler.shared().branch('cm')

    @profiler.wrap
    def __init__(self, constraints_group, num_classes):
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
        batch, num = pred.shape[0], pred.shape[1]
        cons = self.pos_head.shape[0]
        return batch, num, cons

    @staticmethod
    @profiler.wrap
    def from_symmetric(preds):
        return (preds + 1) / 2

    @staticmethod
    @profiler.wrap
    def to_symmetric(preds):
        return 2 * preds - 1

    @profiler.wrap
    def to_minimal(self, tensor, atoms=None):
        if atoms is None: atoms = self.atoms
        return tensor[:, atoms].reshape(tensor.shape[0], len(atoms))

    @profiler.wrap
    def from_minimal(self, tensor, init, atoms=None):
        if atoms is None: atoms = self.atoms
        return init.index_copy(1, atoms, tensor)

    # Get constraints with full sat body and those with unsat head
    @profiler.wrap
    def active_constraints(self, goal):
        symm_goal = ConstraintsModule.to_symmetric(goal)
        full_body = torch.matmul(symm_goal, self.symm_body) == self.literals_count
        unsat_head = torch.matmul(symm_goal, self.symm_head) == -1
        return full_body, unsat_head

        # Apply constraints together with 3D tensors

    @profiler.wrap
    def apply_tensor(self, preds, active_constraints=None, body_mask=None):
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
        if iterative:
            return self.apply_iter(preds)
        else:
            return self.apply_tensor(preds)

    @profiler.wrap
    def apply_goal(self, preds, goal, iterative):
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
    cm, preds = cm.to(device), preds.to(device)
    if not goal is None: goal = goal.to(device)

    iter = cm(preds, goal=goal, iterative=True)
    tens = cm(preds, goal=goal, iterative=False)
    assert torch.isclose(iter, tens).all()
    return iter.cpu()
