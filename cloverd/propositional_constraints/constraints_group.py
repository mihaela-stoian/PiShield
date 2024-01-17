import numpy as np
import networkx as nx
from cloverd.propositional_constraints.constraint import Constraint
from cloverd.propositional_constraints.literal import Literal


class ConstraintsGroup:
    def __init__(self, arg):
        if isinstance(arg, str):
            # ConstraintGroup(string)
            with open(arg, 'r') as f:
                self.constraints = [Constraint(line) for line in f]
        else:
            # ConstraintGroup([Constraint])
            self.constraints = arg

        # Keep the initial order of constraints for coherent_with
        self.constraints_list = self.constraints
        self.constraints = frozenset(self.constraints_list)

    def __add__(self, other):
        return ConstraintsGroup(self.constraints.union(other.constraints))

    def __str__(self):
        return '\n'.join([str(constraint) for constraint in sorted(self.constraints)])

    def __iter__(self):
        return iter(self.constraints)

    def __eq__(self, other):
        if not isinstance(other, ConstraintsGroup): return False
        return self.constraints == other.constraints

    def __len__(self):
        return len(self.constraints)

    def head_encoded(self, num_classes):
        pos_head = []
        neg_head = []

        for constraint in self.constraints:
            pos, neg = constraint.head_encoded(num_classes)
            pos_head.append(pos)
            neg_head.append(neg)

        return np.array(pos_head), np.array(neg_head)

    def body_encoded(self, num_classes):
        pos_body = []
        neg_body = []

        for constraint in self.constraints:
            pos, neg = constraint.body_encoded(num_classes)
            pos_body.append(pos)
            neg_body.append(neg)

        return np.array(pos_body), np.array(neg_body)

    def encoded(self, num_classes):
        head = self.head_encoded(num_classes)
        body = self.body_encoded(num_classes)
        return head, body

    def coherent_with(self, preds):
        coherent = [constraint.coherent_with(preds) for constraint in self.constraints_list]
        return np.array(coherent).transpose()

    def atoms(self):
        atoms = set()
        for constraint in self.constraints:
            atoms = atoms.union(constraint.atoms())
        return atoms

    def heads(self):
        heads = set()
        for constraint in self.constraints:
            heads.add(constraint.head.atom)
        return heads

    def graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.atoms())

        for constraint in self.constraints:
            for lit in constraint.body:
                x = lit.atom
                y = constraint.head.atom
                G.add_edge(x, y)
                G[x][y]['body'] = lit.positive
                G[x][y]['head'] = constraint.head.positive

        return G

    def duograph(self):
        atoms = self.atoms()
        pos_atoms = [str(Literal(atom, True)) for atom in atoms]
        neg_atoms = [str(Literal(atom, False)) for atom in atoms]

        G = nx.DiGraph()
        G.add_nodes_from(pos_atoms + neg_atoms)

        for constraint in self.constraints:
            for lit in constraint.body:
                G.add_edge(str(lit), str(constraint.head))

        return G

    def stratify(self):
        G = self.graph()

        for node in G.nodes():
            G.nodes[node]['deps'] = 0
            G.nodes[node]['constraints'] = []

        for x, y in G.edges():
            G.nodes[y]['deps'] += 1

        for constraint in self.constraints:
            G.nodes[constraint.head.atom]['constraints'].append(constraint)

        result = []
        ready = [node for node in G.nodes() if G.nodes[node]['deps'] == 0]
        while len(ready) > 0:
            resolved = [cons for node in ready for cons in G.nodes[node]['constraints']]
            if len(resolved) > 0:
                result.append(ConstraintsGroup(resolved))

            next = []
            for node in ready:
                for other in G[node]:
                    G.nodes[other]['deps'] -= 1
                    if G.nodes[other]['deps'] == 0:
                        next.append(other)

            ready = next

        return result
