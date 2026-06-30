"""Collections of propositional constraints.

A :class:`ConstraintsGroup` bundles a set of :class:`Constraint` Horn rules, supports
encoding them into matrices, checking coherency of predictions, building dependency
graphs over the variables, and stratifying the constraints into ordered layers so that
each head is only resolved after the variables it depends on.
"""

import numpy as np
import networkx as nx
from pishield.propositional_requirements.constraint import Constraint
from pishield.propositional_requirements.literal import Literal


class ConstraintsGroup:
    """A set of Horn constraints with encoding and stratification utilities.

    Attributes:
        constraints: A frozenset of the :class:`Constraint` objects in the group.
        constraints_list: The constraints in their original insertion order, kept so
            that coherency results have a stable column order.
    """

    def __init__(self, arg):
        """Build a constraints group from a file or a list of constraints.

        Args:
            arg: Either a path to a constraints file (one ``head :- body`` rule per
                line) or an iterable of :class:`Constraint` objects.
        """
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
        """Return the union of this group with another.

        Args:
            other: The other ConstraintsGroup.

        Returns:
            A new ConstraintsGroup containing constraints from both.
        """
        return ConstraintsGroup(self.constraints.union(other.constraints))

    def __str__(self):
        """Return the sorted constraints, one ``head :- body`` rule per line."""
        return '\n'.join([str(constraint) for constraint in sorted(self.constraints)])

    def __iter__(self):
        """Iterate over the constraints in the group."""
        return iter(self.constraints)

    def __eq__(self, other):
        """Return True if both groups contain the same set of constraints.

        Args:
            other: The object to compare against.

        Returns:
            True if ``other`` is an equal ConstraintsGroup, False otherwise.
        """
        if not isinstance(other, ConstraintsGroup): return False
        return self.constraints == other.constraints

    def __len__(self):
        """Return the number of constraints in the group."""
        return len(self.constraints)

    def head_encoded(self, num_classes):
        """Encode all constraint heads into stacked positive/negative matrices.

        Args:
            num_classes: The number of variables (row width of the encodings).

        Returns:
            A tuple ``(pos_head, neg_head)`` of arrays, each row encoding one
            constraint's head (see :meth:`Constraint.head_encoded`).
        """
        pos_head = []
        neg_head = []

        for constraint in self.constraints:
            pos, neg = constraint.head_encoded(num_classes)
            pos_head.append(pos)
            neg_head.append(neg)

        return np.array(pos_head), np.array(neg_head)

    def body_encoded(self, num_classes):
        """Encode all constraint bodies into stacked positive/negative matrices.

        Args:
            num_classes: The number of variables (row width of the encodings).

        Returns:
            A tuple ``(pos_body, neg_body)`` of arrays, each row encoding one
            constraint's body (see :meth:`Constraint.body_encoded`).
        """
        pos_body = []
        neg_body = []

        for constraint in self.constraints:
            pos, neg = constraint.body_encoded(num_classes)
            pos_body.append(pos)
            neg_body.append(neg)

        return np.array(pos_body), np.array(neg_body)

    def encoded(self, num_classes):
        """Encode both heads and bodies of all constraints.

        Args:
            num_classes: The number of variables (encoding width).

        Returns:
            A tuple ``(head, body)`` where each element is the ``(pos, neg)`` pair
            returned by :meth:`head_encoded` and :meth:`body_encoded`.
        """
        head = self.head_encoded(num_classes)
        body = self.body_encoded(num_classes)
        return head, body

    def coherent_with(self, preds):
        """Check which constraints each prediction satisfies.

        Args:
            preds: A 2D array of predicted probabilities, shape (batch, num_classes).

        Returns:
            A boolean array of shape (batch, num_constraints), True where the
            corresponding constraint is satisfied (columns in insertion order).
        """
        coherent = [constraint.coherent_with(preds) for constraint in self.constraints_list]
        return np.array(coherent).transpose()

    def atoms(self):
        """Return the set of all variable indices used across the constraints."""
        atoms = set()
        for constraint in self.constraints:
            atoms = atoms.union(constraint.atoms())
        return atoms

    def heads(self):
        """Return the set of variable indices that appear as a constraint head."""
        heads = set()
        for constraint in self.constraints:
            heads.add(constraint.head.atom)
        return heads

    def graph(self):
        """Build a directed dependency graph over the variables.

        Adds an edge from each body atom to the head atom of every constraint,
        annotating it with the body and head polarities.

        Returns:
            A ``networkx.DiGraph`` whose nodes are atom indices and whose edges carry
            ``'body'`` and ``'head'`` polarity attributes.
        """
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
        """Build a directed graph over signed literals (atom and its negation).

        Unlike :meth:`graph`, each atom appears as two nodes (positive and negative
        literal), and an edge is added from each body literal to the head literal.

        Returns:
            A ``networkx.DiGraph`` whose nodes are literal strings.
        """
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
        """Partition the constraints into ordered dependency layers (strata).

        Performs a topological-style sweep of the dependency graph: variables with no
        unresolved dependencies are processed first, and the constraints whose heads
        they are form a stratum, repeated until all variables are consumed. Each
        stratum can be corrected without violating an earlier one.

        Returns:
            A list of :class:`ConstraintsGroup` objects, ordered from the constraints
            that should be applied first to those applied last.
        """
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
