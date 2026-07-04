"""Data model for hierarchical requirements.

A hierarchical requirement is a subsumption ``child -> parent`` over binary
variables (class labels): whenever a child class holds, its parent class must
hold too. :class:`Hierarchy` stores the classes and the direct child-parent
edges, validates that they form a DAG, and exposes the transitively-closed
descendants matrix ``R`` consumed by the hierarchical Shield Layer (following
C-HMCNN [3]).
"""

from typing import List, Tuple

import networkx as nx
import numpy as np


class Hierarchy:
    """A class hierarchy: named classes plus direct child->parent edges.

    The hierarchy is a directed acyclic graph (a class may have several
    parents). Class names are only used for reporting and for aligning the
    classes with the columns of the tensors to correct: the i-th class name
    corresponds to the i-th variable.

    Attributes:
        class_names: The class names, in variable-index order.
        edges: The direct edges as sorted ``(child_index, parent_index)`` pairs.
        graph: A ``networkx.DiGraph`` over the variable indices whose edges
            point from each class to its parents (descendant -> ancestor).
    """

    def __init__(self, class_names: List[str], edges: List[Tuple[int, int]]):
        """Build and validate the hierarchy.

        Args:
            class_names: The class names, in variable-index order.
            edges: The direct edges as ``(child_index, parent_index)`` pairs.

        Raises:
            Exception: If an edge index is out of range, an edge is a
                self-loop, or the edges contain a cycle.
        """
        self.class_names = list(class_names)
        num_classes = len(self.class_names)

        for child, parent in edges:
            if not (0 <= child < num_classes and 0 <= parent < num_classes):
                raise Exception(f'Edge ({child}, {parent}) references a variable outside 0..{num_classes - 1}!')
            if child == parent:
                raise Exception(f'Edge ({child}, {parent}) is a self-loop!')
        self.edges = sorted(set(edges))

        # Edges point from each class to its parents (descendant -> ancestor), as in C-HMCNN.
        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_classes))
        graph.add_edges_from(self.edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise Exception('The hierarchical requirements contain a cycle: a hierarchy must be a DAG!')
        self.graph = graph

    @property
    def num_classes(self) -> int:
        """The number of classes (variables) in the hierarchy."""
        return len(self.class_names)

    def ancestors(self, index: int) -> set:
        """Return the indices of all (strict) ancestors of a class.

        Args:
            index: The variable index of the class.

        Returns:
            The set of variable indices of the class's ancestors.
        """
        # Edges point descendant -> ancestor, so the graph-descendants of a node are its ancestors.
        return set(nx.descendants(self.graph, index))

    def adjacency_matrix(self) -> np.ndarray:
        """Return the direct-edge adjacency matrix.

        Returns:
            An array ``A`` of shape (num_classes, num_classes) with
            ``A[child, parent] = 1`` for every direct edge.
        """
        adjacency = np.zeros((self.num_classes, self.num_classes))
        for child, parent in self.edges:
            adjacency[child, parent] = 1
        return adjacency

    def descendants_matrix(self) -> np.ndarray:
        """Return the transitively-closed descendants matrix ``R``.

        This is the matrix consumed by the hierarchical Shield Layer: the
        corrected value of class ``i`` is the maximum of the predictions over
        the classes marked in row ``i``.

        Returns:
            An array ``R`` of shape (num_classes, num_classes) with
            ``R[i, j] = 1`` iff ``j`` is ``i`` itself or a descendant of ``i``.
        """
        R = np.zeros((self.num_classes, self.num_classes))
        np.fill_diagonal(R, 1)
        for index in range(self.num_classes):
            for ancestor in self.ancestors(index):
                R[index, ancestor] = 1
        # R[i, j] currently marks the ancestors of i; transpose so rows mark descendants.
        return R.transpose()

    def to_requirements_lines(self) -> List[str]:
        """Return the hierarchy as textual Horn rules.

        Returns:
            One ``parent :- child`` line per direct edge, over variable indices.
            The lines are also valid propositional requirements.
        """
        return [f'{parent} :- {child}' for child, parent in self.edges]

    def __str__(self):
        """Return the hierarchy as ``parent :- child`` rules, one per line."""
        return '\n'.join(self.to_requirements_lines())

    def __eq__(self, other):
        """Return True if both hierarchies have the same classes and edges."""
        if not isinstance(other, Hierarchy):
            return False
        return self.class_names == other.class_names and self.edges == other.edges
