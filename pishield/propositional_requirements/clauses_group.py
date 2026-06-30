"""Collections of propositional clauses.

A :class:`ClausesGroup` holds a set of :class:`Clause` disjunctions and provides the
machinery that turns requirements into the Shield Layer: variable elimination by
resolution, clause compaction, centrality-based ordering, and stratification into
ordered :class:`ConstraintsGroup` layers (optionally enforcing strong coherency).
"""

import numpy as np
import networkx as nx
from pishield.propositional_requirements.literal import Literal
from pishield.propositional_requirements.clause import Clause
from pishield.propositional_requirements.constraints_group import ConstraintsGroup
from pishield.propositional_requirements.strong_coherency import strong_coherency_constraint_preprocessing


class ClausesGroup:
    """A set of clauses supporting resolution and stratification.

    Attributes:
        clauses: A frozenset of the :class:`Clause` objects in the group.
        clauses_list: The clauses in their original order, kept so that coherency
            results have a stable column order.
    """

    def __init__(self, clauses):
        """Build a clauses group from an iterable of clauses.

        Args:
            clauses: An iterable of :class:`Clause` objects.
        """
        # ClausesGroup([Clause])
        self.clauses = frozenset(clauses)
        self.clauses_list = clauses

    @classmethod
    def from_constraints_group(cls, group):
        """Build a clauses group from a :class:`ConstraintsGroup`.

        Args:
            group: The constraints group to convert (each constraint becomes a clause).

        Returns:
            The equivalent ClausesGroup.
        """
        return cls([Clause.from_constraint(cons) for cons in group])

    def __len__(self):
        """Return the number of clauses in the group."""
        return len(self.clauses)

    def __eq__(self, other):
        """Return True if both groups contain the same set of clauses.

        Args:
            other: The object to compare against.

        Returns:
            True if ``other`` is an equal ClausesGroup, False otherwise.
        """
        if not isinstance(other, ClausesGroup): return False
        return self.clauses == other.clauses

    def __add__(self, other):
        """Return the union of this group with another.

        Args:
            other: The other ClausesGroup.

        Returns:
            A new ClausesGroup containing clauses from both.
        """
        return ClausesGroup(self.clauses.union(other.clauses))

    def __str__(self):
        """Return the clauses, one per line."""
        return '\n'.join([str(clause) for clause in self.clauses])

    def __hash__(self):
        """Return a hash based on the set of clauses."""
        return hash(self.clauses)

    def __iter__(self):
        """Iterate over the clauses in the group."""
        return iter(self.clauses)

    @classmethod
    def random(cls, max_clauses, num_classes, coherent_with=np.array([]), min_clauses=0):
        """Build a random clauses group, optionally filtered by coherency.

        Generates up to ``max_clauses`` random clauses; if ``coherent_with`` is given,
        only clauses satisfied by every one of those predictions are kept, and
        generation is repeated until at least ``min_clauses`` survive.

        Args:
            max_clauses: The number of random clauses to generate per attempt.
            num_classes: The number of variables available.
            coherent_with: Optional array of predictions the clauses must satisfy.
            min_clauses: The minimum number of clauses required in the result.

        Returns:
            A randomly generated ClausesGroup.
        """
        assert min_clauses <= max_clauses
        clauses = [Clause.random(num_classes) for i in range(max_clauses)]

        if len(coherent_with) > 0:
            keep = cls(clauses).coherent_with(coherent_with).all(axis=0)
            clauses = np.array(clauses)[keep].tolist()

        found = len(clauses)
        if found < min_clauses:
            other = cls.random(max_clauses - found, num_classes, coherent_with=coherent_with,
                               min_clauses=min_clauses - found)
            return cls(clauses) + other
        else:
            return cls(clauses)

    def add_detection_label(self, forced=False):
        """Insert a detection variable at index 0 and shift the others up.

        Each clause's atoms are shifted up by one and the negative literal ``n0`` added
        (so the clause is vacuously satisfied when the detection variable is off).
        When ``forced``, extra clauses tie the detection variable to every atom.

        Args:
            forced: If True, also add ``0 n{x+1}`` clauses for every atom.

        Returns:
            A new ClausesGroup including the detection variable.
        """
        n0 = Literal(0, False)
        clauses = [clause.shift_add_n0() for clause in self]
        forced = [Clause(f"0 n{x + 1}") for x in self.atoms()] if forced else []
        return ClausesGroup(clauses + forced)

    def compacted(self):
        """Remove clauses subsumed by a smaller clause.

        Sorts clauses from longest to shortest and drops any clause that is a superset
        of (i.e. subsumed by) a retained clause.

        Returns:
            A new, subsumption-free ClausesGroup.
        """
        clauses = list(self.clauses)
        clauses.sort(reverse=True, key=len)
        compacted = []

        for clause in clauses:
            compacted = [c for c in compacted if not clause.is_subset(c)]
            compacted.append(clause)

        # print(f"compacted {len(clauses) - len(compacted)} out of {len(clauses)}")
        return ClausesGroup(compacted)

    def resolution(self, atom):
        """Eliminate a variable by resolution, returning its constraints.

        Splits the clauses into those containing the positive literal, the negative
        literal, and neither; resolves each positive against each negative clause to
        produce the remaining clauses (compacted), and turns the positive and negative
        clauses into constraints whose head is the eliminated literal.

        Args:
            atom: The variable index to eliminate.

        Returns:
            A tuple ``(constraints, next_clauses)`` where ``constraints`` is a
            :class:`ConstraintsGroup` defining the eliminated atom and ``next_clauses``
            is the remaining :class:`ClausesGroup` over the other atoms.
        """
        pos = Literal(atom, True)
        neg = Literal(atom, False)

        # Split clauses in three categories
        pos_clauses, neg_clauses, other_clauses = set(), set(), set()
        for clause in self.clauses:
            if pos in clause:
                pos_clauses.add(clause)
            elif neg in clause:
                neg_clauses.add(clause)
            else:
                other_clauses.add(clause)

        # Apply resolution on positive and negative clauses
        resolution_clauses = [c1.resolution(c2, literal=pos) for c1 in pos_clauses for c2 in neg_clauses]
        resolution_clauses = {clause for clause in resolution_clauses if clause != None}
        next_clauses = ClausesGroup(other_clauses.union(resolution_clauses)).compacted()

        # Compute constraints 
        pos_constraints = [clause.fix_head(pos) for clause in pos_clauses]
        neg_constraints = [clause.fix_head(neg) for clause in neg_clauses]
        constraints = ConstraintsGroup(pos_constraints + neg_constraints)

        return constraints, next_clauses

    def graph(self):
        """Build a bipartite graph linking clauses to their atoms.

        Returns:
            A ``networkx.Graph`` with ``kind='atom'`` and ``kind='clause'`` nodes and
            an edge between each clause and every atom it mentions.
        """
        G = nx.Graph()
        G.add_nodes_from(self.atoms(), kind='atom')
        G.add_nodes_from(self.clauses, kind='clause')

        for clause in self.clauses:
            for lit in clause:
                G.add_edge(clause, lit.atom)

        return G

    @staticmethod
    def centrality_measures():
        """Return the names of the supported centrality measures.

        Returns:
            A list of measure names usable with :meth:`centrality`.
        """
        return ['degree', 'eigenvector', 'katz', 'closeness', 'betweenness']

    def centrality(self, centrality):
        """Compute a centrality score per atom to guide the elimination order.

        Args:
            centrality: The centrality measure name (one of
                :meth:`centrality_measures`), optionally prefixed with ``'rev-'`` to
                invert the resulting scores.

        Returns:
            A dict mapping each graph node to its (possibly reversed) centrality score.

        Raises:
            Exception: If the centrality measure name is unknown.
        """
        G = self.graph()

        if centrality.startswith('rev-'):
            centrality = centrality[4:]
            rev = True
        else:
            rev = False

        if centrality == 'degree':
            result = nx.algorithms.centrality.degree_centrality(G)
        elif centrality == 'eigenvector':
            result = nx.algorithms.centrality.eigenvector_centrality_numpy(G)
        elif centrality == 'katz':
            result = nx.algorithms.centrality.katz_centrality_numpy(G)
        elif centrality == 'closeness':
            result = nx.algorithms.centrality.closeness_centrality(G)
        elif centrality == 'betweenness':
            result = nx.algorithms.centrality.betweenness_centrality(G)
        else:
            raise Exception(f"Unknown centrality {centrality}")

        # Normalize results
        if rev:
            values = np.array([result[node] for node in result])
            mini, maxi = values.min(), values.max()
            for node in result: result[node] = maxi - (result[node] - mini)

        return result

    def stratify(self, centrality):
        """Convert the clauses into ordered constraint strata for the Shield Layer.

        Repeatedly eliminates atoms (in an order determined by ``centrality``) via
        resolution, accumulating the resulting constraints and optionally rewriting
        them to enforce strong coherency, then stratifies the accumulated constraints.

        Args:
            centrality: Either a centrality measure name (str) used to order atoms, an
                explicit iterable of atom indices giving the order, or None to use the
                order atoms appear in the constraints.

        Returns:
            A list of :class:`ConstraintsGroup` strata (the output of
            :meth:`ConstraintsGroup.stratify`).

        Raises:
            Exception: If the clauses are unsatisfiable (clauses remain after
                eliminating every atom).
        """
        # Centrality guides the inference order
        if not isinstance(centrality, str):
            atoms = centrality
        else:
            centrality = self.centrality(centrality)
            atoms = list(self.atoms())
            atoms.sort(key=lambda x: centrality[x])
        if centrality is None:  # get atoms in the order they appear in constraints
            atoms = list(self.atoms())

        # Apply resolution repeatedly
        atoms = atoms[::-1]
        group = ConstraintsGroup([])
        clauses = self

        for atom in atoms:
            # print(f"Eliminating %{atom} from %{len(clauses)} clauses\n")
            constraints, clauses = clauses.resolution(atom)
            if len(constraints.constraints_list):
                strongly_coherent_constraints = strong_coherency_constraint_preprocessing(constraints.constraints_list, atoms)
                if strongly_coherent_constraints is not None:
                    constraints = strongly_coherent_constraints
            group = group + constraints

        if len(clauses):
            raise Exception("Unsatisfiable set of clauses")

        return group.stratify()

    def coherent_with(self, preds):
        """Check which clauses each prediction satisfies.

        Args:
            preds: A 2D array of predicted probabilities, shape (batch, num_classes).

        Returns:
            A boolean array of shape (batch, num_clauses), True where the corresponding
            clause is satisfied (columns in insertion order).
        """
        answer = [clause.coherent_with(preds) for clause in self.clauses_list]
        answer = np.array(answer).reshape(len(self.clauses_list), preds.shape[0])
        return answer.transpose()

    def atoms(self):
        """Return the set of all variable indices used across the clauses."""
        result = set()
        for clause in self.clauses:
            result = result.union(clause.atoms())
        return result
