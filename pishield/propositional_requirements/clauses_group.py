import numpy as np
import networkx as nx
from pishield.propositional_requirements.literal import Literal
from pishield.propositional_requirements.clause import Clause
from pishield.propositional_requirements.constraints_group import ConstraintsGroup
from pishield.propositional_requirements.strong_coherency import strong_coherency_constraint_preprocessing


class ClausesGroup:
    def __init__(self, clauses):
        # ClausesGroup([Clause])
        self.clauses = frozenset(clauses)
        self.clauses_list = clauses

    @classmethod
    def from_constraints_group(cls, group):
        return cls([Clause.from_constraint(cons) for cons in group])

    def __len__(self):
        return len(self.clauses)

    def __eq__(self, other):
        if not isinstance(other, ClausesGroup): return False
        return self.clauses == other.clauses

    def __add__(self, other):
        return ClausesGroup(self.clauses.union(other.clauses))

    def __str__(self):
        return '\n'.join([str(clause) for clause in self.clauses])

    def __hash__(self):
        return hash(self.clauses)

    def __iter__(self):
        return iter(self.clauses)

    @classmethod
    def random(cls, max_clauses, num_classes, coherent_with=np.array([]), min_clauses=0):
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
        n0 = Literal(0, False)
        clauses = [clause.shift_add_n0() for clause in self]
        forced = [Clause(f"0 n{x + 1}") for x in self.atoms()] if forced else []
        return ClausesGroup(clauses + forced)

    def compacted(self):
        clauses = list(self.clauses)
        clauses.sort(reverse=True, key=len)
        compacted = []

        for clause in clauses:
            compacted = [c for c in compacted if not clause.is_subset(c)]
            compacted.append(clause)

        # print(f"compacted {len(clauses) - len(compacted)} out of {len(clauses)}")
        return ClausesGroup(compacted)

    def resolution(self, atom):
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
        G = nx.Graph()
        G.add_nodes_from(self.atoms(), kind='atom')
        G.add_nodes_from(self.clauses, kind='clause')

        for clause in self.clauses:
            for lit in clause:
                G.add_edge(clause, lit.atom)

        return G

    @staticmethod
    def centrality_measures():
        return ['degree', 'eigenvector', 'katz', 'closeness', 'betweenness']

    def centrality(self, centrality):
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
        answer = [clause.coherent_with(preds) for clause in self.clauses_list]
        answer = np.array(answer).reshape(len(self.clauses_list), preds.shape[0])
        return answer.transpose()

    def atoms(self):
        result = set()
        for clause in self.clauses:
            result = result.union(clause.atoms())
        return result
