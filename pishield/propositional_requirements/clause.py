"""Propositional clauses.

A clause is a disjunction of literals (e.g. ``y_0 or not y_1``), represented as an
unordered set of :class:`Literal` objects. Clauses are the normalised form into which
requirements are converted before stratification, and they support the logical
operations (resolution, subsumption, coherency checks) used to build the Shield Layer.
"""

import numpy as np
from pishield.propositional_requirements.literal import Literal
from pishield.propositional_requirements.constraint import Constraint


class Clause:
    """A disjunction of literals.

    Attributes:
        literals: A frozenset of the :class:`Literal` objects in the disjunction.
    """

    def __init__(self, literals):
        """Build a clause from literals.

        Args:
            literals: Either a whitespace-separated string of literals (e.g.
                ``'0 n1 2'``) or an iterable of :class:`Literal` objects.
        """
        if isinstance(literals, str):
            # Clause(string)
            literals = [Literal(lit) for lit in literals.split(' ')]
            self.literals = frozenset(literals)
        else:
            # Clause([Literals])
            self.literals = frozenset(literals)

    def __len__(self):
        """Return the number of literals in the clause."""
        return len(self.literals)

    def __iter__(self):
        """Iterate over the literals in the clause."""
        return iter(self.literals)

    def __eq__(self, other):
        """Return True if both clauses contain the same set of literals.

        Args:
            other: The object to compare against.

        Returns:
            True if ``other`` is an equal Clause, False otherwise.
        """
        if not isinstance(other, Clause): return False
        return self.literals == other.literals

    def __hash__(self):
        """Return a hash based on the set of literals."""
        return hash(self.literals)

    def __str__(self):
        """Return the literals sorted and joined by spaces."""
        return ' '.join([str(literal) for literal in sorted(self.literals)])

    @classmethod
    def from_constraint(cls, constraint):
        """Build the clause equivalent to a Horn constraint ``head :- body``.

        The constraint ``head :- b1, b2`` is logically ``head or not b1 or not b2``,
        so each body literal is negated and the head kept as-is.

        Args:
            constraint: The :class:`Constraint` to convert.

        Returns:
            The equivalent Clause.
        """
        body = [lit.neg() for lit in constraint.body]
        return cls([constraint.head] + body)

    @classmethod
    def random(cls, num_classes):
        """Build a random clause over ``num_classes`` variables.

        Args:
            num_classes: The number of available variables (atom indices).

        Returns:
            A randomly generated Clause.
        """
        atoms_count = np.random.randint(low=1, high=num_classes, size=1)
        atoms = np.random.randint(num_classes, size=atoms_count)

        pos = atoms[np.random.randint(2, size=atoms_count) == 1]
        literals = [Literal(atom, atom in pos) for atom in atoms]
        return cls(literals)

    def shift_add_n0(self):
        """Shift every atom up by one and add the negative literal ``n0``.

        Used to make room for a detection variable at index 0.

        Returns:
            A new Clause with shifted atoms plus the ``n0`` literal.
        """
        n0 = Literal(0, False)
        return Clause([Literal(lit.atom + 1, lit.positive) for lit in self] + [n0])

    def fix_head(self, head):
        """Turn the clause into a constraint by designating one literal as the head.

        The remaining literals become the (negated) body, inverting
        :meth:`from_constraint`.

        Args:
            head: The literal of this clause to use as the constraint head.

        Returns:
            A :class:`Constraint` with the given head.

        Raises:
            Exception: If ``head`` is not a literal of this clause.
        """
        if not head in self.literals:
            raise Exception('Head not in clause')
        body = [lit.neg() for lit in self.literals if lit != head]
        return Constraint(head, body)

    def always_true(self):
        """Return True if the clause is a tautology.

        A clause is always true when it contains both a literal and its negation.

        Returns:
            True if the clause is tautological, False otherwise.
        """
        for literal in self.literals:
            if literal.neg() in self.literals:
                return True
        return False

    def resolution_on(self, other, literal):
        """Resolve this clause with another on a given literal.

        Args:
            other: The other clause to resolve with.
            literal: The literal to resolve on; it and its negation are removed
                from the union of the two clauses.

        Returns:
            The resolvent Clause, or None if the resolvent is a tautology.
        """
        result = self.literals.union(other.literals).difference({literal, literal.neg()})
        result = Clause(result)
        return None if result.always_true() else result

    def resolution(self, other, literal=None):
        """Resolve this clause with another, finding a pivot literal if needed.

        Args:
            other: The other clause to resolve with.
            literal: The literal to resolve on. If None, the first literal of this
                clause whose negation appears in ``other`` is used.

        Returns:
            The resolvent Clause, or None if no pivot is found or the resolvent is
            a tautology.
        """
        if literal != None:
            return self.resolution_on(other, literal)

        for lit in self.literals:
            if lit.neg() in other.literals:
                return self.resolution_on(other, lit)

        return None

    def always_false(self):
        """Return True if the clause is empty (and hence unsatisfiable)."""
        return len(self) == 0

    def coherent_with(self, preds):
        """Check whether predictions satisfy this clause.

        A clause is satisfied when at least one of its literals is true; here this is
        relaxed to a probability above 0.5 for any literal.

        Args:
            preds: A 2D array of predicted probabilities, shape (batch, num_classes).

        Returns:
            A boolean array of length batch, True where the clause is satisfied.
        """
        pos = [lit.atom for lit in self.literals if lit.positive]
        neg = [lit.atom for lit in self.literals if not lit.positive]

        preds = np.concatenate((preds[:, pos], 1 - preds[:, neg]), axis=1)
        preds = preds.max(axis=1)
        return preds > 0.5

    def is_subset(self, other):
        """Return True if this clause's literals are a subset of ``other``'s.

        Args:
            other: The clause to test against.

        Returns:
            True if every literal of this clause is in ``other``.
        """
        return self.literals.issubset(other.literals)

    def atoms(self):
        """Return the set of variable indices appearing in the clause."""
        return {lit.atom for lit in self.literals}
