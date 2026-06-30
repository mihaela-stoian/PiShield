"""Propositional constraints (Horn rules).

A constraint is a Horn rule of the form ``head :- body``, where ``head`` is a single
literal and ``body`` is a conjunction of literals: the rule states that if every body
literal holds then the head must hold. Constraints can be parsed from the ``head :- body``
or disjunctive (``y_0 or not y_1``) textual forms and are the requirement representation
that the Shield Layer and Shield Loss consume.
"""

import numpy as np
from pishield.propositional_requirements.literal import Literal


class Constraint:
    """A Horn rule ``head :- body`` over propositional literals.

    Attributes:
        head: The :class:`Literal` implied by the constraint.
        body: A frozenset of :class:`Literal` objects forming the conjunction that
            must hold for the head to be implied.
    """

    def __init__(self, *args):
        """Build a constraint from a head/body pair or from a textual rule.

        Two calling conventions are supported:
          * ``Constraint(head: Literal, body: Iterable[Literal])`` builds it directly.
          * ``Constraint(text: str)`` parses a rule in either ``head :- body`` form or
            disjunctive ``head or lit or ...`` form (the latter parsed with reversed
            signs to match the ``:-`` convention).

        Args:
            *args: Either two positional arguments ``(head, body)`` or a single string.
        """
        if len(args) == 2:
            # Constraint(Literal, [Literal])
            self.head = args[0]
            self.body = frozenset(args[1])
        else:
            # Constraint(string)
            if ':-' in args[0]:
                line = args[0].split(' ')
                if line[2] == ':-':
                    line = line[1:]
                assert line[1] == ':-'
                self.head = Literal(line[0])
                self.body = frozenset(Literal(lit) for lit in line[2:])
            elif 'or' in args[0]:
                # Constraint(string)
                line = args[0].split(' or ')
                self.head = Literal(line[0])
                self.body = frozenset(Literal(lit, reversed_sign=True) for lit in line[1:])

    def __eq__(self, other):
        """Return True if both constraints have equal heads and bodies.

        Args:
            other: The object to compare against.

        Returns:
            True if ``other`` is an equal Constraint, False otherwise.
        """
        if not isinstance(other, Constraint): return False
        return self.head == other.head and self.body == other.body

    def __lt__(self, other):
        """Order constraints by head, breaking ties by body.

        Args:
            other: The constraint to compare against.

        Returns:
            True if this constraint sorts before ``other``.
        """
        if self.head == other.head:
            return self.body < other.body
        else:
            return self.head < other.head

    def __hash__(self):
        """Return a hash based on the head and body."""
        return hash((self.head, self.body))

    def __str__(self):
        """Return the constraint in ``head :- body`` form with a sorted body."""
        return str(self.head) + " :- " + ' '.join([str(lit) for lit in sorted(self.body)])

    def head_encoded(self, num_classes):
        """One-hot encode the head literal into positive and negative vectors.

        Args:
            num_classes: The number of variables (length of the encoding vectors).

        Returns:
            A tuple ``(pos_head, neg_head)`` of length-``num_classes`` arrays; exactly
            one entry is set to 1 depending on the head's atom and polarity.
        """
        pos_head = np.zeros(num_classes)
        neg_head = np.zeros(num_classes)
        if self.head.positive:
            pos_head[self.head.atom] = 1
        else:
            neg_head[self.head.atom] = 1
        return pos_head, neg_head

    def body_encoded(self, num_classes):
        """Multi-hot encode the body literals into positive and negative vectors.

        Args:
            num_classes: The number of variables (length of the encoding vectors).

        Returns:
            A tuple ``(pos_body, neg_body)`` of length-``num_classes`` integer arrays
            marking which atoms appear positively and negatively in the body.
        """
        pos_body = np.zeros(num_classes, dtype=int)
        neg_body = np.zeros(num_classes, dtype=int)
        for lit in self.body:
            if lit.positive:
                pos_body[lit.atom] = 1
            else:
                neg_body[lit.atom] = 1
        return pos_body, neg_body

    def where(self, cond, opt1, opt2):
        """Differentiable select between two options.

        Args:
            cond: A 0/1 mask (or probability) selecting between the options.
            opt1: The value used where ``cond`` is 1.
            opt2: The value used where ``cond`` is 0.

        Returns:
            ``opt2 + cond * (opt1 - opt2)``.
        """
        return opt2 + cond * (opt1 - opt2)

    def coherent_with(self, preds):
        """Check whether predictions satisfy this constraint.

        The body truth value is the minimum over its literals (a product/Goedel-style
        conjunction); the constraint is satisfied when ``body <= head``.

        Args:
            preds: A 2D array of predicted probabilities, shape (batch, num_classes).

        Returns:
            A boolean array of length batch, True where the constraint is satisfied.
        """
        num_classes = preds.shape[1]
        pos_body, neg_body = self.body_encoded(num_classes)
        pos_body = preds[:, pos_body.astype(bool)]
        neg_body = 1 - preds[:, neg_body.astype(bool)]
        body = np.min(np.concatenate((pos_body, neg_body), axis=1), axis=1)

        head = preds[:, self.head.atom]
        if not self.head.positive:
            head = 1 - head

        return body <= head

    def atoms(self):
        """Return the set of variable indices in the head and body."""
        return {lit.atom for lit in self.body.union({self.head})}
