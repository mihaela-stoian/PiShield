"""Propositional literals.

A literal is a single variable (atom) together with a sign: either the variable
asserted positively or its negation. Literals are the atomic building blocks of
clauses and constraints in the propositional requirements subpackage.
"""


class Literal:
    """A propositional literal: a variable (atom) and a polarity.

    Attributes:
        atom: The integer index of the variable referenced by the literal.
        positive: True if the literal asserts the variable, False if it negates it.
    """

    def __init__(self, *args, reversed_sign=False):
        """Build a literal from either an (atom, polarity) pair or a string.

        Two calling conventions are supported:
          * ``Literal(atom: int, positive: bool)`` builds the literal directly.
          * ``Literal(text: str)`` parses a textual literal. Supported forms are
            ``'y_3'``/``'y_not 3'`` (label notation) and ``'3'``/``'n3'`` (index
            notation), where the ``n`` prefix denotes a negative literal.

        Args:
            *args: Either two positional arguments ``(atom, positive)`` or a single
                string to parse.
            reversed_sign: When parsing ``'y_'`` style strings, flips the polarity so
                the sign matches the ``head :- body`` convention used elsewhere.
        """
        if len(args) == 2:
            # Literal(int, bool)
            self.atom = args[0]
            self.positive = args[1]
        else:
            # Literal(string)
            plain = args[0]
            if 'y_' in plain:
                if 'not ' in plain:
                    self.atom = int(plain[6:])
                    if reversed_sign:
                        self.positive = True   # set to True, to account for the :- format, which the code uses
                    else:
                        self.positive = False   # set to True, to account for the :- format, which the code uses
                else:
                    self.atom = int(plain[2:])
                    if reversed_sign:
                        self.positive = False  # set to False, to account for the :- format, which the code uses
                    else:
                        self.positive = True
            else:
                if 'n' in plain:
                    self.atom = int(plain[1:])
                    self.positive = False
                else:
                    self.atom = int(plain)
                    self.positive = True

    def __str__(self):
        """Return the index notation of the literal (``'3'`` or ``'n3'``)."""
        return str(self.atom) if self.positive else 'n' + str(self.atom)

    def __hash__(self):
        """Return a hash based on the atom and its polarity."""
        return hash((self.atom, self.positive))

    def __eq__(self, other):
        """Return True if both literals share the same atom and polarity.

        Args:
            other: The object to compare against.

        Returns:
            True if ``other`` is an equal Literal, False otherwise.
        """
        if not isinstance(other, Literal): return False
        return self.atom == other.atom and self.positive == other.positive

    def __lt__(self, other):
        """Order literals by atom index, breaking ties by polarity.

        Args:
            other: The literal to compare against.

        Returns:
            True if this literal sorts before ``other``.
        """
        if self.atom == other.atom:
            return self.positive < other.positive
        else:
            return self.atom < other.atom

    def neg(self):
        """Return a new literal with the opposite polarity.

        Returns:
            A Literal over the same atom with flipped sign.
        """
        return Literal(self.atom, not self.positive)
