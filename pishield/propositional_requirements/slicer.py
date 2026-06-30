"""Partial application of the Shield Layer.

The Slicer lets the Shield Layer apply only a prefix of its stratified correction
modules to a subset of the variables (atoms), which is used to gradually enable the
requirements during training.
"""


class Slicer:
    """Selects a subset of atoms and a prefix of correction modules.

    Attributes:
        atoms: The list of variable (atom) indices the slicer restricts to.
        modules: The number of leading correction modules to keep.
    """

    def __init__(self, atoms, modules):
        """Store the atoms and module count to slice to.

        Args:
            atoms: An iterable of variable indices to keep.
            modules: The number of leading correction modules to keep.
        """
        self.atoms = list(atoms)
        self.modules = modules
        print(f"Created slicer for {modules} modules (atoms {atoms})")

    def slice_atoms(self, tensor):
        """Select the slicer's columns from a tensor.

        Args:
            tensor: A 2D tensor indexed by variable in its columns.

        Returns:
            The tensor restricted to the slicer's atom columns.
        """
        return tensor[:, self.atoms]

    def slice_modules(self, modules):
        """Return the leading prefix of correction modules.

        Args:
            modules: The full ordered sequence of correction modules.

        Returns:
            The first ``self.modules`` of them.
        """
        return modules[:self.modules]
