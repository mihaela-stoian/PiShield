class Slicer:
    def __init__(self, atoms, modules):
        self.atoms = list(atoms)
        self.modules = modules
        print(f"Created slicer for {modules} modules (atoms {atoms})")

    def slice_atoms(self, tensor):
        return tensor[:, self.atoms]

    def slice_modules(self, modules):
        return modules[:self.modules]
