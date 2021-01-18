from openmdao.api import Problem
from omtools.api import Group


class ExampleUnusedInputs(Group):
    def setup(self):
        # These inputs are unused; no components will be constructed
        a = self.declare_input('a', val=10)
        b = self.declare_input('b', val=5)
        c = self.declare_input('c', val=2)


prob = Problem()
prob.model = ExampleUnusedInputs()
prob.setup(force_alloc_complex=True)
prob.run_model()
