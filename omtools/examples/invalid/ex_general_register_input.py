from openmdao.api import Problem
from omtools.api import Group


class ErrorRegisterInput(Group):
    def setup(self):
        a = self.declare_input('a', val=10)
        # This will raise a TypeError
        self.register_output('a', a)


prob = Problem()
prob.model = ErrorRegisterInput()
prob.setup(force_alloc_complex=True)
prob.run_model()
