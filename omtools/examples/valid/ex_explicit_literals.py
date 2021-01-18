from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import omtools.api as ot
import numpy as np


class ExampleLiterals(Group):
    def setup(self):
        x = self.declare_input('x', val=3)
        y = -2 * x**2 + 4 * x + 3
        self.register_output('y', y)


prob = Problem()
prob.model = ExampleLiterals()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('y', prob['y'].shape)
print(prob['y'])
