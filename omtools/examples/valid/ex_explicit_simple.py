from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import omtools.api as ot
import numpy as np


class ExampleSimple(Group):
    def setup(self):
        x1 = self.create_indep_var('x1', val=10, dv=True)
        x2 = self.declare_input('x2', val=3)
        self.register_output('y', x1 + x2)


prob = Problem()
prob.model = ExampleSimple()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('y', prob['y'].shape)
print(prob['y'])
