from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import omtools.api as ot
import numpy as np


class ExampleCycles(Group):
    def setup(self):
        # x == (3 + x - 2 * x**2)**(1 / 4)
        group = Group()
        x = group.create_output('x')
        x.define((3 + x - 2 * x**2)**(1 / 4))
        group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add_subsystem('cycle_1', group)

        # x == ((x + 3 - x**4) / 2)**(1 / 4)
        group = Group()
        x = group.create_output('x')
        x.define(((x + 3 - x**4) / 2)**(1 / 4))
        group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add_subsystem('cycle_2', group)

        # x == 0.5 * x
        group = Group()
        x = group.create_output('x')
        x.define(0.5 * x)
        group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add_subsystem('cycle_3', group)


prob = Problem()
prob.model = ExampleCycles()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('cycle_1.x', prob['cycle_1.x'].shape)
print(prob['cycle_1.x'])
print('cycle_2.x', prob['cycle_2.x'].shape)
print(prob['cycle_2.x'])
print('cycle_3.x', prob['cycle_3.x'].shape)
print(prob['cycle_3.x'])
