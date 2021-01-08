from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group, ImplicitComponent
from omtools.core.expression import Expression


class Example(ImplicitComponent):
    def setup(self):
        g = self.group

        # Implicit component with composite residuals
        group = Group()
        group.create_indep_var('a', val=1)
        group.create_indep_var('b', val=-4)
        group.create_indep_var('c', val=3)
        g.add_subsystem('sys', group, promotes=['*'])

        x = g.create_implicit_output('x')
        a = g.declare_input('a')
        b = g.declare_input('b')
        c = g.declare_input('c')
        y = a * x**2 + b * x + c

        x.define_residual(y)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(solve_subsystems=False)


prob = Problem()
prob.model = Group()
prob.model.add_subsystem('example', Example(), promotes=['*'])
prob.setup(force_alloc_complex=True)

prob.set_val('x', 1.9)
prob.run_model()
print(prob['x'])
x1 = np.array(prob['x'])

prob.set_val('x', 2.1)
prob.run_model()
print(prob['x'])
x2 = np.array(prob['x'])
