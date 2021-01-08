from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver

import omtools.api as ot
from omtools.api import Group, ImplicitComponent
from omtools.core.expression import Expression
from openmdao.api import Problem
import numpy as np


class Example(ImplicitComponent):
    def setup(self):
        g = self.group

        c = g.declare_input('c', val=3)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        group = Group()
        a = group.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        group.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        g.add_subsystem('coeff_a', group, promotes=['*'])

        a = g.declare_input('a')

        group = Group()
        group.create_indep_var('b', val=-4)
        g.add_subsystem('coeff_b', group, promotes=['*'])

        b = g.declare_input('b')
        y = g.create_implicit_output('y')
        z = a * y**2 + b * y + c
        y.define_residual(
            z,
            linear_solver=ScipyKrylov(),
            nonlinear_solver=NewtonSolver(
                solve_subsystems=False,
                maxiter=100,
            ),
        )


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
print(prob['y'])
