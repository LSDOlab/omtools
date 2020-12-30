from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
    def setup(self):
        # Implicit component with composite residuals
        group = Group()
        group.create_indep_var('a', val=1)
        group.create_indep_var('b', val=-4)
        group.create_indep_var('c', val=3)
        self.add_subsystem('sys', group, promotes=['*'])

        x = self.create_implicit_output('x')
        a = self.declare_input('a')
        b = self.declare_input('b')
        c = self.declare_input('c')
        y = a * x**2 + b * x + c

        # TODO: set bracket between model runs
        x.define_residual_bracketed(
            y,
            x1=0,
            x2=2,
        )


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
print(prob['x'])
