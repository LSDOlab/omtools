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
        group.create_indep_var('a', val=[1, -1])
        group.create_indep_var('b', val=[-4, 4])
        group.create_indep_var('c', val=[3, -3])
        g.add_subsystem('sys', group, promotes=['*'])

        x = g.create_implicit_output('x', shape=(2, ))
        a = g.declare_input('a', shape=(2, ))
        b = g.declare_input('b', shape=(2, ))
        c = g.declare_input('c', shape=(2, ))
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
print(prob['x'])
