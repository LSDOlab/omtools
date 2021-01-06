from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver

import omtools.api as ot
from omtools.api import Group, ImplicitGroup
from omtools.core.expression import Expression
from openmdao.api import Problem
import numpy as np


class Example(ImplicitGroup):
    def setup(self):
        c = self.declare_input('c', val=[3, -3])

        # a == (3 + a - 2 * a**2)**(1 / 4)
        group = Group()
        a = group.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        group.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        self.add_subsystem('coeff_a', group, promotes=['*'])

        # store positive and negative values of `a` in an array
        ap = self.declare_input('a')
        an = -ap
        a = self.create_output('vec_a', shape=(2, ))
        a[0] = ap
        a[1] = an

        group = Group()
        group.create_indep_var('b', val=[-4, 4])
        self.add_subsystem('coeff_b', group, promotes=['*'])

        b = self.declare_input('b', shape=(2, ))
        y = self.create_implicit_output('y', shape=(2, ))
        z = a * y**2 + b * y + c
        y.define_residual_bracketed(
            z,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
print(prob['y'])