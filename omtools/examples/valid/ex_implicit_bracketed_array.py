from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import numpy as np


class ExampleBracketedArray(ImplicitComponent):
    def setup(self):
        g = self.group

        with g.create_group('sys') as group:
            group.create_indep_var('a', val=[1, -1])
            group.create_indep_var('b', val=[-4, 4])
            group.create_indep_var('c', val=[3, -3])
        a = g.declare_input('a', shape=(2, ))
        b = g.declare_input('b', shape=(2, ))
        c = g.declare_input('c', shape=(2, ))

        x = g.create_implicit_output('x', shape=(2, ))
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


prob = Problem()
prob.model = Group()
prob.model.add_subsystem('example', ExampleBracketedArray())
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x', prob['x'].shape)
print(prob['x'])
