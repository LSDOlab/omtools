from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import numpy as np


class ExampleWithSubsystemsBracketedArray(ImplicitComponent):
    def setup(self):
        g = self.group

        # define a subsystem (this is a very simple example)
        group = Group()
        p = group.create_indep_var('p', val=[7, -7])
        q = group.create_indep_var('q', val=[8, -8])
        r = p + q
        group.register_output('r', r)

        # add child system
        g.add_subsystem('R', group, promotes=['*'])
        # declare output of child system as input to parent system
        r = g.declare_input('r', shape=(2, ))

        c = g.declare_input('c', val=[18, -18])

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with g.create_group('coeff_a') as group:
            a = group.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            group.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)

        # store positive and negative values of `a` in an array
        ap = g.declare_input('a')
        an = -ap
        a = g.create_output('vec_a', shape=(2, ))
        a[0] = ap
        a[1] = an

        with g.create_group('coeff_b') as group:
            group.create_indep_var('b', val=[-4, 4])

        b = g.declare_input('b', shape=(2, ))
        y = g.create_implicit_output('y', shape=(2, ))
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(
            z,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


prob = Problem()
prob.model = Group()
prob.model.add_subsystem('example', ExampleWithSubsystemsBracketedArray())
prob.setup(force_alloc_complex=True)
prob.run_model()

print('y', prob['y'].shape)
print(prob['y'])
