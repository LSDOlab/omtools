from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import numpy as np


class ExampleWithSubsystemsInternalN2(ImplicitComponent):
    def setup(self):
        g = self.group

        # define a subsystem (this is a very simple example)
        group = Group()
        p = group.create_indep_var('p', val=7)
        q = group.create_indep_var('q', val=8)
        r = p + q
        group.register_output('r', r)

        # add child system
        g.add_subsystem('R', group, promotes=['*'])
        # declare output of child system as input to parent system
        r = g.declare_input('r')

        c = g.declare_input('c', val=18)

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
        z = a * y**2 + b * y + c - r
        y.define_residual(z)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
        )


prob = Problem()
prob.model = Group()
prob.model.add_subsystem(
    'example', 
    ExampleWithSubsystemsInternalN2(
    n2=True,
))
prob.setup(force_alloc_complex=True)
prob.run_model()
