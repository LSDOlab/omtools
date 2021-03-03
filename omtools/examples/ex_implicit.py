from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import numpy as np


class ExampleApplyNonlinear(ImplicitComponent):
    def setup(self):
        g = self.group

        with g.create_group('sys') as group:
            group.create_indep_var('a', val=1)
            group.create_indep_var('b', val=-4)
            group.create_indep_var('c', val=3)
        a = g.declare_input('a')
        b = g.declare_input('b')
        c = g.declare_input('c')

        x = g.create_implicit_output('x')
        y = a * x**2 + b * x + c

        x.define_residual(y)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(solve_subsystems=False)


class ExampleBracketedScalar(ImplicitComponent):
    """
    :param var: x
    """
    def setup(self):
        g = self.group

        with g.create_group('sys') as group:
            group.create_indep_var('a', val=1)
            group.create_indep_var('b', val=-4)
            group.create_indep_var('c', val=3)
        a = g.declare_input('a')
        b = g.declare_input('b')
        c = g.declare_input('c')

        x = g.create_implicit_output('x')
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=0,
            x2=2,
        )


class ExampleBracketedArray(ImplicitComponent):
    """
    :param var: x
    """
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


class ExampleWithSubsystems(ImplicitComponent):
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


class ExampleWithSubsystemsBracketedScalar(ImplicitComponent):
    """
    :param var: y
    """
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
        with g.create_group('coeff_a') as group:
            a = group.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            group.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)

        a = g.declare_input('a')

        with g.create_group('coeff_b') as group:
            group.create_indep_var('b', val=-4)

        b = g.declare_input('b')
        y = g.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(z, x1=0, x2=2)


class ExampleWithSubsystemsBracketedArray(ImplicitComponent):
    """
    :param var: y
    """
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


class ExampleWithSubsystemsInternalN2(ImplicitComponent):
    """
    :param option: n2=True
    """
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
