from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import omtools.api as ot
import numpy as np


class ExampleLiterals(Group):
    """
    :param var: y
    """
    def setup(self):
        x = self.declare_input('x', val=3)
        y = -2 * x**2 + 4 * x + 3
        self.register_output('y', y)


class ExampleBinaryOperations(Group):
    """
    :param var: y1
    :param var: y2
    :param var: y3
    :param var: y4
    :param var: y5
    :param var: y6
    :param var: y7
    :param var: y8
    :param var: y9
    :param var: y10
    """
    def setup(self):
        # declare inputs with default values
        x1 = self.declare_input('x1', val=2)
        x2 = self.declare_input('x2', val=3)
        x3 = self.declare_input('x3', val=np.arange(7))

        # Elementwise addition
        y1 = x2 + x1
        self.register_output('y1', y1)

        # Elementwise subtraction
        self.register_output('y2', x2 - x1)

        # Elementwise multitplication
        self.register_output('y3', x1 * x2)

        # Elementwise division
        self.register_output('y4', x1 / x2)
        self.register_output('y5', x1 / 3)
        self.register_output('y6', 2 / x2)

        # Elementwise Power
        y8 = x2**2
        self.register_output('y8', y8)
        self.register_output('y7', x1**2)

        # Adding other expressions
        self.register_output('y9', y1 + y8)

        # Array of powers
        y10 = x3**(2 * np.ones(7))
        self.register_output('y10', y10)


class ExampleCycles(Group):
    """
    :param var: cycle_1.x
    :param var: cycle_2.x
    :param var: cycle_3.x
    """
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


class ExampleNoRegisteredOutput(Group):
    """
    :param var: prod
    """
    def setup(self):
        group = Group()
        a = group.declare_input('a', val=2)
        b = group.create_indep_var('b', val=12)
        group.register_output('prod', a * b)
        self.add_subsystem('sys', group, promotes=['*'])

        # These expressions do not lead to constructing any Component
        # objects
        x1 = self.declare_input('x1')
        x2 = self.declare_input('x2')
        y1 = x2 + x1
        y2 = x2 - x1
        y3 = x1 * x2
        y5 = x2**2


class ExampleUnary(Group):
    def setup(self):
        x = self.declare_input('x', val=np.pi)
        y = self.declare_input('y', val=1)
        self.register_output('arccos', ot.arccos(y))
        self.register_output('arcsin', ot.arcsin(y))
        self.register_output('arctan', ot.arctan(x))
        self.register_output('cos', ot.cos(x))
        self.register_output('cosec', ot.cosec(y))
        self.register_output('cosech', ot.cosech(x))
        self.register_output('cosh', ot.cosh(x))
        self.register_output('cotan', ot.cotan(y))
        self.register_output('cotanh', ot.cotanh(x))
        self.register_output('exp', ot.exp(x))
        self.register_output('log', ot.log(x))
        self.register_output('log10', ot.log10(x))
        self.register_output('sec', ot.sec(x))
        self.register_output('sech', ot.sech(x))
        self.register_output('sin', ot.sin(x))
        self.register_output('sinh', ot.sinh(x))
        self.register_output('tan', ot.tan(x))
        self.register_output('tanh', ot.tanh(x))


class ExampleWithSubsystems(Group):
    """
    :param var: prod
    :param var: y1
    :param var: y2
    :param var: y3
    :param var: y4
    :param var: y5
    :param var: y6
    """
    def setup(self):
        # Create independent variable
        x1 = self.create_indep_var('x1', val=40)

        # Powers
        y4 = x1**2

        # Create subsystem that depends on previously created
        # independent variable
        group = Group()
        # This value is overwritten by connection
        a = group.declare_input('x1', val=2)
        b = group.create_indep_var('x2', val=12)
        group.register_output('prod', a * b)
        self.add_subsystem('sys', group, promotes=['*'])

        # declare inputs with default values
        # This value is overwritten by connection
        x2 = self.declare_input('x2', val=3)

        # Simple addition
        y1 = x2 + x1
        self.register_output('y1', y1)

        # Simple subtraction
        self.register_output('y2', x2 - x1)

        # Simple multitplication
        self.register_output('y3', x1 * x2)

        # Powers
        y5 = x2**2

        # register outputs in reverse order to how they are defined
        self.register_output('y5', y5)
        self.register_output('y6', y1 + y5)
        self.register_output('y4', y4)
