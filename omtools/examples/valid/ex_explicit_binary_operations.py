from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import omtools.api as ot
import numpy as np


class ExampleBinaryOperations(Group):
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


prob = Problem()
prob.model = ExampleBinaryOperations()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('y1', prob['y1'].shape)
print(prob['y1'])
print('y2', prob['y2'].shape)
print(prob['y2'])
print('y3', prob['y3'].shape)
print(prob['y3'])
print('y4', prob['y4'].shape)
print(prob['y4'])
print('y5', prob['y5'].shape)
print(prob['y5'])
print('y6', prob['y6'].shape)
print(prob['y6'])
print('y7', prob['y7'].shape)
print(prob['y7'])
print('y8', prob['y8'].shape)
print(prob['y8'])
print('y9', prob['y9'].shape)
print(prob['y9'])
print('y10', prob['y10'].shape)
print(prob['y10'])
