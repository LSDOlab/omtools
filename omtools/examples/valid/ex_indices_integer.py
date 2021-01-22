from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group


class ExampleInteger(Group):
    def setup(self):
        a = self.declare_input('a', val=0)
        b = self.declare_input('b', val=1)
        c = self.declare_input('c', val=2)
        d = self.declare_input('d', val=7.4)
        e = self.declare_input('e', val=np.pi)
        f = self.declare_input('f', val=9)
        g = e + f
        x = self.create_output('x', shape=(7, ))
        x[0] = a
        x[1] = b
        x[2] = c
        x[3] = d
        x[4] = e
        x[5] = f
        x[6] = g

        # Get value from indices
        self.register_output('x0', x[0])
        self.register_output('x6', x[6])


prob = Problem()
prob.model = ExampleInteger()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x', prob['x'].shape)
print(prob['x'])
print('x0', prob['x0'].shape)
print(prob['x0'])
print('x6', prob['x6'].shape)
print(prob['x6'])
