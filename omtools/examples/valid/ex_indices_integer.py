from openmdao.api import Problem
from omtools.api import Group
import numpy as np


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


prob = Problem()
prob.model = ExampleInteger()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x', prob['x'].shape)
print(prob['x'])
