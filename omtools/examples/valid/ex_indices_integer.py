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

        # get value from indices
        z = self.create_output('z', shape=(3, ))
        z[0:3] = ot.expand(x[2], (3, ))
        self.register_output('x0', x[0])
        self.register_output('x0_5', x[0:5])
        self.register_output('x3_', x[3:])
        self.register_output('x6', x[6])
        self.register_output('x2_4', x[2:4])


prob = Problem()
prob.model = ExampleInteger()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x', prob['x'].shape)
print(prob['x'])
print('x0', prob['x0'].shape)
print(prob['x0'])
print('x0_5', prob['x0_5'].shape)
print(prob['x0_5'])
print('x3_', prob['x3_'].shape)
print(prob['x3_'])
print('x6', prob['x6'].shape)
print(prob['x6'])
print('x2_4', prob['x2_4'].shape)
print(prob['x2_4'])
print('z', prob['z'].shape)
print(prob['z'])
