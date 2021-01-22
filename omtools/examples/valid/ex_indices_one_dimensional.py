from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group


class ExampleOneDimensional(Group):
    def setup(self):
        n = 20
        u = self.declare_input('u',
                               shape=(n, ),
                               val=np.arange(n).reshape((n, )))
        v = self.declare_input('v',
                               shape=(n - 4, ),
                               val=np.arange(n - 4).reshape((n - 4, )))
        w = self.declare_input('w',
                               shape=(4, ),
                               val=16 + np.arange(4).reshape((4, )))
        x = self.create_output('x', shape=(n, ))
        x[0:n] = 2 * (u + 1)
        y = self.create_output('y', shape=(n, ))
        y[0:n - 4] = 2 * (v + 1)
        y[n - 4:n] = w - 3

        # Get value from indices
        z = self.create_output('z', shape=(3, ))
        z[0:3] = ot.expand(x[2], (3, ))
        self.register_output('x0_5', x[0:5])
        self.register_output('x3_', x[3:])
        self.register_output('x2_4', x[2:4])


prob = Problem()
prob.model = ExampleOneDimensional()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x', prob['x'].shape)
print(prob['x'])
print('y', prob['y'].shape)
print(prob['y'])
print('z', prob['z'].shape)
print(prob['z'])
print('x0_5', prob['x0_5'].shape)
print(prob['x0_5'])
print('x3_', prob['x3_'].shape)
print(prob['x3_'])
print('x2_4', prob['x2_4'].shape)
print(prob['x2_4'])
