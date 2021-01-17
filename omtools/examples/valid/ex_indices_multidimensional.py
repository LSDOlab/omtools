from openmdao.api import Problem
from omtools.api import Group
import numpy as np


class ExampleMultidimensional(Group):
    def setup(self):
        # Works with two dimensional arrays
        z = self.declare_input('z',
                               shape=(2, 3),
                               val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        x[0:2, 0:3] = z

        # Also works with higher dimensional arrays
        p = self.declare_input('p',
                               shape=(5, 2, 3),
                               val=np.arange(30).reshape((5, 2, 3)))
        q = self.create_output('q', shape=(5, 2, 3))
        q[0:5, 0:2, 0:3] = p


prob = Problem()
prob.model = ExampleMultidimensional()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x', prob['x'].shape)
print(prob['x'])
print('q', prob['q'].shape)
print(prob['q'])
