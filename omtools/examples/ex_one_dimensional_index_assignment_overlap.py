from omtools.api import Group
from openmdao.api import Problem
import numpy as np


class Example(Group):
    def setup(self):
        n = 20
        x = self.declare_input('x',
                               shape=(n - 4, ),
                               val=np.arange(n - 4).reshape((n - 4, )))
        y = self.declare_input('y',
                               shape=(4, ),
                               val=16 + np.arange(4).reshape((4, )))
        z = self.create_output('z', shape=(n, ))
        z[0:n - 4] = 2 * (x + 1)
        # This triggers an error
        z[n - 5:n - 1] = y - 3


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
