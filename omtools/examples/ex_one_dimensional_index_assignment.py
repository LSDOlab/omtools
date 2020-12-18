from omtools.api import Group
from openmdao.api import Problem
import numpy as np


class Example(Group):
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


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
