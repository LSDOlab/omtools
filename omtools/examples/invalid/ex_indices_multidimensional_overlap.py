from openmdao.api import Problem
from omtools.api import Group
import numpy as np


class ErrorMultidimensionalOverlap(Group):
    def setup(self):
        z = self.declare_input('z',
                               shape=(2, 3),
                               val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        x[0:2, 0:3] = z
        # This triggers an error
        x[0:2, 0:3] = z


prob = Problem()
prob.model = ErrorMultidimensionalOverlap()
prob.setup(force_alloc_complex=True)
prob.run_model()
