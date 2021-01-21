from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group


class ErrorMultidimensionalOutOfRange(Group):
    def setup(self):
        z = self.declare_input('z',
                               shape=(2, 3),
                               val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        # This triggers an error
        x[0:3, 0:3] = z


prob = Problem()
prob.model = ErrorMultidimensionalOutOfRange()
prob.setup(force_alloc_complex=True)
prob.run_model()
