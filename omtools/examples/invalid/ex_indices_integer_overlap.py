from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group


class ErrorIntegerOverlap(Group):
    def setup(self):
        a = self.declare_input('a', val=0)
        b = self.declare_input('b', val=1)
        x = self.create_output('x', shape=(2, ))
        x[0] = a
        # This triggers an error
        x[0] = b


prob = Problem()
prob.model = ErrorIntegerOverlap()
prob.setup(force_alloc_complex=True)
prob.run_model()
