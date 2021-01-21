from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group


class ErrorIntegerOutOfRange(Group):
    def setup(self):
        a = self.declare_input('a', val=0)
        x = self.create_output('x', shape=(1, ))
        # This triggers an error
        x[1] = a


prob = Problem()
prob.model = ErrorIntegerOutOfRange()
prob.setup(force_alloc_complex=True)
prob.run_model()
