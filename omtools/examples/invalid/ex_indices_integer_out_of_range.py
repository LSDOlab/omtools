from openmdao.api import Problem
from omtools.api import Group
import numpy as np


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
