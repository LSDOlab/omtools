from omtools.api import Group
from openmdao.api import Problem
import numpy as np


class Example(Group):
    def setup(self):
        a = self.declare_input('a', val=0)
        x = self.create_output('x', shape=(1, ))
        # This triggers an error
        x[1] = a


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
