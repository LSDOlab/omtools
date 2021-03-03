from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group


class ErrorIntegerReuse(Group):
    def setup(self):
        a = self.declare_input('a', val=4)
        b = self.declare_input('b', val=3)
        x = self.create_output('x', shape=(2, ))
        x[0] = a
        x[1] = b
        y = self.create_output('y', shape=(2, ))
        y[0] = x[0]
        y[1] = x[0]


prob = Problem()
prob.model = ErrorIntegerReuse()
prob.setup(force_alloc_complex=True)
prob.run_model()
