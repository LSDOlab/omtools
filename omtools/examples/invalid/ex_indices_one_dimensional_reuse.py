from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group


class ErrorOneDimensionalReuse(Group):
    def setup(self):
        n = 8
        u = self.declare_input('u',
                               shape=(n, ),
                               val=np.arange(n).reshape((n, )))
        v = self.create_output('v', shape=(n, ))
        v[:4] = u[:4]
        v[4:] = u[:4]


prob = Problem()
prob.model = ErrorOneDimensionalReuse()
prob.setup(force_alloc_complex=True)
prob.run_model()
