from openmdao.api import Problem
import omtools.api as ot
from omtools.api import Group
import numpy as np


class ErrorScalarIncorrectOrder(Group):
    def setup(self):
        scalar = self.declare_input('scalar', val=1.)
        expanded_scalar = ot.expand((2, 3), scalar)
        self.register_output('expanded_scalar', expanded_scalar)


prob = Problem()
prob.model = ErrorScalarIncorrectOrder()
prob.setup(force_alloc_complex=True)
prob.run_model()
