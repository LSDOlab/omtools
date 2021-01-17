from openmdao.api import Problem
import omtools.api as ot
from omtools.api import Group
import numpy as np


class ErrorScalarIndices(Group):
    def setup(self):
        scalar = self.declare_input('scalar', val=1.)
        expanded_scalar = ot.expand(scalar, (2, 3), '->ij')
        self.register_output('expanded_scalar', expanded_scalar)


prob = Problem()
prob.model = ErrorScalarIndices()
prob.setup(force_alloc_complex=True)
prob.run_model()
