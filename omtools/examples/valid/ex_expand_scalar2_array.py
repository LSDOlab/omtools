from openmdao.api import Problem
import omtools.api as ot
from omtools.api import Group
import numpy as np


class ExampleScalar2Array(Group):
    def setup(self):
        # Expanding a scalar into an array
        scalar = self.declare_input('scalar', val=1.)
        expanded_scalar = ot.expand(scalar, (2, 3))
        self.register_output('expanded_scalar', expanded_scalar)


prob = Problem()
prob.model = ExampleScalar2Array()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('scalar', prob['scalar'].shape)
print(prob['scalar'])
print('expanded_scalar', prob['expanded_scalar'].shape)
print(prob['expanded_scalar'])
