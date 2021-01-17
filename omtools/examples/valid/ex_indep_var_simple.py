from openmdao.api import Problem
from omtools.api import Group
import numpy as np


class ExampleSimple(Group):
    def setup(self):
        z = self.create_indep_var('z', val=10)


prob = Problem()
prob.model = ExampleSimple()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('z', prob['z'].shape)
print(prob['z'])
