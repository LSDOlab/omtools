from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMultipleVector(Group):
    def setup(self):
        n = 3

        # Declare a vector of length 3 as input
        v1 = self.declare_input('v1', val=np.arange(n))

        # Declare another vector of length 3 as input
        v2 = self.declare_input('v2', val=np.arange(n, 2 * n))

        # Output the elementwise average of vectors v1 and v2
        self.register_output('multiple_vector_average', ot.average(v1, v2))


prob = Problem()
prob.model = ExampleMultipleVector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('v1', prob['v1'].shape)
print(prob['v1'])
print('v2', prob['v2'].shape)
print(prob['v2'])
print('multiple_vector_average', prob['multiple_vector_average'].shape)
print(prob['multiple_vector_average'])
