from openmdao.api import Problem
import omtools.api as ot
from omtools.api import Group
import numpy as np


class ExampleArray2HigherArray(Group):
    def setup(self):
        # Expanding an array into a higher-rank array
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_input('array', val=val)
        expanded_array = ot.expand(array, (2, 4, 3, 1), 'ij->iajb')
        self.register_output('expanded_array', expanded_array)


prob = Problem()
prob.model = ExampleArray2HigherArray()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('array', prob['array'].shape)
print(prob['array'])
print('expanded_array', prob['expanded_array'].shape)
print(prob['expanded_array'])
