import omtools.api as ot
from omtools.api import Group
from openmdao.api import Problem
import numpy as np


class Example(Group):
    def setup(self):
        # Test scalar expansion
        scalar = self.declare_input('scalar', val=1.)
        expanded_scalar = ot.expand(scalar, (2, 3))
        self.register_output('expanded_scalar', expanded_scalar)

        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_input('array', val=val)
        expanded_array = ot.expand(array, (2, 4, 3, 1), [1, 3])
        self.register_output('expanded_array', expanded_array)


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('scalar', prob['scalar'].shape)
print(prob['scalar'])
print('expanded_scalar', prob['expanded_scalar'].shape)
print(prob['expanded_scalar'])

print()

print('array', prob['array'].shape)
print(prob['array'])
print('expanded_array', prob['expanded_array'].shape)
print(prob['expanded_array'])