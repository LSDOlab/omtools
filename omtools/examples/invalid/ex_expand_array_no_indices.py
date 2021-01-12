from openmdao.api import Problem
import omtools.api as ot
from omtools.api import Group
import numpy as np


class ErrorArrayNoIndices(Group):
    def setup(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_input('array', val=val)
        expanded_array = ot.expand(array, (2, 4, 3, 1))
        self.register_output('expanded_array', expanded_array)


prob = Problem()
prob.model = ErrorArrayNoIndices()
prob.setup(force_alloc_complex=True)
prob.run_model()
