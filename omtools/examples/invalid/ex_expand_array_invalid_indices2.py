from openmdao.api import Problem
import omtools.api as ot
from omtools.api import Group
import numpy as np


class ErrorArrayInvalidIndices2(Group):
    def setup(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_input('array', val=val)
        expanded_array = ot.expand(array, (2, 4, 3, 1), 'ij->ijab')
        self.register_output('expanded_array', expanded_array)


prob = Problem()
prob.model = ErrorArrayInvalidIndices2()
prob.setup(force_alloc_complex=True)
prob.run_model()
