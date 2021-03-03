from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMatrix(Group):
    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input(
            'M1',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Compute the transpose of mat
        self.register_output('axes_reordered_matrix',
                             ot.reorder_axes(mat, 'ij->ji'))


prob = Problem()
prob.model = ExampleMatrix()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('M1', prob['M1'].shape)
print(prob['M1'])
print('axes_reordered_matrix', prob['axes_reordered_matrix'].shape)
print(prob['axes_reordered_matrix'])
