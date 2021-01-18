from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleReorderAxes(Group):
    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input(
            'M1',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input(
            'T1',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute the transpose of mat
        self.register_output('axes_reordered_matrix',
                             ot.reorder_axes(mat, 'ij->ji'))

        # Compute a new tensor by reordering axes of tens; reordering is
        # given by 'ijkl->ljki'
        self.register_output('axes_reordered_tensor',
                             ot.reorder_axes(tens, 'ijkl->ljki'))


prob = Problem()
prob.model = ExampleReorderAxes()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('axes_reordered_matrix', prob['axes_reordered_matrix'].shape)
print(prob['axes_reordered_matrix'])
print('axes_reordered_tensor', prob['axes_reordered_tensor'].shape)
print(prob['axes_reordered_tensor'])
