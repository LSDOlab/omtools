from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleTensor(Group):
    def setup(self):

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input(
            'T1',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute a new tensor by reordering axes of tens; reordering is
        # given by 'ijkl->ljki'
        self.register_output('axes_reordered_tensor',
                             ot.reorder_axes(tens, 'ijkl->ljki'))


prob = Problem()
prob.model = ExampleTensor()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('T1', prob['T1'].shape)
print(prob['T1'])
print('axes_reordered_tensor', prob['axes_reordered_tensor'].shape)
print(prob['axes_reordered_tensor'])
