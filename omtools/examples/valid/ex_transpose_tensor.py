from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleTensor(Group):
    def setup(self):

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input(
            'Tens',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute the transpose of tens
        self.register_output('tensor_transpose', ot.transpose(tens))


prob = Problem()
prob.model = ExampleTensor()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('Tens', prob['Tens'].shape)
print(prob['Tens'])
print('tensor_transpose', prob['tensor_transpose'].shape)
print(prob['tensor_transpose'])
